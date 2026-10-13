# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XAttention-style antidiagonal block scoring (Xu et al., ICML 2025).

Training-free page-scoring rule adapted to this repository's InfLLMv2
block-sparse contract: 64-token pages, Top-K page selection, and the
unchanged Stage-2 block-sparse kernels. Only the *scoring rule* changes
relative to the native Sparse (actual-query) selector: a page pair's
importance is the softmax-normalized mass on its within-page
antidiagonals instead of the page max-pool of token scores.

Reference: "XAttention: Block Sparse Attention with Antidiagonal
Scoring" (ICML 2025), Algorithm 1 and the find_blocks formula. For a
query page of block size B and key page p, the antidiagonal pairs are
(row r, col c) with r + c = B - 1, and the pair's score is the sum of
softmax probabilities on those entries. The paper's stride-S row gather
is a permutation of all rows (the S phases cover every residue), so it
does not change which pairs exist; it enters here only through the
sqrt(d_h * S) scale, as in Algorithm 1 line 12.

Implementation contract (performance-matched to the other baseline
scorers): pass A (softmax statistics) is batched bmm over key chunks —
the same O(L^2 d) matmul work the native Sparse selector performs —
and pass B (antidiagonal extraction) is one bmm per query-page tile
plus a flat-index block-diagonal gather, replacing the earlier
per-page-pair Python loop (~2M iterations at 128K) with a loop over
query-page tiles. All products run in fp32; the pass-B bmm is chunked
over the key width so its transient score matrix stays within a 2 GiB
budget (~34 GB unchunked at 128K).

Memory strategy: peak transient memory is
O(n_kv * L * key_chunk) for pass A (online rescale) and
O(n_heads * B * L) for pass B (one bmm row-block), never the
(n_heads, L, L) grid.

The repository's shared top-k contract is page-causal: query row i (in
query page i // B) may select key pages p <= i // B, and the softmax key
range for a query page's rows ends at that page's last row. The packed
output right-aligns each batch item to max_seqlen_q, matching the q_idx
grid that `compressed_attention` masks against.
"""

from __future__ import annotations

import torch

__all__ = [
    "xattention_page_scores",
    "xattention_token_scores_packed",
    "xattention_token_scores_padded",
    "xattention_token_scores_kv_packed",
]

_KEY_CHUNK = 16384
_Q_PAGE_TILE = 32


def _softmax_stats(
    q_tile: torch.Tensor,
    k: torch.Tensor,
    k_visible: torch.Tensor,
    scale: float,
    key_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online (max, sumexp) over each row's causal key range.

    Args:
        q_tile: (n_heads, rows, d) queries of a tile of query pages.
        k: (n_heads, k_len, d) keys already mapped to query heads.
        k_visible: (rows,) causal key count per row (page-causal
            contract: row r's page ends at k_visible of the LAST row in
            its page; rows within one page share their page's range).
        scale: 1 / sqrt(d_h * S).
        key_chunk: key chunk size for the streaming pass.

    Returns:
        (row_max, row_sumexp), each (n_heads, rows), float32.
    """
    n_heads, rows, _ = q_tile.shape
    row_max = torch.full(
        (n_heads, rows), float("-inf"), device=q_tile.device, dtype=torch.float32
    )
    row_sum = torch.zeros((n_heads, rows), device=q_tile.device, dtype=torch.float32)
    k_max = int(k_visible.max().item()) if rows else 0
    for c0 in range(0, k_max, key_chunk):
        c1 = min(c0 + key_chunk, k_max)
        logits = torch.bmm(q_tile.float(), k[:, c0:c1].float().transpose(1, 2)) * scale
        # Mask keys beyond each row's own page-causal frontier.
        visible = (
            torch.arange(c0, c1, device=q_tile.device)[None, :] < k_visible[:, None]
        )
        logits = logits.masked_fill(~visible[None], float("-inf"))
        chunk_max = logits.amax(dim=-1)
        new_max = torch.maximum(row_max, chunk_max)
        rescale = torch.exp(row_max - new_max)
        row_sum = row_sum * rescale + torch.exp(logits - new_max[..., None]).sum(dim=-1)
        row_max = new_max
    return row_max, row_sum


def _reverse_pages(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Reverse rows within every page: (n_heads, L, d) -> same shape.

    After this transform, the antidiagonal pairing (row r of query page
    p with column p*B + B-1-r of key page p') becomes the block
    diagonal: reversed query row i' pairs with reversed key column
    p'*B + i'.
    """
    n_heads, L, d = x.shape
    n_pages = (L + block_size - 1) // block_size
    pad = n_pages * block_size - L
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    x = x.reshape(n_heads, n_pages, block_size, d).flip(dims=[2])
    return x.reshape(n_heads, n_pages * block_size, d)[..., :L, :]


def xattention_page_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_len: int,
    k_len: int,
    block_size: int,
    stride: int,
    key_chunk: int = _KEY_CHUNK,
    q_page_tile: int = _Q_PAGE_TILE,
) -> torch.Tensor:
    """Antidiagonal page scores for one sequence.

    Args:
        q: (q_len, n_heads, d) queries for one sequence (RoPE-applied,
            the same tensors the native Sparse selector scores).
        k: (k_len, n_kv, d) keys for the same sequence; n_kv must divide
            n_heads (GQA groups are expanded via the HF head mapping).
        q_len, k_len: sequence lengths; must match q/k shapes.
        block_size: InfLLMv2 page size (64).
        stride: antidiagonal stride S (the Algorithm 1 scale factor).
        key_chunk: streaming key chunk for the softmax statistics pass.
        q_page_tile: query pages processed per bmm in pass B.

    Returns:
        (n_heads, q_len, n_pages) float32 page scores; entry (h, i, p)
        is the XAttention importance of key page p for query row i.
        Pages beyond the page-causal frontier (p > i // block_size)
        carry -inf so the shared top-k masking stays exact.
    """
    if int(q.shape[0]) != q_len or int(k.shape[0]) != k_len:
        raise ValueError(
            f"q/k leading dims do not match q_len/k_len: "
            f"{tuple(q.shape)} vs {q_len}, {tuple(k.shape)} vs {k_len}"
        )
    n_heads = int(q.shape[1])
    n_kv = int(k.shape[1])
    head_dim = int(q.shape[2])
    if n_kv > n_heads or n_heads % n_kv != 0:
        raise ValueError(f"k head count {n_kv} must divide q head count {n_heads}")
    if int(k.shape[2]) != head_dim:
        raise ValueError("q and k head_dim mismatch")
    if block_size <= 0 or stride <= 0:
        raise ValueError("block_size and stride must be positive")

    n_pages = max((k_len + block_size - 1) // block_size, 1)
    q_pages = max((q_len + block_size - 1) // block_size, 1)
    device = q.device
    scale = 1.0 / float(head_dim * stride) ** 0.5

    q_t = q.permute(1, 0, 2)  # (n_heads, q_len, d)
    k_t = k.permute(1, 0, 2)  # (n_kv, k_len, d)
    if n_kv != n_heads:
        # HF GQA mapping: query head h attends with key head h // group.
        group = n_heads // n_kv
        h2kv = torch.arange(n_heads, device=device) // group
        k_t = k_t.index_select(0, h2kv)  # (n_heads, k_len, d)

    # Reverse rows within the KEYS' pages: the antidiagonal pairing
    # (q row i with key column p_k*B + B-1-i) becomes the block diagonal
    # of q @ k_rev^T.
    k_rev = _reverse_pages(k_t, block_size).float()

    # Page-causal key range per query row: row i's page is i // B; its
    # last row is min((i//B + 1) * B, q_len) - 1, and the shared
    # page-causal contract lets that page see keys up to its last row.
    row_page = torch.arange(q_len, device=device) // block_size
    # A query page's causal key range ends at its last actual row:
    # min((page+1)*B, q_len), matching the reference's r1. Clamp to
    # k_len for the shorter-key case.
    k_visible_rows = torch.clamp((row_page + 1) * block_size, max=min(q_len, k_len))

    page_scores = torch.full(
        (n_heads, q_pages, n_pages),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )

    # The ragged tail page (q_len not a multiple of B) is handled by
    # padding the query to full pages: padding rows are masked out of
    # the softmax stats (they see no keys) and out of the page-score
    # sum, so they contribute exactly zero.
    q_padded_pages = q_pages
    q_pad = q_padded_pages * block_size - q_len
    q_t_padded = q_t.float()
    if q_pad:
        q_t_padded = torch.nn.functional.pad(q_t_padded, (0, 0, 0, q_pad))
    # Rows' page-causal key counts, with padding rows seeing nothing.
    kv_rows = torch.zeros(q_padded_pages * block_size, device=device)
    kv_rows[:q_len] = k_visible_rows
    # Padding rows: k_visible = 0 -> softmax over empty set. Represent
    # as -inf max / sum=epsilon so exp(diag - (-inf)) = 0.
    empty = kv_rows == 0

    # Keys reversed within pages, padded to the page grid once.
    k_pad_full = n_pages * block_size
    k_pad_t = k_rev
    if k_pad_full > k_len:
        k_pad_t = torch.nn.functional.pad(k_rev, (0, 0, 0, k_pad_full - k_len))

    # Adaptive tile sizing: keep pass-A's transient logits (n_heads *
    # tile_rows * key_chunk * 4B) within ~1 GiB so the scorer coexists
    # with the resident model and KV cache on one card.
    stats_budget = 1 << 30
    _rows = q_page_tile * block_size
    _chunk = key_chunk
    while n_heads * _rows * _chunk * 4 > stats_budget and _rows > block_size:
        _rows //= 2
    while n_heads * _rows * _chunk * 4 > stats_budget and _chunk > 1024:
        _chunk //= 2
    eff_tile = max(_rows // block_size, 1)

    for t0 in range(0, q_padded_pages, eff_tile):
        t1 = min(t0 + eff_tile, q_padded_pages)
        rows = slice(t0 * block_size, t1 * block_size)
        q_tile = q_t_padded[:, rows]  # (n_heads, p_tile*B, d)
        row_max, row_sum = _softmax_stats(
            q_tile,
            k_t.float(),
            kv_rows[rows],
            scale,
            _chunk,
        )

        # Restrict the bmm to causally visible key pages for the tile's
        # LAST page (max page index in the tile).
        last_page = t1 - 1
        visible_pages = min(last_page + 1, n_pages)
        i_idx = torch.arange(block_size, device=device)
        pk_idx = torch.arange(visible_pages, device=device)
        # Only the KEYS are reversed within pages: q[i] pairs with
        # k_rev[p_k*B + i] = k[p_k*B + B-1-i] — the antidiagonal. Gather
        # per key-width chunk directly into the (n_heads, p_tile, vp, B)
        # diag tensor (0.5 GiB at 128K) instead of materializing the full
        # (n_heads, tile_rows, k_width) score matrix (~34 GiB at 128K):
        # the block-diagonal gather never needs the off-diagonal majority.
        p_tile = t1 - t0
        diag = torch.empty(
            n_heads,
            p_tile,
            visible_pages,
            block_size,
            device=device,
            dtype=torch.float32,
        )
        chunk_pages = max((1 << 30) // max(n_heads * p_tile * block_size * 4, 1), 1)
        for pc0 in range(0, visible_pages, chunk_pages):
            pc1 = min(pc0 + chunk_pages, visible_pages)
            part = (
                torch.bmm(
                    q_t_padded[:, rows],
                    k_pad_t[:, pc0 * block_size : pc1 * block_size].transpose(1, 2),
                )
                * scale
            )  # (n_heads, tile_rows, (pc1-pc0)*B)
            part = part.reshape(n_heads, p_tile, block_size, pc1 - pc0, block_size)
            # (i, local pk) diagonal within the chunk.
            ip = torch.arange(block_size, device=device)
            g = part[:, :, ip, :, ip]  # advanced indexing: (B, n_heads, p_tile, cp)
            diag[:, :, pc0:pc1, :] = g.permute(1, 2, 3, 0)

        # Pair validity, exactly the reference keep filter: pair (i, p_k)
        # contributes iff the paired column p_k*B + B-1-i is strictly
        # below the query page's causal frontier min((p_q+1)*B, q_len,
        # k_len) AND the row is below the key page's width
        # ell(p_k) = min(B, k_len - p_k*B). Covers k_len != q_len.
        ell = torch.clamp(k_len - pk_idx * block_size, min=0, max=block_size)
        pq_front = torch.clamp(
            (torch.arange(t0, t1, device=device) + 1) * block_size,
            max=min(q_len, k_len),
        )
        col_pos = (
            pk_idx[None, None, :] * block_size + (block_size - 1) - i_idx[None, :, None]
        )  # (1, B, vp)
        invalid = (col_pos >= pq_front[:, None, None]) | (
            i_idx[None, :, None] >= ell[None, None, :]
        )
        diag = diag.masked_fill(invalid.permute(0, 2, 1)[None], float("-inf"))

        # Convert to probabilities with each row's online stats.
        rm = row_max.reshape(n_heads, p_tile, block_size)
        rs = row_sum.reshape(n_heads, p_tile, block_size)
        # Guard empty rows (padding): their stats are -inf/0; use a safe
        # exponent so they contribute zero.
        safe_rm = torch.where(torch.isinf(rm), torch.zeros_like(rm), rm)
        probs = (
            torch.exp(diag - safe_rm[..., None, :])
            / torch.where(rs == 0, torch.ones_like(rs), rs)[..., None, :]
        )
        # Zero out padding rows' contributions.
        row_empty = empty[rows].reshape(p_tile, block_size)
        probs = probs.masked_fill(row_empty[None, :, None, :], 0.0)
        tile_scores = probs.sum(dim=-1)  # (n_heads, p_tile, visible_pages)
        # Causal frontier per query page p_q: key pages p <= p_q.
        pq_idx = torch.arange(t0, t1, device=device)[:, None]
        tile_scores = tile_scores.masked_fill(
            pk_idx[None, :] > pq_idx[None, :, :], float("-inf")
        )
        page_scores[:, t0:t1, :visible_pages] = tile_scores

    # GQA-reduce on the page grid before the row expansion: the expanded
    # (n_heads, q_len, n_pages) would be ~34 GB at 128K fp32, while the
    # reduced (n_kv, ...) form is what all consumers (tf_page_topk, the
    # token broadcast) actually use. The per-KV-head mean matches the HF
    # group convention used everywhere else in this scorer.
    reduce_grid = n_heads * q_len * n_pages * 4 > (1 << 30)
    n_kv_in = int(k.shape[1])
    if reduce_grid and n_kv_in != n_heads:
        group = n_heads // n_kv_in
        page_scores = page_scores.reshape(n_kv_in, group, q_pages, n_pages).mean(dim=1)

    # Expand per query row: row i inherits its query page's score vector.
    row_page_clamped = torch.clamp(
        torch.arange(q_len, device=device) // block_size, max=q_pages - 1
    )
    result = page_scores[:, row_page_clamped, :]  # (n_heads, q_len, n_pages)

    # Page-level causal mask matching the shared top-k contract: row i
    # lives in query page i // B and may select key pages p <= i // B.
    row_page_idx = torch.arange(q_len, device=device)[:, None] // block_size
    key_page_idx = torch.arange(n_pages, device=device)[None, :]
    result = result.masked_fill(key_page_idx > row_page_idx, float("-inf"))
    return result


def _page_scores_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: int,
    stride: int,
) -> torch.Tensor:
    """Packed varlen page scores: right-aligned rows, loop over items.

    Args:
        q: (total_q, n_heads, head_dim) unpad queries.
        k: (total_k, n_kv, head_dim) unpad keys.
        cu_seqlens_q, cu_seqlens_k: cumulative lengths per item.
        max_seqlen_q, max_seqlen_k: padded lengths used for the shared
            right-aligned row-to-page mapping.

    Returns:
        (n_heads, total_q, n_pages) float32 page scores, -inf beyond the
        page-causal frontier; n_pages = ceil(max_seqlen_k / block_size).
    """
    batch = int(cu_seqlens_q.numel()) - 1
    n_heads = int(q.shape[1])
    n_pages = max((int(max_seqlen_k) + block_size - 1) // block_size, 1)
    device = q.device
    if int(q.shape[0]) != int(cu_seqlens_q[-1].item()) - int(cu_seqlens_q[0].item()):
        raise NotImplementedError(
            "packed q rows differ from cu_seqlens span; right-aligned or "
            "offset packed layouts are unsupported."
        )
    result = torch.full(
        (n_heads, int(q.shape[0]), n_pages),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )
    for b in range(batch):
        qs = int(cu_seqlens_q[b].item())
        qe = int(cu_seqlens_q[b + 1].item())
        ks = int(cu_seqlens_k[b].item())
        ke = int(cu_seqlens_k[b + 1].item())
        q_len = qe - qs
        k_len = ke - ks
        if q_len <= 0 or k_len <= 0:
            result[:, qs:qe, :].fill_(float("-inf"))
            continue
        out = xattention_page_scores(
            q[qs:qe],
            k[ks:ke],
            q_len=q_len,
            k_len=k_len,
            block_size=block_size,
            stride=stride,
        )
        # Right-align the key grid so every batch item spans the same
        # n_pages as the padded contract; the shared q_idx mask governs
        # the final page visibility.
        n_pages_b = out.shape[-1]
        if n_pages_b < n_pages:
            pad = torch.full(
                (n_heads, q_len, n_pages - n_pages_b),
                float("-inf"),
                device=device,
                dtype=out.dtype,
            )
            out = torch.cat([out, pad], dim=-1)
        result[:, qs:qe, :].copy_(out)
    return result


def xattention_token_scores_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: int,
    stride: int,
) -> torch.Tensor:
    """Token-level scores on the shared precomputed contract (batch-1).

    Broadcasts page scores (from `_page_scores_packed`) to token slots:
    exactly max_seqlen_k columns, with the GQA reduction to KV heads on
    the page grid before the broadcast.

    Shape: (n_kv, total_q, max_seqlen_k) — batch-1 single-sequence
    inputs only.
    """
    if int(cu_seqlens_q.numel()) - 1 != 1 or int(cu_seqlens_k.numel()) - 1 != 1:
        raise NotImplementedError(
            "XAttention prefill scoring is verified for single-item (batch-1) "
            "packed inputs only; multi-item packed prefill needs a GPU check "
            "of the shared pooling kernel's precomputed-score grid layout."
        )
    n_heads = int(q.shape[1])
    n_kv = int(k.shape[1])
    group = 1
    if n_kv != n_heads:
        if n_heads % n_kv != 0:
            raise ValueError(f"n_heads {n_heads} not divisible by n_kv {n_kv}")
        group = n_heads // n_kv
    page = _page_scores_packed(
        q,
        k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_size=block_size,
        stride=stride,
    )  # (n_heads, total_q, n_pages)
    if group > 1:
        # HF convention: query head h belongs to KV group h // group.
        grouped = page.reshape(n_kv, group, page.shape[1], page.shape[2])
        page = grouped.mean(dim=1)  # (n_kv, total_q, n_pages)
    token = page.repeat_interleave(block_size, dim=-1)
    token = token[:, :, : int(max_seqlen_k)]
    return token.to(q.device)


def xattention_token_scores_padded(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: int,
    stride: int,
) -> torch.Tensor:
    """Token-level scores for the shared compressed_attention contract.

    Same output as `xattention_token_scores_packed` (kept for API
    compatibility): page scores broadcast to token slots after the GQA
    reduction; (n_kv, total_q, max_seqlen_k), batch-1 only.
    """
    return xattention_token_scores_packed(
        q,
        k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_size=block_size,
        stride=stride,
    )


def xattention_token_scores_kv_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: int,
    stride: int,
) -> torch.Tensor:
    """Per-KV-head token scores matching the shared Stage-1 contract.

    `compressed_attention` and `max_pooling_1d_varlen` consume scores
    with one row per KV head (the GQA group shares one page selection).
    Delegates to `xattention_token_scores_padded`, which reduces GQA
    groups on the page grid before broadcasting to token slots.

    GQA note: the reduction is a per-group *mean* of the antidiagonal
    softmax masses. InfiniGen's token-score kernel uses a pre-softmax
    *sum* across the GQA group; both are per-KV-head constant factors
    for the per-row page top-k (ranking-invariant), but the two
    training-free rows use different GQA conventions — recorded in the
    integration doc for the paper's comparability appendix.

    Args:
        q: (total_q, n_heads, head_dim) unpad queries.
        k: (total_k, n_kv, head_dim) unpad keys.

    Returns:
        (n_kv, total_q, max_seqlen_k) token-level scores.
    """
    return xattention_token_scores_padded(
        q,
        k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_size=block_size,
        stride=stride,
    )
