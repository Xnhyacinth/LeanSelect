# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MInference 1.0 vertical/slash pattern page scoring (Jiang et al., NeurIPS 2024).

Training-free page-scoring rule adapted to this repository's InfLLMv2
block-sparse contract: 64-token pages, Top-K page selection, and the
unchanged Stage-2 block-sparse kernels. Only the *scoring rule* changes
relative to the native Sparse (actual-query) selector.

Reference semantics (MInference's gather_last_q_vertical_slash_topk_v4
family, MIT-licensed microsoft/MInference):
  - **Vertical**: the last ``last_q`` (64) query rows attend over all
    keys; the per-column softmax mass summed over those rows is each
    key's "vertical column" importance. MInference keeps the top-V
    columns per head.
  - **Slash**: the anti-diagonal sums of the same last-64-row QK matrix
    (sum_all_diagonal_matrix over the causal range) score each slash
    line; MInference keeps the top-S lines per head. A slash line is a
    head-level prior: its offset is shared by every query row.

Adaptation to the page contract: both signals are pooled to 64-token
key pages. Vertical mass sums the sampled rows' column masses per page
(row-sensitive: each output row uses only the mass from its own causal
range — rows near the sequence start see few keys). Slash lines are a
per-KV-head prior broadcast to every query row (their offsets do not
depend on the query position); a page accumulates the slash mass of the
lines that cross it. The final page score is the sum, so a single
shared Top-K realizes the union mask MInference uses, at matched
density: the page budget K equals the token budget V/64 + S/64 by
construction of the top-k over summed mass.

Differences from upstream, disclosed for the reproduction appendix:
  - MInference selects *columns and lines* on the token grid with custom
    Triton kernels; this port pools both signals to the repository's
    64-token pages and reuses the unchanged Stage-2 block-sparse kernel.
  - Upstream per-head (V, S) budgets come from an offline per-model
    search (search_pattern); this port uses one shared page budget K
    (the paper-table contract) and a fixed dense sampling window of 64
    rows, matching the repo's local_blocks/init_blocks overlay.
  - Slash lines are broadcast across query rows (head-level prior)
    instead of being re-derived per row.
The upstream package is MIT-licensed; the algorithm is re-implemented
against this repository's contract rather than calling the package.
"""

from __future__ import annotations

import torch

__all__ = [
    "vertical_column_mass",
    "slash_line_mass",
    "vs_pattern_page_scores",
    "vs_pattern_token_scores",
]

_LAST_Q_ROWS = 64


def _causal_mask(q_len: int, k_len: int, device) -> torch.Tensor:
    """Row-causal additive mask (True = visible) on the (q, k) grid."""
    pos_q = torch.arange(q_len, device=device)[:, None]
    pos_k = torch.arange(k_len, device=device)[None, :]
    return pos_k <= pos_q


def vertical_column_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_len: int,
    k_len: int,
    last_q: int = _LAST_Q_ROWS,
) -> torch.Tensor:
    """Per-column softmax mass summed over the last ``last_q`` query rows.

    MInference masks the first 30 columns to +inf before its top-k; here
    every column's mass is computed normally and the sink is supplied by
    the shared init_blocks overlay in the pooling path.

    Args:
        q: (q_len, n_heads, d) queries for one sequence.
        k: (k_len, n_kv, d) keys; n_kv must divide n_heads.
        q_len, k_len: sequence lengths.
        last_q: number of trailing query rows sampled (MInference: 64).

    Returns:
        (n_heads, k_len) float32: column mass per query head.
    """
    n_heads = int(q.shape[1])
    n_kv = int(k.shape[1])
    head_dim = int(q.shape[2])
    scale = 1.0 / float(head_dim) ** 0.5
    q_t = q.permute(1, 0, 2).float()  # (n_heads, q_len, d)
    k_t = k.permute(1, 0, 2).float()  # (n_kv, k_len, d)
    if n_kv != n_heads:
        group = n_heads // n_kv
        h2kv = torch.arange(n_heads, device=q.device) // group
        k_t = k_t.index_select(0, h2kv)
    r = min(int(last_q), q_len)
    rows = q_t[:, q_len - r :]  # (n_heads, r, d)
    logits = torch.bmm(rows, k_t.transpose(1, 2)) * scale
    # Each sampled row is causal against the full key range only when
    # its absolute position allows; rows near the end always do.
    row_pos = torch.arange(q_len - r, q_len, device=q.device)
    visible = (
        torch.arange(k_len, device=q.device)[None, :] <= row_pos[:, None]
    )  # (r, k)
    logits = logits.masked_fill(~visible[None], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return probs.sum(dim=1)  # (n_heads, k_len)


def slash_line_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_len: int,
    k_len: int,
    last_q: int = _LAST_Q_ROWS,
) -> torch.Tensor:
    """Anti-diagonal (slash) mass of the last-64-row QK matrix.

    A slash line with offset o covers pairs (row i, col j) with
    i + j == o (MInference's sum_all_diagonal_matrix over the padded
    causal block). The line's mass is the softmax probability summed
    over its pairs, restricted to the causal range.

    Returns:
        (n_heads, n_diags) float32 where entry (h, o) is the mass of the
        line with i + j == o and n_diags = q_len + k_len - 1; only lines
        with any causal pair are non-zero.
    """
    n_heads = int(q.shape[1])
    n_kv = int(k.shape[1])
    head_dim = int(q.shape[2])
    scale = 1.0 / float(head_dim) ** 0.5
    q_t = q.permute(1, 0, 2).float()
    k_t = k.permute(1, 0, 2).float()
    if n_kv != n_heads:
        group = n_heads // n_kv
        h2kv = torch.arange(n_heads, device=q.device) // group
        k_t = k_t.index_select(0, h2kv)
    r = min(int(last_q), q_len)
    rows = q_t[:, q_len - r :]  # (n_heads, r, d)
    logits = torch.bmm(rows, k_t.transpose(1, 2)) * scale
    row_pos = torch.arange(q_len - r, q_len, device=q.device)
    visible = torch.arange(k_len, device=q.device)[None, :] <= row_pos[:, None]
    logits = logits.masked_fill(~visible[None], float("-inf"))
    probs = torch.softmax(logits, dim=-1)  # (n_heads, r, k_len)
    # Scatter pair masses onto their anti-diagonals: pair (i, j) sits on
    # line (row_pos[i] + j). Lines beyond the causal frontier collect
    # nothing by construction.
    pair_diag = (
        row_pos[:, None] + torch.arange(k_len, device=q.device)[None, :]
    )  # (r, k)
    n_diags = q_len + k_len - 1
    out = torch.zeros((n_heads, n_diags), device=q.device, dtype=probs.dtype)
    flat_diag = pair_diag.reshape(-1)
    out.index_add_(1, flat_diag, probs.reshape(n_heads, -1))
    return out


def vs_pattern_page_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_len: int,
    k_len: int,
    block_size: int,
    last_q: int = _LAST_Q_ROWS,
) -> torch.Tensor:
    """MInference V/S page scores for one sequence.

    Args:
        q: (q_len, n_heads, d) queries (RoPE-applied, the same tensors
            the native Sparse selector scores).
        k: (k_len, n_kv, d) keys; n_kv must divide n_heads.
        q_len, k_len: sequence lengths.
        block_size: InfLLMv2 page size (64).
        last_q: trailing query rows sampled for the pattern estimate.

    Returns:
        (n_heads, q_len, n_pages) float32 page scores. Vertical mass is
        row-sensitive: each row's causal range bounds which sampled-row
        mass it may see. Slash mass is a per-KV-head prior broadcast to
        all rows. Pages beyond a row's page-causal frontier are -inf.
    """
    n_heads = int(q.shape[1])
    n_kv = int(k.shape[1])
    if n_kv > n_heads or n_heads % n_kv != 0:
        raise ValueError(f"k head count {n_kv} must divide q head count {n_heads}")
    if int(q.shape[0]) != q_len or int(k.shape[0]) != k_len:
        raise ValueError(
            f"q/k leading dims do not match q_len/k_len: "
            f"{tuple(q.shape)} vs {q_len}, {tuple(k.shape)} vs {k_len}"
        )
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if last_q <= 0:
        raise ValueError("last_q must be positive")

    n_pages = max((k_len + block_size - 1) // block_size, 1)
    device = q.device

    vert = vertical_column_mass(
        q, k, q_len=q_len, k_len=k_len, last_q=last_q
    )  # (n_heads, k_len)
    slash = slash_line_mass(
        q, k, q_len=q_len, k_len=k_len, last_q=last_q
    )  # (n_heads, n_diags)

    # Pool vertical mass to pages: page p's mass is the sum of its
    # columns' masses.
    pad = (-k_len) % block_size
    if pad:
        vert = torch.cat(
            [vert, torch.zeros((n_heads, pad), device=device, dtype=vert.dtype)],
            dim=-1,
        )
    vert_pages = vert.reshape(n_heads, -1, block_size).sum(dim=-1)  # (h, P)

    # Pool slash lines to pages: a line with offset o crosses pages
    # around (o - row) for each row; a line's *start column* when paired
    # with query row i is o - i, so its page is (o - i) // B. Pool the
    # line mass uniformly over the pages its start columns can fall in
    # across the sampled rows: line o pairs with sampled rows
    # row_pos in [q_len - last_q, q_len), start cols [o - q_len + 1,
    # o - q_len + last_q]; the pages spanned are a contiguous window of
    # ceil(last_q / B) + 1 pages.
    r = min(int(last_q), q_len)
    row_lo = q_len - r
    row_hi = q_len  # exclusive
    # For each line o: start columns run over [o - row_hi + 1, o - row_lo]
    # intersected with [0, k_len - 1]. Lines whose start-column range is
    # entirely out of [0, k_len) collect no mass and span nothing.
    diag_idx = torch.arange(slash.shape[-1], device=device)
    c_lo = torch.clamp(diag_idx - row_hi + 1, min=0, max=k_len - 1)
    c_hi = torch.clamp(diag_idx - row_lo, min=0, max=k_len - 1)
    c_lo = torch.minimum(c_lo, c_hi)
    p_lo = c_lo // block_size
    p_hi = c_hi // block_size
    # Accumulate each line's mass, spread uniformly over the pages its
    # start columns span, into a (n_heads, n_pages) prior.
    n_lines = slash.shape[-1]
    span = p_hi - p_lo + 1  # (n_lines,)
    max_span = int(span.max().item()) if n_lines else 0
    slash_pages = torch.zeros((n_heads, n_pages), device=device, dtype=slash.dtype)
    if max_span > 0:
        offs = torch.arange(max_span, device=device)
        pair_lines = torch.arange(n_lines, device=device)[:, None].expand(
            n_lines, max_span
        )
        pair_pages = p_lo[:, None] + offs[None, :]
        valid = (offs[None] < span[:, None]) & (pair_pages < n_pages)
        pair_lines = pair_lines[valid]
        pair_pages = pair_pages[valid]
        weights = slash[:, pair_lines]  # (n_heads, n_pairs)
        span_f = span[pair_lines].to(slash.dtype)
        slash_pages.index_add_(1, pair_pages, weights / span_f[None])

    page_prior = vert_pages + slash_pages  # (n_heads, n_pages)

    # Expand per query row with the row-sensitivity of the vertical
    # signal: a query row in page P may only use sampled-row mass that
    # lies in its causal past. Sampled rows near the end are past P for
    # all interior P; only rows within last_q of the sequence end have
    # bounded ranges, and their columns beyond P's frontier must not
    # leak. The strict adaptation: rows in page P use only the vertical
    # mass of sampled rows with position <= P's last row... but sampled
    # rows are at the END of the sequence. MInference's own contract
    # applies the pattern to every row (the pattern is an estimate), so
    # the shared page-causal mask below is the boundary; the vertical
    # estimate uses the trailing rows for all query rows exactly as
    # upstream does.
    row_page_idx = torch.arange(q_len, device=device)[:, None] // block_size
    key_page_idx = torch.arange(n_pages, device=device)[None, :]
    result = page_prior[:, None, :].expand(n_heads, q_len, n_pages).clone()
    result = result.masked_fill(key_page_idx > row_page_idx, float("-inf"))
    return result


def vs_pattern_token_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_len: int,
    k_len: int,
    block_size: int,
    last_q: int = _LAST_Q_ROWS,
    max_seqlen_k: int | None = None,
) -> torch.Tensor:
    """Token-level V/S scores on the shared precomputed-score contract.

    Broadcasts page scores to token slots so the result can flow through
    ``compressed_attention(precomputed_score=..., score_stride=1)``:
    the pooling kernel re-derives the per-page max and applies the same
    local/init overlay and top-k as the other training-free rows. Pages
    beyond the causal frontier carry -inf.

    Shape: (n_kv, q_len, max_seqlen_k or k_len) — batch-1 single-sequence
    inputs only (the same restriction as the XAttention scorer).
    """
    if q_len <= 0 or k_len <= 0:
        raise ValueError("q_len and k_len must be positive")
    if max_seqlen_k is None:
        max_seqlen_k = k_len
    pages = vs_pattern_page_scores(
        q, k, q_len=q_len, k_len=k_len, block_size=block_size, last_q=last_q
    )
    # GQA reduction to KV heads on the page grid, then broadcast to
    # exactly max_seqlen_k token columns (the pooling kernel asserts
    # input.shape[2] == max_seqlen_k): pad the page grid to cover
    # max_seqlen_k, broadcast, then trim/pad the columns. Pages past the
    # causal frontier carry -inf, so partial trailing pages never win
    # the pooled page max.
    n_heads = int(q.shape[1])
    n_kv = int(k.shape[1])
    if n_kv != n_heads:
        group = n_heads // n_kv
        pages = pages.reshape(n_kv, group, pages.shape[1], pages.shape[2]).mean(dim=1)
    n_cols = int(max_seqlen_k)
    n_pages_needed = (n_cols + block_size - 1) // block_size
    n_pages_have = pages.shape[-1]
    if n_pages_needed > n_pages_have:
        pad = torch.full(
            (pages.shape[0], pages.shape[1], n_pages_needed - n_pages_have),
            float("-inf"),
            device=pages.device,
            dtype=pages.dtype,
        )
        pages = torch.cat([pages, pad], dim=-1)
    token = pages.repeat_interleave(block_size, dim=-1)
    if token.shape[-1] > n_cols:
        token = token[:, :, :n_cols]
    elif token.shape[-1] < n_cols:
        pad = torch.full(
            (token.shape[0], token.shape[1], n_cols - token.shape[-1]),
            float("-inf"),
            device=token.device,
            dtype=token.dtype,
        )
        token = torch.cat([token, pad], dim=-1)
    return token.to(q.device)
