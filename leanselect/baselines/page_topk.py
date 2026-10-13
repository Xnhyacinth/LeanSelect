# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct page-level Top-K for the training-free selector baselines.

The compressed_attention path materializes a token-level score grid of
shape (n_kv, total_q, max_seqlen_k); at 96K/128K prefill that grid alone
is tens of GiB and OOMs. This module reproduces the shared pooling +
top-k contract directly on the *page* grid (O(n_kv, total_q, n_pages)),
exactly matching what max_pooling_1d_varlen + the top-k tail produce for
the training-free scorers:

  - pool a page's score from its token column window
    [k*stride - padding, k*stride - padding + kernel_size) — for the
    training-free scorers every token in a page carries the page's
    score, so the pooled page max equals the page score (pages whose
    window is fully past the causal range read -inf / the init+local
    +inf overlay);
  - the init (sink) pages and the local window pages are forced by
    writing +inf (they must win top-k);
  - top-k with sorted indices, masked by the row's page-causal q_idx
    (page p selectable for row i only when p <= i // B), int32, -1 pad.

Output: topk_idx identical in shape/semantics to compressed_attention's
return value, consumed unchanged by Stage 2.
"""

from __future__ import annotations

import torch

__all__ = ["tf_page_topk"]


def tf_page_topk(
    page_scores: torch.Tensor,
    *,
    topk: int,
    block_size: int,
    stride: int,
    init_blocks: int,
    local_blocks: int,
    q_len: int,
    k_len: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> torch.Tensor:
    """Page-grid top-k on the shared selection contract.

    Args:
        page_scores: (n_kv, q_len, n_pages_scored) per-KV-head page
            scores with -inf past each row's page-causal frontier. May
            be narrower than the padded page grid (caller's scoring
            length); padded pages read -inf.
        topk: requested page budget (already the effective
            selection K; local/init pages come out of it, as in the
            shared path).
        block_size: page size (64).
        stride: the pooling kernel's *token* stride parameter as passed
            to max_pooling_1d_varlen (score_stride=1 in the TF path;
            kept for interface completeness).
        init_blocks, local_blocks: forced sink and window pages.
        q_len, k_len: packed lengths of this batch-1 prefill.
        max_seqlen_q, max_seqlen_k: padded grid lengths (batch-1:
            equal to q_len/k_len).

    Returns:
        (n_kv, q_len, k_take) int32 sorted page indices with -1 where
        masked, matching compressed_attention's topk_idx contract.
    """
    n_kv = int(page_scores.shape[0])
    device = page_scores.device
    # Padded page grid, exactly as the pooled output would span.
    total_len = int(max_seqlen_q) + 0  # cache_len = 0 at prefill
    out_len = max((int(max_seqlen_k) + block_size - 1) // block_size, 1)
    n_scored = int(page_scores.shape[-1])
    if n_scored < out_len:
        pad = torch.full(
            (n_kv, page_scores.shape[1], out_len - n_scored),
            float("-inf"),
            device=device,
            dtype=page_scores.dtype,
        )
        grid = torch.cat([page_scores, pad], dim=-1)
    else:
        grid = page_scores[:, :, :out_len]

    # Row page index (padded, right-aligned rows like q_idx).
    q_idx = torch.arange(q_len, device=device)
    if int(max_seqlen_q) != q_len:
        q_idx = q_idx + (int(max_seqlen_q) - q_len)
    off_bq = q_idx // block_size  # (q_len,)

    # Replicate the pooling window exactly: the kernel pools page p
    # from token columns [p*B - padding, p*B - padding + kernel_size)
    # with padding=1, kernel_size=B (score_stride=1) — i.e. page p's
    # pooled value is max(page[p], page[p-1]) when every token in a
    # page carries the page score (the training-free broadcast).
    pooled = grid.clone()
    if out_len > 1:
        pooled[:, :, 1:] = torch.maximum(grid[:, :, 1:], grid[:, :, :-1])

    # Forced overlay from the kernel: init pages and the local window
    # are written +inf so they always win top-k.
    off_bk = torch.arange(out_len, device=device)
    force_inf = (off_bk < init_blocks) | (
        (off_bq[:, None] >= off_bk[None, :])
        & (off_bq[:, None] <= off_bk[None, :] + local_blocks)
    )
    grid = pooled.masked_fill(force_inf[None], float("inf"))

    k_take = min(int(topk), out_len)
    topk_idx = grid.topk(k_take, dim=-1).indices.sort(-1).values
    # Page-causal mask: page p selectable for row i only when p <= i//B
    # (the same q_idx comparison compressed_attention performs).
    topk_idx = topk_idx.masked_fill(topk_idx > off_bq[None, :, None], -1)
    return topk_idx.to(torch.int32)
