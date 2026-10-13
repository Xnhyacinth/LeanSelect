# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StreamingLLM-style recency prior as a page scorer (Xiao et al., ICLR 2024).

Training-free static page-scoring rule for the systems comparison
table: page score = -(page distance from the query row's page). Top-K
then selects the K nearest pages (plus the repository's forced
init/sink and local window overlay, which reproduces StreamingLLM's
attention-sink + sliding-window design at the page level). The row's
causal frontier masks the future, as in every other row.

This is the "no query-awareness" lower-bound control: the score never
looks at Q or K. It isolates how much of a training-free selector's
quality is simply recency. The original StreamingLLM evicts tokens
outside sink+window during decode and does not speed up prefill; this
port places the same static prior into the repository's shared
block-sparse prefill contract (64-token pages, K=96, unchanged
Stage-2 kernels) so it is directly comparable with the other
training-free rows at matched density.
"""

from __future__ import annotations

import torch

__all__ = ["recency_page_scores", "recency_token_scores"]


def recency_page_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_len: int,
    k_len: int,
    block_size: int,
) -> torch.Tensor:
    """Recency page scores: score = -(distance in pages).

    Args:
        q: (q_len, n_heads, d) queries (unused — static prior; kept for
            interface parity with the other scorers).
        k: (k_len, n_kv, d) keys (unused).
        q_len, k_len: sequence lengths.
        block_size: InfLLMv2 page size (64).

    Returns:
        (n_heads, q_len, n_pages) float32 page scores where
        entry (h, i, p) = -(i // B - p), and -inf past the causal
        frontier (p > i // B). All heads share one score grid.
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
    n_pages = max((k_len + block_size - 1) // block_size, 1)
    device = q.device
    row_page = torch.arange(q_len, device=device)[:, None] // block_size
    key_page = torch.arange(n_pages, device=device)[None, :]
    # Score = -(page distance): nearer pages score higher; the forced
    # init (sink) pages get a small floor so the shared overlay's
    # init_blocks pages survive ties.
    scores = -(row_page - key_page).float()
    scores = scores.masked_fill(key_page > row_page, float("-inf"))
    return (
        scores[:, None, :].expand(q_len, n_heads, n_pages).permute(1, 0, 2).contiguous()
    )


def recency_token_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_len: int,
    k_len: int,
    block_size: int,
    max_seqlen_k: int | None = None,
) -> torch.Tensor:
    """Token-level recency scores on the shared precomputed contract.

    Broadcasts page scores to token slots (exactly k_len columns) so the
    result flows through
    ``compressed_attention(precomputed_score=..., score_stride=1)``;
    the shared pooling re-derives the per-page max and applies the same
    local/init overlay and top-k as the other rows.

    Shape: (n_kv, q_len, max_seqlen_k or k_len) — batch-1 single-sequence
    inputs only.
    """
    n_heads = int(q.shape[1])
    n_kv = int(k.shape[1])
    if n_kv > n_heads or n_heads % n_kv != 0:
        raise ValueError(f"k head count {n_kv} must divide q head count {n_heads}")
    if max_seqlen_k is None:
        max_seqlen_k = k_len
    pages = recency_page_scores(
        q, k, q_len=q_len, k_len=k_len, block_size=block_size
    )  # (n_heads, q_len, n_pages); all heads identical
    # Broadcast to exactly max_seqlen_k token columns (the pooling
    # kernel asserts input.shape[2] == max_seqlen_k): pad the page grid
    # to cover max_seqlen_k, broadcast, then trim/pad the columns.
    n_cols = int(max_seqlen_k)
    n_pages_needed = (n_cols + block_size - 1) // block_size
    n_pages_have = pages.shape[-1]
    if n_pages_needed > n_pages_have:
        pad = torch.full(
            (1, pages.shape[1], n_pages_needed - n_pages_have),
            float("-inf"),
            device=pages.device,
            dtype=pages.dtype,
        )
        pages = torch.cat([pages[:1], pad], dim=-1)
    token = pages[:1].repeat_interleave(block_size, dim=-1)
    if token.shape[-1] > n_cols:
        token = token[:, :, :n_cols]
    elif token.shape[-1] < n_cols:
        pad = torch.full(
            (1, token.shape[1], n_cols - token.shape[-1]),
            float("-inf"),
            device=token.device,
            dtype=token.dtype,
        )
        token = torch.cat([token, pad], dim=-1)
    return token.expand(n_kv, -1, -1).contiguous().to(q.device)
