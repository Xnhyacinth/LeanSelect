"""Compressed-key and overlapping-page geometry shared by training and inference."""
from __future__ import annotations

import torch


def serve_dynamic_k(
    topk: int,
    n_pages: int,
    forced_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
    """Retrieved-page count after init/local consume Top-K slots.

    Serve sets forced pages to +inf then takes ``topk``; they occupy slots.
    Train/recall must hinge the remaining ``topk - n_forced`` pages, not 96
    extra non-forced pages.
    """
    k_cap = min(max(int(topk), 0), max(int(n_pages), 0))
    if forced_mask is None:
        return k_cap
    n_forced = forced_mask.to(dtype=torch.long).sum(dim=-1)
    n_free = int(n_pages) - n_forced
    return (k_cap - n_forced).clamp(min=0).clamp(max=n_free)


def variable_k_topk_mask(
    scores: torch.Tensor,
    k: torch.Tensor | int,
) -> torch.Tensor:
    """True on the per-row top-``k`` finite pages. ``k`` may be [... ] or int."""
    nb = int(scores.shape[-1])
    if torch.is_tensor(k):
        k_t = k.to(device=scores.device, dtype=torch.long)
        while k_t.ndim < scores.ndim - 1:
            k_t = k_t.unsqueeze(0)
        k_t = k_t.expand(*scores.shape[:-1])
    else:
        k_t = torch.full(
            scores.shape[:-1],
            int(k),
            device=scores.device,
            dtype=torch.long,
        )
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(
        -1,
        order,
        torch.arange(nb, device=scores.device, dtype=order.dtype).expand_as(order),
    )
    return (rank < k_t.unsqueeze(-1)) & torch.isfinite(scores)


def group_size(num_q_heads: int, num_kv_heads: int) -> int:
    heads = int(num_q_heads)
    n_kv = int(num_kv_heads)
    if n_kv <= 0 or heads <= 0 or heads % n_kv != 0:
        raise ValueError(f"H={heads} must be a positive multiple of n_kv={n_kv}")
    return heads // n_kv


def uniform_group_queries(
    q: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """Mean of vanilla Q heads inside each GQA group. q (..., H, D) → (..., n_kv, D)."""
    n_kv = int(num_kv_heads)
    heads = int(q.shape[-2])
    dim = int(q.shape[-1])
    gs = group_size(heads, n_kv)
    return q.reshape(*q.shape[:-2], n_kv, gs, dim).mean(dim=-2)


def infllm_atom_count(
    seq_len: int, kernel_size: int = 32, kernel_stride: int = 16
) -> int:
    if min(seq_len, kernel_size, kernel_stride) <= 0:
        raise ValueError("sequence and compression geometry must be positive")
    if seq_len < kernel_size:
        return 1
    return (seq_len - kernel_size) // kernel_stride + 1


def compress_keys_mean(
    k: torch.Tensor,
    *,
    kernel_size: int = 32,
    kernel_stride: int = 16,
) -> torch.Tensor:
    """InfLLM CompressK: mean-pool overlapping windows.

    k: [B, S, n_kv, D] → [B, C, n_kv, D]
    """
    if k.ndim != 4:
        raise ValueError(f"k must be [B, S, n_kv, D], got {tuple(k.shape)}")
    seq_len = int(k.shape[1])
    if min(seq_len, kernel_size, kernel_stride) <= 0:
        raise ValueError("sequence and compression geometry must be positive")
    if seq_len < kernel_size:
        return k.mean(dim=1, keepdim=True)
    return k.unfold(1, kernel_size, kernel_stride).mean(dim=-1)


def expand_kv_to_q_heads(k_atoms: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """[B, C, n_kv, D] → [B, C, H, D] by repeating GQA groups."""
    n_kv = int(k_atoms.shape[2])
    heads = int(num_q_heads)
    if n_kv == heads:
        return k_atoms
    if heads % n_kv != 0:
        raise ValueError(f"H={heads} is not a multiple of n_kv={n_kv}")
    return k_atoms.repeat_interleave(heads // n_kv, dim=2)


def vanilla_atom_scores(q: torch.Tensor, k_4d: torch.Tensor) -> torch.Tensor:
    """Teacher atom scores: Q_h · K_{kv(h)}. Same softmax-sum contract.

    q: [B, Q, H, D], k_4d: [B, C, n_kv|H, D] → [B, Q, H, C].
    One grouped bmm per KV group (does not materialize H copies of K).
    """
    if q.ndim != 4 or k_4d.ndim != 4:
        raise ValueError(
            f"q must be [B,Q,H,D] and k [B,C,n_kv,D], got {tuple(q.shape)} / {tuple(k_4d.shape)}"
        )
    if q.shape[0] != k_4d.shape[0] or q.shape[-1] != k_4d.shape[-1]:
        raise ValueError("q/k batch or head_dim mismatch")
    batch, n_q, heads, dim = q.shape
    n_atoms = int(k_4d.shape[1])
    n_kv = int(k_4d.shape[2])
    gs = group_size(heads, n_kv)
    q_b = (
        q.view(batch, n_q, n_kv, gs, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * n_kv, n_q * gs, dim)
    )
    k_b = k_4d.permute(0, 2, 1, 3).reshape(batch * n_kv, n_atoms, dim)
    dots = torch.bmm(q_b.float(), k_b.float().transpose(1, 2))
    return (
        dots.view(batch, n_kv, n_q, gs, n_atoms)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, n_q, heads, n_atoms)
        .to(dtype=q.dtype)
    )


def pool_atoms_to_pages(
    atom_scores: torch.Tensor,
    *,
    seq_len: int,
    block_size: int = 64,
    kernel_stride: int = 16,
    reduce: str = "max",
) -> torch.Tensor:
    """InfLLM 5-atom page window. ``reduce`` is ``max`` (vanilla) or ``lse`` (mass).

    atom_scores: [..., C] → [..., N_b]
    page k gathers atoms [k * (B/s) - 1, k * (B/s) - 1 + 5).
    """
    if min(seq_len, block_size, kernel_stride) <= 0:
        raise ValueError("page geometry must be positive")
    if atom_scores.ndim < 1 or atom_scores.shape[-1] == 0 or not atom_scores.is_floating_point():
        raise ValueError("atom_scores must be floating point with a nonempty atom dimension")
    if block_size % kernel_stride != 0:
        raise ValueError(
            f"block_size={block_size} must be divisible by kernel_stride={kernel_stride}"
        )
    if reduce not in ("max", "lse"):
        raise ValueError(f"reduce must be max or lse, got {reduce!r}")
    pool_stride = block_size // kernel_stride
    kernel = pool_stride + 1
    padding = 1
    n_pages = (int(seq_len) + block_size - 1) // block_size
    n_atoms = int(atom_scores.shape[-1])
    if reduce == "max":
        # Keep the large Stage-1 matmul on cuBLAS/Tensor Cores, then pool its
        # regular width-5/stride-4 grid without materializing the old
        # ``[..., pages, 5]`` gather tensor.
        padded_len = (n_pages - 1) * pool_stride + kernel
        right = max(padded_len - (n_atoms + padding), 0)
        flat = atom_scores.reshape(-1, 1, n_atoms)
        flat = torch.nn.functional.pad(
            flat, (padding, right), mode="constant", value=float("-inf")
        )
        pooled = torch.nn.functional.max_pool1d(
            flat, kernel_size=kernel, stride=pool_stride
        )
        return pooled[..., :n_pages].reshape(*atom_scores.shape[:-1], n_pages)

    device = atom_scores.device
    starts = torch.arange(n_pages, device=device) * pool_stride - padding
    idx = starts[:, None] + torch.arange(kernel, device=device)
    valid = (idx >= 0) & (idx < n_atoms)
    idx_clamped = idx.clamp(0, max(n_atoms - 1, 0))
    gathered = atom_scores[..., idx_clamped]
    gathered = gathered.masked_fill(~valid, float("-inf"))
    return torch.logsumexp(gathered.float(), dim=-1).to(dtype=atom_scores.dtype)


def apply_atom_causal_mask(
    atom_scores: torch.Tensor,
    query_abs_positions: torch.Tensor,
    *,
    kernel_stride: int = 16,
) -> torch.Tensor:
    """Hide atoms whose window starts after the query token.

    Matches NOSI vanilla Stage-1: compressed atom ``i`` covers tokens
    ``[i * stride, ...)`` and is visible iff ``i * stride <= q_pos``.
    atom_scores: [B, Q, ..., C], query_abs_positions: [Q] or [B, Q]
    """
    n_atoms = int(atom_scores.shape[-1])
    atom_start = torch.arange(n_atoms, device=atom_scores.device) * int(kernel_stride)
    pos = query_abs_positions.to(device=atom_scores.device)
    if pos.ndim == 1:
        pos = pos.view(1, -1).expand(atom_scores.shape[0], -1)
    while pos.ndim < atom_scores.ndim - 1:
        pos = pos.unsqueeze(-1)
    visible = atom_start <= pos.unsqueeze(-1)
    return atom_scores.masked_fill(~visible, float("-inf"))


def forced_page_mask(
    query_abs_positions: torch.Tensor,
    *,
    num_pages: int,
    block_size: int = 64,
    init_blocks: int = 1,
    local_blocks: int = 32,
) -> torch.Tensor:
    """True on InfLLM init + sliding local pages (always selected at serve).

    Returns [B, Q, N_b] (B=1 when positions are 1D).
    """
    pos = query_abs_positions.long()
    if pos.ndim == 1:
        pos = pos.view(1, -1)
    q_page = (pos // int(block_size)).unsqueeze(-1)
    pages = torch.arange(int(num_pages), device=pos.device).view(1, 1, -1)
    local = (q_page >= pages) & (q_page <= pages + int(local_blocks))
    return (pages < int(init_blocks)) | local
