# This file is copied/adapted from NOSI / NOSA (https://github.com/thunlp/NOSA).
# Copyright (c) 2026 THUNLP.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

import math
import torch
import torch.nn.functional as F


def compute_skewing_matrix(q_act, k_act, num_heads, num_kv_heads, head_dim):
    """Compute per-KV-head skewing matrix A from post-RoPE Q/K activations.

    For GQA: Q-heads in each group are averaged before SVD (matching CLO impl).
    Returns one skewing matrix per KV-head.

    Args:
        q_act: (bsz, num_heads, seq_len, head_dim) — post-RoPE Q activations
        k_act: (bsz, num_kv_heads, seq_len, head_dim) — post-RoPE K activations
        num_heads: total number of Q heads (32 for MiniCPM)
        num_kv_heads: number of KV heads (2 for MiniCPM)
        head_dim: dimension per head (128 for MiniCPM)

    Returns:
        A: (num_kv_heads, head_dim, head_dim) — skewing matrix per KV-head
    """
    group_size = num_heads // num_kv_heads
    device = q_act.device
    dtype = q_act.dtype
    A = torch.zeros(num_kv_heads, head_dim, head_dim, device=device, dtype=dtype)
    for kv_h in range(num_kv_heads):
        q_group = q_act[0, kv_h * group_size : (kv_h + 1) * group_size]
        q_avg = q_group.mean(dim=0)
        (_, sq, vq) = torch.svd(q_avg.float())
        (_, sk, _) = torch.svd(k_act[0, kv_h].float())
        s = sq * sk
        (_, ind) = s.sort()
        vq = vq.to(dtype)
        a = torch.zeros(head_dim, head_dim, device=device, dtype=dtype)
        A[kv_h] = a.scatter(-1, ind.unsqueeze(0).expand(head_dim, -1), vq)
    return A


def compute_partial_weight_indices(
    q_act, k_act, A, num_heads, num_kv_heads, head_dim, num_channels=32
):
    """Compute per-KV-head partial dimension indices (matching CLO implementation).

    For each KV-head: skew Q (averaged across GQA group) and K, take abs-sum
    over sequence, sum across Q-heads in group, select top num_channels dims.

    Args:
        q_act: (bsz, num_heads, seq_len, head_dim) — post-RoPE Q
        k_act: (bsz, num_kv_heads, seq_len, head_dim) — post-RoPE K
        A: (num_kv_heads, head_dim, head_dim) — skewing matrices (per KV-head)
        num_heads: total Q heads
        num_kv_heads: KV heads
        head_dim: head dimension
        num_channels: number of dimensions to select (default 32)

    Returns:
        indices: (num_kv_heads, num_channels) — selected dimension indices per KV-head
    """
    group_size = num_heads // num_kv_heads
    device = q_act.device
    A = A.to(device=device, dtype=q_act.dtype)
    indices = torch.zeros(num_kv_heads, num_channels, dtype=torch.long, device=device)
    for kv_h in range(num_kv_heads):
        q_group = q_act[0, kv_h * group_size : (kv_h + 1) * group_size]
        q_skewed = torch.matmul(q_group, A[kv_h])
        q_importance = q_skewed.abs().sum(dim=1).sum(dim=0)
        (_, topk) = torch.topk(q_importance, num_channels)
        indices[kv_h] = topk
    return indices


def infinigen_token_scores_hf(
    prev_hidden,
    q_proj_weight,
    skewing_matrix,
    partial_indices,
    partial_key_cache,
    cos,
    sin,
    position_ids,
    num_heads,
    num_kv_heads,
    head_dim,
    num_channels,
    apply_rope_fn,
    a_partial=None,
):
    """Compute InfiniGen token-level scores for HF accuracy paths.

    Runtime einsum skewing (CLO style). Speculates Q from previous layer's
    attention input (already post-layernorm), applies RoPE, skews, extracts
    partial dims, scores against partial key cache. Pre-softmax SUM across
    GQA group.

    Args:
        prev_hidden: (B, 1, D) — previous layer's attention input (post-layernorm)
        q_proj_weight: (num_heads * head_dim, D) — Q projection weight
        skewing_matrix: (num_kv_heads, head_dim, head_dim) — per-KV-head skewing
        partial_indices: (num_kv_heads, num_channels) — selected dim indices
        partial_key_cache: (B, seq_len, H_kv, nc) or (seq_len, H_kv, nc) — partial K on GPU
        cos: RoPE cosine tensor
        sin: RoPE sine tensor
        position_ids: (B, 1) — current position
        num_heads: total Q heads
        num_kv_heads: KV heads
        head_dim: head dimension
        num_channels: number of partial dimensions
        apply_rope_fn: function(q, k, cos, sin, pos) -> (q_rotated, _)

    Returns:
        token_scores: (num_kv_heads, B, seq_len) — pre-softmax aggregated scores
    """
    bsz = prev_hidden.shape[0]
    group_size = num_heads // num_kv_heads
    spec_q = F.linear(prev_hidden, q_proj_weight)
    spec_q = spec_q.view(bsz, 1, num_heads, head_dim).transpose(1, 2)
    (spec_q, _) = apply_rope_fn(
        spec_q, spec_q[:, :num_kv_heads], cos, sin, position_ids
    )
    spec_q_grouped = spec_q.view(bsz, num_kv_heads, group_size, 1, head_dim)
    skew = skewing_matrix.to(spec_q.dtype).to(spec_q.device)
    spec_q_skewed = torch.einsum("bhgsd,hde->bhgse", spec_q_grouped, skew)
    p_idx = partial_indices.to(spec_q.device)
    idx_q = (
        p_idx.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bsz, -1, group_size, 1, -1)
    )
    spec_q_partial = torch.gather(spec_q_skewed, -1, idx_q)
    if partial_key_cache.dim() == 3:
        p_k = partial_key_cache.unsqueeze(0).expand(bsz, -1, -1, -1).permute(0, 2, 1, 3)
    else:
        p_k = partial_key_cache.permute(0, 2, 1, 3)
    scores = torch.matmul(spec_q_partial, p_k.unsqueeze(2).transpose(-1, -2))
    scores = scores / math.sqrt(num_channels)
    score_agg = scores.sum(dim=2).squeeze(2)
    return score_agg.permute(1, 0, 2).contiguous()


def infinigen_build_partial_cache(full_k, skewing_matrix, partial_indices):
    """Build partial key cache from full post-RoPE keys during prefill.

    Applies skewing matrix per KV-head, then extracts partial dimensions.
    Supports both unbatched and batched inputs.

    Args:
        full_k: (seq_len, H_kv, D) or (B, seq_len, H_kv, D) — full post-RoPE keys
        skewing_matrix: (H_kv, D, D) — per-KV-head skewing
        partial_indices: (H_kv, nc) — selected dim indices

    Returns:
        partial_k: (seq_len, H_kv, nc) or (B, seq_len, H_kv, nc) — matching input batch dims
    """
    batched = full_k.dim() == 4
    if batched:
        (B, seq_len, num_kv_heads, head_dim) = full_k.shape
    else:
        (seq_len, num_kv_heads, head_dim) = full_k.shape
        B = 1
        full_k = full_k.unsqueeze(0)
    skew = skewing_matrix.to(full_k.dtype).to(full_k.device)
    p_idx = partial_indices.to(full_k.device)
    nc = p_idx.shape[1]
    k_t = full_k.permute(0, 2, 1, 3).reshape(B * num_kv_heads, seq_len, head_dim)
    skew_exp = (
        skew.unsqueeze(0)
        .expand(B, -1, -1, -1)
        .reshape(B * num_kv_heads, head_dim, head_dim)
    )
    k_skewed = torch.bmm(k_t, skew_exp)
    idx_exp = p_idx.unsqueeze(1).expand(-1, seq_len, -1)
    idx_exp = (
        idx_exp.unsqueeze(0)
        .expand(B, -1, -1, -1)
        .reshape(B * num_kv_heads, seq_len, nc)
    )
    partial_k = torch.gather(k_skewed, -1, idx_exp)
    partial_k = partial_k.view(B, num_kv_heads, seq_len, nc).permute(0, 2, 1, 3)
    if not batched:
        return partial_k.squeeze(0).contiguous()
    return partial_k.contiguous()


def infinigen_update_partial_cache(
    partial_key_cache, new_k, position, skewing_matrix, partial_indices
):
    """Append single decode token to partial key cache.

    Supports both unbatched and batched caches.

    Args:
        partial_key_cache: (max_seq, H_kv, nc) or (B, max_seq, H_kv, nc)
        new_k: (H_kv, D) or (B, H_kv, D) — new key(s)
        position: int or (B,) tensor — position(s) to write at
        skewing_matrix: (H_kv, D, D) — per-KV-head skewing
        partial_indices: (H_kv, nc) — selected dim indices
    """
    skew = skewing_matrix.to(new_k.dtype).to(new_k.device)
    p_idx = partial_indices.to(new_k.device)
    if new_k.dim() == 2:
        new_k_skewed = torch.einsum("hd,hde->he", new_k, skew)
        new_k_partial = torch.gather(new_k_skewed, -1, p_idx)
        if isinstance(position, int) and position < partial_key_cache.shape[-3]:
            if partial_key_cache.dim() == 3:
                partial_key_cache[position] = new_k_partial
            else:
                partial_key_cache[:, position] = new_k_partial
    else:
        new_k_skewed = torch.einsum("bhd,hde->bhe", new_k, skew)
        new_k_partial = torch.gather(
            new_k_skewed, -1, p_idx.unsqueeze(0).expand(new_k.shape[0], -1, -1)
        )
        if isinstance(position, int):
            pos = position
            if pos < partial_key_cache.shape[-3]:
                if partial_key_cache.dim() == 4:
                    partial_key_cache[:, pos] = new_k_partial
                else:
                    partial_key_cache[pos] = new_k_partial[0]
        else:
            for b in range(new_k.shape[0]):
                pos = int(position[b].item())
                if pos < partial_key_cache.shape[-3]:
                    if partial_key_cache.dim() == 4:
                        partial_key_cache[b, pos] = new_k_partial[b]
                    else:
                        partial_key_cache[pos] = new_k_partial[b]


def calibrate_infinigen(model, input_ids, max_new_tokens, *, backbone=None, seed=42):
    """Calibrate once per task with the paper's backbone-specific input recipe.

    Pass the first canonical task prompt before filtering completed examples.
    MiniCPM uses its first 2,048 tokens; NOSA uses a private seeded vocabulary
    draw. Repeated calls recalibrate from scratch without advancing generation RNG.
    """
    import hashlib

    backbone = backbone or getattr(model, "leanselect_backbone", None)
    if backbone not in {"minicpm", "nosa"}:
        raise ValueError("Calibration requires a minicpm or nosa backbone")
    if int(max_new_tokens) <= 0:
        raise ValueError("Calibration requires a positive generation budget")
    device = torch.device(model.device)
    if backbone == "minicpm":
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.shape[0] < 1
            or input_ids.shape[1] < 1
            or input_ids.dtype != torch.long
        ):
            raise ValueError(
                "MiniCPM calibration requires canonical Long input_ids [B,L]"
            )
        calibration = input_ids[:1, :2048].to(device)
        recipe = {
            "form": "first canonical task prompt, first 2048 tokens",
            "seed": None,
        }
    else:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        vocab_size = int(model.config.vocab_size)
        calibration = torch.randint(
            0,
            vocab_size,
            (1, 2048),
            device=device,
            dtype=torch.long,
            generator=generator,
        )
        recipe = {
            "form": "uniform torch.randint(0,vocab,(1,2048))",
            "seed": int(seed),
            "low": 0,
            "high": vocab_size,
            "generator": "private torch.Generator",
            "generator_device": device.type,
        }
    attns = model._get_infinigen_attns()
    if not attns:
        raise ValueError("Model has no InfiniGen attention adapters")
    for attn in attns:
        attn._ig_warmed_up = False
        attn._ig_skewing_matrix = None
        attn._ig_partial_indices = None
        attn._ig_partial_key_cache = None
        attn._ig_prev_hidden_states = None
        attn._ig_attn_input = None
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        model.infinigen_warmup(
            input_ids=calibration, max_new_tokens=int(max_new_tokens)
        )
    if not all(getattr(attn, "_ig_warmed_up", False) for attn in attns):
        raise RuntimeError("InfiniGen calibration did not initialize every adapter")
    model._cleanup_infinigen_request_state()
    receipt = dict(
        recipe,
        backbone=backbone,
        shape=list(calibration.shape),
        dtype=str(calibration.dtype),
        input_sha256=hashlib.sha256(
            calibration.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
    )
    model.leanselect_calibration = receipt
    return receipt
