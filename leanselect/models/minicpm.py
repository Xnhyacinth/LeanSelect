# Copyright 2025 The OpenBMB Team. All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This code is based on the original HuggingFace MiniCPM implementation with
# InfLLM V2 future prediction integration for sparse attention.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MiniCPM backbone with native sparse attention and LeanSelect."""

from .configuration_minicpm import MiniCPMConfig
from leanselect.baselines.xattention import xattention_token_scores_kv_packed
from leanselect.baselines.minference import vs_pattern_token_scores
from leanselect.baselines.recency import recency_token_scores
from leanselect.baselines.infinigen import (
    infinigen_build_partial_cache,
    infinigen_token_scores_hf,
    infinigen_update_partial_cache,
)

_XATTENTION_AVAILABLE = True
_INFINIGEN_AVAILABLE = True
import contextlib
import math
import os
import re
import warnings
from functools import lru_cache
import torch
from leanselect.ops import profiling as nvtx
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.generation.utils import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import is_flash_attn_greater_or_equal_2_10, logging

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

    flash_attn_available = True
except ImportError:
    flash_attn_available = False
    flash_attn_func = None
    flash_attn_varlen_func = None
    index_first_axis = None
    pad_input = None
    unpad_input = None
_TF_TOKEN_GRID_CELL_LIMIT = 1 << 30
_TRAINING_FREE_SCORERS = {
    "xattention": (xattention_token_scores_kv_packed, _XATTENTION_AVAILABLE),
    "minference": (vs_pattern_token_scores, vs_pattern_token_scores is not None),
    "recency": (recency_token_scores, recency_token_scores is not None),
}
try:
    from infllm_v2 import (
        infllmv2_attn_stage1,
        infllmv2_attn_varlen_func,
        infllmv2_attn_with_kvcache,
        max_pooling_1d,
        max_pooling_1d_varlen,
    )

    try:
        from infllm_v2 import infllmv2_attn_stage1_kl_teacher
    except ImportError:
        infllmv2_attn_stage1_kl_teacher = None
    infllm_ops_available = True
except ImportError:
    infllmv2_attn_varlen_func = None
    infllmv2_attn_stage1 = None
    infllmv2_attn_stage1_kl_teacher = None
    infllmv2_attn_with_kvcache = None
    max_pooling_1d = None
    max_pooling_1d_varlen = None
    infllm_ops_available = False
logger = logging.get_logger(__name__)
_SKIP_KL_LOSS = False


def set_skip_kl_loss(skip: bool):
    """Set global flag to skip KL loss computation (used during grad checkpoint recompute)."""
    global _SKIP_KL_LOSS
    _SKIP_KL_LOSS = skip


@contextlib.contextmanager
def skip_kl_loss(enabled: bool = True):
    """Context manager to temporarily toggle KL-loss computation."""
    global _SKIP_KL_LOSS
    prev = _SKIP_KL_LOSS
    _SKIP_KL_LOSS = enabled
    try:
        yield
    finally:
        _SKIP_KL_LOSS = prev


def _call_infllmv2_stage1_compat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_v: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
) -> torch.Tensor:
    """Call stage1 with the repo API and fail fast on stale infllm_v2 builds."""
    try:
        return infllmv2_attn_stage1(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_v=cu_seqlens_v,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )
    except TypeError as exc:
        if "cu_seqlens_v" not in str(exc):
            raise
        raise RuntimeError(
            "MiniCPM training requires the repo infllm_v2 extension with cu_seqlens_v support. Build the extension with third_party/infllmv2/install.py."
        ) from exc


def _call_infllmv2_stage1_kl_teacher_compat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_v: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    causal_stride: int,
    query_offset: int,
) -> torch.Tensor:
    """Call the training-only KL-teacher stage1 op and fail fast on stale builds."""
    if infllmv2_attn_stage1_kl_teacher is None:
        raise RuntimeError(
            "MiniCPM KL-teacher training requires the repo infllm_v2 extension with infllmv2_attn_stage1_kl_teacher support. Build the extension with third_party/infllmv2/install.py."
        )
    try:
        return infllmv2_attn_stage1_kl_teacher(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_v=cu_seqlens_v,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal_stride=causal_stride,
            query_offset=query_offset,
            causal=causal,
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "MiniCPM KL-teacher training requires the rebuilt repo infllm_v2 extension with the training-only stage1 KL-teacher op. Build the extension with third_party/infllmv2/install.py."
        ) from exc


def _leanselect_sampled_stage1_teacher(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    *,
    windows: tuple[tuple[int, int], ...],
    kernel_stride: int,
    training_kernel_stride: int | None = None,
    deployed_atom_count: int | None = None,
    align_pooling: str = "max",
) -> torch.Tensor:
    """Evaluate sampled actual-Q scores on the deployed atom grid."""
    if q.ndim != 3 or k.ndim != 3 or k2.ndim != 3:
        raise ValueError(
            "LeanSelect teacher expects Q/K/K2 tensors with three dimensions"
        )
    if min(int(q.shape[0]), int(k.shape[0]), int(k2.shape[0]), int(kernel_stride)) <= 0:
        raise ValueError("LeanSelect teacher geometry must be positive")
    if not windows:
        raise ValueError("LeanSelect teacher requires at least one query window")
    if int(kernel_stride) != 16:
        raise ValueError("production InfLLM Stage-1 uses a fixed causal stride of 16")
    if q.shape[1] % k.shape[1] != 0 or q.shape[2] != k.shape[2]:
        raise ValueError("LeanSelect teacher Q/K head geometry is incompatible")
    if q.device != k.device or q.device != k2.device:
        raise ValueError("LeanSelect teacher Q/K/K2 must share one device")
    teacher_stride = int(
        kernel_stride if training_kernel_stride is None else training_kernel_stride
    )
    if teacher_stride != kernel_stride:
        if q.shape[1] // k.shape[1] != 16:
            raise ValueError(
                "fine LeanSelect KL teacher requires 16 query heads per GQA group"
            )
        if deployed_atom_count is None or deployed_atom_count <= 0:
            raise ValueError("fine LeanSelect teacher requires the deployed atom count")
        cu_q = torch.empty(2, device=q.device, dtype=torch.int32)
        cu_k = torch.tensor([0, len(k)], device=q.device, dtype=torch.int32)
        aligned_windows = []
        previous_end = 0
        for start, end in windows:
            if not previous_end <= start < end <= len(q):
                raise ValueError(
                    "LeanSelect teacher windows must be ordered and disjoint"
                )
            previous_end = end
            (cu_q[0], cu_q[1]) = (0, end - start)
            probabilities = _call_infllmv2_stage1_kl_teacher_compat(
                q[start:end].contiguous(),
                k.contiguous(),
                k.contiguous(),
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                cu_seqlens_v=cu_k,
                max_seqlen_q=end - start,
                max_seqlen_k=len(k),
                causal=True,
                causal_stride=teacher_stride,
                query_offset=start,
            )[:, : end - start, : len(k)]
            aligned_windows.append(
                _align_probabilities_to_target_grid_1d(
                    probabilities.reshape(-1, 1, len(k)),
                    source_stride=teacher_stride,
                    target_stride=kernel_stride,
                    target_length=deployed_atom_count,
                    mode=align_pooling,
                ).reshape(k.shape[1], end - start, deployed_atom_count)
            )
        return torch.cat(aligned_windows, dim=1).to(q.dtype)
    rows = []
    previous_end = 0
    for start, end in windows:
        (start, end) = (int(start), int(end))
        if start < previous_end or end <= start or end > int(q.shape[0]):
            raise ValueError("LeanSelect teacher windows must be ordered and disjoint")
        rows.append(torch.arange(start, end, dtype=torch.long, device=q.device))
        previous_end = end
    read_points = torch.cat(rows)
    sampled_q = q.index_select(0, read_points)
    group_size = q.shape[1] // k.shape[1]
    grouped_q = (
        sampled_q.float()
        .reshape(sampled_q.shape[0], k.shape[1], group_size, q.shape[2])
        .permute(1, 2, 0, 3)
    )
    logits = torch.matmul(grouped_q, k.float().permute(1, 2, 0).unsqueeze(1)) * q.shape[
        2
    ] ** (-0.5)
    compressed_query_rows = int((int(q.shape[0]) - 16 + 1) / 16)
    key_offset = int(k.shape[0]) - int(compressed_query_rows)
    visible_counts = (
        torch.div(read_points - 16 + 1, 16, rounding_mode="trunc") + key_offset
    )
    visible_counts = visible_counts.clamp(min=0, max=int(k.shape[0]))
    key_ids = torch.arange(k.shape[0], device=q.device)
    logits = logits.masked_fill(
        key_ids.view(1, 1, 1, -1) >= visible_counts.view(1, 1, -1, 1), float("-inf")
    )
    probabilities = torch.softmax(logits, dim=-1).sum(dim=1)
    probabilities = torch.where(
        torch.isnan(probabilities), torch.zeros_like(probabilities), probabilities
    )
    return probabilities.to(q.dtype)


def _pool_kl_target_alignment_1d(
    values: torch.Tensor, *, kernel_size: int, stride: int, padding: int = 0, mode: str
) -> torch.Tensor:
    """Pool finer KL targets onto the pred compressed grid."""
    if mode == "max":
        return F.max_pool1d(
            values, kernel_size=kernel_size, stride=stride, padding=padding
        )
    if mode == "mean":
        return F.avg_pool1d(
            values, kernel_size=kernel_size, stride=stride, padding=padding
        )
    raise ValueError(f"Unsupported kl_target_align_pooling={mode!r}")


def _align_probabilities_to_target_grid_1d(
    values: torch.Tensor,
    *,
    source_stride: int,
    target_stride: int,
    target_length: int,
    mode: str,
) -> torch.Tensor:
    """Align finer probability tensors onto a coarser compressed grid."""
    if target_stride % source_stride != 0:
        raise ValueError(
            f"Cannot align source stride {source_stride} to target stride {target_stride}: target stride must be divisible by source stride."
        )
    ratio = target_stride // source_stride
    if ratio == 1:
        aligned = values
    else:
        aligned = _pool_kl_target_alignment_1d(
            values, kernel_size=ratio + 1, stride=ratio, padding=1, mode=mode
        )
    pooled_len = aligned.shape[-1]
    if pooled_len > target_length:
        aligned = aligned[..., :target_length]
    elif pooled_len < target_length:
        aligned = F.pad(aligned, (0, target_length - pooled_len), value=0.0)
    return aligned


def _overlapping_page_atom_union(
    page_ids: torch.Tensor, *, atoms_per_page: int, atom_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand overlapping InfLLM pages to a unique, fixed-width atom union."""
    width = int(atoms_per_page) + 1
    offsets = torch.arange(width, device=page_ids.device, dtype=torch.long)
    expanded = page_ids.to(torch.long).unsqueeze(-1) * int(atoms_per_page) - 1
    expanded = expanded + offsets
    valid = (
        (page_ids.unsqueeze(-1) >= 0) & (expanded >= 0) & (expanded < int(atom_count))
    )
    sentinel = torch.full_like(expanded, int(atom_count))
    sorted_atoms = torch.where(valid, expanded, sentinel).flatten(-2).sort(-1).values
    sorted_valid = sorted_atoms < int(atom_count)
    unique = sorted_valid.clone()
    unique[..., 1:] &= sorted_atoms[..., 1:] != sorted_atoms[..., :-1]
    return (sorted_atoms.clamp(max=max(int(atom_count) - 1, 0)), unique)


def _uses_lightweight_selector(
    selection_method: str, *, is_decode: bool, training_stage: int = 0
) -> bool:
    """Select with LeanSelect or Uniform during prefill."""
    if int(training_stage) != 0:
        return False
    if selection_method in {"uniform", "leanselect"}:
        return not is_decode
    return False


def _uses_forecast_selector(
    selection_method: str, *, enabled: bool, is_decode: bool
) -> bool:
    """Mix/covering decode is always the vanilla actual-Q Sparse baseline."""
    if selection_method in {"leanselect"}:
        return False
    return bool(enabled) and (not (selection_method == "uniform" and bool(is_decode)))


def _projects_forecast_in_attention(
    selection_method: str,
    *,
    q_len: int,
    layer_idx: int,
    num_hidden_layers: int,
    training_stage: int,
) -> bool:
    """Whether Forecast is produced at the attention input read point."""
    if selection_method in {"leanselect"}:
        return False
    if int(training_stage) != 0:
        return True
    return True


def _streaming_sparse_prefill_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    q_len: int,
    k_len: int,
    block_size: int,
    kernel_stride: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    score_tile_bytes: int = 2 << 30,
) -> torch.Tensor:
    """Exact vanilla Sparse prefill in bounded query tiles."""
    batch = int(cu_seqlens_q.numel()) - 1
    n_kv = int(k.shape[1])
    bytes_per_query = max(batch * n_kv * int(k_len) * q.element_size(), 1)
    tile_q = max(1024, int(score_tile_bytes) // bytes_per_query)
    tile_q = min(int(q_len), max(1024, tile_q // 1024 * 1024))
    n_pages = max((int(q_len) + int(block_size) - 1) // int(block_size), 1)
    k_take = min(int(topk), n_pages)
    result = torch.empty(
        (n_kv, batch, int(q_len), k_take), dtype=torch.int32, device=q.device
    )
    q_packed = q.reshape(batch, int(q_len), q.shape[1], q.shape[2])
    k_packed = k.reshape(batch, int(k_len), k.shape[1], k.shape[2])
    cache_lens = torch.empty(batch, dtype=torch.int32, device=q.device)
    stride = int(kernel_stride)
    full_q_grid = (int(q_len) - stride + 1) // stride
    for q0 in range(0, int(q_len), tile_q):
        q1 = min(q0 + tile_q, int(q_len))
        width = q1 - q0
        overlap = stride if q0 else 0
        call_q0 = q0 - overlap
        call_width = width + overlap
        q_tile = (
            q_packed[:, call_q0:q1]
            .reshape(batch * call_width, q.shape[1], q.shape[2])
            .contiguous()
        )
        cu_q_tile = (
            torch.arange(batch + 1, dtype=cu_seqlens_q.dtype, device=q.device)
            * call_width
        )
        if q0:
            tile_q_grid = max(call_width - stride + 1, 0) // stride
            visible_k = int(k_len) - full_q_grid + q0 // stride - 1 + tile_q_grid
        else:
            tile_q_grid = max(width - stride + 1, 0) // stride
            visible_k = int(k_len) - full_q_grid + tile_q_grid
        visible_k = min(max(visible_k, 0), int(k_len))
        if visible_k == 0:
            score = torch.zeros(
                n_kv, batch * call_width, int(k_len), device=q.device, dtype=q.dtype
            )
        else:
            k_tile = (
                k_packed[:, :visible_k]
                .reshape(batch * visible_k, k.shape[1], k.shape[2])
                .contiguous()
            )
            cu_k_tile = (
                torch.arange(batch + 1, dtype=cu_seqlens_k.dtype, device=q.device)
                * visible_k
            )
            compact_score = _call_infllmv2_stage1_compat(
                q_tile,
                k_tile,
                k_tile,
                cu_seqlens_q=cu_q_tile,
                cu_seqlens_k=cu_k_tile,
                cu_seqlens_v=cu_k_tile,
                max_seqlen_q=call_width,
                max_seqlen_k=visible_k,
                causal=True,
            )
            if visible_k == int(k_len):
                score = compact_score
            else:
                score = torch.zeros(
                    n_kv,
                    batch * call_width,
                    int(k_len),
                    device=q.device,
                    dtype=compact_score.dtype,
                )
                score[..., :visible_k].copy_(compact_score[..., :visible_k])
        if overlap:
            score = score.reshape(n_kv, batch, call_width, int(k_len))[
                :, :, overlap:
            ].reshape(n_kv, batch * width, int(k_len))
            cu_q_tile = (
                torch.arange(batch + 1, dtype=cu_seqlens_q.dtype, device=q.device)
                * width
            )
        if score.dtype != torch.bfloat16:
            score = score.to(torch.bfloat16)
        cache_lens.fill_(q0)
        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_q_tile,
            cu_seqlens_k,
            cache_lens,
            int(q_len) - q0,
            int(k_len),
            local_blocks=int(local_blocks),
            init_blocks=int(init_blocks),
            block_size=int(block_size),
            stride=int(kernel_stride),
        )
        tile = block_score.topk(k_take, dim=-1).indices.sort(-1).values
        q_pages = (
            torch.arange(q0, q1, device=q.device, dtype=torch.long) // int(block_size)
        ).repeat(batch)
        tile.masked_fill_(tile > q_pages.view(1, batch * width, 1), -1)
        result[:, :, q0:q1].copy_(
            tile.reshape(n_kv, batch, width, k_take).to(torch.int32)
        )
    return result.reshape(n_kv, batch * int(q_len), k_take)


def _streaming_indexer_prefill_topk(
    q_unpad: torch.Tensor,
    k_compressed: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    block_size: int,
    kernel_stride: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    score_tile_bytes: int = 2 << 30,
) -> torch.Tensor:
    """Bounded-memory SparDA prefill with the unchanged learned indexer."""
    batch = int(cu_seqlens_q.numel()) - 1
    n_kv = int(q_unpad.shape[1])
    head_dim = int(q_unpad.shape[2])
    n_pages = max((int(max_seqlen_q) + int(block_size) - 1) // int(block_size), 1)
    k_take = min(int(topk), n_pages)
    result = torch.empty(
        n_kv, int(q_unpad.shape[0]), k_take, dtype=torch.int32, device=q_unpad.device
    )
    for b in range(batch):
        qs = int(cu_seqlens_q[b].item())
        qe = int(cu_seqlens_q[b + 1].item())
        ks = int(cu_seqlens_k[b].item())
        ke = int(cu_seqlens_k[b + 1].item())
        q_len = qe - qs
        k_len = ke - ks
        if q_len <= 0 or k_len <= 0:
            result[:, qs:qe].fill_(-1)
            continue
        bytes_per_query = max(n_kv * k_len * q_unpad.element_size(), 1)
        tile_q = max(1024, int(score_tile_bytes) // bytes_per_query)
        tile_q = min(q_len, max(1024, tile_q // 1024 * 1024))
        k_t = k_compressed[ks:ke].permute(1, 2, 0).contiguous()
        cu_k = torch.tensor([0, k_len], dtype=torch.int32, device=q_unpad.device)
        for q0 in range(0, q_len, tile_q):
            q1 = min(q0 + tile_q, q_len)
            width = q1 - q0
            q_tile = q_unpad[qs + q0 : qs + q1].permute(1, 0, 2)
            score = torch.bmm(q_tile, k_t)
            q_pos = torch.arange(q0, q1, dtype=torch.long, device=q_unpad.device).view(
                width, 1
            )
            k_pos = (
                torch.arange(k_len, dtype=torch.long, device=q_unpad.device)
                * int(kernel_stride)
            ).view(1, k_len)
            score.masked_fill_(k_pos > q_pos, float("-inf"))
            cu_q = torch.tensor([0, width], dtype=torch.int32, device=q_unpad.device)
            cache = torch.tensor([q0], dtype=torch.int32, device=q_unpad.device)
            block_score = max_pooling_1d_varlen(
                score.contiguous(),
                cu_q,
                cu_k,
                cache,
                int(max_seqlen_q) - q0,
                k_len,
                local_blocks=int(local_blocks),
                init_blocks=int(init_blocks),
                block_size=int(block_size),
                stride=int(kernel_stride),
            )
            tile = block_score.topk(k_take, dim=-1).indices.sort(-1).values
            q_pages = (
                torch.arange(q0, q1, device=q_unpad.device) + int(max_seqlen_q) - q_len
            ) // int(block_size)
            tile.masked_fill_(tile > q_pages.view(1, width, 1), -1)
            result[:, qs + q0 : qs + q1].copy_(tile.to(torch.int32))
    return result


def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens=None,
    precomputed_score: torch.Tensor = None,
    score_stride: int = None,
) -> torch.Tensor:
    """Compute compressed attention scores and return top-k block indices."""
    with torch.no_grad():
        device = q.device if precomputed_score is None else precomputed_score.device
        batch_size = cu_seqlens_q.shape[0] - 1
        is_prefilling = cache_lens is None or (cache_lens == 0).all().item()
        if is_prefilling:
            cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
            q_idx = torch.cat(
                [
                    (
                        torch.arange(
                            cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=device
                        )
                        + max_seqlen_q
                        - (cu_seqlens_q[i + 1] - cu_seqlens_q[i])
                    )
                    // block_size
                    for i in range(batch_size)
                ],
                dim=0,
            )
        else:
            q_idx = cache_lens // block_size
        if is_prefilling and precomputed_score is None:
            q_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            k_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            q_len0 = int(q_lengths[0].item()) if q_lengths.numel() else 0
            k_len0 = int(k_lengths[0].item()) if k_lengths.numel() else 0
            equal_pack = bool(
                q_len0 > 1
                and (q_lengths == q_len0).all().item()
                and (k_lengths == k_len0).all().item()
            )
            score_bytes = (
                int(k.shape[1]) * int(q.shape[0]) * max(k_len0, 1) * q.element_size()
            )
            score_budget = int(
                os.environ.get("LEANSELECT_SCORE_TILE_BYTES", str(2 << 30))
            )
            if equal_pack and score_bytes > score_budget:
                return _streaming_sparse_prefill_topk(
                    q,
                    k,
                    k2,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    cu_seqlens_k2=cu_seqlens_k2,
                    q_len=q_len0,
                    k_len=k_len0,
                    block_size=int(block_size),
                    kernel_stride=int(kernel_stride),
                    topk=int(topk),
                    init_blocks=int(init_blocks),
                    local_blocks=int(local_blocks),
                    score_tile_bytes=score_budget,
                )
        if precomputed_score is not None:
            score = precomputed_score
        else:
            nvtx.range_push("stage 1")
            score = _call_infllmv2_stage1_compat(
                q.contiguous(),
                k.contiguous(),
                k2.contiguous(),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                cu_seqlens_v=cu_seqlens_k2,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=is_prefilling,
            )
            nvtx.range_pop()
        score = score[:, : q_idx.shape[0], :]
        if score.dtype != torch.bfloat16:
            score = score.to(torch.bfloat16)
        nvtx.range_push("pooling")
        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            max_seqlen_k,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=score_stride if score_stride is not None else kernel_stride,
        )
        topk = min(topk, block_score.shape[-1])
        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx > q_idx[None, :, None]] = -1
        topk_idx = topk_idx.to(torch.int32)
        nvtx.range_pop()
    return topk_idx


@lru_cache(maxsize=16)
def calc_chunks_with_stride(cu_seqlen, chunk_size, kernel_stride):
    """Compute the chunks that require Sparse attention, with stride support."""
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    max_seq_len = torch.max(batch_sizes)
    if max_seq_len < chunk_size:
        empty_indices = torch.tensor([], dtype=torch.long, device=cu_seqlen.device)
        cu_seqlens_compressed = torch.zeros(
            len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
        )
        return (empty_indices, cu_seqlens_compressed)
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1
    chunk_start_offsets = torch.arange(
        0,
        max_num_chunks_per_seq * kernel_stride,
        kernel_stride,
        device=cu_seqlen.device,
    )
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]
    chunk_end_in_seq = chunk_start_in_seq + chunk_size
    valid_chunk_mask = chunk_end_in_seq <= seq_starts[:, None] + batch_sizes[:, None]
    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]
    del chunk_start_in_seq
    chunk_indices = torch.arange(0, chunk_size, device=cu_seqlen.device)[None, :]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices
    filtered_indices = filtered_indices.view(-1)
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    )
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    del (
        num_filtered_chunks_per_batch,
        chunk_start_offsets,
        seq_starts,
        chunk_end_in_seq,
        valid_chunk_mask,
        chunk_indices,
    )
    return (filtered_indices, cu_seqlens_compressed)


def _unpad_one_tensor(hidden_states, attention_mask):
    """Unpad a single tensor using attention mask."""
    (indices, cu_seqlens, max_seqlen_in_batch) = _get_unpad_data(attention_mask)
    (batch_size, seq_len) = hidden_states.shape[:2]
    remaining_dims = hidden_states.shape[2:]
    reshaped_states = hidden_states.reshape(batch_size * seq_len, *remaining_dims)
    unpadded_states = index_first_axis(reshaped_states, indices)
    return (unpadded_states, indices, cu_seqlens, max_seqlen_in_batch)


def _get_unpad_data(attention_mask):
    """Get unpadding data for variable length sequences."""
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (indices, cu_seqlens, max_seqlen_in_batch)


def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    chunk = int(os.environ.get("LEANSELECT_PREFILL_NORM_CHUNK_TOKENS", "0"))
    tokens = hidden.numel() // hidden.shape[-1]
    if not torch.is_grad_enabled() and chunk > 0 and (tokens > chunk):
        flat = hidden.reshape(tokens, hidden.shape[-1])
        output = torch.empty_like(flat)
        for start in range(0, tokens, chunk):
            stop = min(start + chunk, tokens)
            row = flat[start:stop]
            variance = row.float().pow(2).mean(dim=-1, keepdim=True)
            normalized = (row * torch.rsqrt(variance + eps)).to(old_dtype)
            output[start:stop].copy_(normalized * weight)
        return output.view_as(hidden)
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class MiniCPMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


ALL_LAYERNORM_LAYERS.append(MiniCPMRMSNorm)


class MiniCPMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (
            torch.arange(0, self.dim, 2).float().to(device) / self.dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.float32,
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MiniCPMLinearScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class MiniCPMDynamicNTKScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                self.scaling_factor * seq_len / self.max_position_embeddings
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / base ** (
                torch.arange(0, self.dim, 2).float().to(device) / self.dim
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class MiniCPMLongRoPE(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with LongRoPE scaling for extended context."""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        short_factor=None,
        long_factor=None,
        original_max_position_embeddings=None,
    ):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        scale = max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = math.sqrt(
            1 + math.log(scale) / math.log(self.original_max_position_embeddings)
        )
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(
                self.long_factor, dtype=torch.float32, device=device
            )
        else:
            ext_factors = torch.tensor(
                self.short_factor, dtype=torch.float32, device=device
            )
        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype),
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos().to(dtype) * self.scaling_factor, persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin().to(dtype) * self.scaling_factor, persistent=False
        )


_MINICPM41_128K_LONGROPE_FACTOR = [
    0.9982316082870437,
    1.033048153422584,
    1.0749920956484724,
    1.1255096879436193,
    1.1863348602111476,
    1.259543828902579,
    1.3476188888731149,
    1.4535223827776373,
    1.5807816745852985,
    1.7335856049489526,
    1.9168922912975785,
    2.1365471404135326,
    2.3994084200118646,
    2.713475511863602,
    3.0880118452194134,
    3.533650295140154,
    4.062463396503134,
    4.687974098908333,
    5.425075306704039,
    6.289818967956352,
    7.29902962722721,
    8.6357018163639,
    10.210822723989212,
    12.053807765671676,
    14.193944598909404,
    16.65780676784363,
    19.463620727694074,
    22.628311203524586,
    26.150106147261315,
    30.02526691405111,
    34.23183327975347,
    38.73811934094828,
    43.502489489729555,
    48.47627117965394,
    53.61139491762471,
    58.857366522037935,
    64.16798299215064,
    69.51359464319125,
    74.86555458220285,
    80.21497790341579,
    85.55322183307433,
    90.89611806932027,
    96.26245306514224,
    101.68269304046481,
    107.18619510219668,
    112.82253283014026,
    118.63764063163615,
    119.88866203644656,
    120.9462882391725,
    121.837565139014,
    122.58663780572562,
    123.2147719894291,
    123.74049454862576,
    124.17980424685767,
    124.54641761955492,
    124.85202548028222,
    125.10654406389756,
    125.31835105170659,
    125.49450117164764,
    125.64091910903052,
    125.76256945356558,
    125.86360463815589,
    125.94749252260765,
    126.01712561287873,
]


def apply_minicpm41_128k_longrope(config):
    """Mutate config to use official MiniCPM4.1-8B 128K LongRoPE factors."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return config
    scaling_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if scaling_type != "longrope":
        return config
    rope_scaling["rope_type"] = "longrope"
    rope_scaling["long_factor"] = list(_MINICPM41_128K_LONGROPE_FACTOR)
    rope_scaling["short_factor"] = list(_MINICPM41_128K_LONGROPE_FACTOR)
    rope_scaling["original_max_position_embeddings"] = 65536
    config.rope_scaling = rope_scaling
    return config


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    orig_dtype = k.dtype
    chunk = int(os.environ.get("LEANSELECT_PREFILL_ROPE_CHUNK_TOKENS", "0"))
    if (
        not torch.is_grad_enabled()
        and chunk > 0
        and (q.shape[-2] > chunk)
        and (position_ids.ndim == 2)
    ):
        q_embed = torch.empty_like(q)
        k_embed = torch.empty_like(k)
        for start in range(0, q.shape[-2], chunk):
            stop = min(start + chunk, q.shape[-2])
            cos_chunk = cos[position_ids[:, start:stop]].unsqueeze(unsqueeze_dim)
            sin_chunk = sin[position_ids[:, start:stop]].unsqueeze(unsqueeze_dim)
            q_chunk = q[..., start:stop, :].float()
            k_chunk = k[..., start:stop, :].float()
            q_embed[..., start:stop, :].copy_(
                (q_chunk * cos_chunk + rotate_half(q_chunk) * sin_chunk).to(
                    dtype=orig_dtype
                )
            )
            k_embed[..., start:stop, :].copy_(
                (k_chunk * cos_chunk + rotate_half(k_chunk) * sin_chunk).to(
                    dtype=orig_dtype
                )
            )
        return (q_embed, k_embed)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    q_embed = q_fp32 * cos + rotate_half(q_fp32) * sin
    k_embed = k_fp32 * cos + rotate_half(k_fp32) * sin
    return (q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype))


def apply_rotary_pos_emb_q_only(q, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to query tensor only."""
    orig_dtype = q.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    q_embed = q_fp32 * cos + rotate_half(q_fp32) * sin
    return q_embed.to(dtype=orig_dtype)


class MiniCPMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.inference_chunk_tokens = int(
            os.environ.get("LEANSELECT_PREFILL_FFN_CHUNK_TOKENS", "0")
        )
        if self.inference_chunk_tokens < 0:
            raise ValueError("LEANSELECT_PREFILL_FFN_CHUNK_TOKENS must be non-negative")

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)
            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            chunk = self.inference_chunk_tokens
            if (
                not torch.is_grad_enabled()
                and chunk
                and (x.numel() // x.shape[-1] > chunk)
            ):
                flat = x.reshape(-1, x.shape[-1])
                down_flat = torch.empty_like(flat)
                for start in range(0, flat.shape[0], chunk):
                    stop = min(start + chunk, flat.shape[0])
                    x_chunk = flat[start:stop]
                    gated = self.act_fn(self.gate_proj(x_chunk))
                    gated.mul_(self.up_proj(x_chunk))
                    down_flat[start:stop].copy_(self.down_proj(gated))
                down_proj = down_flat.view_as(x)
            else:
                down_proj = self.down_proj(
                    self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                )
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (for GQA/MQA)."""
    (batch, num_key_value_heads, slen, head_dim) = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class CompressK(torch.nn.Module):
    """Module for compressing key (K) representations using mean pooling."""

    def __init__(self, head_num_k, head_dim, kernel_size, kernel_stride=16):
        """Args:"""
        super().__init__()
        self.kernel_size = kernel_size
        self.head_num_k = head_num_k
        self.head_dim = head_dim
        self.kernel_stride = kernel_stride

    def forward(self, k: torch.Tensor, cu_seqlens):
        """Forward pass for compressing the key (K) tensor."""
        (filtered_k_indices, cu_seqlens_compressed) = calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )
        if filtered_k_indices.numel() == 0:
            compressed_k = k.new_empty(0, self.head_num_k, self.head_dim)
            return (compressed_k, cu_seqlens_compressed)
        filtered_k = k.index_select(0, filtered_k_indices.view(-1))
        filtered_k = filtered_k.view(
            filtered_k.shape[0] // self.kernel_size,
            self.kernel_size,
            self.head_num_k,
            self.head_dim,
        )
        compressed_k = filtered_k.mean(dim=1)
        return (compressed_k, cu_seqlens_compressed)


class InfLLMv2CacheLayer(DynamicLayer):
    """Cache layer for InfLLM V2 sparse attention with compressed key storage."""

    def __init__(self):
        super().__init__()
        self.no_rope_keys = torch.tensor([], dtype=torch.float32)
        self.compress_k_cache = []
        self.cached_compressed_cu_seqlens = torch.tensor([], dtype=torch.int32)
        self.compress_k_cache_varlen = torch.tensor([], dtype=torch.float32)
        self.no_compress_k_cache = []
        self.compress_k2_cache = []
        self.cached_compressed_cu_seqlens2 = torch.tensor([], dtype=torch.int32)
        self.compress_k2_cache_varlen = torch.tensor([], dtype=torch.float32)
        self.no_compress_k2_cache = []
        self.compress_k_cache_kl_target = []
        self.cached_compressed_cu_seqlens_kl_target = torch.tensor(
            [], dtype=torch.int32
        )
        self.compress_k_cache_varlen_kl_target = torch.tensor([], dtype=torch.float32)
        self.no_compress_k_cache_kl_target = []
        self.compress_k2_cache_kl_target = []
        self.cached_compressed_cu_seqlens2_kl_target = torch.tensor(
            [], dtype=torch.int32
        )
        self.compress_k2_cache_varlen_kl_target = torch.tensor([], dtype=torch.float32)
        self.no_compress_k2_cache_kl_target = []

    def _infer_empty_tensor_shape(self, key_states):
        """Infer device/dtype/shape from key_states or KV cache for creating empty tensors."""
        (device, dtype, num_kv_heads, head_dim) = (None, None, None, None)
        keys = getattr(self, "keys", None)
        if isinstance(keys, torch.Tensor) and keys.numel() > 0 and (keys.dim() == 4):
            (device, dtype) = (keys.device, keys.dtype)
            (num_kv_heads, head_dim) = (keys.shape[1], keys.shape[-1])
        if device is None:
            for k in (
                key_states if isinstance(key_states, (list, tuple)) else [key_states]
            ):
                if isinstance(k, torch.Tensor):
                    (device, dtype) = (k.device, k.dtype)
                    (num_kv_heads, head_dim) = (k.shape[-2], k.shape[-1])
                    break
        if device is None:
            device = (
                keys.device if isinstance(keys, torch.Tensor) else torch.device("cpu")
            )
            dtype = keys.dtype if isinstance(keys, torch.Tensor) else torch.float32
            (num_kv_heads, head_dim) = (0, 0)
        return (device, dtype, num_kv_heads, head_dim)

    def _update_compress_cache(
        self, key_states, cu_seqlens, cache_list, cu_attr, varlen_attr
    ):
        """Generic compressed key cache update (used by all compress_k variants)."""
        if (
            cu_seqlens is None
            and cache_list
            and isinstance(key_states, (list, tuple))
            and all((key is None for key in key_states))
        ):
            return (getattr(self, varlen_attr), getattr(self, cu_attr))
        if len(cache_list) == 0:
            if cu_seqlens is not None:
                setattr(self, cu_attr, cu_seqlens.clone())
                setattr(self, varlen_attr, key_states)
                split_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
                cache_list[:] = list(torch.split(key_states, split_sizes))
            elif isinstance(key_states, (list, tuple)):
                batch_size = len(key_states)
                (device, dtype, num_kv_heads, head_dim) = (
                    self._infer_empty_tensor_shape(key_states)
                )
                empty = torch.empty(
                    (0, num_kv_heads, head_dim), device=device, dtype=dtype
                )
                cache_list[:] = [
                    k if isinstance(k, torch.Tensor) else empty for k in key_states
                ]
                setattr(
                    self,
                    varlen_attr,
                    torch.cat(cache_list, dim=0) if batch_size > 0 else empty,
                )
                seq_lens = torch.tensor(
                    [t.shape[0] for t in cache_list], dtype=torch.int32, device=device
                )
                cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
                if batch_size > 0:
                    cu[1:] = torch.cumsum(seq_lens, dim=0)
                setattr(self, cu_attr, cu)
            else:
                if not isinstance(key_states, torch.Tensor):
                    raise TypeError(f"Unexpected key_states type: {type(key_states)}")
                device = key_states.device
                cu = torch.tensor(
                    [0, key_states.shape[0]], dtype=torch.int32, device=device
                )
                setattr(self, cu_attr, cu)
                setattr(self, varlen_attr, key_states)
                cache_list[:] = [key_states]
        else:
            for index, k in enumerate(key_states):
                if k is not None:
                    cache_list[index] = torch.cat([cache_list[index], k], dim=0)
            new_seq_lens = torch.tensor(
                [t.shape[0] for t in cache_list], dtype=torch.int32
            )
            new_cumsum = torch.cumsum(new_seq_lens, dim=0, dtype=torch.int32)
            varlen = torch.cat(cache_list, dim=0)
            setattr(self, varlen_attr, varlen)
            setattr(
                self,
                cu_attr,
                torch.cat([torch.tensor([0], dtype=torch.int32), new_cumsum]).to(
                    varlen.device
                ),
            )
        return (getattr(self, varlen_attr), getattr(self, cu_attr))

    @staticmethod
    def _update_no_compress_cache(
        key_states, no_compress_cache, kernel_size, kernel_stride
    ):
        """Generic uncompressed key buffer update for incremental compression."""
        k_chunk_list = []
        for index, k in enumerate(key_states):
            if len(no_compress_cache) <= index:
                no_compress_cache.append(k)
            else:
                no_compress_cache[index] = torch.cat(
                    [no_compress_cache[index], k], dim=0
                )
                current_len = no_compress_cache[index].shape[0]
                if current_len >= kernel_size:
                    k_chunk_list.append(no_compress_cache[index][:kernel_size])
                    no_compress_cache[index] = no_compress_cache[index][kernel_stride:]
                else:
                    k_chunk_list.append(None)
        return k_chunk_list

    def update_no_rope_key(self, key_states):
        """Update the no-rope key cache."""
        if self.no_rope_keys.numel() == 0:
            self.no_rope_keys = key_states
        else:
            self.no_rope_keys = torch.cat([self.no_rope_keys, key_states], dim=1)
        return self.no_rope_keys

    def update_compress_k(self, key_states, cu_seqlens=None):
        return self._update_compress_cache(
            key_states,
            cu_seqlens,
            self.compress_k_cache,
            "cached_compressed_cu_seqlens",
            "compress_k_cache_varlen",
        )

    def update_compress_k_kl_target(self, key_states, cu_seqlens=None):
        return self._update_compress_cache(
            key_states,
            cu_seqlens,
            self.compress_k_cache_kl_target,
            "cached_compressed_cu_seqlens_kl_target",
            "compress_k_cache_varlen_kl_target",
        )

    def update_compress_k2(self, key_states, cu_seqlens=None):
        return self._update_compress_cache(
            key_states,
            cu_seqlens,
            self.compress_k2_cache,
            "cached_compressed_cu_seqlens2",
            "compress_k2_cache_varlen",
        )

    def update_compress_k2_kl_target(self, key_states, cu_seqlens=None):
        return self._update_compress_cache(
            key_states,
            cu_seqlens,
            self.compress_k2_cache_kl_target,
            "cached_compressed_cu_seqlens2_kl_target",
            "compress_k2_cache_varlen_kl_target",
        )

    def update_no_compress_k(self, key_states, kernel_size=32, kernel_stride=16):
        return self._update_no_compress_cache(
            key_states, self.no_compress_k_cache, kernel_size, kernel_stride
        )

    def update_no_compress_k_kl_target(
        self, key_states, kernel_size=32, kernel_stride=16
    ):
        return self._update_no_compress_cache(
            key_states, self.no_compress_k_cache_kl_target, kernel_size, kernel_stride
        )

    def update_no_compress_k2(self, key_states, kernel_size=128, kernel_stride=64):
        return self._update_no_compress_cache(
            key_states, self.no_compress_k2_cache, kernel_size, kernel_stride
        )

    def update_no_compress_k2_kl_target(
        self, key_states, kernel_size=128, kernel_stride=64
    ):
        return self._update_no_compress_cache(
            key_states, self.no_compress_k2_cache_kl_target, kernel_size, kernel_stride
        )


class InfLLMv2Cache(DynamicCache):
    """Dynamic cache for InfLLM V2 with compressed key storage per layer."""

    def __init__(self, config=None, num_hidden_layers: int | None = None) -> None:
        super().__init__()
        self.layers = (
            [InfLLMv2CacheLayer() for _ in range(num_hidden_layers)]
            if num_hidden_layers
            else []
        )
        self._seen_tokens = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def update_no_rope_key(self, key_states, layer_idx, cache_kwargs=None):
        return self.layers[layer_idx].update_no_rope_key(key_states)

    def update_compress_k(
        self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None
    ):
        return self.layers[layer_idx].update_compress_k(key_states, cu_seqlens)

    def update_no_compress_k(
        self, key_states, layer_idx, kernel_size=32, kernel_stride=16, cache_kwargs=None
    ):
        return self.layers[layer_idx].update_no_compress_k(
            key_states, kernel_size, kernel_stride
        )

    def update_compress_k2(
        self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None
    ):
        return self.layers[layer_idx].update_compress_k2(key_states, cu_seqlens)

    def update_no_compress_k2(
        self,
        key_states,
        layer_idx,
        kernel_size=128,
        kernel_stride=64,
        cache_kwargs=None,
    ):
        return self.layers[layer_idx].update_no_compress_k2(
            key_states, kernel_size, kernel_stride
        )

    def update_compress_k_kl_target(
        self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None
    ):
        return self.layers[layer_idx].update_compress_k_kl_target(
            key_states, cu_seqlens
        )

    def update_no_compress_k_kl_target(
        self, key_states, layer_idx, kernel_size=32, kernel_stride=16, cache_kwargs=None
    ):
        return self.layers[layer_idx].update_no_compress_k_kl_target(
            key_states, kernel_size, kernel_stride
        )

    def update_compress_k2_kl_target(
        self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None
    ):
        return self.layers[layer_idx].update_compress_k2_kl_target(
            key_states, cu_seqlens
        )

    def update_no_compress_k2_kl_target(
        self,
        key_states,
        layer_idx,
        kernel_size=128,
        kernel_stride=64,
        cache_kwargs=None,
    ):
        return self.layers[layer_idx].update_no_compress_k2_kl_target(
            key_states, kernel_size, kernel_stride
        )

    def crop(self, max_length):
        for layer in self.layers:
            layer.crop(max_length)

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices):
        for layer in self.layers:
            layer.batch_select_indices(indices)


class MiniCPMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MiniCPMConfig, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended."
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MiniCPMRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling.get(
                "rope_type", self.config.rope_scaling.get("type", "linear")
            )
            scaling_factor = self.config.rope_scaling.get("factor", None)
            if scaling_type == "linear":
                self.rotary_emb = MiniCPMLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = MiniCPMDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "longrope":
                self.rotary_emb = MiniCPMLongRoPE(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    short_factor=self.config.rope_scaling["short_factor"],
                    long_factor=self.config.rope_scaling["long_factor"],
                    base=self.rope_theta,
                    original_max_position_embeddings=self.config.rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if "padding_mask" in kwargs:
            warnings.warn("Passing `padding_mask` is deprecated.")
        (bsz, q_len, _) = hidden_states.size()
        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim // self.config.pretraining_tp
            )
            query_slices = self.q_proj.weight.split(
                self.num_heads * self.head_dim // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)
            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)
            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if hasattr(past_key_value, "get_usable_length"):
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx
                )
            elif hasattr(past_key_value, "get_seq_length"):
                kv_seq_len += past_key_value.get_seq_length()
        (cos, sin) = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)
        (query_states, key_states) = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            (key_states, value_states) = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return (attn_output, attn_weights, past_key_value)


class MiniCPMFlashAttention2(MiniCPMAttention):
    """MiniCPM flash attention module with proper unpadding support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if "padding_mask" in kwargs:
            warnings.warn("Passing `padding_mask` is deprecated.")
            attention_mask = kwargs.pop("padding_mask")
        (bsz, q_len, _) = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if hasattr(past_key_value, "get_usable_length"):
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx
                )
            elif hasattr(past_key_value, "get_seq_length"):
                kv_seq_len += past_key_value.get_seq_length()
        (cos, sin) = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)
        (query_states, key_states) = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            (key_states, value_states) = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        dropout_rate = self.attention_dropout if self.training else 0.0
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32. We will cast back in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return (attn_output, attn_weights, past_key_value)

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """Flash Attention with proper unpadding for variable length sequences."""
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
            (cu_seqlens_q, cu_seqlens_k) = cu_seq_lens
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k) = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        (indices_k, cu_seqlens_k, max_seqlen_in_batch_k) = _get_unpad_data(
            attention_mask
        )
        (batch_size, kv_seq_len, num_key_value_heads, head_dim) = key_layer.shape
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            (query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q) = unpad_input(
                query_layer, attention_mask
            )
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class MiniCPMSdpaAttention(MiniCPMAttention):
    """MiniCPM attention using torch.nn.functional.scaled_dot_product_attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if output_attentions:
            logger.warning_once(
                "MiniCPMModel is using MiniCPMSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        (bsz, q_len, _) = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if hasattr(past_key_value, "get_usable_length"):
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx
                )
            elif hasattr(past_key_value, "get_seq_length"):
                kv_seq_len += past_key_value.get_seq_length()
        (cos, sin) = self.rotary_emb(value_states, seq_len=kv_seq_len)
        (query_states, key_states) = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            (key_states, value_states) = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal and attention_mask is None and (q_len > 1),
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return (attn_output, None, past_key_value)


class MiniCPMInfLLMv2Attention(MiniCPMAttention):
    """MiniCPM attention using InfLLM V2 kernels with q_future prediction training support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.config._attn_implementation == "flash_attention_2", (
            "Only flash_attention_2 is supported for sparse attention"
        )
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        sparse_config = getattr(self.config, "sparse_config", None) or {}
        self.sparse_config = sparse_config
        self.kernel_size = sparse_config.get("kernel_size", 32)
        self.kernel_stride = sparse_config.get("kernel_stride", 16)
        self.init_blocks = sparse_config.get("init_blocks", 1)
        self.block_size = sparse_config.get("block_size", 64)
        self.window_size = sparse_config.get("window_size", 2048)
        self.dense_len = sparse_config.get("dense_len", 8192)
        self.local_blocks = self.window_size // self.block_size
        self.topk = sparse_config.get("topk", 64) + self.window_size // self.block_size
        self.use_nope = sparse_config.get("use_nope", False)
        self.use_q_future_for_topk = sparse_config.get("use_q_future_for_topk", True)
        self.use_q_future_decode_only = sparse_config.get(
            "use_q_future_decode_only", False
        )
        self.selection_method = str(
            sparse_config.get("selection_method", "current")
        ).lower()
        if self.selection_method not in {
            "current",
            "minference",
            "uniform",
            "leanselect",
            "recency",
            "xattention",
        }:
            raise ValueError(f"Unsupported selection_method={self.selection_method!r}")
        if self.selection_method in _TRAINING_FREE_SCORERS and (
            not _TRAINING_FREE_SCORERS[self.selection_method][1]
        ):
            raise RuntimeError(
                f"selection_method={self.selection_method!r} requested but its scorer module failed to import; refusing to silently fall back to the native Sparse selector."
            )
        self.leanselect_total_topk_blocks = int(self.topk)
        declared_phase_topk = sparse_config.get("leanselect_total_topk_blocks")
        if (
            self.selection_method in {"leanselect", "uniform"}
            and declared_phase_topk is not None
            and (int(declared_phase_topk) != self.leanselect_total_topk_blocks)
        ):
            raise ValueError(
                f"leanselect_total_topk_blocks must equal the effective InfLLM budget topk + local_blocks: declared={declared_phase_topk}, effective={self.leanselect_total_topk_blocks}"
            )
        self.leanselect = None
        self.leanselect_supervision_rows = int(
            sparse_config.get("leanselect_supervision_rows", 64)
        )
        self.leanselect_worst_group_weight = float(
            sparse_config.get("leanselect_worst_group_weight", 0.1)
        )
        if self.selection_method == "leanselect":
            from leanselect.selector import LeanSelectSelector

            self.leanselect = LeanSelectSelector(
                num_heads=self.num_heads,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                router_heads=int(sparse_config.get("leanselect_router_heads", 4)),
                router_dim=int(sparse_config.get("leanselect_router_dim", 32)),
            )
        _tr_ks = sparse_config.get("training_kernel_size", None)
        _tr_kst = sparse_config.get("training_kernel_stride", None)
        self.training_kernel_size = int(self.kernel_size if _tr_ks is None else _tr_ks)
        self.training_kernel_stride = int(
            self.kernel_stride if _tr_kst is None else _tr_kst
        )
        self.kl_target_align_pooling = str(
            sparse_config.get("kl_target_align_pooling", "max")
        ).lower()
        if self.kl_target_align_pooling not in {"max", "mean"}:
            raise ValueError(
                f"Unsupported kl_target_align_pooling={self.kl_target_align_pooling!r}"
            )
        self._kl_target_compress_differs = (
            self.training_kernel_size != self.kernel_size
            or self.training_kernel_stride != self.kernel_stride
        )
        self.compress_k = CompressK(
            self.num_key_value_heads,
            self.head_dim,
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride,
        )
        self.compress_k2 = CompressK(
            self.num_key_value_heads,
            self.head_dim,
            kernel_size=self.kernel_size * 4,
            kernel_stride=self.kernel_stride * 4,
        )
        if self._kl_target_compress_differs:
            self.compress_k_kl_target = CompressK(
                self.num_key_value_heads,
                self.head_dim,
                kernel_size=self.training_kernel_size,
                kernel_stride=self.training_kernel_stride,
            )
            self.compress_k2_kl_target = CompressK(
                self.num_key_value_heads,
                self.head_dim,
                kernel_size=self.training_kernel_size * 4,
                kernel_stride=self.training_kernel_stride * 4,
            )
        else:
            self.compress_k_kl_target = self.compress_k
            self.compress_k2_kl_target = self.compress_k2
        create_indexer = sparse_config.get(
            "create_indexer", True
        ) and self.selection_method not in {"leanselect", "uniform"}
        create_current_indexer = bool(sparse_config.get("create_q_curr_indexer", True))
        if create_indexer:
            self.q_future_proj = self._new_indexer_projection()
            if self.layer_idx == 0 and create_current_indexer:
                self.q_curr_proj = self._new_indexer_projection()
        else:
            self.q_future_proj = None
            self.q_curr_proj = None
        self.infinigen_enabled = False
        self.infinigen_num_channels = 32
        self._ig_skewing_matrix = None
        self._ig_partial_indices = None
        self._ig_partial_key_cache = None
        self._ig_warmed_up = False
        self._ig_prev_hidden_states = None
        self._ig_attn_input = None
        self._ig_max_new_tokens = None
        self._path_infinigen = 0
        self._path_fallback = 0
        self.selector_loss_mode = (
            "leanselect" if self.selection_method == "leanselect" else "legacy_kl"
        )
        from leanselect.objective import UtilityLossConfig

        self.utility_loss_config = UtilityLossConfig(
            listwise_weight=float(sparse_config.get("listwise_weight", 1.0)),
            boundary_weight=float(sparse_config.get("boundary_weight", 0.1)),
            temperature=float(sparse_config.get("loss_temperature", 1.0)),
            boundary_margin=float(sparse_config.get("boundary_margin", 0.1)),
        )
        self._last_selector_loss_metrics = None
        self.collect_selection_stats = False
        self.uniform_head_mix = "uniform"
        self._last_selection_stats = None

    def _new_indexer_projection(self) -> nn.Linear:
        projection = nn.Linear(
            self.config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        projection._is_indexer = True
        return projection

    def _forecast_projection(self, q_len: int, *, current: bool) -> nn.Linear | None:
        kind = "curr" if current else "future"
        return getattr(self, f"q_{kind}_proj", None)

    def _project_forecast_queries(
        self, indexer_input: torch.Tensor, *, include_current: bool = True
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        (bsz, q_len, _) = indexer_input.shape
        q_future_proj = self._forecast_projection(q_len, current=False)
        if q_future_proj is None:
            return (None, None)
        output_shape = (bsz, q_len, self.num_key_value_heads, self.head_dim)
        q_future = q_future_proj(indexer_input).view(output_shape)
        q_curr_proj = (
            self._forecast_projection(q_len, current=True) if include_current else None
        )
        q_curr = (
            q_curr_proj(indexer_input).view(output_shape)
            if q_curr_proj is not None
            else None
        )
        return (q_future, q_curr)

    def _apply_forecast_rope(
        self,
        q_forecast: torch.Tensor,
        position_ids: torch.LongTensor,
        token_length: int,
    ) -> torch.Tensor:
        q_forecast_t = q_forecast.transpose(1, 2)
        (cos, sin) = self.rotary_emb(q_forecast_t.float(), seq_len=int(token_length))
        q_forecast_t = apply_rotary_pos_emb_q_only(q_forecast_t, cos, sin, position_ids)
        return q_forecast_t.transpose(1, 2)

    def _reset_selection_diagnostics(self):
        self._last_selection_stats = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        q_future_from_prev: torch.Tensor | None = None,
        training_stage: int = 0,
        **kwargs,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor] | None,
        tuple | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        """Forward pass with InfLLM V2 sparse attention and integrated indexer."""
        self._reset_selection_diagnostics()
        if "padding_mask" in kwargs:
            warnings.warn("Passing `padding_mask` is deprecated.")
            attention_mask = kwargs.pop("padding_mask")
        (bsz, q_len, _) = hidden_states.size()
        cache_owner_id = None if past_key_value is None else id(past_key_value)
        if getattr(self, "infinigen_enabled", False):
            self._ig_attn_input = hidden_states.detach()
        nvtx.range_push("linear")
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        q_future = None
        q_curr = None
        forecast_projection_needed = _projects_forecast_in_attention(
            self.selection_method,
            q_len=q_len,
            layer_idx=self.layer_idx,
            num_hidden_layers=int(self.config.num_hidden_layers),
            training_stage=training_stage,
        )
        if not getattr(self, "infinigen_enabled", False) and forecast_projection_needed:
            indexer_input = (
                hidden_states.detach() if training_stage == 1 else hidden_states
            )
            (q_future, q_curr) = self._project_forecast_queries(
                indexer_input, include_current=True
            )
        if self.use_nope:
            query_states_no_rope = query_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            )
            key_states_no_rope = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            )
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        nvtx.range_pop()
        nvtx.range_push("rope")
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if isinstance(past_key_value, Cache) or hasattr(
                past_key_value, "get_seq_length"
            ):
                kv_seq_len += past_key_value.get_seq_length()
        (cos, sin) = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)
        (query_states, key_states) = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        if q_future is not None:
            q_future_t = q_future.transpose(1, 2)
            q_future_t = apply_rotary_pos_emb_q_only(q_future_t, cos, sin, position_ids)
            q_future = q_future_t.transpose(1, 2)
        if q_curr is not None:
            q_curr_t = q_curr.transpose(1, 2)
            q_curr_t = apply_rotary_pos_emb_q_only(q_curr_t, cos, sin, position_ids)
            q_curr = q_curr_t.transpose(1, 2)
        if (
            getattr(self, "infinigen_enabled", False)
            and getattr(self, "_ig_warmed_up", False)
            and (q_len > 1)
            and (getattr(self, "_ig_skewing_matrix", None) is not None)
            and (getattr(self, "_ig_partial_key_cache", None) is None)
            and (getattr(self, "_ig_partial_indices", None) is None)
            and hasattr(self, "_refresh_partial_indices_from_prefill")
        ):
            self._refresh_partial_indices_from_prefill(query_states, key_states)
        key_states_for_indexer = key_states.transpose(1, 2)
        nvtx.range_pop()
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            (key_states, value_states) = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if (
            self.use_nope
            and past_key_value is not None
            and isinstance(past_key_value, InfLLMv2Cache)
        ):
            key_states_no_rope = past_key_value.update_no_rope_key(
                key_states_no_rope, self.layer_idx
            )
            no_rope_param = {
                "key_states_no_rope": key_states_no_rope,
                "query_states_no_rope": query_states_no_rope,
            }
        else:
            no_rope_param = None
        dropout_rate = self.attention_dropout if self.training else 0.0
        kv_seq_len_current = key_states.shape[1]
        ig_token_scores = None
        if (
            q_len == 1
            and getattr(self, "infinigen_enabled", False)
            and getattr(self, "_ig_warmed_up", False)
            and (self.layer_idx > 0)
            and (getattr(self, "_ig_prev_hidden_states", None) is not None)
            and (getattr(self, "_ig_partial_indices", None) is not None)
            and (getattr(self, "_ig_partial_key_cache", None) is not None)
            and _INFINIGEN_AVAILABLE
        ):
            ig_token_scores = infinigen_token_scores_hf(
                self._ig_prev_hidden_states,
                self.q_proj.weight.data,
                self._ig_skewing_matrix,
                self._ig_partial_indices,
                self._ig_partial_key_cache[:, :kv_seq_len_current],
                cos,
                sin,
                position_ids,
                self.num_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.infinigen_num_channels,
                apply_rotary_pos_emb,
            )
        if (
            getattr(self, "infinigen_enabled", False)
            and getattr(self, "_ig_warmed_up", False)
            and (getattr(self, "_ig_skewing_matrix", None) is not None)
            and (getattr(self, "_ig_partial_indices", None) is not None)
        ):
            if q_len > 1 and getattr(self, "_ig_partial_key_cache", None) is None:
                max_cache_len = kv_seq_len_current + int(
                    getattr(self, "_ig_max_new_tokens", 0)
                )
                self._ig_partial_key_cache = torch.zeros(
                    bsz,
                    max_cache_len,
                    self.num_key_value_heads,
                    self.infinigen_num_channels,
                    dtype=key_states.dtype,
                    device=key_states.device,
                )
            if getattr(self, "_ig_partial_key_cache", None) is not None:
                if q_len == 1:
                    new_k = key_states[:, -1, :, :]
                    infinigen_update_partial_cache(
                        self._ig_partial_key_cache,
                        new_k,
                        kv_seq_len_current - 1,
                        self._ig_skewing_matrix,
                        self._ig_partial_indices,
                    )
                elif q_len > 1:
                    partial_k = infinigen_build_partial_cache(
                        key_states, self._ig_skewing_matrix, self._ig_partial_indices
                    )
                    write_len = min(
                        kv_seq_len_current, self._ig_partial_key_cache.shape[1]
                    )
                    self._ig_partial_key_cache[:, :write_len] = partial_k[:, :write_len]
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32. We will cast back in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        aux_scores = (None, None)
        if training_stage not in (0, 1):
            raise ValueError(
                f"Unsupported training_stage={training_stage}. Expected 0 (inference) or 1 (training)."
            )
        if (
            training_stage == 0
            and (not getattr(self, "infinigen_enabled", False))
            and (kv_seq_len_current < self.dense_len)
        ):
            attn_output = self._flash_attention_forward_dense(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=dropout_rate,
            )
        else:
            (attn_output, aux_scores) = self._sparse_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=dropout_rate,
                no_rope_param=no_rope_param,
                past_key_value=past_key_value,
                q_future_from_prev=q_future_from_prev,
                q_curr=q_curr,
                key_states_for_indexer=key_states_for_indexer,
                training_stage=training_stage,
                ig_token_scores=ig_token_scores,
            )
        nvtx.range_push("o linear")
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        nvtx.range_pop()
        return (
            attn_output,
            None,
            past_key_value,
            aux_scores,
            q_future,
            q_curr,
            key_states_for_indexer,
        )

    def _sparse_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        dropout: float = 0.0,
        softmax_scale: float = None,
        no_rope_param: dict = None,
        past_key_value: Cache | None = None,
        q_future_from_prev: torch.Tensor | None = None,
        q_curr: torch.Tensor | None = None,
        key_states_for_indexer: torch.Tensor | None = None,
        training_stage: int = 0,
        ig_token_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """Sparse attention forward pass with integrated indexer support."""
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1
        aux_scores = (None, None)
        if attention_mask is None:
            raise ValueError("Need attention_mask for sparse attention")
        batch_size = query_states.shape[0]
        q_pred_for_kl_pre = (
            q_future_from_prev if q_future_from_prev is not None else q_curr
        )
        enable_kl_target_cache = (
            training_stage == 1
            and (
                q_pred_for_kl_pre is not None or self.selection_method in {"leanselect"}
            )
            and (not _SKIP_KL_LOSS)
        )
        should_compute_kl = enable_kl_target_cache
        if past_key_value is not None and isinstance(past_key_value, InfLLMv2Cache):
            nvtx.range_push("compress")
            (
                compressed_k,
                compressed_cu_seqlens,
                compressed_k2,
                compressed_cu_seqlens2,
            ) = self.get_compress_k(
                key_states=key_states
                if not self.use_nope
                else no_rope_param["key_states_no_rope"],
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                enable_kl_target_cache=enable_kl_target_cache,
            )
            compressed_k_kl_target = compressed_k
            compressed_cu_seqlens_kl_target = compressed_cu_seqlens
            compressed_k2_kl_target = compressed_k2
            compressed_cu_seqlens2_kl_target = compressed_cu_seqlens2
            if enable_kl_target_cache and self._kl_target_compress_differs:
                layer_cache = past_key_value.layers[self.layer_idx]
                if layer_cache.compress_k_cache_varlen_kl_target.numel() > 0:
                    compressed_k_kl_target = (
                        layer_cache.compress_k_cache_varlen_kl_target
                    )
                    compressed_cu_seqlens_kl_target = (
                        layer_cache.cached_compressed_cu_seqlens_kl_target
                    )
                if layer_cache.compress_k2_cache_varlen_kl_target.numel() > 0:
                    compressed_k2_kl_target = (
                        layer_cache.compress_k2_cache_varlen_kl_target
                    )
                    compressed_cu_seqlens2_kl_target = (
                        layer_cache.cached_compressed_cu_seqlens2_kl_target
                    )
            nvtx.range_pop()
        (
            query_states_unpad,
            key_states_unpad,
            value_states_unpad,
            indices_q,
            cu_seq_lens,
            max_seq_lens,
        ) = self._upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )
        (cu_seqlens_q, cu_seqlens_k) = cu_seq_lens
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) = max_seq_lens
        if no_rope_param is not None:
            if max_seqlen_in_batch_q == 1:
                no_rope_param["query_states_no_rope"] = no_rope_param[
                    "query_states_no_rope"
                ].squeeze(1)
            else:
                (no_rope_param["query_states_no_rope"], _, _, _) = _unpad_one_tensor(
                    no_rope_param["query_states_no_rope"], attention_mask=attention_mask
                )
        if past_key_value is None or not isinstance(past_key_value, InfLLMv2Cache):
            nvtx.range_push("compress")
            (compressed_k, compressed_cu_seqlens) = self.compress_k(
                key_states_unpad, cu_seqlens_k
            )
            (compressed_k2, compressed_cu_seqlens2) = self.compress_k2(
                key_states_unpad, cu_seqlens_k
            )
            compressed_k_kl_target = compressed_k
            compressed_cu_seqlens_kl_target = compressed_cu_seqlens
            compressed_k2_kl_target = compressed_k2
            compressed_cu_seqlens2_kl_target = compressed_cu_seqlens2
            if enable_kl_target_cache and self._kl_target_compress_differs:
                (compressed_k_kl_target, compressed_cu_seqlens_kl_target) = (
                    self.compress_k_kl_target(key_states_unpad, cu_seqlens_k)
                )
                if self.selection_method not in {"leanselect"}:
                    (compressed_k2_kl_target, compressed_cu_seqlens2_kl_target) = (
                        self.compress_k2_kl_target(key_states_unpad, cu_seqlens_k)
                    )
            nvtx.range_pop()
        q_actual = (
            query_states_unpad
            if no_rope_param is None
            else no_rope_param["query_states_no_rope"]
        )
        cache_lens = None
        if max_seqlen_in_batch_q == 1 and max_seqlen_in_batch_k > 1:
            seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            cache_lens = seq_lens_k - 1
        compressed_k_for_indexer = compressed_k.detach()
        use_pred_scores_for_topk = False
        q_pred_for_topk = None
        if training_stage == 0:
            q_pred_for_topk = (
                q_future_from_prev if q_future_from_prev is not None else q_curr
            )
            use_pred_scores_for_topk = (
                _uses_forecast_selector(
                    self.selection_method,
                    enabled=self.use_q_future_for_topk,
                    is_decode=cache_lens is not None,
                )
                and q_pred_for_topk is not None
                and (compressed_k_for_indexer is not None)
                and (compressed_k_for_indexer.numel() > 0)
            )
            if self.use_q_future_decode_only:
                is_decode = cache_lens is not None
                use_pred_scores_for_topk = use_pred_scores_for_topk and is_decode
        elif training_stage == 1:
            use_pred_scores_for_topk = False
        use_aligned_training_topk = (
            training_stage == 1
            and self._kl_target_compress_differs
            and (self.selection_method not in {"leanselect"})
        )
        lightweight_selection_method = _uses_lightweight_selector(
            self.selection_method,
            is_decode=cache_lens is not None,
            training_stage=training_stage,
        )
        if lightweight_selection_method:
            topk_idx = self._lightweight_compressed_attention(
                q_actual,
                compressed_k,
                cu_seqlens_q,
                compressed_cu_seqlens,
                max_seqlen_in_batch_q,
                cache_lens=cache_lens,
                q_pred_for_topk=q_pred_for_topk,
            )
        xa_token_scores = None
        xa_topk_idx = None
        _scorer = _TRAINING_FREE_SCORERS.get(self.selection_method)
        if self.selection_method in _TRAINING_FREE_SCORERS and cache_lens is None:
            (_fn, _available) = _scorer
            if not _available:
                raise RuntimeError(
                    f"selection_method={self.selection_method!r} requested but its scorer module failed to import; refusing to silently fall back to the native Sparse selector."
                )
            if int(cu_seqlens_q.numel()) - 1 != 1:
                raise NotImplementedError(
                    f"selection_method={self.selection_method!r} is verified for batch-1 packed prefill only."
                )
            _n_kv_heads = int(key_states_unpad.shape[1])
            _grid_cells = (
                _n_kv_heads * int(q_actual.shape[0]) * int(max_seqlen_in_batch_k)
            )
            if _grid_cells > _TF_TOKEN_GRID_CELL_LIMIT:
                from leanselect.baselines.page_topk import tf_page_topk as _page_topk_fn

                if self.selection_method == "xattention":
                    from leanselect.baselines.xattention import xattention_page_scores

                    _pages = xattention_page_scores(
                        q_actual,
                        key_states_unpad,
                        q_len=int(q_actual.shape[0]),
                        k_len=int(key_states_unpad.shape[0]),
                        block_size=int(self.block_size),
                        stride=int(self.kernel_stride),
                    )
                elif self.selection_method == "minference":
                    from leanselect.baselines.minference import vs_pattern_page_scores

                    _pages = vs_pattern_page_scores(
                        q_actual,
                        key_states_unpad,
                        q_len=int(q_actual.shape[0]),
                        k_len=int(key_states_unpad.shape[0]),
                        block_size=int(self.block_size),
                    )
                else:
                    from leanselect.baselines.recency import recency_page_scores

                    _pages = recency_page_scores(
                        q_actual,
                        key_states_unpad,
                        q_len=int(q_actual.shape[0]),
                        k_len=int(key_states_unpad.shape[0]),
                        block_size=int(self.block_size),
                    )
                _n_heads = int(_pages.shape[0])
                if _n_kv_heads != _n_heads:
                    _group = _n_heads // _n_kv_heads
                    _pages = _pages.reshape(
                        _n_kv_heads, _group, _pages.shape[1], _pages.shape[2]
                    ).mean(dim=1)
                xa_topk_idx = _page_topk_fn(
                    _pages,
                    topk=int(self.topk),
                    block_size=int(self.block_size),
                    stride=int(self.kernel_stride),
                    init_blocks=int(self.init_blocks),
                    local_blocks=int(self.local_blocks),
                    q_len=int(q_actual.shape[0]),
                    k_len=int(key_states_unpad.shape[0]),
                    max_seqlen_q=int(max_seqlen_in_batch_q),
                    max_seqlen_k=int(max_seqlen_in_batch_k),
                )
            elif self.selection_method == "xattention":
                xa_token_scores = _fn(
                    q_actual,
                    key_states_unpad,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    block_size=int(self.block_size),
                    stride=int(self.kernel_stride),
                )
            elif self.selection_method in ("minference", "recency"):
                xa_token_scores = _fn(
                    q_actual,
                    key_states_unpad,
                    q_len=int(q_actual.shape[0]),
                    k_len=int(key_states_unpad.shape[0]),
                    block_size=int(self.block_size),
                    max_seqlen_k=int(max_seqlen_in_batch_k),
                )
        if ig_token_scores is not None:
            topk_idx = compressed_attention(
                q_actual,
                compressed_k,
                compressed_k2,
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.topk,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_seqlens_k,
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                cache_lens=cache_lens,
                precomputed_score=ig_token_scores,
                score_stride=1,
            )
        elif xa_topk_idx is not None:
            topk_idx = xa_topk_idx
        elif xa_token_scores is not None:
            topk_idx = compressed_attention(
                q_actual,
                compressed_k,
                compressed_k2,
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.topk,
                cu_seqlens_q,
                cu_seqlens_k,
                cu_seqlens_k,
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                cache_lens=cache_lens,
                precomputed_score=xa_token_scores,
                score_stride=1,
            )
        elif lightweight_selection_method:
            pass
        elif use_aligned_training_topk:
            q_topk_unpad = q_actual
            if use_pred_scores_for_topk:
                q_topk_unpad = self._unpad_indexer_query(
                    q_pred_for_topk, attention_mask
                )
            topk_idx = self._compute_training_aligned_topk_idx(
                q_topk_unpad,
                compressed_k_kl_target.detach(),
                cu_seqlens_q,
                compressed_cu_seqlens_kl_target,
                compressed_cu_seqlens,
                max_seqlen_in_batch_q,
                cache_lens,
                causal,
            )
        elif use_pred_scores_for_topk:
            topk_idx = self._indexer_compressed_attention(
                q_pred_for_topk,
                compressed_k_for_indexer,
                attention_mask,
                cu_seqlens_q,
                compressed_cu_seqlens,
                max_seqlen_in_batch_q,
                cache_lens=cache_lens,
            )
        else:
            compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
            max_seqlen_compressed = compressed_seqlens.max().item()
            topk_idx = compressed_attention(
                q_actual,
                compressed_k,
                compressed_k2,
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.topk,
                cu_seqlens_q,
                compressed_cu_seqlens,
                compressed_cu_seqlens2,
                max_seqlen_in_batch_q,
                max_seqlen_compressed,
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                cache_lens=cache_lens,
            )
        if getattr(self, "_collect_block_stats", False):
            self._last_topk_idx = topk_idx.detach()
        q_pred_for_kl = q_pred_for_kl_pre
        if should_compute_kl:
            loss_topk_idx = topk_idx
            (scores_pred, scores_target) = self._compute_indexer_kl_loss(
                q_actual,
                q_pred_for_kl,
                compressed_k_kl_target,
                compressed_k_for_indexer,
                cu_seqlens_q,
                compressed_cu_seqlens_kl_target,
                compressed_cu_seqlens,
                compressed_cu_seqlens,
                attention_mask,
                causal=causal,
                topk_idx=loss_topk_idx,
                compressed_k2=compressed_k2_kl_target,
                compressed_cu_seqlens2=compressed_cu_seqlens2_kl_target,
            )
            aux_scores = (scores_pred, scores_target)
        nvtx.range_push("stage 2")
        attn_output_unpad = self.sparse_forward(
            query_states_unpad,
            key_states_unpad,
            value_states_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_in_batch_q,
            max_seqlen_in_batch_k,
            no_rope_param=no_rope_param,
            topk_idx=topk_idx,
        )
        nvtx.range_pop()
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        return (attn_output, aux_scores)

    def get_compress_k(
        self,
        key_states: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_value: InfLLMv2Cache,
        enable_kl_target_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get compressed key states, handling both prefilling and decoding."""
        is_prefilling = not past_key_value.layers[self.layer_idx].compress_k_cache
        if is_prefilling:
            (unpadded_key_states, indices, cu_seqlens, max_seqlen_in_batch) = (
                _unpad_one_tensor(key_states, attention_mask=attention_mask)
            )
            (compressed_k, compressed_cu_seqlens) = self.compress_k(
                unpadded_key_states, cu_seqlens
            )
            (compressed_k2, compressed_cu_seqlens2) = self.compress_k2(
                unpadded_key_states, cu_seqlens
            )
            past_key_value.update_compress_k(
                compressed_k, self.layer_idx, compressed_cu_seqlens
            )
            past_key_value.update_compress_k2(
                compressed_k2, self.layer_idx, compressed_cu_seqlens2
            )
            if enable_kl_target_cache and self._kl_target_compress_differs:
                (compressed_k_t, compressed_cu_seqlens_t) = self.compress_k_kl_target(
                    unpadded_key_states, cu_seqlens
                )
                (compressed_k2_t, compressed_cu_seqlens2_t) = (
                    self.compress_k2_kl_target(unpadded_key_states, cu_seqlens)
                )
                past_key_value.update_compress_k_kl_target(
                    compressed_k_t, self.layer_idx, compressed_cu_seqlens_t
                )
                past_key_value.update_compress_k2_kl_target(
                    compressed_k2_t, self.layer_idx, compressed_cu_seqlens2_t
                )
            no_compress_k_list = []
            for i in range(len(compressed_cu_seqlens) - 1):
                no_compress_k_start = (
                    compressed_cu_seqlens[i + 1] - compressed_cu_seqlens[i]
                ) * self.kernel_stride
                no_compress_k_list.append(
                    unpadded_key_states[
                        cu_seqlens[i] + no_compress_k_start : cu_seqlens[i + 1]
                    ].clone()
                )
            past_key_value.update_no_compress_k(
                no_compress_k_list,
                self.layer_idx,
                kernel_stride=self.kernel_stride,
                kernel_size=self.kernel_size,
            )
            if enable_kl_target_cache and self._kl_target_compress_differs:
                no_compress_k_list_t = []
                compressed_cu_seqlens_t = past_key_value.layers[
                    self.layer_idx
                ].cached_compressed_cu_seqlens_kl_target
                for i in range(len(compressed_cu_seqlens_t) - 1):
                    no_compress_k_start_t = (
                        compressed_cu_seqlens_t[i + 1] - compressed_cu_seqlens_t[i]
                    ) * self.training_kernel_stride
                    no_compress_k_list_t.append(
                        unpadded_key_states[
                            cu_seqlens[i] + no_compress_k_start_t : cu_seqlens[i + 1]
                        ].clone()
                    )
                past_key_value.update_no_compress_k_kl_target(
                    no_compress_k_list_t,
                    self.layer_idx,
                    kernel_stride=self.training_kernel_stride,
                    kernel_size=self.training_kernel_size,
                )
            no_compress_k2_list = []
            for i in range(len(compressed_cu_seqlens2) - 1):
                no_compress_k2_start = (
                    (compressed_cu_seqlens2[i + 1] - compressed_cu_seqlens2[i])
                    * self.kernel_stride
                    * 4
                )
                no_compress_k2_list.append(
                    unpadded_key_states[
                        cu_seqlens[i] + no_compress_k2_start : cu_seqlens[i + 1]
                    ].clone()
                )
            past_key_value.update_no_compress_k2(
                no_compress_k2_list,
                self.layer_idx,
                kernel_stride=self.kernel_stride * 4,
                kernel_size=self.kernel_size * 4,
            )
            if enable_kl_target_cache and self._kl_target_compress_differs:
                no_compress_k2_list_t = []
                compressed_cu_seqlens2_t = past_key_value.layers[
                    self.layer_idx
                ].cached_compressed_cu_seqlens2_kl_target
                for i in range(len(compressed_cu_seqlens2_t) - 1):
                    no_compress_k2_start_t = (
                        (compressed_cu_seqlens2_t[i + 1] - compressed_cu_seqlens2_t[i])
                        * self.training_kernel_stride
                        * 4
                    )
                    no_compress_k2_list_t.append(
                        unpadded_key_states[
                            cu_seqlens[i] + no_compress_k2_start_t : cu_seqlens[i + 1]
                        ].clone()
                    )
                past_key_value.update_no_compress_k2_kl_target(
                    no_compress_k2_list_t,
                    self.layer_idx,
                    kernel_stride=self.training_kernel_stride * 4,
                    kernel_size=self.training_kernel_size * 4,
                )
        else:
            batch_size = key_states.shape[0]
            key_states_split = list(
                torch.split(key_states[:, -1:].squeeze(1), [1] * batch_size, dim=0)
            )
            no_compress_k_list = past_key_value.update_no_compress_k(
                key_states_split,
                self.layer_idx,
                kernel_stride=self.kernel_stride,
                kernel_size=self.kernel_size,
            )
            new_compressed_k_list = []
            for no_compress_k in no_compress_k_list:
                if no_compress_k is not None:
                    new_compressed_k = no_compress_k.mean(dim=0, keepdim=True)
                    new_compressed_k_list.append(new_compressed_k)
                else:
                    new_compressed_k_list.append(None)
            (compressed_k, compressed_cu_seqlens) = past_key_value.update_compress_k(
                new_compressed_k_list, self.layer_idx
            )
            if enable_kl_target_cache and self._kl_target_compress_differs:
                no_compress_k_list_t = past_key_value.update_no_compress_k_kl_target(
                    key_states_split,
                    self.layer_idx,
                    kernel_stride=self.training_kernel_stride,
                    kernel_size=self.training_kernel_size,
                )
                new_compressed_k_list_t = []
                for no_compress_k in no_compress_k_list_t:
                    if no_compress_k is not None:
                        new_compressed_k = no_compress_k.mean(dim=0, keepdim=True)
                        new_compressed_k_list_t.append(new_compressed_k)
                    else:
                        new_compressed_k_list_t.append(None)
                past_key_value.update_compress_k_kl_target(
                    new_compressed_k_list_t, self.layer_idx
                )
            no_compress_k2_list = past_key_value.update_no_compress_k2(
                key_states_split,
                self.layer_idx,
                kernel_stride=self.kernel_stride * 4,
                kernel_size=self.kernel_size * 4,
            )
            new_compressed_k2_list = []
            for no_compress_k2 in no_compress_k2_list:
                if no_compress_k2 is not None:
                    new_compressed_k2 = no_compress_k2.mean(dim=0, keepdim=True)
                    new_compressed_k2_list.append(new_compressed_k2)
                else:
                    new_compressed_k2_list.append(None)
            (compressed_k2, compressed_cu_seqlens2) = past_key_value.update_compress_k2(
                new_compressed_k2_list, self.layer_idx
            )
            if enable_kl_target_cache and self._kl_target_compress_differs:
                no_compress_k2_list_t = past_key_value.update_no_compress_k2_kl_target(
                    key_states_split,
                    self.layer_idx,
                    kernel_stride=self.training_kernel_stride * 4,
                    kernel_size=self.training_kernel_size * 4,
                )
                new_compressed_k2_list_t = []
                for no_compress_k2 in no_compress_k2_list_t:
                    if no_compress_k2 is not None:
                        new_compressed_k2 = no_compress_k2.mean(dim=0, keepdim=True)
                        new_compressed_k2_list_t.append(new_compressed_k2)
                    else:
                        new_compressed_k2_list_t.append(None)
                past_key_value.update_compress_k2_kl_target(
                    new_compressed_k2_list_t, self.layer_idx
                )
        return (
            compressed_k,
            compressed_cu_seqlens,
            compressed_k2,
            compressed_cu_seqlens2,
        )

    def _get_indexer_scale(self) -> float:
        """Get softmax scaling factor."""
        return 1.0 / math.sqrt(self.head_dim)

    def _compute_indexer_block_scores(
        self,
        q_future: torch.Tensor | None,
        k_compressed: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_k: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute Q @ K scores for inference top-k block selection."""
        num_kv_heads = self.num_key_value_heads
        batch_size = q_future.shape[0]
        seq_len_q = q_future.shape[1]
        head_dim = self.head_dim
        device = q_future.device
        dtype = q_future.dtype
        is_decode = seq_len_q == 1
        k_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        max_seqlen_k = k_seqlens.max().item()
        if is_decode:
            q = q_future.squeeze(1)
            k_padded = torch.zeros(
                batch_size,
                max_seqlen_k,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )
            for b in range(batch_size):
                (k_start, k_end) = (cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item())
                k_len = k_end - k_start
                if k_len > 0:
                    k_padded[b, :k_len] = k_compressed[k_start:k_end]
            k_t = k_padded.permute(0, 2, 3, 1)
            q_bmm = q.view(batch_size * num_kv_heads, 1, head_dim)
            k_bmm = k_t.reshape(batch_size * num_kv_heads, head_dim, max_seqlen_k)
            prev_reduced = (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            )
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            try:
                batch_scores = torch.bmm(q_bmm, k_bmm).view(
                    batch_size, num_kv_heads, max_seqlen_k
                )
            finally:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                    prev_reduced
                )
            score = batch_scores.permute(1, 0, 2)
        else:
            (indices_q, _, _) = _get_unpad_data(attention_mask)
            q_future_flat = q_future.reshape(-1, num_kv_heads, head_dim)
            q_unpad = index_first_axis(q_future_flat, indices_q)
            total_q = q_unpad.shape[0]
            score = torch.full(
                (num_kv_heads, total_q, max_seqlen_k),
                float("-inf"),
                device=device,
                dtype=dtype,
            )
            for b in range(batch_size):
                (q_start, q_end) = (cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item())
                (k_start, k_end) = (cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item())
                (q_len, k_len) = (q_end - q_start, k_end - k_start)
                if q_len == 0 or k_len == 0:
                    continue
                q_b = q_unpad[q_start:q_end]
                k_b = k_compressed[k_start:k_end].permute(1, 0, 2)
                q_b_t = q_b.permute(1, 0, 2)
                k_b_t = k_b.transpose(1, 2)
                prev_reduced = (
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                )
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                    False
                )
                try:
                    raw_scores = torch.bmm(q_b_t, k_b_t)
                finally:
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prev_reduced
                q_positions = torch.arange(q_len, device=device).unsqueeze(1)
                k_positions = (
                    torch.arange(k_len, device=device).unsqueeze(0) * self.kernel_stride
                )
                causal_mask = k_positions > q_positions
                raw_scores = raw_scores.masked_fill(
                    causal_mask.unsqueeze(0), float("-inf")
                )
                score[:, q_start:q_end, :k_len] = raw_scores
        return score

    def _lightweight_compressed_attention(
        self,
        q_unpad: torch.Tensor,
        k_compressed: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        cache_lens: torch.Tensor | None = None,
        q_pred_for_topk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cache_lens is not None:
            raise ValueError("Lightweight selection is prefill only")
        if self.selection_method == "uniform":
            from leanselect.ops.uniform import uniform_prefill_topk_idx

            return uniform_prefill_topk_idx(
                q_unpad,
                k_compressed,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                topk=self.topk,
                block_size=self.block_size,
                kernel_stride=self.kernel_stride,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
            )[0]
        if cache_lens is not None:
            if self.selection_method == "leanselect":
                raise RuntimeError("leanselect is Prefill-only")
        elif self.leanselect is None:
            raise RuntimeError("LeanSelect selector was not initialized")
        if cache_lens is None:
            q_offsets = cu_seqlens_q.detach().to("cpu").tolist()
            k_offsets = cu_seqlens_k.detach().to("cpu").tolist()
            max_pages = max(
                (
                    (q_offsets[b + 1] - q_offsets[b] + self.block_size - 1)
                    // self.block_size
                    for b in range(len(q_offsets) - 1)
                ),
                default=0,
            )
            take = min(self.topk, max_pages)
            topk_idx = torch.full(
                (self.num_key_value_heads, q_unpad.shape[0], take),
                -1,
                dtype=torch.int32,
                device=q_unpad.device,
            )
            for batch_idx in range(len(q_offsets) - 1):
                (q_start, q_end) = q_offsets[batch_idx : batch_idx + 2]
                (k_start, k_end) = k_offsets[batch_idx : batch_idx + 2]
                q_len = q_end - q_start
                if q_len <= 0 or k_end <= k_start:
                    continue
                scores = self.leanselect.score_pages(
                    q_unpad[q_start:q_end],
                    k_compressed[k_start:k_end],
                    seq_len=q_len,
                    block_size=self.block_size,
                    kernel_stride=self.kernel_stride,
                )
                page_ids = torch.arange(scores.shape[-1], device=scores.device)
                query_pages = (
                    torch.arange(q_len, device=scores.device) // self.block_size
                )
                visible = page_ids[None] <= query_pages[:, None]
                local = visible & (
                    page_ids[None] + self.local_blocks >= query_pages[:, None]
                )
                forced = (page_ids < self.init_blocks)[None] | local
                scores = scores.masked_fill(~visible, float("-inf"))
                scores = scores.masked_fill(forced & visible, float("inf"))
                batch_take = min(take, scores.shape[-1])
                (values, indices) = torch.topk(
                    scores, batch_take, dim=-1, largest=True, sorted=False
                )
                invalid = torch.isnan(values) | torch.isneginf(values)
                indices = indices.masked_fill(invalid, -1).to(torch.int32)
                topk_idx[:, q_start:q_end, :batch_take] = indices
            return topk_idx

    def _indexer_compressed_attention(
        self,
        q_future: torch.Tensor,
        k_compressed: torch.Tensor,
        attention_mask: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        cache_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Indexer compressed attention for inference top-k block selection."""
        self._last_selection_stats = None
        is_prefill = cache_lens is None or bool((cache_lens == 0).all().item())
        if is_prefill:
            k_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            max_k = int(k_seqlens.max().item()) if k_seqlens.numel() else 0
            total_q = int(cu_seqlens_q[-1].item())
            score_bytes = (
                self.num_key_value_heads
                * total_q
                * max(max_k, 1)
                * q_future.element_size()
            )
            score_budget = int(
                os.environ.get("LEANSELECT_SCORE_TILE_BYTES", str(2 << 30))
            )
            if score_bytes > score_budget:
                q_unpad = self._unpad_indexer_query(q_future, attention_mask)
                return _streaming_indexer_prefill_topk(
                    q_unpad,
                    k_compressed,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=int(max_seqlen_q),
                    block_size=self.block_size,
                    kernel_stride=self.kernel_stride,
                    topk=self.topk,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                    score_tile_bytes=score_budget,
                )
        nvtx.range_push("stage 1")
        score = self._compute_indexer_block_scores(
            q_future,
            k_compressed,
            attention_mask,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
        )
        nvtx.range_pop()
        k_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        max_seqlen_k = k_seqlens.max().item()
        topk_idx = compressed_attention(
            q=None,
            k=None,
            k2=None,
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride,
            block_size=self.block_size,
            topk=self.topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_k2=None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            cache_lens=cache_lens,
            precomputed_score=score,
        )
        return topk_idx

    def sparse_forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_in_batch_q: int,
        max_seqlen_in_batch_k: int,
        topk_idx: torch.Tensor,
        no_rope_param: dict = None,
    ) -> torch.Tensor:
        """Execute sparse attention with pre-computed top-k block indices."""
        topk_attn_output = infllmv2_attn_varlen_func(
            query_layer,
            key_layer,
            value_layer,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_in_batch_q,
            max_seqlen_in_batch_k,
            dropout_p=0.0,
            deterministic=False,
            softmax_scale=None,
            causal=max_seqlen_in_batch_q != 1,
            return_attn_probs=False,
            topk_idx=topk_idx,
        )
        return topk_attn_output

    def _flash_attention_forward_dense(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        dropout: float = 0.0,
        softmax_scale: float = None,
    ) -> torch.Tensor:
        """Dense flash attention for short sequences."""
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
            (cu_seqlens_q, cu_seqlens_k) = cu_seq_lens
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k) = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        return attn_output

    def _upad_input(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
    ):
        """Unpad inputs for variable length sequences."""
        (indices_k, cu_seqlens_k, max_seqlen_in_batch_k) = _get_unpad_data(
            attention_mask
        )
        (batch_size, kv_seq_len, num_key_value_heads, head_dim) = key_layer.shape
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            (query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q) = unpad_input(
                query_layer, attention_mask
            )
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def _unpad_indexer_query(
        self, q_states: torch.Tensor | None, attention_mask: torch.Tensor
    ) -> torch.Tensor | None:
        if q_states is None:
            return None
        if q_states.shape[1] == 1:
            return q_states.squeeze(1)
        return _unpad_one_tensor(q_states, attention_mask)[0]

    def _build_aligned_training_probs(
        self,
        q_unpad: torch.Tensor,
        compressed_k_train: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        compressed_cu_seqlens_train: torch.Tensor,
        compressed_cu_seqlens_default: torch.Tensor,
        max_seqlen_q: int,
        cache_lens: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        """Compute training-grid stage1 probabilities and align them to the default grid."""
        num_kv_heads = self.num_key_value_heads
        total_q = int(cu_seqlens_q[-1].item())
        default_seqlens = (
            compressed_cu_seqlens_default[1:] - compressed_cu_seqlens_default[:-1]
        )
        max_default_len = int(default_seqlens.max().item())
        aligned = torch.zeros(
            (num_kv_heads, total_q, max_default_len),
            device=q_unpad.device,
            dtype=q_unpad.dtype,
        )
        q_seqlens = cu_seqlens_q.detach().to("cpu").tolist()
        train_seqlens = compressed_cu_seqlens_train.detach().to("cpu").tolist()
        default_seqlens_list = compressed_cu_seqlens_default.detach().to("cpu").tolist()
        cu_q_buf = torch.empty((2,), device=q_unpad.device, dtype=torch.int32)
        cu_k_buf = torch.empty((2,), device=q_unpad.device, dtype=torch.int32)
        for b in range(len(q_seqlens) - 1):
            (q_start, q_end) = (q_seqlens[b], q_seqlens[b + 1])
            (k_start, k_end) = (train_seqlens[b], train_seqlens[b + 1])
            default_len = default_seqlens_list[b + 1] - default_seqlens_list[b]
            q_len = q_end - q_start
            k_len = k_end - k_start
            if q_len == 0 or k_len == 0 or default_len == 0:
                continue
            q_batch = q_unpad[q_start:q_end].contiguous()
            k_batch = compressed_k_train[k_start:k_end].contiguous()
            cu_q_buf[0] = 0
            cu_q_buf[1] = q_len
            cu_k_buf[0] = 0
            cu_k_buf[1] = k_len
            query_offset = 0
            if cache_lens is not None and q_len == 1:
                query_offset = int(cache_lens[b].item())
            probs = _call_infllmv2_stage1_kl_teacher_compat(
                q_batch,
                k_batch,
                k_batch,
                cu_seqlens_q=cu_q_buf,
                cu_seqlens_k=cu_k_buf,
                cu_seqlens_v=cu_k_buf,
                max_seqlen_q=q_len,
                max_seqlen_k=k_len,
                causal=causal,
                causal_stride=self.training_kernel_stride,
                query_offset=query_offset,
            )
            probs = probs[:, :q_len, :k_len].contiguous()
            probs = _align_probabilities_to_target_grid_1d(
                probs.reshape(-1, 1, k_len),
                source_stride=self.training_kernel_stride,
                target_stride=self.kernel_stride,
                target_length=default_len,
                mode=self.kl_target_align_pooling,
            ).reshape(num_kv_heads, q_len, default_len)
            aligned[:, q_start:q_end, :default_len] = probs
        return aligned

    def _compute_training_aligned_topk_idx(
        self,
        q_unpad: torch.Tensor,
        compressed_k_train: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        compressed_cu_seqlens_train: torch.Tensor,
        compressed_cu_seqlens_default: torch.Tensor,
        max_seqlen_q: int,
        cache_lens: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        aligned_probs = self._build_aligned_training_probs(
            q_unpad,
            compressed_k_train,
            cu_seqlens_q,
            compressed_cu_seqlens_train,
            compressed_cu_seqlens_default,
            max_seqlen_q,
            cache_lens,
            causal,
        )
        default_seqlens = (
            compressed_cu_seqlens_default[1:] - compressed_cu_seqlens_default[:-1]
        )
        max_default_len = int(default_seqlens.max().item())
        return compressed_attention(
            q=None,
            k=None,
            k2=None,
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride,
            block_size=self.block_size,
            topk=self.topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=compressed_cu_seqlens_default,
            cu_seqlens_k2=None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_default_len,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            cache_lens=cache_lens,
            precomputed_score=aligned_probs,
            score_stride=self.kernel_stride,
        )

    def _compute_indexer_kl_loss(
        self,
        query_states_unpad: torch.Tensor,
        q_future: torch.Tensor,
        compressed_k_attn: torch.Tensor,
        compressed_k_index: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        compressed_cu_seqlens_attn: torch.Tensor,
        compressed_cu_seqlens_index: torch.Tensor,
        compressed_cu_seqlens_default: torch.Tensor | None,
        attention_mask: torch.Tensor,
        causal: bool = True,
        topk_idx: torch.Tensor = None,
        compressed_k2: torch.Tensor = None,
        compressed_cu_seqlens2: torch.Tensor = None,
    ) -> tuple[torch.Tensor, str]:
        from leanselect.ops.supervision import selector_loss

        return (
            selector_loss(
                self,
                query_states_unpad,
                compressed_k_attn,
                compressed_k_index,
                cu_seqlens_q,
                compressed_cu_seqlens_attn,
                compressed_cu_seqlens_index,
                compressed_cu_seqlens_default,
                attention_mask,
                compressed_k2,
                compressed_cu_seqlens2,
            ),
            "precomputed",
        )


MINICPM_ATTENTION_CLASSES = {
    "eager": MiniCPMAttention,
    "flash_attention_2": MiniCPMFlashAttention2,
    "sdpa": MiniCPMSdpaAttention,
}


def _get_attention_class(config, layer_idx: int):
    """Select the configured attention implementation without a silent fallback."""
    if getattr(config, "sparse_config", None) is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("A sparse MiniCPM configuration requires CUDA")
        return MiniCPMInfLLMv2Attention
    return MINICPM_ATTENTION_CLASSES[config._attn_implementation]


class MiniCPMDecoderLayer(nn.Module):
    """Decoder layer with InfLLM V2 indexer for sparse attention."""

    def __init__(self, config: MiniCPMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        attention_class = _get_attention_class(config, layer_idx)
        self.self_attn = attention_class(config=config, layer_idx=layer_idx)
        self.use_sparse_attn = attention_class == MiniCPMInfLLMv2Attention
        self.indexer_future_params = []
        self.indexer_curr_params = []
        if self.use_sparse_attn:
            attn = self.self_attn
            future_names = ["q_future_proj"]
            curr_names = ["q_curr_proj"]
            for name in future_names:
                projection = getattr(attn, name, None)
                if projection is not None:
                    self.indexer_future_params.extend(projection.parameters())
            for name in curr_names:
                projection = getattr(attn, name, None)
                if projection is not None:
                    self.indexer_curr_params.extend(projection.parameters())
            if getattr(attn, "leanselect", None) is not None:
                self.indexer_curr_params.extend(attn.leanselect.parameters())
        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MiniCPMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        q_future_from_prev: torch.Tensor | None = None,
        training_stage: int = 0,
        **kwargs,
    ) -> tuple:
        """Forward pass for decoder layer with indexer."""
        if "padding_mask" in kwargs:
            warnings.warn("Passing `padding_mask` is deprecated.")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        is_first_layer = self.layer_idx == 0
        (q_future, q_curr) = (None, None)
        aux_scores = None
        if self.use_sparse_attn:
            q_for_scoring = q_future_from_prev if not is_first_layer else None
            attn_result = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                q_future_from_prev=q_for_scoring,
                training_stage=training_stage,
                **kwargs,
            )
            (
                hidden_states,
                self_attn_weights,
                present_key_value,
                aux_scores,
                q_future,
                q_curr,
                key_states_for_indexer,
            ) = attn_result
        else:
            attn_result = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            (hidden_states, self_attn_weights, present_key_value) = attn_result
            aux_scores = None
        hidden_states = residual + hidden_states * (
            self.scale_depth / math.sqrt(self.num_hidden_layers)
        )
        nvtx.range_push("ffn")
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (
            self.scale_depth / math.sqrt(self.num_hidden_layers)
        )
        nvtx.range_pop()
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if self.use_sparse_attn:
            outputs += (q_future, aux_scores)
        return outputs


class MiniCPMPreTrainedModel(PreTrainedModel, GenerationMixin):
    config_class = MiniCPMConfig
    base_model_prefix = "model"
    _no_split_modules = ["MiniCPMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        """Enable or disable gradient checkpointing for decoder layers."""
        if isinstance(module, MiniCPMModel):
            module.gradient_checkpointing = value

    def _init_weights(self, module):
        std = self.config.initializer_range
        sparse_config = getattr(self.config, "sparse_config", None) or {}
        use_kaiming_for_indexer = sparse_config.get("indexer_kaiming_init", True)
        is_indexer = getattr(module, "_is_indexer", False)
        from leanselect.selector import LeanSelectSelector

        if isinstance(module, (LeanSelectSelector,)):
            module.reset_parameters()
        elif isinstance(module, nn.Linear):
            if use_kaiming_for_indexer and is_indexer:
                fan_in = module.weight.shape[1]
                bound = 1.0 / math.sqrt(fan_in)
                module.weight.data.uniform_(-bound, bound)
            else:
                module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MiniCPMModel(MiniCPMPreTrainedModel):
    """MiniCPM decoder model with InfLLM V2 support."""

    def __init__(self, config: MiniCPMConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                MiniCPMDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sparse_attn = (
            getattr(config, "sparse_config", None) is not None
            and torch.cuda.is_available()
        )
        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        training_stage: int = 0,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            (batch_size, seq_length) = input_ids.shape[:2]
        elif inputs_embeds is not None:
            (batch_size, seq_length) = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        past_key_values_length = 0
        use_legacy_cache = False
        if use_cache:
            if past_key_values is not None:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            if isinstance(past_key_values, InfLLMv2Cache):
                past_key_values_length = past_key_values.get_seq_length()
            elif hasattr(past_key_values, "get_usable_length"):
                past_key_values_length = past_key_values.get_usable_length(seq_length)
            elif hasattr(past_key_values, "get_seq_length"):
                past_key_values_length = past_key_values.get_seq_length()
            else:
                past_key_values_length = 0
            if self._use_sparse_attn and past_key_values_length == 0:
                past_key_values = InfLLMv2Cache(
                    config=self.config, num_hidden_layers=self.config.num_hidden_layers
                )
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb
        if self._use_flash_attention_2 or self._use_sparse_attn:
            if attention_mask is None:
                raise ValueError(
                    "attention_mask is required for flash attention or sparse attention"
                )
        elif self._use_sdpa and (not output_attentions):
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        q_future_prev = None
        self._prev_layer_indexer_params = None
        accumulated_kl_loss = None
        kl_loss_count = 0
        use_cache_effective = use_cache and (
            not (self.gradient_checkpointing and self.training)
        )
        _infinigen_active = any(
            (
                getattr(decoder_layer.self_attn, "infinigen_enabled", False)
                for decoder_layer in self.layers
            )
        )
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    output_attentions,
                    False,
                    q_future_prev,
                    training_stage,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache_effective,
                    q_future_from_prev=q_future_prev,
                    training_stage=training_stage,
                )
            hidden_states = layer_outputs[0]
            if _infinigen_active and idx + 1 < len(self.layers):
                next_attn = self.layers[idx + 1].self_attn
                attn = decoder_layer.self_attn
                if (
                    getattr(next_attn, "infinigen_enabled", False)
                    and hasattr(attn, "_ig_attn_input")
                    and (attn._ig_attn_input is not None)
                ):
                    next_attn._ig_prev_hidden_states = attn._ig_attn_input
            if decoder_layer.use_sparse_attn:
                q_future_prev = layer_outputs[-2]
                aux_scores = layer_outputs[-1]
                if aux_scores is not None and training_stage == 1:
                    (scores_pred, scores_target) = aux_scores
                    if scores_target == "precomputed" and scores_pred is not None:
                        if (
                            self.training
                            and hasattr(self, "_accumulate_kl_grads")
                            and self._accumulate_kl_grads
                        ):
                            if (
                                hasattr(self, "_prev_layer_indexer_params")
                                and self._prev_layer_indexer_params is not None
                            ):
                                target_params = self._prev_layer_indexer_params
                            else:
                                target_params = [
                                    p
                                    for p in decoder_layer.indexer_curr_params
                                    if p.requires_grad
                                ]
                            if target_params:
                                scaled_loss = scores_pred / self._kl_grad_scale
                                grads = torch.autograd.grad(
                                    scaled_loss,
                                    target_params,
                                    retain_graph=False,
                                    allow_unused=True,
                                )
                                for param, grad in zip(target_params, grads):
                                    if grad is None:
                                        continue
                                    if param.grad is None:
                                        param.grad = grad
                                    else:
                                        param.grad.add_(grad)
                            if accumulated_kl_loss is None:
                                accumulated_kl_loss = scores_pred.detach()
                                kl_loss_count = 1
                            else:
                                accumulated_kl_loss = (
                                    accumulated_kl_loss + scores_pred.detach()
                                )
                                kl_loss_count += 1
                            self._kl_grads_need_allreduce = True
                        elif accumulated_kl_loss is None:
                            accumulated_kl_loss = scores_pred
                            kl_loss_count = 1
                        else:
                            accumulated_kl_loss = accumulated_kl_loss + scores_pred
                            kl_loss_count += 1
                if (
                    self.training
                    and hasattr(self, "_accumulate_kl_grads")
                    and self._accumulate_kl_grads
                ):
                    self._prev_layer_indexer_params = [
                        p
                        for p in decoder_layer.indexer_future_params
                        if p.requires_grad
                    ]
                del aux_scores
                if training_stage == 1:
                    try:
                        del scores_pred, scores_target
                    except NameError:
                        pass
            if use_cache_effective:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = None
        if use_cache_effective and next_decoder_cache is not None:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        avg_kl_loss = None
        if accumulated_kl_loss is not None and kl_loss_count > 0:
            avg_kl_loss = accumulated_kl_loss / kl_loss_count
        if not return_dict:
            result = tuple(
                (
                    v
                    for v in [
                        hidden_states,
                        next_cache,
                        all_hidden_states,
                        all_self_attns,
                    ]
                    if v is not None
                )
            )
            if avg_kl_loss is not None:
                result = result + (avg_kl_loss,)
            return result
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        output.kl_loss = avg_kl_loss
        return output


class MiniCPMForCausalLM(MiniCPMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniCPMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        training_stage: int = 0,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        """Args:"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if "num_logits_to_keep" in kwargs:
            if isinstance(logits_to_keep, int) and logits_to_keep == 0:
                logits_to_keep = kwargs.pop("num_logits_to_keep")
            else:
                kwargs.pop("num_logits_to_keep")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training_stage=training_stage,
        )
        hidden_states = outputs[0]
        hidden_states_full = hidden_states
        chunk_size_cfg = int(getattr(self.config, "chunk_size", 0))
        if (
            self.training
            and labels is not None
            and (chunk_size_cfg > 0)
            and isinstance(logits_to_keep, int)
            and (logits_to_keep == 0)
        ):
            slice_indices = slice(-1, None)
        else:
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int) and logits_to_keep > 0
                else logits_to_keep
            )
        hidden_states_for_logits = hidden_states_full
        if isinstance(slice_indices, slice) or isinstance(slice_indices, torch.Tensor):
            if (
                isinstance(slice_indices, slice)
                and slice_indices != slice(None)
                or isinstance(slice_indices, torch.Tensor)
            ):
                hidden_states_for_logits = hidden_states_for_logits[
                    :, slice_indices, :
                ].contiguous()

        def _lm_head(h: torch.Tensor) -> torch.Tensor:
            """Project hidden states to vocab logits (supports pretraining_tp)."""
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(
                    self.vocab_size // self.config.pretraining_tp, dim=0
                )
                logits_parts = [
                    F.linear(h, lm_head_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
                return torch.cat(logits_parts, dim=-1)
            scale = self.config.hidden_size / self.config.dim_model_base
            return self.lm_head(h / scale)

        loss = None
        if labels is not None:
            (bsz, seq_len, _) = hidden_states_full.shape
            if seq_len <= 1:
                loss = None
            elif self.training and chunk_size_cfg > 0:
                hs = hidden_states_full[:, :-1, :]
                tgt = labels[:, 1:].to(hs.device)
                chunk_size = chunk_size_cfg
                total_loss = hs.new_zeros((), dtype=torch.float32)
                total_tokens = 0
                loss_fct = CrossEntropyLoss(reduction="sum")
                for start in range(0, hs.shape[1], chunk_size):
                    end = min(start + chunk_size, hs.shape[1])
                    hs_chunk = hs[:, start:end, :]
                    tgt_chunk = tgt[:, start:end]
                    logits_chunk = _lm_head(hs_chunk)
                    loss_chunk = loss_fct(
                        logits_chunk.reshape(-1, self.config.vocab_size),
                        tgt_chunk.reshape(-1),
                    )
                    total_loss = total_loss + loss_chunk
                    total_tokens += tgt_chunk.numel()
                loss = total_loss / max(total_tokens, 1)
            else:
                logits_full = _lm_head(hidden_states_full)
                if not self.training:
                    logits_full = logits_full.float()
                shift_logits = logits_full[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
                )
        logits = _lm_head(hidden_states_for_logits)
        if not self.training:
            logits = logits.float()
        kl_loss = getattr(outputs, "kl_loss", None)
        if not return_dict:
            output = (logits,) + outputs[1:]
            result = (loss,) + output if loss is not None else output
            if kl_loss is not None:
                result = result + (kl_loss,)
            return result
        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        result.kl_loss = kl_loss
        return result

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        use_sparse_attn = (
            getattr(self.config, "sparse_config", None) is not None
            and torch.cuda.is_available()
        )
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                if use_sparse_attn and cache_length == 0:
                    past_key_values = InfLLMv2Cache(
                        config=self.config,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )
                if hasattr(past_key_values, "seen_tokens"):
                    past_length = past_key_values.seen_tokens
                else:
                    past_length = cache_length
                max_cache_length = (
                    past_key_values.get_max_length()
                    if hasattr(past_key_values, "get_max_length")
                    else None
                )
            else:
                raise ValueError(
                    "You must use the new past_key_values format, such as the Cache class, instead of the old tuple format."
                )
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            if (
                max_cache_length is not None
                and attention_mask is not None
                and (cache_length + input_ids.shape[1] > max_cache_length)
            ):
                attention_mask = attention_mask[:, -max_cache_length:]
        else:
            pass
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    (
                        past_state.index_select(0, beam_idx.to(past_state.device))
                        for past_state in layer_past
                    )
                ),
            )
        return reordered_past

    @torch.inference_mode()
    def chat(
        self,
        tokenizer,
        query: str,
        history: list[dict] = None,
        role: str = "user",
        max_length: int = 4096,
        num_beams=1,
        do_sample=True,
        top_p=0.8,
        temperature=0.3,
        logits_processor=None,
        **kwargs,
    ):
        if history is None:
            history = []
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "top_p": top_p,
            "temperature": temperature,
            "logits_processor": logits_processor,
            **kwargs,
        }
        history.append({"role": role, "content": query})
        history_str = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=False
        )
        inputs = tokenizer(history_str, return_tensors="pt").to(self.device)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]) : -1]
        response = tokenizer.decode(outputs)
        pattern = re.compile(".*?(?=<AI>|<用户>)", re.DOTALL)
        matches = pattern.findall(response)
        if len(matches) > 0:
            response = matches[0]
        history.append({"role": "assistant", "content": response})
        return (response, history)
