# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""NOSA backbone with native sparse attention and LeanSelect."""

from leanselect.baselines.infinigen import (
    compute_skewing_matrix,
    compute_partial_weight_indices,
    infinigen_token_scores_hf,
    infinigen_build_partial_cache,
    infinigen_update_partial_cache,
)

_INFINIGEN_AVAILABLE = True
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from functools import lru_cache
from leanselect.ops import profiling as nvtx

LONG_CONTEXT_ENABLED = False
LONG_CONTEXT_ROPE_THETA = None
logger = logging.get_logger(__name__)


def _leanselect_sampled_stage1_teacher(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    windows: tuple[tuple[int, int], ...],
    kernel_stride: int,
    training_kernel_stride: int | None = None,
    deployed_atom_count: int | None = None,
    align_pooling: str = "max",
) -> torch.Tensor:
    """Evaluate sampled actual-Q scores on NOSA's deployed atom grid."""
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(
            "NOSA LeanSelect teacher expects Q/K tensors with three dimensions"
        )
    if min(int(q.shape[0]), int(k.shape[0]), int(kernel_stride)) <= 0:
        raise ValueError("NOSA LeanSelect teacher geometry must be positive")
    if not windows:
        raise ValueError("NOSA LeanSelect teacher requires at least one query window")
    if int(kernel_stride) != 16:
        raise ValueError("production NOSA Stage-1 uses a fixed causal stride of 16")
    if q.shape[1] % k.shape[1] != 0 or q.shape[2] != k.shape[2]:
        raise ValueError("NOSA LeanSelect teacher Q/K head geometry is incompatible")
    if q.device != k.device:
        raise ValueError("NOSA LeanSelect teacher Q/K must share one device")
    teacher_stride = int(
        kernel_stride if training_kernel_stride is None else training_kernel_stride
    )
    if teacher_stride != kernel_stride:
        if q.shape[1] // k.shape[1] != 16:
            raise ValueError(
                "fine NOSA LeanSelect KL teacher requires 16 query heads per GQA group"
            )
        if deployed_atom_count is None or deployed_atom_count <= 0:
            raise ValueError(
                "fine NOSA LeanSelect teacher requires the deployed atom count"
            )
        cu_q = torch.empty(2, device=q.device, dtype=torch.int32)
        cu_k = torch.tensor([0, len(k)], device=q.device, dtype=torch.int32)
        aligned_windows = []
        previous_end = 0
        for start, end in windows:
            if not previous_end <= start < end <= len(q):
                raise ValueError(
                    "NOSA LeanSelect teacher windows must be ordered and disjoint"
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
            raise ValueError(
                "NOSA LeanSelect teacher windows must be ordered and disjoint"
            )
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
    key_offset = int(k.shape[0]) - compressed_query_rows
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


def _sdpa_dense_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
) -> torch.Tensor:
    key_states = repeat_kv(key_states, query_states.shape[1] // key_states.shape[1])
    value_states = repeat_kv(
        value_states, query_states.shape[1] // value_states.shape[1]
    )
    sdpa_is_causal = is_causal
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            (_, kv_len) = attention_mask.shape
            q_len = query_states.shape[2]
            attn_keep = attention_mask.to(torch.bool)[:, None, None, :]
            if q_len > 1:
                query_positions = torch.arange(
                    q_len, device=query_states.device, dtype=torch.int64
                )
                key_positions = torch.arange(
                    kv_len, device=query_states.device, dtype=torch.int64
                )
                past_len = max(kv_len - q_len, 0)
                causal_keep = (
                    key_positions.view(1, 1, 1, kv_len)
                    <= query_positions.view(1, 1, q_len, 1) + past_len
                )
                attn_keep = attn_keep & causal_keep
                sdpa_is_causal = False
            attention_mask = attn_keep
        elif attention_mask.dtype != torch.bool and (
            not torch.is_floating_point(attention_mask)
        ):
            attention_mask = attention_mask.to(torch.bool)
        elif (
            torch.is_floating_point(attention_mask)
            and attention_mask.dtype != query_states.dtype
        ):
            attention_mask = attention_mask.to(query_states.dtype)
    if attention_mask is not None and query_states.device.type == "cuda":
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
    return F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=sdpa_is_causal,
    )


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape"""
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = (
                causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            )
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[
                :, :, :, :mask_length
            ].masked_fill(padding_mask, min_dtype)
    return causal_mask


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-06):
        """LlamaRMSNorm is equivalent to T5LayerNorm"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the `config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        (inv_freq, self.attention_scaling) = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """dynamic RoPE layers should recompute `inv_freq` in the following situations:"""
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            (inv_freq, self.attention_scaling) = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use `LlamaRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use `LlamaRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


import contextlib

_SKIP_KL_LOSS = False


def set_skip_kl_loss(skip: bool):
    global _SKIP_KL_LOSS
    _SKIP_KL_LOSS = skip


@contextlib.contextmanager
def skip_kl_loss(enabled: bool = True):
    global _SKIP_KL_LOSS
    prev = _SKIP_KL_LOSS
    _SKIP_KL_LOSS = enabled
    try:
        yield
    finally:
        _SKIP_KL_LOSS = prev


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return (q_embed, k_embed)


def apply_rotary_pos_emb_q_only(q, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to query tensor only."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return q * cos + rotate_half(q) * sin


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

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
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,"""
    (batch, num_key_value_heads, slen, head_dim) = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@lru_cache(maxsize=16)
def calc_chunks_with_stride(cu_seqlen, chunk_size, kernel_stride):
    """Compute the chunks that require Sparse attention, with stride support."""
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    max_seq_len = torch.max(batch_sizes)
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


class CompressK(torch.nn.Module):
    def __init__(self, head_num_k, head_dim, kernel_size, kernel_stride=16):
        """Module for compressing key (K) representations."""
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
        filtered_k = k.index_select(0, filtered_k_indices.view(-1))
        filtered_k = filtered_k.view(
            filtered_k.shape[0] // self.kernel_size,
            self.kernel_size,
            self.head_num_k,
            self.head_dim,
        )
        compressed_k = filtered_k.mean(dim=1)
        return (compressed_k, cu_seqlens_compressed)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` when creating this class."
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
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
        self.kernel_size = 32
        self.kernel_stride = 16
        self.init_blocks = 1
        self.block_size = 64
        self.window_size = 1024
        self.local_blocks = self.window_size // self.block_size
        self.select_blocks = 24
        self.topk = 96
        self.use_nope = False
        self.dense_len = 0
        self.force_dense_inference = False
        self.compress_k = CompressK(
            self.num_key_value_heads,
            self.head_dim,
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride,
        )
        self.A = nn.Parameter(torch.zeros(self.num_key_value_heads))
        self.delta = nn.Linear(
            self.num_key_value_heads * self.head_dim,
            self.num_key_value_heads,
            bias=config.attention_bias,
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        sparse_config = getattr(config, "sparse_config", None) or {}
        self.kernel_size = int(sparse_config.get("kernel_size", self.kernel_size))
        self.kernel_stride = int(sparse_config.get("kernel_stride", self.kernel_stride))
        self.init_blocks = int(sparse_config.get("init_blocks", self.init_blocks))
        self.block_size = int(sparse_config.get("block_size", self.block_size))
        self.window_size = int(sparse_config.get("window_size", self.window_size))
        self.local_blocks = self.window_size // self.block_size
        self.select_blocks = int(sparse_config.get("select_blocks", self.select_blocks))
        self.topk = int(sparse_config.get("topk", 80)) + self.local_blocks
        self.dense_len = int(sparse_config.get("dense_len", 8192))
        self.force_dense_inference = bool(
            sparse_config.get("force_dense_inference", False)
        )
        self.use_nope = bool(sparse_config.get("use_nope", self.use_nope))
        self.compress_k.kernel_size = self.kernel_size
        self.compress_k.kernel_stride = self.kernel_stride
        self.selection_method = str(
            sparse_config.get("selection_method", "current")
        ).lower()
        if self.selection_method not in {"current", "leanselect"}:
            raise ValueError(
                f"Unsupported NOSA selection_method={self.selection_method!r}"
            )
        self.leanselect_total_topk_blocks = int(self.topk)
        declared_phase_topk = sparse_config.get("leanselect_total_topk_blocks")
        if (
            self.selection_method == "leanselect"
            and declared_phase_topk is not None
            and (int(declared_phase_topk) != self.leanselect_total_topk_blocks)
        ):
            raise ValueError(
                f"leanselect_total_topk_blocks must equal NOSA's effective final budget: declared={declared_phase_topk}, effective={self.leanselect_total_topk_blocks}"
            )
        self.leanselect = None
        self.leanselect_supervision_rows = int(
            sparse_config.get("leanselect_supervision_rows", 64)
        )
        self.leanselect_worst_group_weight = float(
            sparse_config.get("leanselect_worst_group_weight", 0.1)
        )
        if self.leanselect_supervision_rows <= 0:
            raise ValueError("leanselect_supervision_rows must be positive")
        if self.leanselect_worst_group_weight < 0:
            raise ValueError("leanselect_worst_group_weight must be non-negative")
        if self.selection_method == "leanselect":
            from leanselect.selector import LeanSelectSelector

            self.leanselect = LeanSelectSelector(
                num_heads=self.num_heads,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                router_heads=int(sparse_config.get("leanselect_router_heads", 4)),
                router_dim=int(sparse_config.get("leanselect_router_dim", 32)),
            )
        from leanselect.objective import UtilityLossConfig

        self.leanselect_loss_config = UtilityLossConfig(
            listwise_weight=float(sparse_config.get("listwise_weight", 1.0)),
            boundary_weight=float(sparse_config.get("boundary_weight", 0.1)),
            temperature=float(sparse_config.get("loss_temperature", 1.0)),
            boundary_margin=float(sparse_config.get("boundary_margin", 0.1)),
        )
        self._last_selector_loss_metrics = None
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
        if self._kl_target_compress_differs:
            self.compress_k_kl_target = CompressK(
                self.num_key_value_heads,
                self.head_dim,
                kernel_size=self.training_kernel_size,
                kernel_stride=self.training_kernel_stride,
            )
        else:
            self.compress_k_kl_target = self.compress_k
        self.infinigen_enabled = sparse_config.get("infinigen_enabled", False)
        self.infinigen_num_channels = int(
            sparse_config.get("infinigen_num_channels", 32)
        )
        self._ig_skewing_matrix = None
        self._ig_partial_indices = None
        self._ig_partial_key_cache = None
        self._ig_warmed_up = False
        self._ig_prev_hidden_states = None
        self.use_q_future_for_topk = sparse_config.get("use_q_future_for_topk", False)
        self.use_q_future_decode_only = sparse_config.get(
            "use_q_future_decode_only", False
        )
        create_indexer = sparse_config.get("create_indexer", self.use_q_future_for_topk)
        if self.infinigen_enabled:
            create_indexer = False
        if create_indexer:
            self.q_future_proj = self._new_indexer_projection()
            if self.layer_idx == 0:
                self.q_curr_proj = self._new_indexer_projection()
        else:
            self.q_future_proj = None
            self.q_curr_proj = None

    def _new_indexer_projection(self) -> nn.Linear:
        projection = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        projection._is_indexer = True
        return projection

    def _forecast_projection(self, q_len: int, *, current: bool) -> Optional[nn.Linear]:
        kind = "curr" if current else "future"
        return getattr(self, f"q_{kind}_proj", None)

    def _project_forecast_queries(
        self, indexer_input: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        (bsz, q_len, _) = indexer_input.shape
        q_future_proj = self._forecast_projection(q_len, current=False)
        if q_future_proj is None:
            return (None, None)
        output_shape = (bsz, q_len, self.num_key_value_heads, self.head_dim)
        q_future = q_future_proj(indexer_input).view(output_shape)
        q_curr_proj = self._forecast_projection(q_len, current=True)
        q_curr = (
            q_curr_proj(indexer_input).view(output_shape)
            if q_curr_proj is not None
            else None
        )
        return (q_future, q_curr)

    def _refresh_partial_indices_from_prefill(
        self, query_states: torch.Tensor, key_states: torch.Tensor
    ) -> None:
        """Recompute request-specific InfiniGen channels from the current prefill."""
        if self._ig_skewing_matrix is None:
            return
        self._ig_partial_indices = compute_partial_weight_indices(
            query_states,
            key_states,
            self._ig_skewing_matrix,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
            num_channels=self.infinigen_num_channels,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed `position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be removed and `position_embeddings` will be mandatory."
            )
            (cos, sin) = self.rotary_emb(value_states, position_ids)
        else:
            (cos, sin) = position_embeddings
        (query_states, key_states) = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            (key_states, value_states) = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
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
        attn_output = attn_output.reshape(bsz, q_len, -1)
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


class LlamaFlashAttention2(LlamaAttention):
    """Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )
        output_attentions = False
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
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed `position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be removed and `position_embeddings` will be mandatory."
            )
            (cos, sin) = self.rotary_emb(value_states, position_ids)
        else:
            (cos, sin) = position_embeddings
        (query_states, key_states) = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            (key_states, value_states) = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        dropout_rate = self.attention_dropout if self.training else 0.0
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return (attn_output, attn_weights, past_key_value)


try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
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
except ImportError:
    infllmv2_attn_stage1 = None
    infllmv2_attn_stage1_kl_teacher = None
    infllmv2_attn_varlen_func = None
    infllmv2_attn_with_kvcache = None
    max_pooling_1d = None
    max_pooling_1d_varlen = None


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
    """Call stage1 against either the repo build or the legacy NOSA package."""
    if infllmv2_attn_stage1 is None:
        raise RuntimeError(
            "infllm_v2 is not available; install or expose the CUDA extension first."
        )
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
        return infllmv2_attn_stage1(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
        )


def _call_infllmv2_stage1_kl_teacher_compat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_v: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal_stride: int,
    query_offset: int,
    causal: bool,
) -> torch.Tensor:
    """Call the training-only KL-teacher stage1 op and fail fast on stale builds."""
    if infllmv2_attn_stage1_kl_teacher is None:
        raise RuntimeError(
            "NOSA KL-teacher training requires the rebuilt repo infllm_v2 extension with infllmv2_attn_stage1_kl_teacher support. Build the extension with third_party/infllmv2/install.py."
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
    except TypeError as exc:
        if "cu_seqlens_v" not in str(exc):
            raise
        raise RuntimeError(
            "NOSA KL-teacher training requires the repo infllm_v2 extension with the training-only stage1 KL-teacher op. Build the extension with third_party/infllmv2/install.py."
        ) from exc


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (indices, cu_seqlens, max_seqlen_in_batch)


def _unpad_one_tensor(hidden_states, attention_mask):
    (indices, cu_seqlens, max_seqlen_in_batch) = _get_unpad_data(attention_mask)
    (batch_size, seq_len) = hidden_states.shape[:2]
    remaining_dims = hidden_states.shape[2:]
    reshaped_states = hidden_states.reshape(batch_size * seq_len, *remaining_dims)
    unpadded_states = index_first_axis(reshaped_states, indices)
    return (unpadded_states, indices, cu_seqlens, max_seqlen_in_batch)


def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cis: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    select_blocks: int = 24,
    cache_lens: torch.Tensor = None,
    cu_seqlens_k_ori: torch.Tensor = None,
    max_seqlen_in_batch_k_ori: int = None,
    q_pred: torch.Tensor = None,
    ig_token_scores: torch.Tensor = None,
    precomputed_qk_score: torch.Tensor = None,
    precomputed_score_stride: int = None,
    precomputed_cu_seqlens_k: torch.Tensor = None,
    precomputed_max_seqlen_k: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention between query and compressed key and value. Compute attention output and topk block idx used in topk_sparse_attention."""
    with torch.no_grad():
        batch_size = cu_seqlens_q.shape[0] - 1
        is_prefilling = cache_lens is None or (cache_lens == 0).all().item()
        if is_prefilling:
            cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=q.device)
            q_idx = torch.cat(
                [
                    (
                        torch.arange(
                            cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device
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
        nvtx.range_push("stage1")
        score_stride = kernel_stride
        pool_cu_seqlens_k = cu_seqlens_k
        pool_max_seqlen_k = max_seqlen_k
        if precomputed_qk_score is not None:
            score = precomputed_qk_score
            score_stride = (
                kernel_stride
                if precomputed_score_stride is None
                else int(precomputed_score_stride)
            )
            if precomputed_cu_seqlens_k is not None:
                pool_cu_seqlens_k = precomputed_cu_seqlens_k
            if precomputed_max_seqlen_k is not None:
                pool_max_seqlen_k = int(precomputed_max_seqlen_k)
        elif ig_token_scores is not None:
            score = ig_token_scores
            score_stride = 1
            if cu_seqlens_k_ori is not None:
                pool_cu_seqlens_k = cu_seqlens_k_ori
                pool_max_seqlen_k = max_seqlen_in_batch_k_ori
        elif q_pred is not None:
            batch_size_s = cu_seqlens_q.shape[0] - 1
            num_kv_heads = q_pred.shape[1]
            total_q = q_pred.shape[0]
            k_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            max_k = k_seqlens.max().item()
            score = torch.full(
                (num_kv_heads, total_q, max_k),
                float("-inf"),
                device=q_pred.device,
                dtype=q_pred.dtype,
            )
            for b in range(batch_size_s):
                (q_s, q_e) = (int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1]))
                (k_s, k_e) = (int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1]))
                (q_len_b, k_len_b) = (q_e - q_s, k_e - k_s)
                if q_len_b == 0 or k_len_b == 0:
                    continue
                q_b = q_pred[q_s:q_e].permute(1, 0, 2)
                k_b_t = k[k_s:k_e].permute(1, 2, 0)
                raw = torch.bmm(q_b, k_b_t)
                if is_prefilling and q_len_b > 1:
                    q_pos = torch.arange(q_len_b, device=raw.device)
                    k_pos = torch.arange(k_len_b, device=raw.device) * kernel_stride
                    raw = raw.masked_fill(
                        (k_pos[None, :] > q_pos[:, None]).unsqueeze(0), float("-inf")
                    )
                score[:, q_s:q_e, :k_len_b] = raw
        else:
            score = _call_infllmv2_stage1_compat(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                cu_seqlens_v=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=is_prefilling,
            )
        nvtx.range_pop()
        nvtx.range_push("mean_pooling")
        from leanselect.ops.cis_pooling import nosa_mean_pooling

        score_cis = nosa_mean_pooling(
            cis.squeeze(-1), cu_seqlens_k_ori, max_seqlen_in_batch_k_ori
        )
        score = score[:, : q_idx.shape[0], :]
        score_cis = score_cis[:, : q_idx.shape[0], :]
        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            pool_cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            pool_max_seqlen_k,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=score_stride,
        )
        block_score_cis = max_pooling_1d_varlen(
            score_cis.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            max_seqlen_k,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=kernel_stride,
        )
        raw_block_score_cis = block_score_cis
        j_idx = torch.arange(
            block_score_cis.shape[-1], device=block_score_cis.device
        ).unsqueeze(0)
        ninf_mask = j_idx > q_idx.unsqueeze(1)
        block_score_cis = block_score_cis.masked_fill(
            ninf_mask.unsqueeze(0), float("-inf")
        )
        qk_select = min(
            init_blocks + local_blocks + select_blocks, block_score.shape[-1]
        )
        topk_idx_qk = block_score.topk(qk_select, dim=-1).indices
        scatter_mask = torch.zeros_like(block_score_cis, dtype=torch.bool)
        scatter_mask.scatter_(2, topk_idx_qk, True)
        block_score_cis = block_score_cis.masked_fill(scatter_mask, float("inf"))
        topk = min(topk, block_score.shape[-1])
        topk_idx = block_score_cis.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx > q_idx[None, :, None]] = -1
        topk_idx = topk_idx.to(torch.int32)
        nvtx.range_pop()
    return topk_idx


class LlamaSdpaAttention(LlamaAttention):
    """Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        q_future_from_prev: Optional[torch.Tensor] = None,
        training_stage: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                'LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        nvtx.range_push("linear")
        (bsz, q_len, _) = hidden_states.size()
        if self.infinigen_enabled:
            self._ig_attn_input = hidden_states.detach()
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
        dt_states = self.delta(value_states.transpose(1, 2).flatten(2, 3))
        dt_states = self.A * F.softplus(dt_states)
        cis = dt_states.to(hidden_states.dtype)
        if cis.shape[1] > 1:
            self.cis = cis
        else:
            self.cis = torch.cat((self.cis, cis), dim=1)
            cis = self.cis
        cis = cis.flatten(0, 1)
        nvtx.range_pop()
        nvtx.range_push("rope")
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed `position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be removed and `position_embeddings` will be mandatory."
            )
            (cos, sin) = self.rotary_emb(value_states, position_ids)
        else:
            (cos, sin) = position_embeddings
        (query_states, key_states) = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if (
            self.infinigen_enabled
            and self._ig_warmed_up
            and (q_len > 1)
            and (self._ig_partial_key_cache is None)
            and (self._ig_partial_indices is None)
        ):
            self._refresh_partial_indices_from_prefill(query_states, key_states)
        q_future = None
        q_curr = None
        if not self.infinigen_enabled:
            indexer_input = (
                hidden_states.detach() if training_stage == 1 else hidden_states
            )
            (q_future, q_curr) = self._project_forecast_queries(indexer_input)
            if q_future is not None:
                q_future = q_future.transpose(1, 2)
                q_future = apply_rotary_pos_emb_q_only(q_future, cos, sin)
                q_future = q_future.transpose(1, 2)
            if q_curr is not None:
                q_curr = q_curr.transpose(1, 2)
                q_curr = apply_rotary_pos_emb_q_only(q_curr, cos, sin)
                q_curr = q_curr.transpose(1, 2)
        if training_stage not in (0, 1):
            raise ValueError(
                f"Unsupported training_stage={training_stage}. Expected 0 (inference) or 1 (training)."
            )
        _indexer_q = q_future_from_prev if q_future_from_prev is not None else q_curr
        q_pred_topk = None
        if training_stage == 0 and self.use_q_future_for_topk:
            q_pred_topk = _indexer_q
            if self.use_q_future_decode_only and q_len != 1:
                q_pred_topk = None
        q_pred_kl = _indexer_q if training_stage == 1 else None
        nvtx.range_pop()
        nvtx.range_push("comp")
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            (key_states, value_states) = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        if query_states.device.type == "cuda":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        kv_seq_len = key_states.shape[2]
        use_dense_inference = (
            training_stage == 0
            and (not self.infinigen_enabled)
            and (
                self.force_dense_inference
                or (self.dense_len > 0 and kv_seq_len < self.dense_len)
            )
        )
        if use_dense_inference:
            attn_output = _sdpa_dense_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout_p=0.0,
                is_causal=attention_mask is None and q_len > 1,
            )
            nvtx.range_pop()
            nvtx.range_push("o")
            attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
            return (attn_output, None, past_key_value, q_future, None)
        ig_token_scores = None
        if (
            q_len == 1
            and self.infinigen_enabled
            and self._ig_warmed_up
            and (self.layer_idx > 0)
            and (self._ig_prev_hidden_states is not None)
            and (self._ig_partial_indices is not None)
            and (self._ig_partial_key_cache is not None)
            and _INFINIGEN_AVAILABLE
        ):
            kv_seq_len = key_states.shape[2]
            ig_token_scores = infinigen_token_scores_hf(
                self._ig_prev_hidden_states,
                self.q_proj.weight.data,
                self._ig_skewing_matrix,
                self._ig_partial_indices,
                self._ig_partial_key_cache[:, :kv_seq_len],
                cos,
                sin,
                position_ids,
                self.num_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.infinigen_num_channels,
                apply_rotary_pos_emb,
            )
            q_pred_topk = None
        if (
            self.infinigen_enabled
            and self._ig_warmed_up
            and (self._ig_partial_key_cache is None)
            and (self._ig_skewing_matrix is not None)
            and (self._ig_partial_indices is not None)
            and (q_len > 1)
        ):
            max_cache_len = key_states.shape[2] + self._ig_max_new_tokens
            self._ig_partial_key_cache = torch.zeros(
                bsz,
                max_cache_len,
                self.num_key_value_heads,
                self.infinigen_num_channels,
                dtype=getattr(self, "_ig_cache_dtype", hidden_states.dtype),
                device=getattr(self, "_ig_cache_device", hidden_states.device),
            )
        if (
            self.infinigen_enabled
            and self._ig_warmed_up
            and (self._ig_partial_key_cache is not None)
        ):
            kv_seq_len = key_states.shape[2]
            if q_len == 1:
                new_k = key_states[:, :, -1, :]
                infinigen_update_partial_cache(
                    self._ig_partial_key_cache,
                    new_k,
                    kv_seq_len - 1,
                    self._ig_skewing_matrix,
                    self._ig_partial_indices,
                )
            elif q_len > 1 and self._ig_partial_key_cache is not None:
                full_k = key_states.permute(0, 2, 1, 3)
                partial_k = infinigen_build_partial_cache(
                    full_k, self._ig_skewing_matrix, self._ig_partial_indices
                )
                max_len = self._ig_partial_key_cache.shape[1]
                write_len = min(kv_seq_len, max_len)
                self._ig_partial_key_cache[:, :write_len] = partial_k[:, :write_len]
        (attn_output, kl_loss) = self._sparse_attention_forward(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            cis,
            attention_mask,
            q_len,
            dropout=0.0,
            no_rope_param=None,
            past_key_value=None,
            q_pred=q_pred_topk,
            q_pred_kl=q_pred_kl,
            training_stage=training_stage,
            ig_token_scores=ig_token_scores,
        )
        nvtx.range_push("o")
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return (attn_output, None, past_key_value, q_future, kl_loss)

    def _sparse_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        cis,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        no_rope_param=None,
        past_key_value=None,
        q_pred=None,
        q_pred_kl=None,
        training_stage=0,
        ig_token_scores=None,
    ):
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            if past_key_value != None:
                (compressed_k, compressed_cu_seqlens) = self.get_compress_k(
                    key_states=key_states
                    if self.use_nope == False
                    else no_rope_param["key_states_no_rope"],
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                )
            (
                query_states,
                key_states,
                value_states,
                cis,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states,
                key_states,
                value_states,
                cis,
                attention_mask,
                query_length,
            )
            (cu_seqlens_q, cu_seqlens_k) = cu_seq_lens
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k) = max_seq_lens

            def _unpad_q_tensor(t):
                if t is None:
                    return None
                if max_seqlen_in_batch_q == 1:
                    return t.squeeze(1)
                return _unpad_one_tensor(t, attention_mask=attention_mask)[0]

            q_pred_unpad = _unpad_q_tensor(q_pred)
            q_pred_kl_unpad = _unpad_q_tensor(q_pred_kl)
            if no_rope_param != None:
                if max_seqlen_in_batch_q == 1:
                    no_rope_param["query_states_no_rope"] = no_rope_param[
                        "query_states_no_rope"
                    ].squeeze(1)
                else:
                    (no_rope_param["query_states_no_rope"], _, _, _) = (
                        _unpad_one_tensor(
                            no_rope_param["query_states_no_rope"],
                            attention_mask=attention_mask,
                        )
                    )
            compressed_k_kl_target = None
            compressed_cu_seqlens_kl_target = None
            if past_key_value == None:
                (compressed_k, compressed_cu_seqlens) = self.compress_k(
                    key_states, cu_seqlens_k
                )
                if (
                    self._kl_target_compress_differs
                    and training_stage == 1
                    and (
                        q_pred_kl_unpad is not None
                        or self.selection_method == "leanselect"
                    )
                ):
                    (compressed_k_kl_target, compressed_cu_seqlens_kl_target) = (
                        self.compress_k_kl_target(key_states, cu_seqlens_k)
                    )
            nvtx.range_pop()
            (attn_output_unpad, kl_loss) = self.sparse_forward(
                query_states,
                key_states,
                value_states,
                cis,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,
                no_rope_param=no_rope_param,
                compressed_k=compressed_k,
                compressed_cu_seqlens=compressed_cu_seqlens,
                compressed_k_kl_target=compressed_k_kl_target,
                compressed_cu_seqlens_kl_target=compressed_cu_seqlens_kl_target,
                q_pred=q_pred_unpad,
                q_pred_kl=q_pred_kl_unpad,
                training_stage=training_stage,
                ig_token_scores=ig_token_scores,
            )
            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            raise ValueError("Need attention mask")
        return (attn_output, kl_loss)

    def get_compress_k(self, key_states, attention_mask, past_key_value):
        """Get compressed key states and corresponding cumulative sequence lengths."""
        is_prefilling = key_states.shape[1] >= self.dense_len and (
            not past_key_value.layers[self.layer_idx].compress_k_cache
        )
        if is_prefilling:
            (unpadded_key_states, indices, cu_seqlens, max_seqlen_in_batch) = (
                _unpad_one_tensor(key_states, attention_mask=attention_mask)
            )
            (compressed_k, compressed_cu_seqlens) = self.compress_k(
                unpadded_key_states, cu_seqlens
            )
            past_key_value.update_compress_k(
                compressed_k, self.layer_idx, compressed_cu_seqlens
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
        return (compressed_k, compressed_cu_seqlens)

    def _build_aligned_training_probs(
        self,
        q_unpad: torch.Tensor,
        compressed_k_train: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        compressed_cu_seqlens_train: torch.Tensor,
        compressed_cu_seqlens_default: torch.Tensor,
        max_seqlen_q: int,
        cache_lens: Optional[torch.Tensor],
        is_prefill: bool,
    ) -> torch.Tensor:
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
            (q_s, q_e) = (q_seqlens[b], q_seqlens[b + 1])
            (k_s, k_e) = (train_seqlens[b], train_seqlens[b + 1])
            default_len = default_seqlens_list[b + 1] - default_seqlens_list[b]
            q_len = q_e - q_s
            k_len = k_e - k_s
            if q_len == 0 or k_len == 0 or default_len == 0:
                continue
            q_batch = q_unpad[q_s:q_e].contiguous()
            k_batch = compressed_k_train[k_s:k_e].contiguous()
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
                causal=is_prefill,
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
            aligned[:, q_s:q_e, :default_len] = probs
        return aligned

    def _compute_pure_qk_topk_idx(
        self,
        score: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        cache_lens: Optional[torch.Tensor],
        score_stride: int,
    ) -> torch.Tensor:
        """Compute a pure-QK top-k block mask for KL filtering."""
        batch_size = cu_seqlens_q.shape[0] - 1
        is_prefilling = cache_lens is None or (cache_lens == 0).all().item()
        if is_prefilling:
            effective_cache_lens = torch.zeros(
                batch_size, dtype=torch.int32, device=score.device
            )
            q_idx = torch.cat(
                [
                    (
                        torch.arange(
                            cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=score.device
                        )
                        + max_seqlen_q
                        - (cu_seqlens_q[i + 1] - cu_seqlens_q[i])
                    )
                    // self.block_size
                    for i in range(batch_size)
                ],
                dim=0,
            )
        else:
            effective_cache_lens = cache_lens
            q_idx = effective_cache_lens // self.block_size
        score = score[:, : q_idx.shape[0], :]
        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            effective_cache_lens,
            max_seqlen_q,
            max_seqlen_k,
            local_blocks=self.local_blocks,
            init_blocks=self.init_blocks,
            block_size=self.block_size,
            stride=score_stride,
        )
        topk = min(
            self.init_blocks + self.local_blocks + self.select_blocks,
            block_score.shape[-1],
        )
        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx > q_idx[None, :, None]] = -1
        return topk_idx.to(torch.int32)

    def _leanselect_atom_scores(
        self,
        query_layer: torch.Tensor,
        compressed_k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        compressed_cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Build NOSA's query-aware atom scores with the LeanSelect factorization."""
        if self.leanselect is None:
            raise RuntimeError("NOSA LeanSelect selector was not initialized")
        q_offsets = cu_seqlens_q.detach().to("cpu").tolist()
        k_offsets = compressed_cu_seqlens.detach().to("cpu").tolist()
        max_k = max(
            (k_offsets[i + 1] - k_offsets[i] for i in range(len(k_offsets) - 1))
        )
        scores = query_layer.new_full(
            (self.num_key_value_heads, int(query_layer.shape[0]), int(max_k)),
            float("-inf"),
        )
        for batch_index in range(len(q_offsets) - 1):
            (q_start, q_end) = q_offsets[batch_index : batch_index + 2]
            (k_start, k_end) = k_offsets[batch_index : batch_index + 2]
            if q_end <= q_start or k_end <= k_start:
                continue
            query_features = self.leanselect.project_queries(query_layer[q_start:q_end])
            key_features = self.leanselect.project_atom_keys(
                compressed_k[k_start:k_end]
            )
            scores[:, q_start:q_end, : k_end - k_start] = torch.einsum(
                "gnf,gaf->gna", query_features, key_features
            )
        return scores

    def _compute_leanselect_prefill_loss(
        self,
        query_layer: torch.Tensor,
        compressed_k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        compressed_cu_seqlens: torch.Tensor,
        compressed_k_kl_target: Optional[torch.Tensor] = None,
        compressed_cu_seqlens_kl_target: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Train the Q-aware total-41 shortlist while preserving CIS final K96."""
        if self.leanselect is None:
            raise RuntimeError("NOSA LeanSelect training requires a selector")
        from leanselect.pages import forced_page_mask, pool_atoms_to_pages
        from leanselect.selector import (
            leanselect_informative_history_start,
            leanselect_supervision_windows,
        )
        from leanselect.objective import leanselect_per_gqa_plan_loss

        q_offsets = cu_seqlens_q.detach().to("cpu").tolist()
        k_offsets = compressed_cu_seqlens.detach().to("cpu").tolist()
        if self._kl_target_compress_differs:
            if (
                compressed_k_kl_target is None
                or compressed_cu_seqlens_kl_target is None
            ):
                raise ValueError(
                    "NOSA LeanSelect fine teacher requires fine compressed keys"
                )
            fine_offsets = compressed_cu_seqlens_kl_target.detach().to("cpu").tolist()
        weighted_loss = query_layer.new_zeros((), dtype=torch.float32)
        supervised_rows = 0
        last_metrics = None
        qk_shortlist = self.init_blocks + self.local_blocks + self.select_blocks
        for batch_index in range(len(q_offsets) - 1):
            (q_start, q_end) = q_offsets[batch_index : batch_index + 2]
            (k_start, k_end) = k_offsets[batch_index : batch_index + 2]
            q_len = q_end - q_start
            if q_len <= 0 or k_end <= k_start:
                continue
            minimum_history = leanselect_informative_history_start(
                total_topk_blocks=qk_shortlist, block_size=self.block_size
            )
            minimum_start = (
                minimum_history
                if q_len >= minimum_history + self.leanselect_supervision_rows
                else 0
            )
            windows = leanselect_supervision_windows(
                q_len, self.leanselect_supervision_rows, minimum_start=minimum_start
            )
            read_points = torch.cat(
                [
                    torch.arange(start, end, device=query_layer.device)
                    for (start, end) in windows
                ]
            )
            q_batch = query_layer[q_start:q_end]
            k_batch = compressed_k[k_start:k_end]
            teacher_keys = (
                compressed_k_kl_target[
                    fine_offsets[batch_index] : fine_offsets[batch_index + 1]
                ]
                if self._kl_target_compress_differs
                else k_batch
            )
            with torch.no_grad():
                teacher_atoms = _leanselect_sampled_stage1_teacher(
                    q_batch,
                    teacher_keys,
                    windows=windows,
                    kernel_stride=self.kernel_stride,
                    training_kernel_stride=self.training_kernel_stride,
                    deployed_atom_count=len(k_batch),
                    align_pooling=self.kl_target_align_pooling,
                )
                teacher_atoms = teacher_atoms / (
                    teacher_atoms.sum(dim=-1, keepdim=True) + 1e-08
                )
                teacher_pages = pool_atoms_to_pages(
                    teacher_atoms,
                    seq_len=q_len,
                    block_size=self.block_size,
                    kernel_stride=self.kernel_stride,
                    reduce="max",
                )
            student_pages = self.leanselect.score_pages(
                q_batch.index_select(0, read_points),
                k_batch,
                seq_len=q_len,
                block_size=self.block_size,
                kernel_stride=self.kernel_stride,
            )
            page_ids = torch.arange(student_pages.shape[-1], device=query_layer.device)
            valid_mask = page_ids[None] <= (read_points // self.block_size)[:, None]
            forced_mask = forced_page_mask(
                read_points,
                num_pages=student_pages.shape[-1],
                block_size=self.block_size,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
            ).squeeze(0)
            (batch_loss, metrics) = leanselect_per_gqa_plan_loss(
                student_pages,
                teacher_pages,
                topk=qk_shortlist,
                valid_mask=valid_mask,
                forced_mask=forced_mask,
                worst_group_weight=self.leanselect_worst_group_weight,
                config=self.leanselect_loss_config,
            )
            row_count = int(read_points.numel())
            weighted_loss = weighted_loss + batch_loss.float() * row_count
            supervised_rows += row_count
            last_metrics = metrics
        if not supervised_rows:
            return None
        self._last_selector_loss_metrics = dict(last_metrics or {})
        self._last_selector_loss_metrics["supervision_rows"] = weighted_loss.new_tensor(
            supervised_rows
        )
        self._last_selector_loss_metrics["nosa_qk_shortlist"] = (
            weighted_loss.new_tensor(qk_shortlist)
        )
        return weighted_loss / supervised_rows

    def sparse_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        cis,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_in_batch_q,
        max_seqlen_in_batch_k,
        no_rope_param=None,
        compressed_k=None,
        compressed_cu_seqlens=None,
        compressed_k_kl_target=None,
        compressed_cu_seqlens_kl_target=None,
        q_pred=None,
        q_pred_kl=None,
        training_stage=0,
        ig_token_scores=None,
    ):
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
        cache_lens = None
        if max_seqlen_in_batch_q == 1 and max_seqlen_in_batch_k > 1:
            seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            cache_lens = seq_lens_k - 1
        is_prefill = False
        if max_seqlen_in_batch_q == 1 and max_seqlen_in_batch_k > 1:
            is_prefill = False
        else:
            is_prefill = True
        leanselect_score = None
        if (
            self.selection_method == "leanselect"
            and is_prefill
            and (training_stage == 0)
        ):
            leanselect_score = self._leanselect_atom_scores(
                query_layer, compressed_k, cu_seqlens_q, compressed_cu_seqlens
            )
        use_aligned_training_topk = (
            training_stage == 1
            and self._kl_target_compress_differs
            and (self.selection_method != "leanselect")
            and (compressed_k_kl_target is not None)
            and (compressed_cu_seqlens_kl_target is not None)
        )
        aligned_qk_probs = None
        if use_aligned_training_topk:
            q_topk = (
                query_layer
                if no_rope_param is None
                else no_rope_param["query_states_no_rope"]
            )
            aligned_qk_probs = self._build_aligned_training_probs(
                q_topk,
                compressed_k_kl_target.detach(),
                cu_seqlens_q,
                compressed_cu_seqlens_kl_target,
                compressed_cu_seqlens,
                max_seqlen_in_batch_q,
                cache_lens,
                is_prefill,
            )
            topk_idx = compressed_attention(
                query_layer
                if no_rope_param is None
                else no_rope_param["query_states_no_rope"],
                compressed_k,
                compressed_k.clone(),
                cis,
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.topk,
                cu_seqlens_q,
                compressed_cu_seqlens,
                max_seqlen_in_batch_q,
                compressed_seqlens.max().item(),
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                select_blocks=self.select_blocks,
                cache_lens=cache_lens,
                cu_seqlens_k_ori=cu_seqlens_k,
                max_seqlen_in_batch_k_ori=max_seqlen_in_batch_k,
                precomputed_qk_score=aligned_qk_probs,
                precomputed_score_stride=self.kernel_stride,
                precomputed_cu_seqlens_k=compressed_cu_seqlens,
                precomputed_max_seqlen_k=compressed_seqlens.max().item(),
            )
        elif leanselect_score is not None:
            topk_idx = compressed_attention(
                query_layer
                if no_rope_param is None
                else no_rope_param["query_states_no_rope"],
                compressed_k,
                compressed_k.clone(),
                cis,
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.topk,
                cu_seqlens_q,
                compressed_cu_seqlens,
                max_seqlen_in_batch_q,
                compressed_seqlens.max().item(),
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                select_blocks=self.select_blocks,
                cache_lens=cache_lens,
                cu_seqlens_k_ori=cu_seqlens_k,
                max_seqlen_in_batch_k_ori=max_seqlen_in_batch_k,
                precomputed_qk_score=leanselect_score,
                precomputed_score_stride=self.kernel_stride,
                precomputed_cu_seqlens_k=compressed_cu_seqlens,
                precomputed_max_seqlen_k=compressed_seqlens.max().item(),
            )
        else:
            topk_idx = compressed_attention(
                query_layer
                if no_rope_param is None
                else no_rope_param["query_states_no_rope"],
                compressed_k,
                compressed_k.clone(),
                cis,
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.topk,
                cu_seqlens_q,
                compressed_cu_seqlens,
                max_seqlen_in_batch_q,
                compressed_seqlens.max().item(),
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                select_blocks=self.select_blocks,
                cache_lens=cache_lens,
                cu_seqlens_k_ori=cu_seqlens_k,
                max_seqlen_in_batch_k_ori=max_seqlen_in_batch_k,
                q_pred=q_pred,
                ig_token_scores=ig_token_scores,
            )
        kl_loss = None
        if (
            training_stage == 1
            and (not _SKIP_KL_LOSS)
            and (self.selection_method == "leanselect")
            and is_prefill
        ):
            kl_loss = self._compute_leanselect_prefill_loss(
                query_layer,
                compressed_k,
                cu_seqlens_q,
                compressed_cu_seqlens,
                compressed_k_kl_target=compressed_k_kl_target,
                compressed_cu_seqlens_kl_target=compressed_cu_seqlens_kl_target,
            )
        elif training_stage == 1 and (not _SKIP_KL_LOSS) and (q_pred_kl is not None):
            kl_topk_idx = topk_idx
            with torch.no_grad():
                if use_aligned_training_topk and aligned_qk_probs is not None:
                    kl_topk_idx = self._compute_pure_qk_topk_idx(
                        aligned_qk_probs,
                        cu_seqlens_q,
                        compressed_cu_seqlens,
                        max_seqlen_in_batch_q,
                        compressed_seqlens.max().item(),
                        cache_lens,
                        self.kernel_stride,
                    )
                else:
                    q_actual_kl = (
                        query_layer
                        if no_rope_param is None
                        else no_rope_param["query_states_no_rope"]
                    )
                    kl_score = _call_infllmv2_stage1_compat(
                        q_actual_kl.contiguous(),
                        compressed_k.contiguous(),
                        compressed_k.contiguous(),
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=compressed_cu_seqlens,
                        cu_seqlens_v=compressed_cu_seqlens,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=compressed_seqlens.max().item(),
                        causal=is_prefill,
                    )
                    kl_topk_idx = self._compute_pure_qk_topk_idx(
                        kl_score,
                        cu_seqlens_q,
                        compressed_cu_seqlens,
                        max_seqlen_in_batch_q,
                        compressed_seqlens.max().item(),
                        cache_lens,
                        self.kernel_stride,
                    )
            kl_compressed_k_pred = compressed_k.detach()
            kl_cu_seqlens_k_pred = compressed_cu_seqlens
            kl_compressed_k_target = kl_compressed_k_pred
            kl_cu_seqlens_k_target = kl_cu_seqlens_k_pred
            kl_target_stride = self.kernel_stride
            if use_aligned_training_topk:
                kl_compressed_k_target = compressed_k_kl_target.detach()
                kl_cu_seqlens_k_target = compressed_cu_seqlens_kl_target
                kl_target_stride = self.training_kernel_stride
            elif (
                compressed_k_kl_target is not None
                and compressed_cu_seqlens_kl_target is not None
            ):
                kl_compressed_k_target = compressed_k_kl_target.detach()
                kl_cu_seqlens_k_target = compressed_cu_seqlens_kl_target
                kl_target_stride = self.training_kernel_stride
            kl_loss = self._compute_indexer_kl_loss(
                query_layer
                if no_rope_param is None
                else no_rope_param["query_states_no_rope"],
                q_pred_kl,
                kl_compressed_k_target,
                kl_compressed_k_pred,
                cu_seqlens_q,
                kl_cu_seqlens_k_target,
                kl_cu_seqlens_k_pred,
                compressed_cu_seqlens,
                max_seqlen_in_batch_q,
                is_prefill,
                target_kernel_stride=kl_target_stride,
                topk_idx=kl_topk_idx,
            )
        if getattr(self, "_collect_block_stats", False):
            self._last_topk_idx = topk_idx.detach()
        nvtx.range_push("attn")
        attn_dtype = query_layer.dtype
        cis = torch.exp(cis).to(attn_dtype)
        if key_layer.dtype != attn_dtype:
            key_layer = key_layer.to(attn_dtype)
        scaled_v = (value_layer * cis[:, :, None]).to(attn_dtype)
        hdim = value_layer.shape[-1]
        (topk_attn_output, lse, _) = infllmv2_attn_varlen_func(
            query_layer,
            key_layer,
            scaled_v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_in_batch_q,
            max_seqlen_in_batch_k,
            dropout_p=0,
            deterministic=False,
            softmax_scale=None,
            causal=True,
            return_attn_probs=True,
            topk_idx=topk_idx,
        )
        lse = lse.reshape(-1, query_layer.shape[1])
        fake_v = cis[:, :, None].repeat(1, 1, hdim).to(attn_dtype)
        (topk_attn_output_fake, lse_fake, _) = infllmv2_attn_varlen_func(
            query_layer,
            key_layer,
            fake_v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_in_batch_q,
            max_seqlen_in_batch_k,
            dropout_p=0,
            deterministic=False,
            softmax_scale=None,
            causal=True,
            return_attn_probs=True,
            topk_idx=topk_idx,
        )
        real_denominator = topk_attn_output_fake[:, :, :1]
        topk_attn_output = topk_attn_output / real_denominator
        nvtx.range_pop()
        return (topk_attn_output, kl_loss)

    def _compute_indexer_kl_loss(
        self,
        q_actual,
        q_pred,
        compressed_k_target,
        compressed_k_pred,
        cu_seqlens_q,
        compressed_cu_seqlens_target,
        compressed_cu_seqlens_pred,
        compressed_cu_seqlens_default,
        max_seqlen_q,
        is_prefill,
        target_kernel_stride=None,
        topk_idx=None,
    ):
        raise ValueError(
            "Training supports LeanSelect; SparDA is an evaluation baseline"
        )

    def _upad_input(
        self, query_layer, key_layer, value_layer, cis, attention_mask, query_length
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
        cis = index_first_axis(cis, indices_k)
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
            cis,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        q_future_from_prev: Optional[torch.Tensor] = None,
        training_stage: int = 0,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_result = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            q_future_from_prev=q_future_from_prev,
            training_stage=training_stage,
            **kwargs,
        )
        if len(attn_result) == 5:
            (hidden_states, self_attn_weights, present_key_value, q_future, kl_loss) = (
                attn_result
            )
        else:
            (hidden_states, self_attn_weights, present_key_value) = attn_result[:3]
            (q_future, kl_loss) = (None, None)
        hidden_states = residual + hidden_states
        nvtx.range_pop()
        nvtx.range_push("ffn")
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        nvtx.range_pop()
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        outputs += (q_future, kl_loss)
        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        sparse_config = getattr(self.config, "sparse_config", None) or {}
        use_kaiming_for_indexer = sparse_config.get("indexer_kaiming_init", True)
        is_indexer = getattr(module, "_is_indexer", False)
        if isinstance(module, nn.Linear):
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


class LlamaModel(LlamaPreTrainedModel):
    """Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]"""

    def __init__(self, config: LlamaConfig):
        if LONG_CONTEXT_ROPE_THETA is not None:
            config.rope_theta = LONG_CONTEXT_ROPE_THETA
            logger.info(
                f"[LongContext] rope_theta increased: {LONG_CONTEXT_ROPE_THETA} (no YaRN, direct theta scaling)"
            )
        elif LONG_CONTEXT_ENABLED:
            config.rope_scaling = {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": config.max_position_embeddings,
            }
            logger.info(
                f"[LongContext] YaRN enabled: factor=4.0, base_ctx={config.max_position_embeddings}, effective_ctx={config.max_position_embeddings * 4}"
            )
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        training_stage: int = 0,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return_legacy_cache = False
        if use_cache and (not isinstance(past_key_values, Cache)):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class (https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        q_future_prev = None
        accumulated_kl_loss = None
        kl_loss_count = 0
        _infinigen_active = any(
            (
                getattr(dl.self_attn, "infinigen_enabled", False)
                and getattr(dl.self_attn, "_ig_warmed_up", False)
                for dl in self.layers
            )
        )
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    q_future_prev,
                    training_stage,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
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
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            q_future_prev = layer_outputs[-2]
            layer_kl = layer_outputs[-1]
            if layer_kl is not None:
                accumulated_kl_loss = (
                    layer_kl
                    if accumulated_kl_loss is None
                    else accumulated_kl_loss + layer_kl
                )
                kl_loss_count += 1
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
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
            return result + (avg_kl_loss,) if avg_kl_loss is not None else result
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        output.kl_loss = avg_kl_loss
        return output

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)
        if (
            self.config._attn_implementation == "sdpa"
            and (not using_static_cache)
            and (not output_attentions)
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None
        (dtype, device) = (input_tensor.dtype, input_tensor.device)
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and (attention_mask.device.type == "cuda")
            and (not output_attentions)
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )
        return causal_mask


class SparseLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
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

    @torch.no_grad()
    def infinigen_warmup(
        self,
        input_ids=None,
        warmup_text=None,
        tokenizer=None,
        warmup_seq_len=2048,
        max_new_tokens=None,
    ):
        """InfiniGen warmup: compute per-layer skewing matrices and partial indices."""
        if not _INFINIGEN_AVAILABLE:
            raise ImportError(
                "InfiniGen controllers not available. Install the baseline runtime dependencies."
            )
        if max_new_tokens is None:
            raise ValueError("max_new_tokens is required for InfiniGen warmup")
        if input_ids is None and warmup_text is not None and (tokenizer is not None):
            input_ids = tokenizer(warmup_text, return_tensors="pt").input_ids
        if input_ids is None:
            raise ValueError("Must provide input_ids or warmup_text+tokenizer")
        input_ids = input_ids[:1, :warmup_seq_len].to(self.device)
        attention_mask = torch.ones_like(input_ids)
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        hidden_states_list = outputs.hidden_states
        for layer_idx, decoder_layer in enumerate(self.model.layers):
            attn = decoder_layer.self_attn
            if not getattr(attn, "infinigen_enabled", False):
                continue
            hs = hidden_states_list[layer_idx]
            hs_normed = decoder_layer.input_layernorm(hs)
            q = attn.q_proj(hs_normed)
            k = attn.k_proj(hs_normed)
            (bsz, seq_len, _) = hs_normed.shape
            q = q.view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(
                1, 2
            )
            pos_ids = torch.arange(seq_len, device=q.device).unsqueeze(0)
            (cos, sin) = attn.rotary_emb(q.to(torch.float32), pos_ids)
            (q, k) = apply_rotary_pos_emb(q, k, cos, sin)
            A = compute_skewing_matrix(
                q, k, attn.num_heads, attn.num_key_value_heads, attn.head_dim
            )
            attn._ig_skewing_matrix = A
            attn._ig_partial_indices = None
            attn._ig_partial_key_cache = None
            attn._ig_cache_dtype = hs.dtype
            attn._ig_cache_device = hs.device
            attn._ig_max_new_tokens = max_new_tokens
            attn._ig_warmed_up = True
        logger.info("InfiniGen warmup complete for NOSA model")

    def _get_infinigen_attns(self):
        """Return attention modules with InfiniGen enabled."""
        return [
            decoder_layer.self_attn
            for decoder_layer in self.model.layers
            if getattr(decoder_layer.self_attn, "infinigen_enabled", False)
        ]

    def _resolve_generation_budget(self, kwargs, prompt_len=None) -> int:
        max_new_tokens = kwargs.get("max_new_tokens")
        gc = getattr(self, "generation_config", None)
        if max_new_tokens is None and gc is not None:
            max_new_tokens = getattr(gc, "max_new_tokens", None)
        if max_new_tokens is None and prompt_len is not None:
            max_length = kwargs.get("max_length")
            if max_length is None and gc is not None:
                max_length = getattr(gc, "max_length", None)
            if max_length is not None:
                max_new_tokens = int(max_length) - int(prompt_len)
        if max_new_tokens is None or int(max_new_tokens) <= 0:
            raise ValueError(
                "InfiniGen requires a positive decode budget. Pass max_new_tokens to generate(), set generation_config.max_new_tokens, or provide max_length larger than the prompt length."
            )
        return int(max_new_tokens)

    def _prepare_infinigen_request_state(self, max_new_tokens: int):
        """Reset per-request InfiniGen state and set this call's decode budget."""
        for attn in self._get_infinigen_attns():
            attn._ig_max_new_tokens = int(max_new_tokens)
            attn._ig_partial_indices = None
            attn._ig_partial_key_cache = None
            attn._ig_prev_hidden_states = None
            attn._ig_attn_input = None

    def _cleanup_infinigen_request_state(self):
        """Release per-request InfiniGen state after generation."""
        for attn in self._get_infinigen_attns():
            attn._ig_partial_indices = None
            attn._ig_partial_key_cache = None
            attn._ig_prev_hidden_states = None
            attn._ig_attn_input = None

    def generate(self, *args, **kwargs):
        """Generate with per-request InfiniGen partial-cache allocation."""
        attns = self._get_infinigen_attns()
        if not attns:
            return super().generate(*args, **kwargs)
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        prompt_len = input_ids.shape[-1] if input_ids is not None else None
        max_new_tokens = self._resolve_generation_budget(kwargs, prompt_len=prompt_len)
        need_warmup = any((not getattr(attn, "_ig_warmed_up", False) for attn in attns))
        if need_warmup:
            if input_ids is None:
                raise ValueError(
                    "InfiniGen auto-warmup requires input_ids. Pass input_ids to generate() before first use."
                )
            self.infinigen_warmup(input_ids=input_ids, max_new_tokens=max_new_tokens)
        self._prepare_infinigen_request_state(max_new_tokens)
        try:
            return super().generate(*args, **kwargs)
        finally:
            self._cleanup_infinigen_request_state()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        training_stage: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
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
            cache_position=cache_position,
            training_stage=training_stage,
        )
        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            if labels is None and (not is_torchdynamo_compiling()):
                logger.warning_once(
                    "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
                )
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()
        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
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
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {
                "input_ids": input_ids.clone(memory_format=torch.contiguous_format),
                "inputs_embeds": None,
            }
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                (batch_size, sequence_length, _) = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                (batch_size, sequence_length) = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device
            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min
            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )
        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep
        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
