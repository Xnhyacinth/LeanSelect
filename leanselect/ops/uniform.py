from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any
import torch
from leanselect.pages import (
    apply_atom_causal_mask,
    forced_page_mask,
    pool_atoms_to_pages,
)
from leanselect.ops.head_mix import uniform_varlen_stage1

DEFAULT_BLOCK_SIZE = 64
DEFAULT_KERNEL_STRIDE = 16
DEFAULT_INIT_BLOCKS = 1
DEFAULT_LOCAL_BLOCKS = 32
DEFAULT_PREFILL_SCORE_TILE_BYTES = 2 << 30


def uniform_prefill_score_tile_bytes() -> int:
    value = int(
        os.environ.get(
            "LEANSELECT_SCORE_TILE_BYTES", str(DEFAULT_PREFILL_SCORE_TILE_BYTES)
        )
    )
    if value <= 0:
        raise ValueError("LEANSELECT_SCORE_TILE_BYTES must be positive")
    return value


def mix_atom_scores(
    q_full: torch.Tensor,
    k_hist: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    num_kv_heads: int,
    causal: bool = False,
    kernel_stride: int = DEFAULT_KERNEL_STRIDE,
    q_abs_offset: int = 0,
    out: torch.Tensor | None = None,
    equal_pack: bool = True,
) -> torch.Tensor:
    """Score compressed history K with uniform or learned actual-Q Mix."""
    return uniform_varlen_stage1(
        q_full,
        k_hist,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=int(max_seqlen_k),
        num_kv_heads=int(num_kv_heads),
        causal=bool(causal),
        kernel_stride=int(kernel_stride),
        q_abs_offset=int(q_abs_offset),
        out=out,
        equal_pack=bool(equal_pack),
    )


@dataclass
class SelectionDiagnostics:
    """Profiling metadata; never runs an additional serving computation."""

    extra: dict[str, Any] = field(default_factory=dict)


def _uniform_prefill_vectorized_topk(
    mix_atoms: torch.Tensor,
    *,
    batch: int,
    q_len: int,
    topk: int,
    block_size: int,
    kernel_stride: int,
    init_blocks: int,
    local_blocks: int,
    query_abs_offset: int = 0,
    sequence_len: int | None = None,
) -> torch.Tensor:
    """Pool, force and Top-K an equal-packed prefill without Python rows."""
    n_kv = int(mix_atoms.shape[0])
    n_atoms = int(mix_atoms.shape[-1])
    atoms = mix_atoms[:, : batch * q_len, :n_atoms].reshape(n_kv, batch, q_len, n_atoms)
    atoms = atoms.permute(1, 2, 0, 3)
    q_pos = torch.arange(q_len, device=mix_atoms.device, dtype=torch.long)
    q_pos = q_pos + int(query_abs_offset)
    atoms = apply_atom_causal_mask(atoms, q_pos, kernel_stride=int(kernel_stride))
    pages = pool_atoms_to_pages(
        atoms,
        seq_len=max(int(sequence_len if sequence_len is not None else q_len), 1),
        block_size=int(block_size),
        kernel_stride=int(kernel_stride),
        reduce="max",
    )
    n_pages = int(pages.shape[-1])
    forced = forced_page_mask(
        q_pos,
        num_pages=n_pages,
        block_size=int(block_size),
        init_blocks=int(init_blocks),
        local_blocks=int(local_blocks),
    )
    forced = forced[:, :, None, :].expand(batch, q_len, n_kv, n_pages)
    pages.masked_fill_(forced, float("inf"))
    k_take = min(int(topk), n_pages)
    result = pages.topk(k_take, dim=-1, largest=True, sorted=True).indices
    query_pages = torch.div(q_pos, int(block_size), rounding_mode="floor")
    result.masked_fill_(result > query_pages[None, :, None, None], -1)
    return (
        result.permute(2, 0, 1, 3)
        .reshape(n_kv, batch * q_len, k_take)
        .contiguous()
        .to(torch.int32)
    )


def _uniform_prefill_streaming_topk(
    q_unpad: torch.Tensor,
    k_compressed: torch.Tensor,
    *,
    cu_seqlens_k: torch.Tensor,
    batch: int,
    q_len: int,
    k_len: int,
    n_kv: int,
    topk: int,
    block_size: int,
    kernel_stride: int,
    init_blocks: int,
    local_blocks: int,
    score_tile_bytes: int = DEFAULT_PREFILL_SCORE_TILE_BYTES,
) -> tuple[torch.Tensor, int]:
    """Query-tiled Tensor-Core prefill that retains only int32 page IDs."""
    bytes_per_query = max(
        int(batch) * int(n_kv) * int(k_len) * q_unpad.element_size(), 1
    )
    tile_q = max(1024, int(score_tile_bytes) // bytes_per_query)
    tile_q = min(int(q_len), max(1024, tile_q // 1024 * 1024))
    n_pages = max((int(q_len) + int(block_size) - 1) // int(block_size), 1)
    k_take = min(int(topk), n_pages)
    result = torch.empty(
        (int(n_kv), int(batch), int(q_len), k_take),
        dtype=torch.int32,
        device=q_unpad.device,
    )
    q_packed = q_unpad.reshape(batch, q_len, q_unpad.shape[1], q_unpad.shape[2])
    for q0 in range(0, int(q_len), tile_q):
        q1 = min(q0 + tile_q, int(q_len))
        width = q1 - q0
        q_tile = q_packed[:, q0:q1].reshape(
            batch * width, q_unpad.shape[1], q_unpad.shape[2]
        )
        q_tile = q_tile.contiguous()
        cu_q_tile = (
            torch.arange(batch + 1, dtype=cu_seqlens_k.dtype, device=q_unpad.device)
            * width
        )
        atoms = mix_atom_scores(
            q_tile,
            k_compressed,
            cu_seqlens_q=cu_q_tile,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max(int(k_len), 1),
            num_kv_heads=int(n_kv),
            causal=False,
            kernel_stride=int(kernel_stride),
        )
        tile_topk = _uniform_prefill_vectorized_topk(
            atoms,
            batch=int(batch),
            q_len=width,
            topk=int(topk),
            block_size=int(block_size),
            kernel_stride=int(kernel_stride),
            init_blocks=int(init_blocks),
            local_blocks=int(local_blocks),
            query_abs_offset=q0,
            sequence_len=int(q_len),
        ).reshape(n_kv, batch, width, k_take)
        result[:, :, q0:q1].copy_(tile_topk)
    return (result.reshape(n_kv, batch * int(q_len), k_take), tile_q)


def uniform_prefill_topk_idx(
    q_unpad: torch.Tensor,
    k_compressed: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    topk: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    kernel_stride: int = DEFAULT_KERNEL_STRIDE,
    init_blocks: int = DEFAULT_INIT_BLOCKS,
    local_blocks: int = DEFAULT_LOCAL_BLOCKS,
) -> tuple[torch.Tensor, SelectionDiagnostics]:
    """Select prefill pages with actual-Q Mix; decode is deliberately rejected."""
    if q_unpad.ndim != 3 or k_compressed.ndim != 3:
        raise ValueError("q_unpad and k_compressed must be 3D")
    batch = int(cu_seqlens_q.numel()) - 1
    q_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    k_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    if int(q_lengths.numel()) == 0 or bool((q_lengths <= 1).any().item()):
        raise ValueError("Uniform is prefill-only; decode must use vanilla Sparse")
    n_kv = int(k_compressed.shape[1])
    max_k = int(k_lengths.max().item())
    q_len0 = int(q_lengths[0].item())
    k_len0 = int(k_lengths[0].item())
    equal_pack = bool(
        (q_lengths == q_len0).all().item() and (k_lengths == k_len0).all().item()
    )
    full_score_bytes = (
        n_kv * int(q_unpad.shape[0]) * max(max_k, 1) * q_unpad.element_size()
    )
    score_tile_bytes = uniform_prefill_score_tile_bytes()
    if equal_pack and full_score_bytes > score_tile_bytes:
        (topk_idx, tile_q) = _uniform_prefill_streaming_topk(
            q_unpad,
            k_compressed,
            cu_seqlens_k=cu_seqlens_k,
            batch=batch,
            q_len=q_len0,
            k_len=k_len0,
            n_kv=n_kv,
            topk=int(topk),
            block_size=int(block_size),
            kernel_stride=int(kernel_stride),
            init_blocks=int(init_blocks),
            local_blocks=int(local_blocks),
            score_tile_bytes=score_tile_bytes,
        )
        return (
            topk_idx,
            SelectionDiagnostics(
                extra={
                    "selector_phase": "prefill",
                    "prefill_impl": "streaming_mix_gemm_pool_topk",
                    "prefill_query_tile": tile_q,
                    "uniform_prefill_head_mix": "uniform",
                }
            ),
        )
    mix_atoms = mix_atom_scores(
        q_unpad,
        k_compressed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max(max_k, 1),
        num_kv_heads=n_kv,
        causal=False,
        kernel_stride=int(kernel_stride),
        equal_pack=equal_pack,
    )
    if equal_pack:
        topk_idx = _uniform_prefill_vectorized_topk(
            mix_atoms,
            batch=batch,
            q_len=q_len0,
            topk=int(topk),
            block_size=int(block_size),
            kernel_stride=int(kernel_stride),
            init_blocks=int(init_blocks),
            local_blocks=int(local_blocks),
        )
    else:
        chunks = []
        max_pages = max(
            (int(q_lengths.max().item()) + int(block_size) - 1) // int(block_size), 1
        )
        output_topk = min(int(topk), max_pages)
        for batch_idx in range(batch):
            q0 = int(cu_seqlens_q[batch_idx].item())
            q1 = int(cu_seqlens_q[batch_idx + 1].item())
            k0 = int(cu_seqlens_k[batch_idx].item())
            k1 = int(cu_seqlens_k[batch_idx + 1].item())
            q_len = q1 - q0
            atoms = mix_atoms[:, q0:q1, : max(k1 - k0, 1)]
            atoms = atoms.permute(1, 0, 2).unsqueeze(0)
            q_pos = torch.arange(q_len, device=q_unpad.device)
            atoms = apply_atom_causal_mask(
                atoms, q_pos, kernel_stride=int(kernel_stride)
            )
            pages = pool_atoms_to_pages(
                atoms,
                seq_len=q_len,
                block_size=int(block_size),
                kernel_stride=int(kernel_stride),
                reduce="max",
            )
            n_pages = int(pages.shape[-1])
            forced = forced_page_mask(
                q_pos,
                num_pages=n_pages,
                block_size=int(block_size),
                init_blocks=int(init_blocks),
                local_blocks=int(local_blocks),
            ).unsqueeze(2)
            pages.masked_fill_(forced.expand_as(pages), float("inf"))
            take = min(int(topk), n_pages)
            selected = pages.topk(take, dim=-1, largest=True, sorted=True).indices[0]
            selected.masked_fill_(
                selected > (q_pos // int(block_size))[:, None, None], -1
            )
            if take < output_topk:
                selected = torch.nn.functional.pad(
                    selected, (0, output_topk - take), value=-1
                )
            chunks.append(selected.permute(1, 0, 2).contiguous().to(torch.int32))
        topk_idx = torch.cat(chunks, dim=1)
    return (
        topk_idx,
        SelectionDiagnostics(
            extra={
                "selector_phase": "prefill",
                "prefill_impl": "mix_gemm_pool_topk",
                "uniform_prefill_head_mix": "uniform",
            }
        ),
    )
