"""Uniform head mixing for the prefill ablation."""
from __future__ import annotations

import torch

from ..pages import uniform_group_queries


def _cu_seqlens_equal_packed(cu_seqlens_q, cu_seqlens_k, *, batch,
                             total_q, total_k, equal_pack=None):
    if batch <= 0 or total_q % batch or total_k % batch:
        return False
    if equal_pack is not None:
        return bool(equal_pack)
    q_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    k_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    return bool((q_lengths == total_q // batch).all()
                and (k_lengths == total_k // batch).all())


def _uniform_grouped_bmm(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    batch: int,
    q_len: int,
    k_len: int,
    n_kv: int,
    max_k: int,
    scale: float,
    causal: bool = False,
    kernel_stride: int = 16,
    q_abs_offset: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grouped decode/prefill Stage-1: one mixed query per KV group."""
    dim = int(q.shape[-1])
    q_eff = uniform_group_queries(q, n_kv)
    q_b = (
        q_eff.view(batch, q_len, n_kv, dim)
        .permute(0, 2, 1, 3)
        .reshape(batch * n_kv, q_len, dim)
    )
    k_b = (
        k.to(dtype=q.dtype)
        .view(batch, k_len, n_kv, dim)
        .permute(0, 2, 1, 3)
        .reshape(batch * n_kv, k_len, dim)
    )
    packed = torch.bmm(q_b, k_b.transpose(1, 2)) * scale
    if causal:
        q_pos = torch.arange(q_len, device=packed.device) + int(q_abs_offset)
        atom_start = torch.arange(k_len, device=packed.device) * int(kernel_stride)
        packed = packed.masked_fill(
            atom_start[None, None, :] > q_pos[None, :, None], float("-inf")
        )
    packed = (
        packed.view(batch, n_kv, q_len, k_len)
        .permute(1, 0, 2, 3)
        .reshape(n_kv, batch * q_len, k_len)
    )
    live = min(k_len, max_k)
    scored = packed[:, :, :live]
    total_q = batch * q_len
    if out is None:
        if live == max_k:
            return scored.contiguous()
        buf = packed.new_full((n_kv, total_q, max_k), float("-inf"))
        buf[:, :, :live] = scored
        return buf
    if out.shape[0] != n_kv or out.shape[1] != total_q or out.shape[2] < live:
        raise ValueError(
            f"Uniform bmm out shape {tuple(out.shape)} incompatible with "
            f"(n_kv={n_kv}, T={total_q}, live={live})"
        )
    out[:, :, :live] = scored
    return out


def uniform_varlen_stage1(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    num_kv_heads: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    kernel_stride: int = 16,
    q_abs_offset: int = 0,
    out: torch.Tensor | None = None,
    equal_pack: bool | None = None,
) -> torch.Tensor:
    """Score compressed keys using the mean query in each GQA group.

    Queries are [total_q, H, D]; keys are [total_k, G, D]. The result is
    [G, total_q, max_seqlen_k], with packed sequences scored independently.
    """
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(
            f"expected q (T,H,D) and k (C,n_kv,D), got {tuple(q.shape)} / {tuple(k.shape)}"
        )
    n_kv = int(k.shape[1])
    if int(num_kv_heads) != n_kv:
        raise ValueError(f"num_kv_heads={num_kv_heads} != compressed K n_kv={n_kv}")
    total_q = int(q.shape[0])
    max_k = int(out.shape[-1]) if out is not None else int(max_seqlen_k)
    scale = float(q.shape[-1]) ** -0.5 if softmax_scale is None else float(softmax_scale)
    batch = int(cu_seqlens_q.shape[0] - 1)
    total_k = int(k.shape[0])
    q_abs_offset = int(q_abs_offset)
    equal = _cu_seqlens_equal_packed(
        cu_seqlens_q,
        cu_seqlens_k,
        batch=batch,
        total_q=total_q,
        total_k=total_k,
        equal_pack=equal_pack,
    )
    if equal:
        q_len = total_q // batch
        k_len = total_k // batch
        if q_len > 0 and k_len > 0:
            return _uniform_grouped_bmm(
                q,
                k,
                batch=batch,
                q_len=q_len,
                k_len=k_len,
                n_kv=n_kv,
                max_k=max_k,
                scale=scale,
                causal=bool(causal),
                kernel_stride=int(kernel_stride),
                q_abs_offset=q_abs_offset,
                out=out,
            )
    q_g = uniform_group_queries(q, n_kv)
    cu_q = cu_seqlens_q.detach().cpu()
    cu_k = cu_seqlens_k.detach().cpu()
    if out is None:
        out = q.new_full((n_kv, total_q, max_k), float("-inf"))
    for i in range(batch):
        q_s = int(cu_q[i].item())
        q_e = int(cu_q[i + 1].item())
        k_s = int(cu_k[i].item())
        k_e = int(cu_k[i + 1].item())
        if q_e <= q_s or k_e <= k_s:
            continue
        k_slice = k[k_s:k_e].to(dtype=q.dtype)
        live = min(int(k_e - k_s), max_k)
        dots = torch.einsum("qgd,cgd->gqc", q_g[q_s:q_e], k_slice) * scale
        row = dots[:, :, :live]
        if causal:
            q_pos = torch.arange(q_e - q_s, device=row.device) + q_abs_offset
            atom_start = torch.arange(live, device=row.device) * int(kernel_stride)
            row = row.masked_fill(
                atom_start[None, None, :] > q_pos[None, :, None], float("-inf")
            )
        out[:, q_s:q_e, :live] = row
    return out
