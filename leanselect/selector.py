"""Lightweight per-GQA selection from frozen attention features."""
from __future__ import annotations

import torch
from torch import nn

LEANSELECT_INITIALIZATION = "gqa_mean_raw_dot_atom_max_orthogonal_frame_v2"
LEANSELECT_SUPERVISION_MAX_WINDOWS = 64
LEANSELECT_SUPERVISION_WINDOW_POLICY = "up_to_64_disjoint_windows_uniform_over_nontrivial_history"
LEANSELECT_CANONICAL_TENSORS = ("head_mix", "kv_head_mix", "q_proj", "k_proj")


def leanselect_supervision_windows(
    sequence_length: int,
    supervision_rows: int,
    *,
    max_windows: int = LEANSELECT_SUPERVISION_MAX_WINDOWS,
    minimum_start: int = 0,
) -> tuple[tuple[int, int], ...]:
    """Choose disjoint contiguous teacher windows spanning the full sequence."""

    sequence_length = int(sequence_length)
    supervision_rows = int(supervision_rows)
    max_windows = int(max_windows)
    minimum_start = int(minimum_start)
    if min(sequence_length, supervision_rows, max_windows) <= 0:
        raise ValueError("LeanSelect supervision geometry must be positive")
    if minimum_start < 0 or minimum_start >= sequence_length:
        raise ValueError("LeanSelect minimum_start must be inside the sequence")
    available_rows = sequence_length - minimum_start
    row_count = min(available_rows, supervision_rows)
    if row_count == available_rows:
        return ((minimum_start, sequence_length),)
    window_count = min(max_windows, row_count)
    width, width_remainder = divmod(row_count, window_count)
    widths = [width + (index < width_remainder) for index in range(window_count)]
    if window_count == 1:
        return ((sequence_length - widths[0], sequence_length),)

    gap, gap_remainder = divmod(available_rows - row_count, window_count - 1)
    gaps = [
        gap + (index >= window_count - 1 - gap_remainder)
        for index in range(window_count - 1)
    ]
    windows = []
    start = minimum_start
    for index, window_width in enumerate(widths):
        end = start + window_width
        windows.append((start, end))
        if index < len(gaps):
            start = end + gaps[index]
    if windows[-1][1] != sequence_length:
        raise RuntimeError("LeanSelect supervision windows do not span the sequence")
    return tuple(windows)


def leanselect_informative_history_start(
    *, total_topk_blocks: int, block_size: int
) -> int:
    """Start supervision only after history is at least twice the Top-K budget."""

    total_topk_blocks = int(total_topk_blocks)
    block_size = int(block_size)
    if min(total_topk_blocks, block_size) <= 0:
        raise ValueError("LeanSelect Top-K and block size must be positive")
    return 2 * total_topk_blocks * block_size


class LeanSelectSelector(nn.Module):
    """Low-rank router that emits one page plan per GQA group."""

    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        router_heads: int = 4,
        router_dim: int = 32,
    ) -> None:
        super().__init__()
        values = {
            "num_heads": int(num_heads),
            "num_kv_heads": int(num_kv_heads),
            "head_dim": int(head_dim),
            "router_heads": int(router_heads),
            "router_dim": int(router_dim),
        }
        if any(value <= 0 for value in values.values()):
            raise ValueError("all LeanSelect dimensions must be positive")
        if values["router_heads"] > values["num_heads"]:
            raise ValueError("router_heads cannot exceed num_heads")
        if values["num_heads"] % values["num_kv_heads"]:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        for name, value in values.items():
            setattr(self, name, value)

        self.head_mix = nn.Parameter(torch.empty(self.num_heads, self.router_heads))
        self.kv_head_mix = nn.Parameter(
            torch.empty(self.num_kv_heads, self.router_heads)
        )
        self.q_proj = nn.Parameter(
            torch.empty(self.router_heads, self.head_dim, self.router_dim)
        )
        self.k_proj = nn.Parameter(
            torch.empty(self.router_heads, self.head_dim, self.router_dim)
        )
        for parameter in self.parameters():
            parameter._is_indexer = True
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            heads_per_group = self.num_heads // self.num_kv_heads
            self.head_mix.fill_(1.0 / heads_per_group)
            self.kv_head_mix.fill_(1.0)
            projection_width = self.router_heads * self.router_dim
            if projection_width <= self.head_dim:
                frame = torch.linalg.qr(
                    torch.randn(
                        self.head_dim, projection_width, dtype=torch.float32
                    ),
                    mode="reduced",
                ).Q
            else:
                frame = torch.linalg.qr(
                    torch.randn(
                        projection_width, self.head_dim, dtype=torch.float32
                    ),
                    mode="reduced",
                ).Q.T
            frame.mul_(self.head_dim**-0.25)
            projection = frame.reshape(
                self.head_dim, self.router_heads, self.router_dim
            ).permute(1, 0, 2)
            self.q_proj.copy_(projection)
            self.k_proj.copy_(projection)

    def canonical_state(self) -> dict[str, torch.Tensor]:
        """Return canonical master tensors; fused serve weights are derived state."""

        return {name: getattr(self, name) for name in LEANSELECT_CANONICAL_TENSORS}

    def fused_weights(
        self, *, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the two deployment matrices without storing duplicate parameters."""

        q_fused = torch.einsum("hr,rdj->hdrj", self.head_mix, self.q_proj).reshape(
            self.num_heads, self.head_dim, -1
        )
        kv_fused = torch.einsum("hr,rdj->hdrj", self.kv_head_mix, self.k_proj).reshape(
            self.num_kv_heads, self.head_dim, -1
        )
        if dtype is not None:
            q_fused = q_fused.to(dtype=dtype)
            kv_fused = kv_fused.to(dtype=dtype)
        return q_fused.contiguous(), kv_fused.contiguous()

    def project_queries(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 3 or tuple(query.shape[1:]) != (
            self.num_heads,
            self.head_dim,
        ):
            raise ValueError("query must have shape [N, num_heads, head_dim]")
        q_fused, _ = self.fused_weights(dtype=query.dtype)
        heads_per_group = self.num_heads // self.num_kv_heads
        grouped_query = query.detach().reshape(
            query.shape[0], self.num_kv_heads, heads_per_group, self.head_dim
        )
        grouped_fused = q_fused.reshape(
            self.num_kv_heads,
            heads_per_group,
            self.head_dim,
            q_fused.shape[-1],
        )
        return torch.einsum("ngsd,gsdf->gnf", grouped_query, grouped_fused)

    def project_atom_keys(self, compressed_keys: torch.Tensor) -> torch.Tensor:
        if compressed_keys.ndim != 3 or tuple(compressed_keys.shape[1:]) != (
            self.num_kv_heads,
            self.head_dim,
        ):
            raise ValueError(
                "compressed_keys must have shape [A, num_kv_heads, head_dim]"
            )
        _, kv_fused = self.fused_weights(dtype=compressed_keys.dtype)
        return torch.einsum(
            "agd,gdf->gaf", compressed_keys.detach(), kv_fused
        )

    def score_pages(
        self,
        query: torch.Tensor,
        compressed_keys: torch.Tensor,
        *,
        seq_len: int,
        block_size: int,
        kernel_stride: int,
    ) -> torch.Tensor:
        from .pages import pool_atoms_to_pages

        atom_scores = torch.einsum(
            "gnf,gaf->gna",
            self.project_queries(query),
            self.project_atom_keys(compressed_keys),
        )
        return pool_atoms_to_pages(
            atom_scores,
            seq_len=int(seq_len),
            block_size=int(block_size),
            kernel_stride=int(kernel_stride),
            reduce="max",
        )
