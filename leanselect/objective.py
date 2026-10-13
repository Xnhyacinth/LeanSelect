"""Residual-utility fitting and strict Top-K boundary supervision."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .pages import variable_k_topk_mask


@dataclass(frozen=True)
class UtilityLossConfig:
    """Weights for utility fitting and the Top-K boundary."""

    listwise_weight: float = 1.0
    boundary_weight: float = 0.1
    temperature: float = 1.0
    boundary_margin: float = 0.1
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in vars(self).values()):
            raise ValueError("loss configuration must be finite")
        if self.listwise_weight < 0 or self.boundary_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.listwise_weight == 0 and self.boundary_weight == 0:
            raise ValueError("at least one utility loss weight must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.boundary_margin < 0:
            raise ValueError("boundary_margin must be non-negative")
        if not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")


DEFAULT_UTILITY_LOSS_CONFIG = UtilityLossConfig()


def _broadcast_mask(
    mask: torch.Tensor | None,
    shape: torch.Size,
    *,
    name: str,
    default: bool,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.full(shape, default, dtype=torch.bool, device=device)
    try:
        return torch.broadcast_to(mask.to(device=device, dtype=torch.bool), shape)
    except RuntimeError as error:
        raise ValueError(f"{name} must broadcast to the score shape") from error


def decision_aligned_utility_loss(
    student_score: torch.Tensor,
    teacher_utility: torch.Tensor,
    *,
    topk: int,
    valid_mask: torch.Tensor | None = None,
    forced_mask: torch.Tensor | None = None,
    config: UtilityLossConfig = DEFAULT_UTILITY_LOSS_CONFIG,
    boundary_regret_weighted: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fit conditional block utility and the fixed-budget decision boundary.

    Every valid, non-forced page remains a separate class.  The listwise term
    is weighted by each row's total utility, so its population minimizer ranks
    pages by conditional expected utility rather than by an equally weighted
    expectation of per-example normalized distributions.
    """
    if (
        student_score.shape != teacher_utility.shape
        or student_score.ndim < 1
        or not student_score.is_floating_point()
        or not teacher_utility.is_floating_point()
    ):
        raise ValueError("student_score and teacher_utility must be matching floats")
    if int(student_score.shape[-1]) <= 0:
        raise ValueError("score tensors must contain at least one page")
    if int(topk) <= 0:
        raise ValueError("topk must be positive")

    valid = _broadcast_mask(
        valid_mask,
        student_score.shape,
        name="valid_mask",
        default=True,
        device=student_score.device,
    )
    forced = _broadcast_mask(
        forced_mask,
        student_score.shape,
        name="forced_mask",
        default=False,
        device=student_score.device,
    )
    candidate = valid & ~forced
    if bool((candidate & ~torch.isfinite(student_score)).any()):
        raise ValueError("student_score must be finite on valid non-forced pages")
    if bool((candidate & ~torch.isfinite(teacher_utility)).any()):
        raise ValueError("teacher_utility must be finite on valid non-forced pages")
    if bool((candidate & (teacher_utility < 0)).any()):
        raise ValueError("teacher_utility must be non-negative")

    raw_score = student_score.float()
    raw_utility = teacher_utility.detach().float().masked_fill(~candidate, 0)
    score = raw_score
    utility = raw_utility
    utility_mass = utility.sum(dim=-1)
    candidate_count = candidate.sum(dim=-1)
    forced_count = (valid & forced).sum(dim=-1)
    deployed_slots = torch.minimum(
        valid.sum(dim=-1),
        torch.full_like(candidate_count, int(topk)),
    )
    dynamic_budget = (deployed_slots - forced_count).clamp(min=0)
    dynamic_budget = torch.minimum(dynamic_budget, candidate_count)
    valid_row = (dynamic_budget > 0) & (utility_mass > config.epsilon)
    differentiable_zero = (
        torch.nan_to_num(
            score[..., :1],
            nan=0,
            posinf=0,
            neginf=0,
        ).sum()
        * 0
    )

    masked_score = score.masked_fill(~candidate, float("-inf"))
    safe_score = torch.where(
        valid_row.unsqueeze(-1),
        masked_score,
        torch.zeros_like(masked_score),
    )
    student_log_probability = F.log_softmax(
        safe_score / config.temperature,
        dim=-1,
    )
    teacher_probability = utility / utility_mass.unsqueeze(-1).clamp_min(config.epsilon)
    teacher_log_probability = teacher_probability.clamp_min(config.epsilon).log()
    listwise_row = torch.where(
        teacher_probability > 0,
        utility * (teacher_log_probability - student_log_probability),
        0,
    ).sum(dim=-1)
    valid_mass = torch.where(valid_row, utility_mass, 0).sum()
    loss_listwise = (
        torch.where(valid_row, listwise_row, 0).sum()
        / valid_mass.clamp_min(config.epsilon)
        + differentiable_zero
    )

    teacher_rank_score = utility.masked_fill(~candidate, float("-inf"))
    positive_mask = variable_k_topk_mask(teacher_rank_score, dynamic_budget)
    student_positive_mask = variable_k_topk_mask(masked_score, dynamic_budget)
    negative_mask = candidate & ~positive_mask
    weakest_positive = score.masked_fill(~positive_mask, float("inf")).amin(dim=-1)
    hardest_negative = score.masked_fill(~negative_mask, float("-inf")).amax(dim=-1)
    weakest_positive_utility = utility.masked_fill(~positive_mask, float("inf")).amin(
        dim=-1
    )
    hardest_negative_utility = utility.masked_fill(~negative_mask, float("-inf")).amax(
        dim=-1
    )
    valid_boundary_row = (
        valid_row
        & positive_mask.any(dim=-1)
        & negative_mask.any(dim=-1)
        & torch.isfinite(weakest_positive)
        & torch.isfinite(hardest_negative)
    )
    safe_positive = torch.where(valid_boundary_row, weakest_positive, 0)
    safe_negative = torch.where(valid_boundary_row, hardest_negative, 0)
    boundary_row = F.softplus(safe_negative - safe_positive + config.boundary_margin)
    boundary_regret = (weakest_positive_utility - hardest_negative_utility).clamp_min(0)
    relative_boundary_regret = boundary_regret / utility_mass.clamp_min(config.epsilon)
    if not boundary_regret_weighted:
        # A finite-precision teacher can split an exact K/K+1 tie arbitrarily.
        # Do not train LeanSelect to reproduce an undefined ordering inside it.
        valid_boundary_row = valid_boundary_row & (boundary_regret > 0)
    boundary_rows = valid_boundary_row.sum()
    boundary_weight = (
        relative_boundary_regret
        if boundary_regret_weighted
        else torch.ones_like(relative_boundary_regret)
    )
    loss_boundary = (
        torch.where(
            valid_boundary_row,
            boundary_row * boundary_weight,
            0,
        ).sum()
        / boundary_rows.clamp_min(1)
        + differentiable_zero
    )

    total = (
        config.listwise_weight * loss_listwise + config.boundary_weight * loss_boundary
    )
    selected_count = dynamic_budget.clamp_min(1)
    teacher_positive_mask = variable_k_topk_mask(
        raw_utility.masked_fill(~candidate, float("-inf")), dynamic_budget
    )
    topk_recall_row = (
        (teacher_positive_mask & student_positive_mask).sum(dim=-1).float()
    )
    topk_recall_row = topk_recall_row / selected_count
    optimal_utility = (raw_utility * teacher_positive_mask).sum(dim=-1)
    captured_utility = (raw_utility * student_positive_mask).sum(dim=-1)
    captured_utility_ratio_row = captured_utility / optimal_utility.clamp_min(
        config.epsilon
    )
    decision_rows = valid_row.sum()
    metrics = {
        "loss": total.detach(),
        "loss_listwise": loss_listwise.detach(),
        "loss_boundary": loss_boundary.detach(),
        "valid_rows": valid_row.sum().detach(),
        "valid_boundary_rows": valid_boundary_row.sum().detach(),
        "mean_dynamic_budget": dynamic_budget.float().mean().detach(),
        "boundary_regret_weight": torch.where(
            valid_boundary_row,
            boundary_weight,
            0,
        )
        .sum()
        .div(boundary_rows.clamp_min(1))
        .detach(),
        "topk_recall": torch.where(valid_row, topk_recall_row, 0)
        .sum()
        .div(decision_rows.clamp_min(1))
        .detach(),
        "captured_utility_ratio": torch.where(
            valid_row,
            captured_utility_ratio_row,
            0,
        )
        .sum()
        .div(decision_rows.clamp_min(1))
        .detach(),
    }
    return total, metrics


def leanselect_per_gqa_plan_loss(
    student_score: torch.Tensor,
    teacher_group_utility: torch.Tensor,
    *,
    topk: int,
    valid_mask: torch.Tensor,
    forced_mask: torch.Tensor,
    worst_group_weight: float = 0.1,
    config: UtilityLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train one decision-aligned page plan per GQA group."""

    if (
        teacher_group_utility.ndim != 3
        or any(size == 0 for size in teacher_group_utility.shape)
        or student_score.shape != teacher_group_utility.shape
    ):
        raise ValueError("student score and teacher utility must match [G, N, P]")
    if not math.isfinite(worst_group_weight) or worst_group_weight < 0:
        raise ValueError("worst_group_weight must be non-negative")
    if config is None:
        config = UtilityLossConfig()
    teacher_group_utility = teacher_group_utility.detach().float()
    per_gqa_loss, metrics = decision_aligned_utility_loss(
        student_score,
        teacher_group_utility,
        topk=int(topk),
        valid_mask=valid_mask,
        forced_mask=forced_mask,
        config=config,
        boundary_regret_weighted=False,
    )
    group_losses = []
    group_metrics = []
    for group_score, group_utility in zip(student_score, teacher_group_utility):
        group_loss, per_group = decision_aligned_utility_loss(
            group_score,
            group_utility,
            topk=int(topk),
            valid_mask=valid_mask,
            forced_mask=forced_mask,
            config=config,
            boundary_regret_weighted=False,
        )
        group_losses.append(group_loss)
        group_metrics.append(per_group)
    worst_loss = torch.stack(group_losses).amax()
    total = per_gqa_loss + float(worst_group_weight) * worst_loss
    result = dict(metrics)
    result["loss"] = total.detach()
    result["loss_worst_group"] = worst_loss.detach()
    result["worst_group_topk_recall"] = torch.stack(
        [item["topk_recall"] for item in group_metrics]
    ).amin()
    result["worst_group_captured_utility_ratio"] = torch.stack(
        [item["captured_utility_ratio"] for item in group_metrics]
    ).amin()
    return total, result
