"""Budget-aligned supervision of reused query and compressed-key features."""

import torch

from leanselect.objective import leanselect_per_gqa_plan_loss
from leanselect.pages import forced_page_mask, pool_atoms_to_pages
from leanselect.selector import (
    leanselect_informative_history_start,
    leanselect_supervision_windows,
)


def selector_loss(
    attn,
    query,
    teacher_keys,
    student_keys,
    cu_query,
    cu_teacher,
    cu_student,
    cu_deployed,
    attention_mask,
    teacher_keys2,
    cu_teacher2,
):
    """Sample teacher rows and average the deployed per-group selection objective."""
    from leanselect.models.minicpm import _leanselect_sampled_stage1_teacher

    if attn.leanselect is None:
        raise ValueError("Selector training requires method='leanselect'")
    if teacher_keys2 is None:
        teacher_keys2, cu_teacher2 = teacher_keys, cu_teacher
    offsets = [
        value.detach().cpu().tolist()
        for value in (cu_query, cu_teacher, cu_student, cu_deployed, cu_teacher2)
    ]
    total_loss = query.new_zeros((), dtype=torch.float32)
    total_rows = 0
    metric_sums = {}
    sequence_lengths = attention_mask.long().sum(-1).tolist()
    for b, sequence_length in enumerate(sequence_lengths):
        q0, q1 = offsets[0][b : b + 2]
        t0, t1 = offsets[1][b : b + 2]
        s0, s1 = offsets[2][b : b + 2]
        d0, d1 = offsets[3][b : b + 2]
        t20, t21 = offsets[4][b : b + 2]
        if q1 <= q0 or t1 <= t0 or s1 <= s0:
            continue
        q_batch = query[q0:q1]
        minimum = leanselect_informative_history_start(
            total_topk_blocks=attn.topk, block_size=attn.block_size
        )
        windows = leanselect_supervision_windows(
            q1 - q0,
            attn.leanselect_supervision_rows,
            minimum_start=minimum
            if q1 - q0 >= minimum + attn.leanselect_supervision_rows
            else 0,
        )
        read_points = torch.cat(
            [torch.arange(start, end, device=query.device) for start, end in windows]
        )
        with torch.no_grad():
            utility_atoms = _leanselect_sampled_stage1_teacher(
                q_batch,
                teacher_keys[t0:t1],
                teacher_keys2[t20:t21],
                windows=windows,
                kernel_stride=attn.kernel_stride,
                training_kernel_stride=attn.training_kernel_stride,
                deployed_atom_count=d1 - d0
                if attn._kl_target_compress_differs
                else t1 - t0,
                align_pooling=attn.kl_target_align_pooling,
            )
            utility_atoms = utility_atoms / (utility_atoms.sum(-1, keepdim=True) + 1e-8)
            teacher_pages = pool_atoms_to_pages(
                utility_atoms,
                seq_len=sequence_length,
                block_size=attn.block_size,
                kernel_stride=attn.kernel_stride,
                reduce="max",
            )
        atom_keys = attn.leanselect.project_atom_keys(student_keys[s0:s1])
        atom_queries = attn.leanselect.project_queries(
            q_batch.index_select(0, read_points)
        )
        student_pages = pool_atoms_to_pages(
            torch.einsum("gnf,gaf->gna", atom_queries, atom_keys),
            seq_len=sequence_length,
            block_size=attn.block_size,
            kernel_stride=attn.kernel_stride,
            reduce="max",
        )
        page_ids = torch.arange(student_pages.shape[-1], device=query.device)
        valid = page_ids[None] <= (read_points // attn.block_size)[:, None]
        forced = forced_page_mask(
            read_points,
            num_pages=student_pages.shape[-1],
            block_size=attn.block_size,
            init_blocks=attn.init_blocks,
            local_blocks=attn.local_blocks,
        ).squeeze(0)
        loss, metrics = leanselect_per_gqa_plan_loss(
            student_pages,
            teacher_pages,
            topk=attn.topk,
            valid_mask=valid,
            forced_mask=forced,
            worst_group_weight=attn.leanselect_worst_group_weight,
            config=attn.utility_loss_config,
        )
        rows = read_points.numel()
        total_loss = total_loss + loss * rows
        total_rows += rows
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0) + value.detach()
    if not total_rows:
        raise ValueError("No nonempty supervision rows in this batch")
    attn._last_selector_loss_metrics = {
        key: value / len(sequence_lengths) for key, value in metric_sums.items()
    }
    attn._last_selector_loss_metrics["supervision_rows"] = total_loss.new_tensor(
        total_rows
    )
    return total_loss / total_rows
