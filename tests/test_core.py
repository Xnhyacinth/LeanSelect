"""Numerical checks for the selector, page decisions, and checkpoint format."""
import copy
import math

import pytest
import torch
import torch.nn.functional as F

from leanselect.selector import (
    LeanSelectSelector,
    leanselect_informative_history_start,
    leanselect_supervision_windows,
)
from leanselect.objective import (
    UtilityLossConfig,
    decision_aligned_utility_loss,
    leanselect_per_gqa_plan_loss,
)
from leanselect.pages import pool_atoms_to_pages
from leanselect.checkpoints import (
    build_leanselect_checkpoint,
    leanselect_checkpoint_config,
    normalize_checkpoint,
    validate_leanselect_checkpoint,
)


def selector():
    torch.manual_seed(7)
    return LeanSelectSelector(num_heads=8, num_kv_heads=2, head_dim=8,
                              router_heads=2, router_dim=4)


def test_folding_matches_explicit_routes_and_detaches_backbone():
    module = selector()
    query = torch.randn(3, 8, 8, requires_grad=True)
    keys = torch.randn(7, 2, 8, requires_grad=True)
    q = module.project_queries(query)
    k = module.project_atom_keys(keys)
    expected = []
    for group in range(2):
        group_score = torch.zeros(3, 7)
        for route in range(2):
            mixed = sum(query[:, head] * module.head_mix[head, route]
                        for head in range(group * 4, (group + 1) * 4))
            q_route = mixed @ module.q_proj[route]
            k_route = (keys[:, group] * module.kv_head_mix[group, route]) @ module.k_proj[route]
            group_score = group_score + q_route @ k_route.T
        expected.append(group_score)
    actual = torch.einsum('gnf,gaf->gna', q, k)
    torch.testing.assert_close(actual, torch.stack(expected))
    actual.square().mean().backward()
    assert query.grad is None and keys.grad is None
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


def test_full_width_initialization_is_scaled_head_mean_dot():
    module = selector()  # R r = d = 8.
    query, keys = torch.randn(3, 8, 8), torch.randn(7, 2, 8)
    actual = torch.einsum('gnf,gaf->gna', module.project_queries(query),
                          module.project_atom_keys(keys))
    expected = torch.einsum('ngd,agd->gna', query.reshape(3, 2, 4, 8).mean(2), keys) / math.sqrt(8)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_page_pool_uses_overlapping_five_atom_window():
    atoms = torch.arange(15).float()
    actual = pool_atoms_to_pages(atoms, seq_len=256)
    expected = torch.tensor([3., 7., 11., 14.])
    torch.testing.assert_close(actual, expected)
    atom = torch.zeros(31)
    atom[19] = 1
    pages = pool_atoms_to_pages(atom, seq_len=512)
    assert pages.nonzero().flatten().tolist() == [4, 5]


def test_loss_matches_residual_mass_kl_and_strict_boundary():
    scores = torch.tensor([[99., .2, -.3, .7]], requires_grad=True)
    teacher = torch.tensor([[100., 5., 3., 1.]], requires_grad=True)
    forced = torch.tensor([[True, False, False, False]])
    loss, metrics = decision_aligned_utility_loss(scores, teacher, topk=3, forced_mask=forced)
    target = torch.tensor([5., 3., 1.]) / 9
    kl = (target * (target.log() - F.log_softmax(scores[0, 1:], -1))).sum()
    boundary = F.softplus(scores[0, 3] - scores[0, 2] + .1)
    torch.testing.assert_close(loss, kl + .1 * boundary)
    assert metrics['mean_dynamic_budget'].item() == 2
    loss.backward()
    assert scores.grad[0, 0] == 0 and teacher.grad is None


def test_worst_group_regularizer_matches_separate_group_losses():
    score = torch.tensor([[[.1, .2, .3]], [[.7, -.2, .4]]], requires_grad=True)
    utility = torch.tensor([[[4., 2., 1.]], [[1., 3., 2.]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    forced = torch.zeros_like(valid)
    overall, _ = decision_aligned_utility_loss(score, utility, topk=1)
    groups = [decision_aligned_utility_loss(s, u, topk=1)[0]
              for s, u in zip(score, utility)]
    loss, _ = leanselect_per_gqa_plan_loss(score, utility, topk=1,
                                          valid_mask=valid, forced_mask=forced)
    torch.testing.assert_close(loss, overall + .1 * torch.stack(groups).max())


@pytest.mark.parametrize('kind', ['zero_mass', 'forced', 'invisible'])
def test_empty_residual_decision_has_finite_zero_gradient(kind):
    score = torch.randn(2, 4, requires_grad=True)
    utility = torch.ones_like(score)
    options = {}
    if kind == 'zero_mass':
        utility.zero_()
    elif kind == 'forced':
        options['forced_mask'] = torch.ones_like(score, dtype=torch.bool)
    else:
        options['valid_mask'] = torch.zeros_like(score, dtype=torch.bool)
    loss, _ = decision_aligned_utility_loss(score, utility, topk=2, **options)
    assert loss == 0
    loss.backward()
    assert torch.equal(score.grad, torch.zeros_like(score))


def test_teacher_tie_has_no_boundary_loss():
    _, metrics = decision_aligned_utility_loss(torch.tensor([[3., -1., 0.]]),
                                              torch.ones(1, 3), topk=1)
    assert metrics['loss_boundary'] == 0
    assert metrics['valid_boundary_rows'] == 0


def checkpoint():
    return build_leanselect_checkpoint(
        {0: selector().canonical_state()}, num_heads=8, num_kv_heads=2,
        head_dim=8, router_heads=2, router_dim=4, total_topk_blocks=6,
        forced_init_blocks=1, forced_local_blocks=2)


def validate(value):
    return validate_leanselect_checkpoint(value, num_layers=1, num_heads=8,
        num_kv_heads=2, head_dim=8, total_topk_blocks=6,
        forced_init_blocks=1, forced_local_blocks=2)


def test_checkpoint_roundtrip_and_legacy_translation(tmp_path):
    original = checkpoint()
    path = tmp_path / 'selector.pt'
    torch.save(original, path)
    restored = torch.load(path, weights_only=True)
    canonical = validate(restored)
    new_selector = selector()
    new_selector.load_state_dict(canonical[0], strict=True)
    for name, tensor in original['leanselect']['layers']['0'].items():
        torch.testing.assert_close(new_selector.state_dict()[name], tensor, rtol=0, atol=0)
    legacy = {'phase_tiny': {**original['leanselect'], 'schema': 'phase_tiny_selector/v3'}}
    translated = normalize_checkpoint(legacy)
    assert 'leanselect' not in legacy
    assert leanselect_checkpoint_config(legacy) == original['leanselect']['config']
    for name in canonical[0]:
        assert validate(translated)[0][name] is legacy['phase_tiny']['layers']['0'][name]


@pytest.mark.parametrize('corrupt', ['shape', 'nonfinite', 'layer', 'dtype', 'role', 'budget'])
def test_checkpoint_rejects_incompatible_artifact(corrupt):
    artifact = copy.deepcopy(checkpoint())
    payload = artifact['leanselect']
    state = payload['layers']['0']
    if corrupt == 'shape':
        state['q_proj'] = state['q_proj'][:, :-1]
    elif corrupt == 'nonfinite':
        state['q_proj'][0, 0, 0] = float('nan')
    elif corrupt == 'layer':
        payload['layers']['2'] = payload['layers'].pop('0')
    elif corrupt == 'dtype':
        state['q_proj'] = state['q_proj'].half()
    elif corrupt == 'role':
        payload['role'] = 'random'
    else:
        payload['config']['total_topk_blocks'] = 7
    with pytest.raises(ValueError):
        validate(artifact)


def test_informative_windows_cover_disjoint_nontrivial_history():
    start = leanselect_informative_history_start(total_topk_blocks=96, block_size=64)
    windows = leanselect_supervision_windows(32768, 257, minimum_start=start)
    assert windows[0][0] == start and windows[-1][1] == 32768
    assert sum(end - begin for begin, end in windows) == 257
    assert all(left[1] <= right[0] for left, right in zip(windows, windows[1:]))
    assert len(windows) == 64


@pytest.mark.parametrize('kwargs', [{'num_heads': 7}, {'router_heads': 9}, {'head_dim': 0}])
def test_rejects_bad_selector_geometry(kwargs):
    config = dict(num_heads=8, num_kv_heads=2, head_dim=8)
    config.update(kwargs)
    with pytest.raises(ValueError):
        LeanSelectSelector(**config)


def test_rejects_bad_inputs():
    module = selector()
    with pytest.raises(ValueError):
        module.project_queries(torch.randn(2, 3, 8))
    with pytest.raises(ValueError):
        module.project_atom_keys(torch.randn(2, 3, 8))
    with pytest.raises(ValueError):
        UtilityLossConfig(boundary_weight=float('nan'))
    with pytest.raises(ValueError):
        pool_atoms_to_pages(torch.ones(3), seq_len=64, kernel_stride=0)
    with pytest.raises(ValueError):
        decision_aligned_utility_loss(torch.ones(2, 3), torch.ones(2, 4), topk=1)


@pytest.mark.parametrize('lengths', [((4, 4), (3, 3)), ((2, 5), (2, 4))])
@pytest.mark.parametrize('causal', [False, True])
def test_uniform_packed_scores_match_independent_sequences(lengths, causal):
    from leanselect.ops.head_mix import uniform_varlen_stage1
    qs, ks = lengths
    q = torch.randn(sum(qs), 8, 4)
    k = torch.randn(sum(ks), 2, 4)
    cuq = torch.tensor([0, qs[0], sum(qs)], dtype=torch.int32)
    cuk = torch.tensor([0, ks[0], sum(ks)], dtype=torch.int32)
    actual = uniform_varlen_stage1(q, k, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
        max_seqlen_k=max(ks), num_kv_heads=2, causal=causal, kernel_stride=2)
    expected = torch.full((2, sum(qs), max(ks)), float('-inf'))
    for i in range(2):
        mean_q = q[cuq[i]:cuq[i + 1]].reshape(qs[i], 2, 4, 4).mean(2)
        result = torch.einsum('qgd,kgd->gqk', mean_q, k[cuk[i]:cuk[i + 1]]) / 2
        if causal:
            visible = torch.arange(ks[i])[None, :] * 2 <= torch.arange(qs[i])[:, None]
            result.masked_fill_(~visible, float('-inf'))
        expected[:, cuq[i]:cuq[i + 1], :ks[i]] = result
    torch.testing.assert_close(actual, expected)
