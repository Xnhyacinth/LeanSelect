import json
from pathlib import Path

import pytest
import torch

from leanselect.objective import UtilityLossConfig, leanselect_per_gqa_plan_loss
from leanselect.selector import LeanSelectSelector
from leanselect.train import ablation_config, freeze_backbone


CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


@pytest.mark.parametrize('backbone', ['minicpm', 'nosa'])
def test_ablation_patches_preserve_teacher_and_page_budget(backbone):
    base = json.loads((CONFIGS / f'{backbone}.json').read_text())
    for path in (CONFIGS / 'ablations').glob('*.json'):
        config = ablation_config(base, json.loads(path.read_text()))
        for key in ('leanselect_total_topk_blocks', 'training_kernel_size',
                    'training_kernel_stride', 'leanselect_supervision_rows'):
            assert config[key] == base[key]
        assert config['leanselect_router_heads'] * config['leanselect_router_dim'] == 128
    assert 'leanselect_freeze_mixing' not in base


def test_fixed_mixing_trains_projections_and_preserves_mixing_and_backbone():
    torch.manual_seed(7)
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(4, 4)
    model.selector = LeanSelectSelector(num_heads=4, num_kv_heads=2, head_dim=4,
                                       router_heads=2, router_dim=2)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    parameters = freeze_backbone(model, freeze_mixing=True)
    assert {id(parameter) for parameter in parameters} == {
        id(model.selector.q_proj), id(model.selector.k_proj)}
    optimizer = torch.optim.AdamW(parameters, lr=.01)
    query, key = torch.randn(8, 4, 4), torch.randn(8, 2, 4)
    loss = (model.selector.project_queries(query) *
            model.selector.project_atom_keys(key)).square().mean()
    loss.backward()
    optimizer.step()
    after = model.state_dict()
    for name in before:
        if name in {'selector.q_proj', 'selector.k_proj'}:
            assert not torch.equal(before[name], after[name])
        else:
            assert torch.equal(before[name], after[name])
    assert model.selector.head_mix.grad is None
    assert model.selector.kv_head_mix.grad is None
    # A subsequent full run can explicitly restore all selector gradients.
    assert len(freeze_backbone(model)) == 4


def test_loss_ablations_remove_the_declared_gradient_terms():
    student = torch.tensor([[[0., 2., 1.]], [[1., 0., 3.]]], requires_grad=True)
    teacher = torch.tensor([[[3., 1., 0.]], [[0., 4., 1.]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    forced = torch.zeros_like(mask)
    gradients = {}
    for name in ('full', 'listwise_only', 'no_boundary', 'no_worst_group'):
        base = json.loads((CONFIGS / 'minicpm.json').read_text())
        config = ablation_config(base, json.loads((CONFIGS / 'ablations' / f'{name}.json').read_text()))
        loss, _ = leanselect_per_gqa_plan_loss(
            student, teacher, topk=1, valid_mask=mask, forced_mask=forced,
            worst_group_weight=config['leanselect_worst_group_weight'],
            config=UtilityLossConfig(boundary_weight=config['boundary_weight']))
        gradients[name] = torch.autograd.grad(loss, student)[0]
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())
    assert not torch.equal(gradients['full'], gradients['no_boundary'])
    assert not torch.equal(gradients['full'], gradients['no_worst_group'])
    assert not torch.equal(gradients['no_boundary'], gradients['listwise_only'])


def test_ablation_configuration_rejects_unknown_or_mistyped_controls():
    with pytest.raises(ValueError, match='unsupported ablation settings'):
        ablation_config({}, {'leanselect_freeze_mixng': True})
    with pytest.raises(ValueError, match='boolean'):
        ablation_config({}, {'leanselect_freeze_mixing': 'false'})
    with pytest.raises(ValueError, match='JSON objects'):
        ablation_config({}, [])


def test_initial_control_cannot_be_reexported_from_a_resumed_run(monkeypatch):
    import sys
    from leanselect.train import main

    monkeypatch.setattr(sys, 'argv', ['train', '--backbone', 'minicpm',
                        '--model', 'models/base', '--data', 'datasets/train',
                        '--config', 'configs/minicpm.json', '--output', 'outputs/test',
                        '--resume', 'outputs/test/training.pt', '--export-initial-selector'])
    with pytest.raises(ValueError, match='fresh training run'):
        main()
