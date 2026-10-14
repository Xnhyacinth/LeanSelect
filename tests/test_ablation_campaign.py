import json
from pathlib import Path

import pytest

from leanselect.ablation_campaign import build_plan, main, run_specs
from leanselect.train import parser as training_parser

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def plan(name, gpus=8, **kwargs):
    return build_plan(name, model='models/base', data='datasets/train', output='outputs/campaign',
                      gpus=gpus, config_root=CONFIGS, **kwargs)


def training_args(payload):
    command = payload['command']
    return training_parser().parse_args(command[command.index('leanselect.train') + 1:])


def test_campaign_covers_twelve_named_controls_with_three_seeds_per_backbone():
    specs = run_specs()
    assert len(specs) == len({spec['name'] for spec in specs}) == 12
    for backbone in ('minicpm', 'nosa'):
        fine = [spec for spec in specs if spec['name'].startswith(f'{backbone}_fine_')]
        assert {spec['seed'] for spec in fine} == {42, 43, 44}
        assert all(spec['teacher'] == [2, 1] and spec['override'] == 'full' for spec in fine)
    controls = {spec['name'] for spec in specs if spec['backbone'] == 'minicpm'
                and not spec['name'].startswith('minicpm_fine_')}
    assert controls == {f'minicpm_{variant}_s42' for variant in
                        ('no_boundary', 'no_worst', 'listwise_only', 'fixed_mixing',
                         'single_subspace', 'coarse')}


@pytest.mark.parametrize('gpus', [1, 2, 4, 8])
def test_commands_parse_as_real_trainer_flags_and_preserve_budget(gpus):
    for spec in run_specs():
        payload = plan(spec['name'], gpus)
        args = training_args(payload)
        assert args.batch_size * args.gradient_accumulation * gpus == payload['global_batch'] == 64
        assert args.steps == 300 and args.save_every == 20
        assert args.seed == spec['seed']
        assert args.learning_rate == 5e-4 and args.weight_decay == .01 and args.max_grad_norm == .5
        assert args.sequence_length == (65536 if args.backbone == 'minicpm' else 32768)
        assert payload['training_tokens'] == 300 * 64 * args.sequence_length
        assert f'--nproc-per-node={gpus}' in payload['command']
        assert payload['command'][1:3] == ['-m', 'torch.distributed.run']
        assert args.output.endswith(spec['name'])
        assert (args.training_kernel_size, args.training_kernel_stride) == tuple(spec['teacher'])


def test_ablation_controls_and_coarse_fine_shared_supervision():
    resolved = {spec['name']: plan(spec['name'])['resolved_sparse_config'] for spec in run_specs()}
    fine = resolved['minicpm_fine_s42']
    coarse = resolved['minicpm_coarse_s42']
    differences = {key for key in fine if fine[key] != coarse[key]}
    assert differences == {'training_kernel_size', 'training_kernel_stride'}
    assert coarse['training_kernel_size'] == 32 and coarse['training_kernel_stride'] == 16
    assert fine['training_kernel_size'] == 2 and fine['training_kernel_stride'] == 1
    assert fine['leanselect_supervision_rows'] == coarse['leanselect_supervision_rows'] == 64
    assert resolved['minicpm_no_boundary_s42']['boundary_weight'] == 0
    assert resolved['minicpm_no_worst_s42']['leanselect_worst_group_weight'] == 0
    assert resolved['minicpm_listwise_only_s42']['boundary_weight'] == 0
    assert resolved['minicpm_listwise_only_s42']['leanselect_worst_group_weight'] == 0
    assert resolved['minicpm_fixed_mixing_s42']['leanselect_freeze_mixing'] is True
    assert resolved['minicpm_single_subspace_s42']['leanselect_router_heads'] == 1
    assert resolved['minicpm_single_subspace_s42']['leanselect_router_dim'] == 128
    assert 'not resolution-only' in plan('minicpm_coarse_s42')['recipe_scope']
    assert 'no best-checkpoint selection' in plan('minicpm_fine_s42')['checkpoint_scope']


def test_dry_run_prints_plan_without_creating_output(tmp_path, capsys):
    destination = tmp_path / 'must-not-exist'
    main(['--run', 'nosa_fine_s44', '--model', 'models/nosa', '--data', 'datasets/nosa',
          '--output', str(destination), '--gpus', '2', '--config-root', str(CONFIGS)])
    payload = json.loads(capsys.readouterr().out)
    assert payload['plan_only'] is True
    assert not destination.exists()
    assert training_args(payload).seed == 44
    with pytest.raises(ValueError, match='GPUs'):
        plan('nosa_fine_s44', 3)


@pytest.mark.parametrize('flags', [['--training-kernel-size', '32'],
                                  ['--training-kernel-size', '32', '--training-kernel-stride', '0']])
def test_teacher_override_rejects_partial_or_nonpositive_geometry_before_cuda(flags, monkeypatch):
    import sys
    from leanselect.train import main as train
    monkeypatch.setattr(sys, 'argv', ['train', '--backbone', 'minicpm', '--model', 'models/base',
                                     '--data', 'datasets/train', '--config', 'configs/minicpm.json',
                                     '--output', 'outputs/test', *flags])
    with pytest.raises(ValueError, match='both be positive'):
        train()


def test_coarse_cli_reaches_model_with_resolved_teacher_config(tmp_path, monkeypatch):
    import sys
    import leanselect.checkpoints as checkpoints
    import leanselect.models as models
    import leanselect.train as train

    model_root = tmp_path / 'base'
    model_root.mkdir()
    (model_root / 'config.json').write_text('{}')
    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setattr(train.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(train.torch.cuda, 'set_device', lambda *args: None)
    monkeypatch.setattr(train, 'batch_indices', lambda *args: [0])
    monkeypatch.setattr(train, 'source_sha256', lambda: 'source')
    monkeypatch.setattr(checkpoints, 'base_model_weights_manifest_sha256', lambda *args: 'base')
    # len is checked before the model loader; retain a tiny real dataset shape.
    class Dataset:
        identity = 'data'
        def __len__(self):
            return 64
    monkeypatch.setattr(train, 'TokenDataset', lambda *args: Dataset())
    captured = {}
    class ReachedLoader(Exception):
        pass
    def load_model(*args, **kwargs):
        captured.update(kwargs['sparse_config'])
        raise ReachedLoader
    monkeypatch.setattr(models, 'load_model', load_model)
    command = plan('minicpm_coarse_s42')['command']
    flags = command[command.index('leanselect.train') + 1:]
    flags[flags.index('--model') + 1] = str(model_root)
    monkeypatch.setattr(sys, 'argv', ['train', *flags])
    with pytest.raises(ReachedLoader):
        train.main()
    assert captured['training_kernel_size'] == 32 and captured['training_kernel_stride'] == 16
    assert captured['leanselect_supervision_rows'] == 64
    assert captured['leanselect_total_topk_blocks'] == 96
