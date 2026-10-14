"""Print one command from the 12-run matched-budget training campaign.

Planning never imports torch, creates outputs, or starts GPU work. Run the
printed command separately after reviewing its model, data, and device choices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys


def run_specs():
    """Shared fine recipe, five mechanism controls, and one coarse contrast."""
    runs = []
    for backbone in ('minicpm', 'nosa'):
        for seed in (42, 43, 44):
            runs.append(dict(name=f'{backbone}_fine_s{seed}', backbone=backbone,
                             seed=seed, override='full', teacher=[2, 1]))
        if backbone == 'minicpm':
            for variant, override in (('no_boundary', 'no_boundary'),
                                      ('no_worst', 'no_worst_group'),
                                      ('listwise_only', 'listwise_only'),
                                      ('fixed_mixing', 'fixed_mixing'),
                                      ('single_subspace', 'direct_h1d128'),
                                      ('coarse', 'full')):
                runs.append(dict(name=f'minicpm_{variant}_s42', backbone=backbone,
                                 seed=42, override=override,
                                 teacher=[32, 16] if variant == 'coarse' else [2, 1]))
    return runs


def build_plan(name, *, model, data, output, gpus, config_root='configs', python=None):
    if gpus not in (1, 2, 4, 8):
        raise ValueError('GPUs must be 1, 2, 4, or 8 to retain global batch 64')
    spec = next((run for run in run_specs() if run['name'] == name), None)
    if spec is None:
        raise ValueError('unknown campaign run')
    config_root = Path(config_root)
    base = config_root / f"{spec['backbone']}.json"
    override = config_root / 'ablations' / f"{spec['override']}.json"
    resolved = {**json.loads(base.read_text()), **json.loads(override.read_text()),
                'training_kernel_size': spec['teacher'][0], 'training_kernel_stride': spec['teacher'][1]}
    expected = {'leanselect_total_topk_blocks': 96, 'leanselect_supervision_rows': 64,
                'listwise_weight': 1., 'loss_temperature': 1., 'boundary_margin': .1}
    if any(resolved.get(key) != value for key, value in expected.items()):
        raise ValueError('shared configuration differs from the matched campaign recipe')
    sequence_length = 65536 if spec['backbone'] == 'minicpm' else 32768
    accumulation = 8 // gpus
    command = [python or sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nnodes=1', f'--nproc-per-node={gpus}', '--module', 'leanselect.train',
               '--backbone', spec['backbone'], '--model', str(model), '--data', str(data),
               '--output', str(Path(output) / name), '--config', str(base),
               '--ablation-config', str(override),
               '--training-kernel-size', str(spec['teacher'][0]),
               '--training-kernel-stride', str(spec['teacher'][1]),
               '--sequence-length', str(sequence_length), '--batch-size', '8',
               '--gradient-accumulation', str(accumulation), '--steps', '300', '--save-every', '20',
               '--learning-rate', '0.0005', '--weight-decay', '0.01', '--max-grad-norm', '0.5',
               '--seed', str(spec['seed'])]
    return {'schema': 'leanselect_ablation_campaign/v1', 'run': spec,
            'plan_only': True, 'optimizer_updates': 300, 'global_batch': 64,
            'global_samples': 19200, 'training_tokens': 19200 * sequence_length,
            'sequence_length': sequence_length, 'gpus': gpus,
            'resolved_sparse_config': resolved,
            'config_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (base, override)},
            'checkpoint_scope': 'seed42 is a current-source reference rerun; no best-checkpoint selection',
            'recipe_scope': 'matched-budget teacher recipe contrast; coarse/fine is not resolution-only',
            'command': command, 'shell_command': shlex.join(command)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', choices=[spec['name'] for spec in run_specs()], required=True)
    parser.add_argument('--model', required=True, help='local frozen backbone for the selected run')
    parser.add_argument('--data', required=True, help='tokenizer-matched training data for the selected backbone')
    parser.add_argument('--output', required=True, help='output root; run name is appended')
    parser.add_argument('--gpus', type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument('--config-root', default='configs', help='directory containing the shared public configurations')
    args = parser.parse_args(argv)
    print(json.dumps(build_plan(args.run, model=args.model, data=args.data, output=args.output,
                                gpus=args.gpus, config_root=args.config_root), indent=2))


if __name__ == '__main__':
    main()
