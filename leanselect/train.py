# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Train only LeanSelect parameters with frozen decoder layers.

Run with python -m leanselect.train or torchrun --module leanselect.train.
Layerwise backward bounds activation memory; gradients are averaged explicitly
across ranks once per update. Checkpoints contain no backbone weights.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist

from .data import TokenDataset, batch_indices, file_sha256, source_sha256
from .selector import LeanSelectSelector


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    torch.save(value, temporary)
    os.replace(temporary, path)


def rng_state():
    state = {'python': random.getstate(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def selectors(model):
    found = {name: module for name, module in model.named_modules()
             if isinstance(module, LeanSelectSelector)}
    if not found:
        raise ValueError('model contains no LeanSelect selectors')
    return found


def freeze_backbone(model, *, freeze_mixing=False):
    model.requires_grad_(False)
    modules = selectors(model)
    for module in modules.values():
        module.float().requires_grad_(True)
        if freeze_mixing:
            module.head_mix.requires_grad_(False)
            module.kv_head_mix.requires_grad_(False)
    return [parameter for module in modules.values() for parameter in module.parameters()
            if parameter.requires_grad]


def ablation_config(config, overrides):
    """Apply a small ablation patch while keeping the backbone configuration shared."""
    allowed = {'leanselect_router_heads', 'leanselect_router_dim',
               'leanselect_freeze_mixing', 'boundary_weight',
               'leanselect_worst_group_weight'}
    if not isinstance(config, dict) or not isinstance(overrides, dict):
        raise ValueError('base and ablation configurations must be JSON objects')
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f'unsupported ablation settings: {sorted(unknown)}')
    result = {**config, **overrides}
    if not isinstance(result.get('leanselect_freeze_mixing', False), bool):
        raise ValueError('leanselect_freeze_mixing must be a boolean')
    return result


def layerwise_backward(model, input_ids, backbone, accumulation):
    """Reuse the deployed runtime's sampled fine-teacher objective."""
    decoder = model.model
    mask = torch.ones_like(input_ids)
    positions = torch.arange(input_ids.shape[-1], device=input_ids.device)[None, :]
    positions = positions.expand(input_ids.shape[0], -1)
    with torch.no_grad():
        hidden = decoder.embed_tokens(input_ids)
        if backbone == 'minicpm':
            hidden = hidden * model.config.scale_emb
        embeddings = decoder.rotary_emb(hidden, positions) if backbone == 'nosa' else None
    total, count = 0.0, 0
    previous = None
    # Preserve each backbone's original training normalization.
    divisor = accumulation * (len(decoder.layers) if backbone == 'minicpm' else 1)
    for layer in decoder.layers:
        kwargs = dict(attention_mask=mask, position_ids=positions,
                      past_key_value=None, use_cache=False,
                      q_future_from_prev=previous, training_stage=1)
        if embeddings is not None:
            kwargs['position_embeddings'] = embeddings
        output = layer(hidden, **kwargs)
        hidden, previous, auxiliary = output[0].detach(), output[-2], output[-1]
        loss = auxiliary
        if backbone == 'minicpm':
            if auxiliary is None or auxiliary[1] != 'precomputed':
                raise RuntimeError('runtime did not return the LeanSelect utility loss')
            loss = auxiliary[0]
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            raise RuntimeError('selector training loss is absent or detached')
        if not torch.isfinite(loss).all():
            raise FloatingPointError('nonfinite selector loss')
        (loss / divisor).backward()
        total += float(loss.detach())
        count += 1
    return total / count


def save_checkpoint(model, optimizer, destination, *, step, microbatch, contract, rank, world):
    local_rng = rng_state()
    states = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(local_rng, states, dst=0)
    else:
        states = [local_rng]
    error = [None]
    if rank == 0:
        try:
            payload = {'format': 'leanselect_training/v1', 'step': step,
                       'microbatch': microbatch, 'contract': contract, 'rng': states,
                       'selectors': {name: {k: v.detach().cpu() for k, v in module.state_dict().items()}
                                     for name, module in selectors(model).items()},
                       'optimizer': optimizer.state_dict()}
            atomic_save(payload, destination)
            export_selector(model, Path(destination).with_name('selector.pt'))
            export_selector(model, Path(destination).with_name(f'selector-{step:06d}.pt'))
        except Exception as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def export_selector(model, destination):
    from .checkpoints import build_leanselect_checkpoint
    layers = {}
    for index, layer in enumerate(model.model.layers):
        matches = [module for module in layer.modules() if isinstance(module, LeanSelectSelector)]
        if len(matches) != 1:
            raise ValueError(f'expected one selector in layer {index}')
        layers[index] = matches[0].canonical_state()
    selector = matches[0]
    attention = model.model.layers[0].self_attn
    config = dict(num_heads=selector.num_heads, num_kv_heads=selector.num_kv_heads,
                  head_dim=selector.head_dim, router_heads=selector.router_heads,
                  router_dim=selector.router_dim,
                  total_topk_blocks=attention.leanselect_total_topk_blocks,
                  forced_init_blocks=attention.init_blocks,
                  forced_local_blocks=attention.local_blocks)
    atomic_save(build_leanselect_checkpoint(layers, **config), destination)


def load_checkpoint(path, model, optimizer, contract, rank):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format') != 'leanselect_training/v1' or saved['contract'] != contract:
        raise ValueError('resume training/data/model configuration differs')
    modules = selectors(model)
    if set(saved['selectors']) != set(modules):
        raise ValueError('resume selector layer names differ')
    for name, module in modules.items():
        module.load_state_dict(saved['selectors'][name], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'][rank])
    return saved['step'], saved['microbatch']


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone', choices=['minicpm', 'nosa'], required=True)
    p.add_argument('--model', required=True, help='local frozen model directory')
    p.add_argument('--data', required=True, help='pretokenized JSONL or saved Dataset')
    p.add_argument('--output', required=True)
    p.add_argument('--config', required=True, help='LeanSelect sparse configuration JSON')
    p.add_argument('--ablation-config', help='JSON overrides from configs/ablations')
    p.add_argument('--export-initial-selector', action='store_true',
                   help='also save selector-000000.pt before the first optimizer update')
    p.add_argument('--resume', help='training.pt checkpoint; not a deployment checkpoint')
    p.add_argument('--sequence-length', type=int, default=65536)
    p.add_argument('--batch-size', type=int, default=8, help='per rank')
    p.add_argument('--gradient-accumulation', type=int, default=2)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--save-every', type=int, default=20)
    p.add_argument('--learning-rate', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--max-grad-norm', type=float, default=.5)
    p.add_argument('--seed', type=int, default=42)
    return p


def main():
    args = parser().parse_args()
    if args.resume and args.export_initial_selector:
        raise ValueError('initial selector export requires a fresh training run')
    if min(args.sequence_length, args.batch_size, args.gradient_accumulation,
           args.steps, args.save_every) <= 0 or args.learning_rate <= 0:
        raise ValueError('training sizes and learning rate must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('the sparse training kernels require CUDA')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        config = ablation_config(json.loads(Path(args.config).read_text()),
                                 json.loads(Path(args.ablation_config).read_text())
                                 if args.ablation_config else {})
        dataset = TokenDataset(args.data, args.sequence_length)
        from .checkpoints import base_model_weights_manifest_sha256
        contract = {k: v for k, v in vars(args).items()
                    if k not in {'output', 'resume', 'steps', 'save_every', 'config',
                                 'ablation_config', 'export_initial_selector', 'model', 'data'}}
        contract.update(config=config, world_size=world, data_sha256=dataset.identity,
                        optimizer={'name': 'AdamW', 'betas': [0.9, 0.95], 'eps': 1e-6},
                        source_sha256=source_sha256(),
                        model_sha256=base_model_weights_manifest_sha256(args.model),
                        model_config_sha256=file_sha256(Path(args.model) / 'config.json'))
        # Validate global batch geometry before loading a large model.
        batch_indices(len(dataset), 0, args.batch_size, rank, world, args.seed)
        from .models import load_model
        model, _ = load_model(args.backbone, args.model, method='leanselect',
                              device=f'cuda:{local_rank}', sparse_config=config, training=True)
        parameters = freeze_backbone(model, freeze_mixing=config.get('leanselect_freeze_mixing', False))
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay,
                                      betas=(0.9, 0.95), eps=1e-6)
        model.train()
        step, microbatch = 0, 0
        if args.resume:
            step, microbatch = load_checkpoint(args.resume, model, optimizer, contract, rank)
        else:
            # Same initialization on every rank, independent sampled teacher rows.
            torch.manual_seed(args.seed + rank)
            random.seed(args.seed + rank)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            (output / 'config.json').write_text(json.dumps(contract, indent=2) + '\n')
            if args.export_initial_selector:
                export_selector(model, output / 'selector-000000.pt')
        while step < args.steps:
            optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            for _ in range(args.gradient_accumulation):
                indices = batch_indices(len(dataset), microbatch, args.batch_size, rank, world, args.seed)
                ids = torch.stack([dataset[index] for index in indices]).to(local_rank)
                loss += layerwise_backward(model, ids, args.backbone, args.gradient_accumulation)
                microbatch += 1
            present = torch.tensor([parameter.grad is not None for parameter in parameters],
                                   device=f'cuda:{local_rank}', dtype=torch.int32)
            if world > 1:
                dist.all_reduce(present, op=dist.ReduceOp.MIN)
            if not bool(present.all()):
                raise RuntimeError('a selector parameter received no gradient on at least one rank')
            for parameter in parameters:
                if world > 1:
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(world)
            norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if rank == 0:
                with (output / 'training.jsonl').open('a') as handle:
                    handle.write(json.dumps({'step': step, 'loss_rank0': loss / args.gradient_accumulation,
                                             'grad_norm': float(norm)}) + '\n')
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(model, optimizer, output / 'training.pt', step=step,
                                microbatch=microbatch, contract=contract, rank=rank, world=world)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
