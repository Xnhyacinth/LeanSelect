"""Measure warmed CUDA generation with identical real input tokens.

Each invocation measures one model/method on one GPU. Launch methods separately
on an otherwise idle device. Forward phases and full generation are measured in
the same request; tokenization, model loading, and warmups are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from .data import file_sha256, read_jsonl, source_sha256

METHODS = ['dense', 'native', 'sparda', 'leanselect', 'uniform', 'infinigen',
           'minference', 'xattention', 'recency']


def model_arguments(parser):
    parser.add_argument('--backbone', choices=['minicpm', 'nosa'], required=True)
    parser.add_argument('--model', required=True, help='local model directory')
    parser.add_argument('--method', choices=METHODS, default='leanselect')
    parser.add_argument('--selector-checkpoint')
    parser.add_argument('--sparda-checkpoint')
    parser.add_argument('--config', help='optional sparse configuration JSON')
    parser.add_argument('--device', default='cuda:0')


def load(args, training=False):
    from .models import load_model
    config = json.loads(Path(args.config).read_text()) if args.config else None
    return load_model(args.backbone, args.model, method=args.method,
                      selector_checkpoint=args.selector_checkpoint,
                      sparda_checkpoint=args.sparda_checkpoint, device=args.device,
                      sparse_config=config, training=training)


def calibrate(args, model, input_ids, max_new_tokens):
    """Prepare the paper's backbone-specific InfiniGen basis once per task."""
    if args.method != 'infinigen':
        return None
    from .baselines.infinigen import calibrate_infinigen
    return calibrate_infinigen(model, input_ids, max_new_tokens, backbone=args.backbone)


def identity(args):
    from .checkpoints import base_model_weights_manifest_sha256
    return {'backbone': args.backbone, 'method': args.method,
            'source_sha256': source_sha256(),
            'base_model_sha256': base_model_weights_manifest_sha256(args.model),
            'model_config_sha256': file_sha256(Path(args.model) / 'config.json'),
            'selector_sha256': file_sha256(args.selector_checkpoint) if args.selector_checkpoint else None,
            'sparda_sha256': file_sha256(args.sparda_checkpoint) if args.sparda_checkpoint else None,
            'sparse_config': json.loads(Path(args.config).read_text()) if args.config else None}


def distribution(values):
    if not values or any(x <= 0 for x in values):
        raise ValueError('timing samples must be positive')
    # Linear interpolation matches NumPy's default quantile definition.
    ordered = sorted(values)
    def quantile(q):
        index = (len(ordered) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {'median': statistics.median(values), 'q1': quantile(.25), 'q3': quantile(.75)}


def last_token_logits_kwargs(model):
    """Use the backbone's actual generation API to project only the last token."""
    import inspect
    parameters = inspect.signature(model.forward).parameters
    backbone = getattr(model, 'leanselect_backbone', None)
    expected = {'minicpm': 'logits_to_keep', 'nosa': 'num_logits_to_keep'}.get(backbone)
    if expected is not None:
        if expected not in parameters:
            raise ValueError(f'{backbone} forward lacks its last-token logits argument')
        return {expected: 1}
    for name in ('logits_to_keep', 'num_logits_to_keep'):
        if name in parameters:
            return {name: 1}
    return {}


@torch.inference_mode()
def measure_request(model, input_ids, output_tokens, synchronize):
    """Time generate itself; its actual forward calls provide phase durations."""
    if output_tokens < 2:
        raise ValueError('at least two output tokens are required for decode timing')
    calls = []
    head_kwargs = last_token_logits_kwargs(model)
    original = model.forward
    def timed_forward(*args, **kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        synchronize()
        begin = time.perf_counter()
        result = original(*args, **kwargs)
        synchronize()
        elapsed = time.perf_counter() - begin
        logits = getattr(result, 'logits', None)
        positions = int(logits.shape[-2]) if logits is not None else None
        if head_kwargs and positions != 1:
            raise RuntimeError('generation did not apply the last-token-only LM head')
        calls.append({'tokens': int(ids.shape[-1]), 'seconds': elapsed, 'logit_positions': positions})
        return result
    # Preserve the forward signature inspected by Transformers generation.
    import functools
    model.forward = functools.wraps(original)(timed_forward)
    try:
        synchronize()
        begin = time.perf_counter()
        output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                do_sample=False, num_beams=1, use_cache=True,
                                max_new_tokens=output_tokens, min_new_tokens=output_tokens, **head_kwargs)
        synchronize()
        wall = time.perf_counter() - begin
    finally:
        model.forward = original
    generated = output[:, input_ids.shape[-1]:]
    if generated.shape != (input_ids.shape[0], output_tokens):
        raise RuntimeError('generation did not produce the fixed output length')
    if len(calls) != output_tokens or calls[0]['tokens'] != input_ids.shape[-1] or any(
            call['tokens'] != 1 for call in calls[1:]):
        raise RuntimeError('expected one full prefill and one-token cached decode calls')
    prefill = calls[0]['seconds']
    decode = sum(x['seconds'] for x in calls[1:])
    return {'prefill_forward_s': prefill, 'decode_forward_s': decode, 'wall_time_s': wall,
            'prefill_tokens_per_s': input_ids.numel() / prefill,
            'decode_tokens_per_s': input_ids.shape[0] * (output_tokens - 1) / decode,
            'request_tokens_per_s': input_ids.shape[0] * output_tokens / wall,
            'generated_ids': generated.cpu().tolist(), 'forward_calls': calls,
            'lm_head_policy': head_kwargs or {'policy': 'model default'}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    model_arguments(p)
    p.add_argument('--inputs', required=True, help='JSONL rows with input_ids (real tokenized prompts)')
    p.add_argument('--length', type=int, required=True, help='explicit prefix length; no synthetic padding')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--output-tokens', type=int, default=4)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if min(args.length, args.batch_size, args.warmups, args.repeats) <= 0:
        raise ValueError('length, batch size, warmups and repeats must be positive')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('efficiency measurement requires CUDA')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('choose a new output path to preserve measurements')
    rows = read_jsonl(args.inputs)
    if len(rows) < args.batch_size or any(len(x['input_ids']) < args.length for x in rows[:args.batch_size]):
        raise ValueError('insufficient real input tokens for the requested batch')
    torch.cuda.set_device(torch.device(args.device))
    provenance = identity(args)
    model, _ = load(args)
    model.eval()
    ids = torch.tensor([x['input_ids'][:args.length] for x in rows[:args.batch_size]],
                       device=args.device, dtype=torch.long)
    synchronize = lambda: torch.cuda.synchronize(args.device)
    calibration = calibrate(args, model, ids, args.output_tokens)
    warmups = [measure_request(model, ids, args.output_tokens, synchronize) for _ in range(args.warmups)]
    torch.cuda.reset_peak_memory_stats(args.device)
    samples = [measure_request(model, ids, args.output_tokens, synchronize) for _ in range(args.repeats)]
    if any(sample['generated_ids'] != warmups[-1]['generated_ids'] for sample in samples):
        raise RuntimeError('deterministic repeated requests produced different tokens')
    metrics = ['prefill_forward_s', 'decode_forward_s', 'wall_time_s',
               'prefill_tokens_per_s', 'decode_tokens_per_s', 'request_tokens_per_s']
    report = {'schema': 'leanselect_efficiency/v1', **provenance,
              'input_file_sha256': file_sha256(args.inputs),
              'input_ids_sha256': hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
              'input_length': args.length, 'batch_size': args.batch_size,
              'calibration': calibration,
              'lm_head_policy': last_token_logits_kwargs(model),
              'effective_sparse_config': model.config.sparse_config,
              'output_tokens_per_request': args.output_tokens, 'warmups': args.warmups,
              'samples': samples, 'summary': {key: distribution([x[key] for x in samples]) for key in metrics},
              'device': torch.cuda.get_device_name(args.device),
              'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device),
              'timing_scope': 'synchronized actual generate forwards; wall includes generation bookkeeping',
              'host_isolation': 'not automatically verified'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
