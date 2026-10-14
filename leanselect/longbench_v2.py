"""Pinned LongBench v2 direct 0-shot generation with local Transformers models.

This protocol generates an answer; it does not score answer-choice likelihoods.
Official prompts and extraction code are read from a pinned upstream checkout.
"""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import platform
import random
import re
from pathlib import Path

import torch
from transformers import GenerationConfig

from .benchmark import calibrate, identity, last_token_logits_kwargs, load, model_arguments
from .data import file_sha256
from .evaluate import (append_record, digest_ids, initialize_output,
                       upstream_identity)

UPSTREAM_COMMIT = '2e00731f8d0bff23dc4325161044d0ed8af94c1e'
DATASET_REVISION = '2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9'
DATASET_SHA256 = '15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2'
OFFICIAL_FILES = {
    'pred.py': 'ab63f77866a1c0dc770582bc3fe6b014c3b4be4667399b0ee267075780c6a138',
    'result.py': 'b6675c1c0b72090c01e12c94d3b176c4991fe38ceed9a301d24d22b1997465bd',
    'prompts/0shot.txt': '68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950',
}
MAX_NEW_TOKENS = 128


def official_protocol(root):
    # v1 and v2 share the same upstream repository and pinned revision.
    upstream = upstream_identity(root, 'longbench')
    if upstream['commit'] != UPSTREAM_COMMIT:
        raise ValueError('official LongBench v2 revision differs')
    for name, expected in OFFICIAL_FILES.items():
        if file_sha256(Path(root) / name) != expected:
            raise ValueError(f'official LongBench v2 source differs: {name}')
    source = ast.parse((Path(root) / 'pred.py').read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_answer')
    scope = {'re': re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'official_extract_answer', 'exec'), scope)
    return (Path(root) / 'prompts/0shot.txt').read_text(), scope['extract_answer'], upstream


def read_examples(path, subset=False):
    path = Path(path)
    if not subset and file_sha256(path) != DATASET_SHA256:
        raise ValueError('full evaluation requires the pinned official data.json; use --subset for other inputs')
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('expected a nonempty JSON array or JSONL file')
    required = ('_id', 'context', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'answer',
                'difficulty', 'length', 'domain', 'sub_domain')
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) for key in required):
            raise ValueError('missing LongBench v2 string fields')
        if row['answer'] not in 'ABCD' or len(row['answer']) != 1:
            raise ValueError('answer must be A, B, C, or D')
        if row['difficulty'] not in {'easy', 'hard'} or row['length'] not in {'short', 'medium', 'long'}:
            raise ValueError('invalid LongBench v2 difficulty/length metadata')
    if len({row['_id'] for row in rows}) != len(rows):
        raise ValueError('duplicate LongBench v2 IDs')
    if not subset and len(rows) != 503:
        raise ValueError('full LongBench v2 evaluation requires 503 examples')
    return rows


def prepare_prompt(row, tokenizer, template, prompt_budget, context_cap):
    prompt = template
    for placeholder, field in (('$DOC$', 'context'), ('$Q$', 'question'),
                               *[(f'$C_{letter}$', f'choice_{letter}') for letter in 'ABCD']):
        prompt = prompt.replace(placeholder, row[field].strip())
    # Match upstream local-tokenizer encoding before the chat template.
    tokens = tokenizer.encode(prompt)
    raw_count = len(tokens)
    truncated = raw_count > prompt_budget
    if truncated:
        half = prompt_budget // 2
        prompt = tokenizer.decode(tokens[:half] + tokens[-half:], skip_special_tokens=True)
    chat = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                         tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(chat, add_special_tokens=False, return_tensors='pt')
    count = encoded['input_ids'].shape[-1]
    if count + MAX_NEW_TOKENS > context_cap:
        raise ValueError('chat input plus generation exceeds --context-cap')
    return encoded, {'raw_prompt_tokens': raw_count, 'truncated': truncated, 'input_tokens': count,
                     'input_ids_sha256': digest_ids(encoded['input_ids'])}


def validate_saved(saved, rows, extract_answer):
    if len(saved) > len(rows):
        raise ValueError('saved predictions exceed dataset size')
    for index, prediction in enumerate(saved):
        row = rows[index]
        for key in ('_id', 'answer', 'difficulty', 'length', 'domain', 'sub_domain'):
            if prediction.get(key) != row[key]:
                raise ValueError('saved prediction metadata differs from dataset')
        if prediction.get('pred') != extract_answer(prediction['response']):
            raise ValueError('saved prediction differs from official answer extraction')
        if prediction.get('judge') != (prediction['pred'] == row['answer']):
            raise ValueError('saved prediction score differs')
        if not 0 <= prediction.get('generated_tokens', -1) <= MAX_NEW_TOKENS:
            raise ValueError('saved generated token count differs')


def summarize(rows):
    """Uncompensated official Overall/Easy/Hard/Short/Medium/Long percentages."""
    groups = {'Overall': rows, **{name.title(): [row for row in rows if row[field] == name]
              for field, names in (('difficulty', ('easy', 'hard')), ('length', ('short', 'medium', 'long')))
              for name in names}}
    return {name: {'count': len(group),
                   'accuracy': round(100 * sum(row['judge'] for row in group) / len(group), 1) if group else None}
            for name, group in groups.items()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument('--upstream', required=True, help='official LongBench checkout at documented revision')
    parser.add_argument('--inputs', required=True, help='pinned official data.json; JSONL allowed with --subset')
    parser.add_argument('--subset', action='store_true', help='explicit partial/custom input; never labeled full benchmark')
    parser.add_argument('--prompt-budget', type=int, default=120000, help='pre-chat first/last token budget')
    parser.add_argument('--context-cap', type=int, default=131072, help='model input plus output capacity')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.prompt_budget < 2 or args.prompt_budget % 2 or args.context_cap <= MAX_NEW_TOKENS:
        parser.error('prompt budget must be positive/even and context capacity must exceed generation length')
    template, extract_answer, upstream = official_protocol(args.upstream)
    rows = read_examples(args.inputs, args.subset)
    for name in ('model', 'config', 'selector_checkpoint', 'sparda_checkpoint'):
        value = getattr(args, name)
        if value:
            setattr(args, name, str(Path(value).resolve()))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, tokenizer = load(args)
    model.eval()
    if model.generation_config.eos_token_id is None:
        raise ValueError('native generation EOS is missing')
    generation = {'temperature': .1, 'max_new_tokens': MAX_NEW_TOKENS, 'do_sample': True,
                  'top_p': 1., 'top_k': 0, 'repetition_penalty': 1., 'min_new_tokens': 0,
                  'num_beams': 1, 'use_cache': True}
    config = GenerationConfig(**generation, eos_token_id=model.generation_config.eos_token_id,
                              bos_token_id=model.generation_config.bos_token_id,
                              pad_token_id=model.generation_config.pad_token_id)
    first_inputs, _ = prepare_prompt(rows[0], tokenizer, template, args.prompt_budget, args.context_cap)
    calibration = calibrate(args, model, first_inputs['input_ids'].to(model.device), MAX_NEW_TOKENS)
    contract = {'benchmark': 'longbench_v2', 'protocol': 'direct_0shot_generation',
                'upstream': upstream, 'official_files_sha256': OFFICIAL_FILES,
                **identity(args), 'dataset_sha256': file_sha256(args.inputs),
                'dataset_repository': 'zai-org/LongBench-v2', 'dataset_revision': DATASET_REVISION,
                'count': len(rows), 'scope': 'subset/custom inputs' if args.subset else 'full official 503 cases',
                'prompt_budget': args.prompt_budget, 'context_cap': args.context_cap,
                'generation': generation, 'seed': args.seed, 'sampling_seed': 'seed + input row index',
                'chat_template': 'native single user message; tokenizer default thinking policy',
                'backend': 'local Transformers adaptation; not bitwise equivalent to upstream API backend',
                'runtime': {'python': platform.python_version(),
                            **{name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy')}},
                'calibration': calibration}
    predictions, saved = initialize_output(args.output, contract, args.resume)
    validate_saved(saved, rows, extract_answer)
    for index, row in enumerate(rows):
        inputs, prompt_identity = prepare_prompt(row, tokenizer, template, args.prompt_budget, args.context_cap)
        if index < len(saved):
            if any(saved[index].get(key) != value for key, value in prompt_identity.items()):
                raise ValueError('resumed input tokenization differs')
            if saved[index].get('seed') != args.seed + index:
                raise ValueError('resumed sampling seed differs')
            continue
        torch.manual_seed(args.seed + index)
        random.seed(args.seed + index)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        output = model.generate(**inputs, generation_config=config, **last_token_logits_kwargs(model))
        generated = output[0, prompt_identity['input_tokens']:]
        if not 0 <= len(generated) <= MAX_NEW_TOKENS:
            raise RuntimeError('generation exceeded the fixed output cap')
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        pred = extract_answer(response)
        record = {key: row[key] for key in ('_id', 'answer', 'difficulty', 'length', 'domain', 'sub_domain')}
        record.update(index=index, response=response, pred=pred, judge=pred == row['answer'],
                      generated_tokens=len(generated), seed=args.seed + index, **prompt_identity)
        append_record(predictions, record)
        saved.append(record)
    validate_saved(saved, rows, extract_answer)
    report = {**contract, 'metrics': summarize(saved), 'predictions_sha256': file_sha256(predictions)}
    (Path(args.output) / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
