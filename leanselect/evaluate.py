# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Quality evaluation using official benchmark prompts and scoring code.

Each invocation evaluates one dataset file/task. Results record the exact input
count and hash; a single task is never labeled a complete benchmark suite.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import torch

from .benchmark import calibrate, identity, load, model_arguments
from .data import file_sha256, read_jsonl
from .metrics import (extract_last_boxed as extract_boxed, normalize_math_answer as normalize_answer,
                      parse_aime_prediction, parse_math_prediction, strip_thinking_blocks)

UPSTREAM = {
    'longbench': ('https://github.com/THUDM/LongBench', '2e00731f8d0bff23dc4325161044d0ed8af94c1e'),
    'ruler': ('https://github.com/NVIDIA/RULER', 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'),
    'helmet': ('https://github.com/princeton-nlp/HELMET', 'aeadacc6593e9a222a84b0660b186a4e23d2efda'),
    'reasoning': ('https://github.com/thunlp/NOSA', '90045b65e6620848f136ae0cf1cc14ab4f40ec68'),
}


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upstream_identity(root, benchmark):
    root = Path(root).resolve()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    expected = UPSTREAM[benchmark][1]
    if commit != expected:
        raise ValueError(f'checkout must use the documented {benchmark} revision {expected}')
    changed = subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--name-only'], text=True)
    if changed.strip():
        raise ValueError('upstream benchmark has modified tracked files')
    return {'repository': UPSTREAM[benchmark][0], 'commit': commit}


def digest_ids(ids):
    return hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def append_record(path, row):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def initialize_output(directory, contract, resume):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / 'run.json'
    if config.exists():
        if not resume or json.loads(config.read_text()) != contract:
            raise ValueError('output exists; resume requires the same full run configuration')
    else:
        if any(directory.iterdir()):
            raise ValueError('new evaluation output directory must be empty')
        config.write_text(json.dumps(contract, indent=2) + '\n')
    predictions = directory / 'predictions.jsonl'
    saved = read_jsonl(predictions) if predictions.exists() and predictions.stat().st_size else []
    if [row['index'] for row in saved] != list(range(len(saved))):
        raise ValueError('saved prediction index/order differs')
    return predictions, saved


def chat(tokenizer, prompt, thinking=False):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=thinking)


def longbench_adapter(root, task, max_length, tokenizer):
    root = Path(root)
    # LongBench v1 files live under LongBench/ in newer upstream checkouts.
    if not (root / 'config/dataset2prompt.json').exists():
        root = root / 'LongBench'
    prompts = json.loads((root / 'config/dataset2prompt.json').read_text())
    lengths = json.loads((root / 'config/dataset2maxlen.json').read_text())
    sys.path.insert(0, str(root))
    official = import_file(root / 'eval.py', '_leanselect_longbench_eval')
    def prepare(row):
        prompt = prompts[task].format(**row)
        ids = tokenizer(prompt, truncation=False, return_tensors='pt').input_ids[0]
        if len(ids) > max_length:
            half = max_length // 2
            prompt = tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(ids[-half:], skip_special_tokens=True)
        if task not in {'trec', 'triviaqa', 'samsum', 'lsht', 'lcc', 'repobench-p'}:
            prompt = chat(tokenizer, prompt, thinking=False)
        return prompt
    def score(rows, saved):
        return {'score': official.scorer(task, [x['prediction'] for x in saved],
                                        [x['answers'] for x in rows], rows[-1]['all_classes'])}
    return prepare, score, lengths[task]


def ruler_adapter(root, task):
    import yaml
    root = Path(root) / 'scripts'
    tasks = yaml.safe_load((root / 'synthetic.yaml').read_text())
    config = tasks[task]
    scoring = import_file(root / 'eval/synthetic/constants.py', '_leanselect_ruler_metrics')
    generation = import_file(root / 'data/synthetic/constants.py', '_leanselect_ruler_generation')
    def score(rows, saved):
        predictions = [re.sub(r'[\x00-\x1f]', '\n', x['prediction'].strip()).strip() for x in saved]
        refs = [row.get('outputs', [row.get('output', '')]) for row in rows]
        if any(not isinstance(ref, list) or not ref for ref in refs):
            raise ValueError('RULER references are missing')
        metric = scoring.TASKS[config['task']]['metric_fn']
        return {'score': metric(predictions, refs)}
    # Inputs are generated by the official RULER preparation script, including
    # its model template and answer prefix. Never apply a second chat template.
    return lambda row: row['input'], score, generation.TASKS[config['task']]['tokens_to_generate']


def reasoning_answer(text, task):
    """Exact prediction extraction used by the paper's reasoning evaluator."""
    stripped = strip_thinking_blocks(text)
    boxed = extract_boxed(stripped)
    if boxed is None:
        boxed = extract_boxed(text)
    if boxed is not None:
        return normalize_answer(boxed)
    parsed = parse_aime_prediction(stripped) if task.startswith('aime') else parse_math_prediction(stripped)
    return normalize_answer(parsed)


def reasoning_adapter(root, task, backbone, tokenizer):
    if task not in {'math-500', 'aime24', 'aime25'}:
        raise ValueError('reasoning task must be math-500, aime24, or aime25')
    prefix = None
    if backbone == 'nosa':
        # Read only the official literal demonstrations, avoiding its model
        # loader and unrelated runtime imports.
        source = Path(root) / 'benchmarks/Reasoning/test.py'
        nodes = ast.parse(source.read_text()).body
        values = [ast.literal_eval(node.value) for node in nodes if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == '_prefix' for target in node.targets)]
        if len(values) != 1 or not isinstance(values[0], str):
            raise ValueError('official NOSA two-shot prefix is unavailable')
        prefix = values[0]
    def prepare(row):
        problem = row.get('problem', row.get('question'))
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError('missing mathematical problem')
        instruction = 'Please reason step by step, and put your final answer within \\boxed{}.'
        if task.startswith('aime'):
            if backbone == 'nosa':
                problem += '\n\nFinal answer should be an integer.'
            else:
                instruction += ' Final answer should be an integer.'
        if backbone == 'nosa':
            return prefix + problem + '\n\n\nAnswer: <think>'
        return chat(tokenizer, instruction + '\n\n' + problem.strip(), thinking=True)
    def score(rows, saved):
        correct = []
        for row, prediction in zip(rows, saved):
            target = row.get('answer')
            if target is None or target == '':
                target = row.get('final_answer')
            if target is None or str(target).strip() == '':
                raise ValueError('reasoning requires an explicit final answer')
            correct.append(reasoning_answer(prediction['prediction'], task) == normalize_answer(target))
        return {'normalized_exact_match': 100 * sum(correct) / len(correct), 'correct': correct}
    return prepare, score, 65536 if backbone == 'minicpm' else 8192


def stop_on_nosa_sequence(prompt_length):
    from transformers import StoppingCriteria, StoppingCriteriaList
    class Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            generated = input_ids[0, prompt_length:].tolist()
            return any(generated[i:i + 3] == [1786, 30062, 59388] for i in range(len(generated) - 2))
    return StoppingCriteriaList([Stop()])


def load_helmet_wrapper(args, config):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, tokenizer = load(args)
    from model_utils import HFModel, LLM
    wrapper = HFModel.__new__(HFModel)
    LLM.__init__(wrapper, args.model, temperature=config.temperature, top_p=config.top_p,
                 max_length=config.input_max_length, generation_max_length=config.generation_max_length,
                 generation_min_length=config.generation_min_length, do_sample=config.do_sample,
                 stop_new_line=config.stop_new_line, use_chat_template=config.use_chat_template,
                 system_message=config.system_message)
    if args.backbone == 'minicpm':
        apply_chat = tokenizer.apply_chat_template
        def without_thinking(*a, **kw):
            kw.setdefault('enable_thinking', False)
            return apply_chat(*a, **kw)
        tokenizer.apply_chat_template = without_thinking
    tokenizer.padding_side = tokenizer.truncation_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper.model, wrapper.tokenizer, wrapper.device = model, tokenizer, model.device
    # Full generation uses the model's native sparse prefill; partial prefill
    # would change which query/page plans are computed.
    wrapper.disable_prefill = True
    eos = model.generation_config.eos_token_id
    wrapper.stop_token_ids = list(eos) if isinstance(eos, list) else [eos]
    if config.stop_new_line:
        wrapper.stop_token_ids = list(set(wrapper.stop_token_ids + [tokenizer.convert_tokens_to_ids(x) for x in ['\n', 'Ċ', 'ĊĊ', '<0x0A>']]))
    wrapper.stop_token_ids = [x for x in wrapper.stop_token_ids if x is not None]
    return wrapper


def helmet(args, upstream):
    """Reuse HELMET's exact input preparation, generation wrapper, and scorer."""
    root = Path(args.upstream).resolve()
    output = Path(args.output).resolve()
    task_config = Path(args.task_config).resolve()
    if args.resume:
        raise ValueError('HELMET uses atomic per-task outputs; choose a new directory after an interrupted task')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('HELMET output directory must be empty')
    sys.path.insert(0, str(root))
    os.chdir(root)
    official = import_file(root / 'eval.py', '_leanselect_helmet_eval')
    old_argv = sys.argv
    sys.argv = [str(root / 'eval.py'), '--config', str(task_config),
                '--model_name_or_path', str(Path(args.model).resolve()),
                '--output_dir', str(output), '--num_workers', '0', '--no_torch_compile']
    try:
        config = official.parse_arguments()
    finally:
        sys.argv = old_argv
    datasets = config.datasets.split(',')
    tests = config.test_files.split(',')
    demos = config.demo_files.split(',')
    lengths = str(config.input_max_length).split(',')
    gen_lengths = str(config.generation_max_length).split(',')
    lengths = lengths * len(datasets) if len(lengths) == 1 else lengths
    gen_lengths = gen_lengths * len(datasets) if len(gen_lengths) == 1 else gen_lengths
    if not len(datasets) == len(tests) == len(demos) == len(lengths) == len(gen_lengths):
        raise ValueError('HELMET task/config list lengths differ')
    if any('alce' in task for task in datasets):
        raise ValueError('this release evaluates HELMET Recall/RAG/ICL; citation tasks require the upstream citation judge')
    provenance = identity(args)
    reports = []
    for task, test, demo, length, gen_length in zip(datasets, tests, demos, lengths, gen_lengths):
        config.datasets, config.test_files, config.demo_files = task, test, demo
        config.input_max_length, config.generation_max_length = int(length), int(gen_length)
        config.seed = args.seed
        wrapper = load_helmet_wrapper(args, config)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        input_hashes = []
        calibration = None
        original_prepare = wrapper.prepare_inputs
        def capture_input(*a, **kw):
            nonlocal calibration
            inputs = original_prepare(*a, **kw)
            if not input_hashes:
                calibration = calibrate(args, wrapper.model, inputs.input_ids, int(gen_length))
            input_hashes.append(digest_ids(inputs.input_ids))
            return inputs
        wrapper.prepare_inputs = capture_input
        try:
            path = official.run_test(config, wrapper, task, test, demo)
        finally:
            wrapper.prepare_inputs = original_prepare
        expected = len(input_hashes)
        report = json.loads(Path(path).read_text())
        if len(report['data']) != expected or expected == 0:
            raise RuntimeError('HELMET omitted generated/scored examples')
        reports.append({'task': task, 'count': expected, 'result': Path(path).name,
                        'sha256': file_sha256(path), 'input_ids_sha256': input_hashes,
                        'calibration': calibration,
                        'test_file_sha256': file_sha256(test),
                        'demo_file_sha256': file_sha256(demo) if demo and Path(demo).is_file() else None,
                        'metrics': report['averaged_metrics']})
        # Bound methods also hold the model; release them before the next load.
        del original_prepare, capture_input, wrapper
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (output / 'summary.json').write_text(json.dumps({'benchmark': 'helmet', 'upstream': upstream,
        'task_config_sha256': file_sha256(task_config), **provenance, 'tasks': reports, 'split': args.split,
        'scope': 'exact tasks and counts listed; no omitted categories inferred'}, indent=2) + '\n')


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    model_arguments(p)
    p.add_argument('--benchmark', choices=UPSTREAM, required=True)
    p.add_argument('--upstream', required=True, help='official benchmark checkout at documented revision')
    p.add_argument('--task', help='one LongBench/RULER task or math-500/aime24/aime25')
    p.add_argument('--inputs', help='one official dataset/generated JSONL')
    p.add_argument('--task-config', help='official HELMET YAML; data paths resolve from upstream directory')
    p.add_argument('--max-length', type=int, default=32768)
    p.add_argument('--max-new-tokens', type=int, help='override recorded in run configuration')
    p.add_argument('--temperature', type=float, help='reasoning default .9 MiniCPM / 1 NOSA')
    p.add_argument('--top-p', type=float, help='reasoning default .95 MiniCPM / 1 NOSA')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', choices=['development', 'test'], default='test')
    p.add_argument('--expected-count', type=int, help='assert exact task file size')
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    # Paths survive HELMET's official working-directory conventions.
    for name in ('model', 'config', 'selector_checkpoint', 'sparda_checkpoint', 'inputs', 'task_config'):
        value = getattr(args, name)
        if value:
            setattr(args, name, str(Path(value).resolve()))
    upstream = upstream_identity(args.upstream, args.benchmark)
    if args.benchmark == 'helmet':
        if args.expected_count is not None:
            p.error('--expected-count applies to a single JSONL task; HELMET uses its official task configuration')
        if not args.task_config:
            p.error('--task-config is required for HELMET')
        helmet(args, upstream)
        return
    if not args.inputs or not args.task:
        p.error('--inputs and --task are required')
    rows = read_jsonl(args.inputs)
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise ValueError('input count differs from expected task size')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, tokenizer = load(args)
    model.eval()
    if args.benchmark == 'longbench':
        prepare, score, cap = longbench_adapter(args.upstream, args.task, args.max_length, tokenizer)
    elif args.benchmark == 'ruler':
        prepare, score, cap = ruler_adapter(args.upstream, args.task)
    else:
        prepare, score, cap = reasoning_adapter(args.upstream, args.task, args.backbone, tokenizer)
    cap = args.max_new_tokens or cap
    temperature = args.temperature if args.temperature is not None else (.9 if args.backbone == 'minicpm' else 1.)
    top_p = args.top_p if args.top_p is not None else (.95 if args.backbone == 'minicpm' else 1.)
    is_reasoning = args.benchmark == 'reasoning'
    sampled = is_reasoning and temperature > 0
    generation = {'max_new_tokens': cap, 'do_sample': sampled, 'num_beams': 1}
    if sampled:
        generation.update(temperature=temperature, top_p=top_p)
    first_inputs = tokenizer(prepare(rows[0]), return_tensors='pt', truncation=False).to(model.device)
    calibration = calibrate(args, model, first_inputs.input_ids, cap)
    contract = {'benchmark': args.benchmark, 'task': args.task, 'upstream': upstream, 'split': args.split,
                **identity(args), 'input_sha256': file_sha256(args.inputs), 'count': len(rows),
                'max_length': args.max_length, 'generation': generation, 'seed': args.seed,
                'calibration': calibration,
                'sampling_seed': 'seed + input index', 'scope': 'all examples in the supplied task file'}
    predictions, saved = initialize_output(args.output, contract, args.resume)
    if len(saved) > len(rows):
        raise ValueError('saved predictions exceed input size')
    for index, row in enumerate(rows):
        prompt = prepare(row)
        encoded = first_inputs if index == 0 else tokenizer(prompt, return_tensors='pt', truncation=False).to(model.device)
        input_hash = digest_ids(encoded.input_ids)
        if index < len(saved):
            if saved[index]['input_ids_sha256'] != input_hash:
                raise ValueError('resume tokenized prompt changed')
            continue
        if args.benchmark == 'ruler' and encoded.input_ids.shape[-1] > args.max_length:
            raise ValueError('prepared RULER prompt exceeds requested context; never silently truncate')
        torch.manual_seed(args.seed + index)
        random.seed(args.seed + index)
        options = dict(generation)
        if is_reasoning and args.backbone == 'nosa':
            options['stopping_criteria'] = stop_on_nosa_sequence(encoded.input_ids.shape[-1])
        result = model.generate(**encoded, **options)
        new_tokens = result[0, encoded.input_ids.shape[-1]:]
        record = {'index': index, 'input_ids_sha256': input_hash,
                  'prediction': tokenizer.decode(new_tokens, skip_special_tokens=True),
                  'input_tokens': encoded.input_ids.shape[-1], 'output_tokens': len(new_tokens)}
        append_record(predictions, record)
        saved.append(record)
    report = {**contract, 'metrics': score(rows, saved), 'predictions_sha256': file_sha256(predictions)}
    (Path(args.output) / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
