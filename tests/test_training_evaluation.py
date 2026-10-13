import json
from types import SimpleNamespace

import pytest
import torch

from leanselect.benchmark import distribution, measure_request
from leanselect.data import TokenDataset, batch_indices
from leanselect.evaluate import (extract_boxed, initialize_output, normalize_answer,
                                 reasoning_adapter, reasoning_answer)
from leanselect.selector import LeanSelectSelector
from leanselect.train import (freeze_backbone, layerwise_backward, load_checkpoint,
                              save_checkpoint)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.leanselect = LeanSelectSelector(num_heads=4, num_kv_heads=2, head_dim=4, router_heads=2, router_dim=2)
        self.self_attn.leanselect_total_topk_blocks = 4
        self.self_attn.init_blocks = 1
        self.self_attn.local_blocks = 1
        self.backbone = torch.nn.Linear(4, 4)

    def forward(self, hidden, **kwargs):
        loss = sum((parameter - torch.rand_like(parameter)).square().mean()
                   for parameter in self.self_attn.leanselect.parameters())
        return self.backbone(hidden), None, (loss, 'precomputed')


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(10, 4)
        self.model.layers = torch.nn.ModuleList([Layer(), Layer()])
        self.config = SimpleNamespace(scale_emb=1.)


def train_update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    layerwise_backward(model, torch.tensor([[1, 2, 3]]), 'minicpm', 1)
    optimizer.step()


def test_resume_replays_rng_optimizer_and_frozen_weights(tmp_path):
    torch.manual_seed(13)
    model = Model()
    parameters = freeze_backbone(model)
    optimizer = torch.optim.AdamW(parameters, lr=.01)
    frozen = {k: value.clone() for k, value in model.state_dict().items() if 'leanselect' not in k}
    train_update(model, optimizer)
    path = tmp_path / 'training.pt'
    save_checkpoint(model, optimizer, path, step=1, microbatch=1, contract={'data': 'same'}, rank=0, world=1)
    train_update(model, optimizer)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    step, cursor = load_checkpoint(path, model, optimizer, {'data': 'same'}, 0)
    assert (step, cursor) == (1, 1)
    train_update(model, optimizer)
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])
    for key, value in frozen.items():
        assert torch.equal(value, model.state_dict()[key])
    saved = torch.load(path, weights_only=True)
    assert 'model_state_dict' not in saved
    with pytest.raises(ValueError, match='configuration differs'):
        load_checkpoint(path, model, optimizer, {'data': 'changed'}, 0)
    deploy = torch.load(tmp_path / 'selector.pt', weights_only=True)
    assert set(deploy['leanselect']['layers']) == {'0', '1'}


def test_distributed_data_cursor_is_disjoint_and_repeatable(tmp_path):
    ranks = [batch_indices(20, 2, 2, rank, 4, 7) for rank in range(4)]
    assert len(set(sum(ranks, []))) == 8
    assert batch_indices(20, 2, 2, 2, 4, 7) == ranks[2]
    assert batch_indices(20, 0, 2, 2, 4, 7) != ranks[2]
    path = tmp_path / 'tokens.jsonl'
    path.write_text(json.dumps({'input_ids': [1, 2, 3]}) + '\n')
    data = TokenDataset(path, 2)
    assert data[0].tolist() == [1, 2]
    with pytest.raises(ValueError, match='fewer'):
        TokenDataset(path, 4)[0]
    with pytest.raises(ValueError, match='global microbatch'):
        batch_indices(3, 0, 2, 0, 2, 7)


def test_math_answers_preserve_nested_boxes_and_wrapped_gold():
    assert extract_boxed(r'work \boxed{\frac{1}{2}}') == r'\frac{1}{2}'
    assert normalize_answer(r'\boxed{204}') == '204'
    assert reasoning_answer(r'<think>guess 400</think>Final answer: \boxed{204}', 'aime24') == '204'
    assert reasoning_answer('Final answer: 007', 'aime25') == '7'
    assert reasoning_answer(r'\boxed{\frac{1}{2}}', 'math-500') == r'\frac{1}{2}'


def test_reasoning_prompts_and_score(tmp_path):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['enable_thinking']
            return messages[0]['content']
    prepare, score, cap = reasoning_adapter(tmp_path, 'aime24', 'minicpm', Tokenizer())
    assert 'integer' in prepare({'question': 'test'})
    assert cap == 65536
    assert score([{'answer': 0}], [{'prediction': r'\boxed{0}'}])['normalized_exact_match'] == 100
    assert score([{'answer': r'\boxed{204}'}], [{'prediction': r'\boxed{204}'}])['normalized_exact_match'] == 100
    folder = tmp_path / 'benchmarks/Reasoning'
    folder.mkdir(parents=True)
    (folder / 'test.py').write_text('_prefix = "literal demonstrations\\nQuestion: "\nraise RuntimeError("must not execute")\n')
    prepare, _, cap = reasoning_adapter(tmp_path, 'math-500', 'nosa', Tokenizer())
    assert prepare({'problem': 'test'}).startswith('literal demonstrations\nQuestion: ')
    assert cap == 8192


def test_resume_rejects_changed_eval_identity(tmp_path):
    contract = {'input_sha256': 'a', 'method': 'leanselect'}
    path, saved = initialize_output(tmp_path, contract, False)
    assert saved == []
    path.write_text(json.dumps({'index': 0}) + '\n')
    assert len(initialize_output(tmp_path, contract, True)[1]) == 1
    with pytest.raises(ValueError, match='configuration'):
        initialize_output(tmp_path, {**contract, 'method': 'native'}, True)


class FakeGeneration(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        return input_ids + 1

    def generate(self, input_ids, max_new_tokens, **kwargs):
        output = input_ids
        for i in range(max_new_tokens):
            ids = input_ids if i == 0 else output[:, -1:]
            new = self.forward(input_ids=ids)[:, -1:]
            output = torch.cat((output, new), dim=-1)
        return output


def test_timing_uses_real_generate_calls_and_restores_forward():
    model = FakeGeneration()
    original = model.forward
    report = measure_request(model, torch.tensor([[1, 2]]), 4, lambda: None)
    assert report['generated_ids'] == [[3, 4, 5, 6]]
    assert [row['tokens'] for row in report['forward_calls']] == [2, 1, 1, 1]
    assert report['wall_time_s'] >= report['prefill_forward_s'] + report['decode_forward_s']
    assert model.forward == original
    assert distribution([1, 2, 3, 4]) == {'median': 2.5, 'q1': 1.75, 'q3': 3.25}


def test_checkpoint_selection_uses_identical_development_only(tmp_path):
    from leanselect.data import file_sha256
    from leanselect.select_checkpoint import select_best
    candidates = []
    common = dict(split='development', method='leanselect', benchmark='ruler', task='niah_single_1',
                  max_length=32768, input_sha256='same', count=10, backbone='minicpm',
                  base_model_sha256='same', model_config_sha256='same', source_sha256='same',
                  generation={'do_sample': False}, seed=42, upstream={'commit': 'same'})
    for i, score in enumerate([40, 60]):
        ckpt = tmp_path / f'candidate{i}.pt'
        ckpt.write_bytes(f'weights{i}'.encode())
        summary = tmp_path / f'dev{i}.json'
        summary.write_text(json.dumps(dict(common, selector_sha256=file_sha256(ckpt), metrics={'score': score})))
        candidates.append({'checkpoint': ckpt.name, 'results': [summary.name]})
    manifest = tmp_path / 'development.json'
    manifest.write_text(json.dumps(candidates))
    receipt = select_best(manifest, tmp_path / 'best.pt')
    assert receipt['selected_development_macro'] == 60
    assert (tmp_path / 'best.pt').read_bytes() == b'weights1'
    bad = json.loads((tmp_path / 'dev1.json').read_text())
    bad['split'] = 'test'
    (tmp_path / 'dev1.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='development summaries'):
        select_best(manifest, tmp_path / 'another.pt')
    bad['split'] = 'development'
    bad['input_sha256'] = 'changed'
    (tmp_path / 'dev1.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='same development protocol'):
        select_best(manifest, tmp_path / 'another.pt')


def test_selection_weights_benchmarks_not_unequal_task_counts(tmp_path):
    from leanselect.data import file_sha256
    from leanselect.select_checkpoint import select_best
    candidates = []
    for index, values in enumerate(([100, 100, 0], [60, 60, 50])):
        checkpoint = tmp_path / f'weights{index}.pt'
        checkpoint.write_bytes(str(index).encode())
        summaries = []
        for task_index, score in enumerate(values):
            report = dict(split='development', method='leanselect',
                          benchmark='ruler' if task_index < 2 else 'longbench', task=str(task_index),
                          max_length=32768, input_sha256=f'task{task_index}', count=10,
                          backbone='minicpm', base_model_sha256='same', model_config_sha256='same',
                          source_sha256='same', generation={}, seed=42, upstream={},
                          selector_sha256=file_sha256(checkpoint), metrics={'score': score})
            filename = tmp_path / f'dev{index}_{task_index}.json'
            filename.write_text(json.dumps(report))
            summaries.append(filename.name)
        candidates.append({'checkpoint': checkpoint.name, 'results': summaries})
    manifest = tmp_path / 'development.json'
    manifest.write_text(json.dumps(candidates))
    receipt = select_best(manifest, tmp_path / 'best.pt')
    # Pooled tasks prefer candidate0 (66.7 > 56.7); benchmark means prefer1.
    assert (tmp_path / 'best.pt').read_bytes() == b'1'
    assert receipt['selected_development_macro'] == 55


class Encoding(dict):
    @property
    def input_ids(self):
        return self['input_ids']

    def to(self, device):
        return Encoding({key: value.to(device) for key, value in self.items()})


class TaskTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        ids = torch.tensor([[int(text), int(text) + 1]])
        return Encoding(input_ids=ids, attention_mask=torch.ones_like(ids))

    def decode(self, ids, **kwargs):
        return str(ids[-1].item())

    def apply_chat_template(self, *args, **kwargs):
        return '1'


def test_resumed_infinigen_reuses_original_task_calibration(tmp_path, monkeypatch):
    import sys
    import leanselect.evaluate as evaluate
    rows = tmp_path / 'input.jsonl'
    rows.write_text(''.join(json.dumps({'input': str(value)}) + '\n' for value in (10, 20)))
    out = tmp_path / 'results'
    seen = []
    generated = []
    class Model:
        device = torch.device('cpu')
        def eval(self):
            return self
        def generate(self, input_ids, **kwargs):
            assert self.calibrated == [[10, 11]]
            generated.append(input_ids.tolist())
            return torch.cat([input_ids, torch.tensor([[1, 2]])], dim=-1)
    def calibrate(args, model, ids, cap):
        model.calibrated = ids.tolist()
        seen.append((model, ids.tolist()))
        return {'input_sha256': evaluate.digest_ids(ids)}
    monkeypatch.setattr(evaluate, 'load', lambda args: (Model(), TaskTokenizer()))
    monkeypatch.setattr(evaluate, 'identity', lambda args: {'method': args.method})
    monkeypatch.setattr(evaluate, 'upstream_identity', lambda *a: {})
    monkeypatch.setattr(evaluate, 'ruler_adapter', lambda *a: (lambda row: row['input'], lambda *a: {'score': 100}, 2))
    monkeypatch.setattr(evaluate, 'calibrate', calibrate)
    argv = ['evaluate', '--backbone', 'minicpm', '--model', str(tmp_path), '--method', 'infinigen',
            '--benchmark', 'ruler', '--upstream', str(tmp_path), '--task', 'niah_single_1',
            '--inputs', str(rows), '--output', str(out)]
    monkeypatch.setattr(sys, 'argv', argv)
    evaluate.main()
    predictions = out / 'predictions.jsonl'
    predictions.write_text(predictions.read_text().splitlines()[0] + '\n')
    monkeypatch.setattr(sys, 'argv', argv + ['--resume'])
    evaluate.main()
    assert len(seen) == 2 and seen[0][0] is not seen[1][0]
    assert [ids for _, ids in seen] == [[[10, 11]], [[10, 11]]]
    assert generated == [[[10, 11]], [[20, 21]], [[20, 21]]]


def test_helmet_loads_fresh_model_and_calibration_for_each_task(tmp_path, monkeypatch):
    import sys
    import types
    import leanselect.evaluate as evaluate
    models = []
    calibrations = []
    class Model:
        device = torch.device('cpu')
        generation_config = SimpleNamespace(eos_token_id=0)
    class HFModel:
        def prepare_inputs(self, sample, data):
            return self.tokenizer(sample['input'])
    class LLM:
        def __init__(self, model_name, **kwargs):
            self.__dict__.update(kwargs)
    monkeypatch.setitem(sys.modules, 'model_utils', SimpleNamespace(HFModel=HFModel, LLM=LLM))
    def load(args):
        model = Model()
        models.append(model)
        return model, TaskTokenizer()
    def calibrate(args, model, ids, cap):
        calibrations.append((model, ids.tolist(), cap))
        return {'input_sha256': evaluate.digest_ids(ids)}
    paths = []
    for name in ['test1.jsonl', 'test2.jsonl', 'task.yaml']:
        path = tmp_path / name
        path.write_text('{}\n')
        paths.append(path)
    config = SimpleNamespace(datasets='task1,task2', test_files=','.join(str(p) for p in paths[:2]),
        demo_files=',', input_max_length='100,200', generation_max_length='3,4',
        temperature=0., top_p=1., generation_min_length=0, do_sample=False,
        stop_new_line=False, use_chat_template=False, system_message=None)
    out = tmp_path / 'output'
    def run_test(config, wrapper, task, test, demo):
        value = '10' if task == 'task1' else '20'
        wrapper.prepare_inputs({'input': value}, {})
        wrapper.prepare_inputs({'input': str(int(value) + 1)}, {})
        assert calibrations[-1][0] is wrapper.model
        path = out / (task + '.json')
        path.write_text(json.dumps({'data': [{}, {}], 'averaged_metrics': {'exact_match': 100}}))
        return path
    official = SimpleNamespace(parse_arguments=lambda: config, run_test=run_test)
    monkeypatch.setattr(evaluate, 'import_file', lambda *a: official)
    monkeypatch.setattr(evaluate, 'load', load)
    monkeypatch.setattr(evaluate, 'calibrate', calibrate)
    monkeypatch.setattr(evaluate, 'identity', lambda args: {'method': args.method})
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(upstream=tmp_path, output=out, task_config=paths[2], resume=False,
                           seed=42, model=str(tmp_path), backbone='minicpm', method='infinigen', split='test')
    evaluate.helmet(args, {})
    assert len(models) == 2 and models[0] is not models[1]
    assert [(ids, cap) for _, ids, cap in calibrations] == [([[10, 11]], 3), ([[20, 21]], 4)]
    report = json.loads((out / 'summary.json').read_text())
    assert all(row['calibration']['input_sha256'] for row in report['tasks'])


@pytest.mark.parametrize('backbone,argument', [('minicpm', 'logits_to_keep'), ('nosa', 'num_logits_to_keep')])
def test_efficiency_applies_backbone_last_token_head(backbone, argument):
    class Model(FakeGeneration):
        leanselect_backbone = backbone
        def forward(self, input_ids, logits_to_keep=0, num_logits_to_keep=0):
            keep = logits_to_keep if backbone == 'minicpm' else num_logits_to_keep
            assert keep == 1
            return SimpleNamespace(logits=torch.ones(1, keep, 4))
        def generate(self, input_ids, max_new_tokens, **kwargs):
            assert kwargs[argument] == 1
            output = input_ids
            for index in range(max_new_tokens):
                self.forward(input_ids=input_ids if index == 0 else output[:, -1:], **{argument: kwargs[argument]})
                output = torch.cat((output, torch.ones(1, 1, dtype=torch.long)), dim=-1)
            return output
    report = measure_request(Model(), torch.tensor([[1, 2]]), 4, lambda: None)
    assert report['lm_head_policy'] == {argument: 1}
    assert all(call['logit_positions'] == 1 for call in report['forward_calls'])


def test_helmet_rejects_ignored_single_file_count_flag(tmp_path, monkeypatch):
    import sys
    import leanselect.evaluate as evaluate
    monkeypatch.setattr(evaluate, 'upstream_identity', lambda *args: {})
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--backbone', 'minicpm', '--model', str(tmp_path),
                                    '--benchmark', 'helmet', '--upstream', str(tmp_path),
                                    '--output', str(tmp_path), '--expected-count', '100'])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 2
