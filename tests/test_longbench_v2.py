import json
from types import SimpleNamespace

import pytest
import torch

from leanselect import longbench_v2 as lbv2


TEMPLATE = '$DOC$\n$Q$\n$C_A$|$C_B$|$C_C$|$C_D$'
EXTRACTOR = '''
def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \\(([A-D])\\)', response)
    if match:
        return match.group(1)
    match = re.search(r'The correct answer is ([A-D])', response)
    return match.group(1) if match else None
'''


def example(identifier='one', **kwargs):
    return dict(_id=identifier, context=' document ', question=' question ', choice_A=' first ',
                choice_B='second', choice_C='third', choice_D='fourth', answer='A',
                difficulty='easy', length='short', domain='domain', sub_domain='subdomain', **kwargs)


class Tokenizer:
    def encode(self, text):
        return [1] + [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(int(value)) for value in ids if int(value) > 1)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        return '[' + messages[0]['content'] + ']'

    def __call__(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False, 'return_tensors': 'pt'}
        ids = torch.tensor([[ord(char) for char in text]])
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


@pytest.fixture
def official(tmp_path, monkeypatch):
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts/0shot.txt').write_text(TEMPLATE)
    # Top-level upstream server configuration must never execute.
    (tmp_path / 'pred.py').write_text('raise RuntimeError("API side effect")\n' + EXTRACTOR)
    (tmp_path / 'result.py').write_text('# official scorer fixture\n')
    monkeypatch.setattr(lbv2, 'upstream_identity', lambda *args: {'commit': lbv2.UPSTREAM_COMMIT})
    monkeypatch.setattr(lbv2, 'OFFICIAL_FILES', {name: lbv2.file_sha256(tmp_path / name)
                                             for name in lbv2.OFFICIAL_FILES})
    return tmp_path


def test_official_extraction_is_exact_without_import_side_effects(official):
    template, extract, _ = lbv2.official_protocol(official)
    assert template == TEMPLATE
    assert extract('**The correct answer is (B)**.') == 'B'
    assert extract('The correct answer is C.') == 'C'
    for text in ('B', '(B)', 'the correct answer is (B)', 'The correct answer is (b)'):
        assert extract(text) is None
    (official / 'prompts/0shot.txt').write_text('changed')
    with pytest.raises(ValueError, match='source differs'):
        lbv2.official_protocol(official)


def test_prompt_matches_official_truncate_before_chat_and_no_duplicate_special_tokens():
    tokenizer = Tokenizer()
    row = example()
    raw = 'document\nquestion\nfirst|second|third|fourth'
    encoded, identity = lbv2.prepare_prompt(row, tokenizer, TEMPLATE, 100, 300)
    assert tokenizer.decode(encoded['input_ids'][0]) == '[' + raw + ']'
    assert identity['raw_prompt_tokens'] == len(raw) + 1
    assert not identity['truncated']
    encoded, identity = lbv2.prepare_prompt(row, tokenizer, TEMPLATE, 10, 300)
    expected = tokenizer.decode(tokenizer.encode(raw)[:5] + tokenizer.encode(raw)[-5:])
    assert tokenizer.decode(encoded['input_ids'][0]) == '[' + expected + ']'
    assert identity['truncated']
    assert identity['input_tokens'] == len(expected) + 2
    with pytest.raises(ValueError, match='context-cap'):
        lbv2.prepare_prompt(row, tokenizer, TEMPLATE, 100, 130)


def test_dataset_enforces_official_identity_and_unique_metadata(tmp_path):
    path = tmp_path / 'examples.json'
    path.write_text(json.dumps([example()]))
    with pytest.raises(ValueError, match='pinned official'):
        lbv2.read_examples(path)
    assert lbv2.read_examples(path, subset=True)[0]['_id'] == 'one'
    path.write_text(json.dumps([example(), example()]))
    with pytest.raises(ValueError, match='duplicate'):
        lbv2.read_examples(path, subset=True)
    row = example()
    row['length'] = 'unknown'
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match='metadata'):
        lbv2.read_examples(path, subset=True)
    path.write_text('[]')
    with pytest.raises(ValueError, match='nonempty'):
        lbv2.read_examples(path, subset=True)


def test_uncompensated_official_aggregate_preserves_group_denominators():
    rows = [dict(difficulty='easy', length='short', judge=True),
            dict(difficulty='hard', length='medium', judge=False),
            dict(difficulty='hard', length='long', judge=False)]
    score = lbv2.summarize(rows)
    assert score['Overall'] == {'count': 3, 'accuracy': 33.3}
    assert score['Easy']['accuracy'] == 100
    assert score['Hard']['accuracy'] == 0
    assert score['Medium']['count'] == 1
    assert lbv2.summarize(rows[:1])['Long'] == {'count': 0, 'accuracy': None}


def test_interrupted_resume_recalibrates_original_input_and_rejects_tampering(official, tmp_path, monkeypatch):
    import sys
    inputs = tmp_path / 'examples.json'
    inputs.write_text(json.dumps([example('one'), example('two')]))
    output = tmp_path / 'output'
    calls, calibrations = [], []
    should_fail = [True]
    class Model:
        device = torch.device('cpu')
        leanselect_backbone = 'minicpm'
        generation_config = SimpleNamespace(eos_token_id=0, bos_token_id=1, pad_token_id=0)

        def eval(self):
            return self

        def forward(self, input_ids, logits_to_keep=0):
            pass

        def generate(self, input_ids, attention_mask, generation_config, logits_to_keep):
            assert self.calibrated == calibrations[0]
            assert logits_to_keep == 1
            assert generation_config.temperature == .1 and generation_config.do_sample
            assert generation_config.top_k == 0 and generation_config.top_p == 1
            assert generation_config.max_new_tokens == 128 and generation_config.eos_token_id == 0
            calls.append(torch.initial_seed())
            if len(calls) == 2 and should_fail[0]:
                raise RuntimeError('interrupted')
            response = torch.tensor([[ord(char) for char in 'The correct answer is (A)']])
            return torch.cat([input_ids, response], dim=-1)
    def calibrate(args, model, input_ids, max_new_tokens):
        model.calibrated = input_ids.tolist()
        calibrations.append(input_ids.tolist())
        return {'input_ids_sha256': lbv2.digest_ids(input_ids)}
    monkeypatch.setattr(lbv2, 'load', lambda args: (Model(), Tokenizer()))
    monkeypatch.setattr(lbv2, 'calibrate', calibrate)
    monkeypatch.setattr(lbv2, 'identity', lambda args: {'method': args.method})
    argv = ['longbench_v2', '--backbone', 'minicpm', '--model', str(tmp_path), '--method', 'infinigen',
            '--upstream', str(official), '--inputs', str(inputs), '--subset', '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(RuntimeError, match='interrupted'):
        lbv2.main()
    assert not (output / 'summary.json').exists()
    should_fail[0] = False
    monkeypatch.setattr(sys, 'argv', argv + ['--resume'])
    lbv2.main()
    assert calls == [42, 43, 43]
    assert len(calibrations) == 2 and calibrations[0] == calibrations[1]
    report = json.loads((output / 'summary.json').read_text())
    assert report['metrics']['Overall'] == {'count': 2, 'accuracy': 100.0}
    assert report['scope'] == 'subset/custom inputs'
    predictions = output / 'predictions.jsonl'
    records = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert all('context' not in row and 'question' not in row for row in records)
    records[0]['pred'] = 'D'
    predictions.write_text(''.join(json.dumps(row) + '\n' for row in records))
    with pytest.raises(ValueError, match='official answer extraction'):
        lbv2.main()


@pytest.mark.parametrize('field,value', [('_id', 'different'), ('answer', 'B'),
                                        ('difficulty', 'hard'), ('judge', False),
                                        ('generated_tokens', 129)])
def test_resume_rejects_forged_metadata_and_scores(official, field, value):
    _, extract, _ = lbv2.official_protocol(official)
    row = example()
    saved = {key: row[key] for key in ('_id', 'answer', 'difficulty', 'length', 'domain', 'sub_domain')}
    saved.update(response='The correct answer is (A)', pred='A', judge=True, generated_tokens=10)
    saved[field] = value
    with pytest.raises(ValueError):
        lbv2.validate_saved([saved], [row], extract)
