# Reproduction

Install the environment and native extension as described in the [README](../README.md). All commands below run from the repository root. Replace the example model and data locations with your local inputs.

## Inputs

Use separate local backbone directories containing the pretrained model, tokenizer, and configuration files. Training accepts an unpadded `input_ids` column in either JSONL or a Hugging Face `Dataset.save_to_disk` directory. Each row must contain at least the requested sequence length; the trainer takes that prefix without padding or repacking.

Keep development and test examples disjoint. Tokenize training data with the corresponding backbone tokenizer. Benchmark datasets, generated RULER inputs, and model weights are obtained separately under their original terms.

## Train

The four-GPU recipe uses eight sequences per GPU and two accumulation microbatches: a global batch of 64. MiniCPM trains at 65,536 tokens; NOSA trains at 32,768 tokens. A 300-update budget processes 1,258,291,200 and 629,145,600 tokens, respectively.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync torchrun \
  --standalone --nproc_per_node=4 --module leanselect.train \
  --backbone minicpm --model models/MiniCPM4.1-8B \
  --data datasets/train-minicpm --config configs/minicpm.json \
  --sequence-length 65536 --batch-size 8 --gradient-accumulation 2 \
  --steps 300 --save-every 20 --output outputs/train-minicpm

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync torchrun \
  --standalone --nproc_per_node=4 --module leanselect.train \
  --backbone nosa --model models/NOSA-8B \
  --data datasets/train-nosa --config configs/nosa.json \
  --sequence-length 32768 --batch-size 8 --gradient-accumulation 2 \
  --steps 300 --save-every 20 --output outputs/train-nosa
```

The trainer freezes the BF16 backbone and updates FP32 selector parameters with AdamW: learning rate `5e-4`, betas `(0.9, 0.95)`, epsilon `1e-6`, weight decay `0.01`, and gradient clipping `0.5`. Both configurations use four width-32 factors, a 96-page budget, and a sampled fine-grained teacher.

The output contains `training.pt` for resuming, `selector.pt` for the latest deployment weights, and periodic `selector-*.pt` candidates. Resume by repeating the same command with `--resume outputs/train-minicpm/training.pt` (or the NOSA path). Model, data, implementation, optimizer settings, and rank count must match the saved run.

## Benchmark sources

Clone the official benchmark repositories into an external working directory and check out these revisions. The evaluation CLI verifies the revision and tracked source files.

| Benchmark | Repository | Revision |
|---|---|---|
| LongBench | [THUDM/LongBench](https://github.com/THUDM/LongBench) | `2e00731f8d0bff23dc4325161044d0ed8af94c1e` |
| RULER | [NVIDIA/RULER](https://github.com/NVIDIA/RULER) | `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a` |
| HELMET | [princeton-nlp/HELMET](https://github.com/princeton-nlp/HELMET) | `aeadacc6593e9a222a84b0660b186a4e23d2efda` |
| Reasoning prompts | [thunlp/NOSA](https://github.com/thunlp/NOSA) | `90045b65e6620848f136ae0cf1cc14ab4f40ec68` |

For example:

```bash
git clone https://github.com/THUDM/LongBench benchmarks/LongBench
git -C benchmarks/LongBench checkout 2e00731f8d0bff23dc4325161044d0ed8af94c1e
```

Follow each upstream repository's data preparation instructions. RULER JSONL must already contain the intended model template and answer prefix; the evaluator uses that prepared input directly. LongBench uses its official task template and scoring function. For the paper's test comparison, apply the same development-example exclusions to every method.

## Select the checkpoint

Evaluate each saved candidate on the same development tasks using `--split development`. For example:

```bash
uv run --no-sync python -m leanselect.evaluate \
  --backbone minicpm --model models/MiniCPM4.1-8B \
  --method leanselect --selector-checkpoint outputs/train-minicpm/selector-000020.pt \
  --benchmark longbench --upstream benchmarks/LongBench --task qasper \
  --inputs datasets/development/qasper.jsonl --split development \
  --output outputs/dev-000020/qasper
```

List every candidate and its development summaries in a JSON manifest. Paths are relative to the manifest file:

```json
[
  {
    "checkpoint": "train-minicpm/selector-000020.pt",
    "results": ["dev-000020/qasper/summary.json", "dev-000020/niah/summary.json"]
  },
  {
    "checkpoint": "train-minicpm/selector-000040.pt",
    "results": ["dev-000040/qasper/summary.json", "dev-000040/niah/summary.json"]
  }
]
```

Include the entire development roster in each `results` list. For MiniCPM, selection averages the RULER and LongBench benchmark means equally; NOSA uses the RULER development mean. The tool checks matching tasks, inputs, model identity, and generation settings, and accepts only development summaries.

```bash
uv run --no-sync python -m leanselect.select_checkpoint \
  --manifest outputs/development.json --output outputs/selector-best.pt
```

Use the selected checkpoint for all subsequent quality and timing evaluations.

## Quality

The following variables select one backbone and one trained selector:

```bash
MODEL=models/MiniCPM4.1-8B
BACKBONE=minicpm
SELECTOR=outputs/selector-best.pt
```

Run individual LongBench and RULER tasks:

```bash
uv run --no-sync python -m leanselect.evaluate \
  --backbone "$BACKBONE" --model "$MODEL" \
  --method leanselect --selector-checkpoint "$SELECTOR" \
  --benchmark longbench --upstream benchmarks/LongBench --task qasper \
  --inputs datasets/test/qasper.jsonl --max-length 32768 \
  --output outputs/longbench/qasper

uv run --no-sync python -m leanselect.evaluate \
  --backbone "$BACKBONE" --model "$MODEL" \
  --method leanselect --selector-checkpoint "$SELECTOR" \
  --benchmark ruler --upstream benchmarks/RULER --task niah_single_1 \
  --inputs datasets/ruler/32768/niah_single_1/validation.jsonl --max-length 32768 \
  --output outputs/ruler/32768/niah_single_1
```

Repeat over the benchmark task roster and context lengths. Each task saves predictions, exact input counts, hashes, and official scores. `--resume` continues a task with the same inputs and configuration. InfiniGen resumes from its original calibration input.

For mathematical reasoning:

```bash
uv run --no-sync python -m leanselect.evaluate \
  --backbone "$BACKBONE" --model "$MODEL" \
  --method leanselect --selector-checkpoint "$SELECTOR" \
  --benchmark reasoning --upstream benchmarks/NOSA --task math-500 \
  --inputs datasets/reasoning/math-500.jsonl --expected-count 500 \
  --output outputs/reasoning/math-500
```

Run `aime24` and `aime25` with their corresponding files and `--expected-count 30`. MiniCPM uses native thinking, temperature `0.9`, top-p `0.95`, and a 65,536-token generation cap. NOSA uses its official two-shot prefix and stop sequence, temperature `1`, top-p `1`, and an 8,192-token cap. Both use one generation per problem and the paper's normalized final-answer exact match.

HELMET additionally requires its metric dependencies:

```bash
uv pip install pytrec-eval==0.5 openai
uv run --no-sync python -m leanselect.evaluate \
  --backbone "$BACKBONE" --model "$MODEL" \
  --method leanselect --selector-checkpoint "$SELECTOR" \
  --benchmark helmet --upstream benchmarks/HELMET \
  --task-config datasets/helmet/recall-32k.yaml \
  --output outputs/helmet/recall
```

Prepare task YAML files from HELMET's configurations with the intended 32K Recall, RAG, and ICL input files, generation limits, and demonstration settings. Data paths in those YAML files resolve from the HELMET checkout. Its official loader, prompt construction, and scorer are reused. Each task loads a fresh model; completed outputs include actual task counts and input hashes.

For baselines, replace `--method leanselect --selector-checkpoint "$SELECTOR"` with the method listed in the README. SparDA also requires `--sparda-checkpoint`. Export its legacy training checkpoint once:

```bash
uv run --no-sync python -m leanselect.baselines.sparda \
  --source checkpoints/sparda-training.pt --output checkpoints/sparda.pt
```

If a trusted legacy file contains pickled training objects, explicitly add `--trust-pickle` during this conversion. Normal evaluation loads the exported tensor-only projections.

## Efficiency

Provide JSONL rows containing real prompt `input_ids`, tokenized with the selected backbone. Each row must contain at least the requested prefix length; a batch uses distinct rows from that file.

```bash
uv run --no-sync python -m leanselect.benchmark \
  --backbone "$BACKBONE" --model "$MODEL" \
  --method leanselect --selector-checkpoint "$SELECTOR" \
  --inputs datasets/timing-inputs.jsonl --length 131072 --batch-size 1 \
  --output-tokens 4 --warmups 1 --repeats 10 \
  --output outputs/timing/leanselect-128k-b1.json
```

Run methods separately on the same idle GPU with identical token inputs, batch size, and output length. The benchmark synchronizes CUDA around actual generation forwards, applies a last-token-only LM head, and records prefill, decode, complete-request wall time, throughput, peak allocation, and all measured samples. Tokenization, loading, InfiniGen calibration, and warmups are excluded from reported times. The calibration recipe and input hash are saved with the result.

Use 32K, 64K, 96K, and 128K for MiniCPM and the supported NOSA lengths from your experiment protocol. Increase `--output-tokens` to measure longer decoding. Keep per-forward phase times and complete-request wall time distinct when comparing results.
