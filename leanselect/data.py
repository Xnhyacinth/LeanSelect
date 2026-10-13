"""Small, deterministic data readers shared by the command-line tools."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError('input JSONL is empty')
    return rows


class TokenDataset:
    """Pretokenized JSONL or a Hugging Face Dataset saved with save_to_disk.

    Each row must contain at least sequence_length tokens. Prefix cropping is
    explicit; padding, repacking, and train/test splitting belong to preparation.
    """
    def __init__(self, path, sequence_length):
        self.path = Path(path)
        self.sequence_length = sequence_length
        if sequence_length < 2:
            raise ValueError('sequence length must be at least two')
        if self.path.is_dir():
            from datasets import load_from_disk
            self.rows = load_from_disk(str(self.path))
            if not hasattr(self.rows, 'column_names') or 'input_ids' not in self.rows.column_names:
                raise ValueError('expected one Dataset with an input_ids column')
            files = sorted(p for p in self.path.rglob('*') if p.is_file())
            entries = [(p.relative_to(self.path).as_posix(), file_sha256(p)) for p in files]
            self.identity = hashlib.sha256(json.dumps(entries).encode()).hexdigest()
        else:
            self.rows = read_jsonl(self.path)
            self.identity = file_sha256(self.path)
        if not len(self.rows):
            raise ValueError('training data is empty')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ids = row['input_ids']
        if len(ids) < self.sequence_length:
            raise ValueError(f'row {index} has fewer than {self.sequence_length} tokens')
        if any(type(x) is not int or x < 0 for x in ids[:self.sequence_length]):
            raise ValueError(f'row {index} has invalid token IDs')
        mask = row.get('attention_mask')
        if mask is not None and any(x != 1 for x in mask[:self.sequence_length]):
            raise ValueError('training expects unpadded sequences')
        return torch.tensor(ids[:self.sequence_length], dtype=torch.long)


def batch_indices(size, microbatch, batch_size, rank, world_size, seed):
    """Rank-disjoint shuffled batches; resume is an exact microbatch cursor."""
    global_batch = batch_size * world_size
    batches_per_epoch = size // global_batch
    if batches_per_epoch == 0:
        raise ValueError('data is smaller than one global microbatch')
    epoch, within = divmod(microbatch, batches_per_epoch)
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(size, generator=generator)
    start = within * global_batch + rank * batch_size
    return order[start:start + batch_size].tolist()


def source_sha256():
    """Hash this installed package's Python implementation for result identity."""
    root = Path(__file__).parent
    entries = [(path.relative_to(root).as_posix(), file_sha256(path))
               for path in sorted(root.rglob('*.py'))]
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()
