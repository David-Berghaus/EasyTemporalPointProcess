from __future__ import annotations

import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from easy_tpp.preprocess.event_tokenizer import BatchEncoding

SequenceDict = Dict[str, Sequence[float]]
SplitLiteral = str

_HF_SPLIT_MAP = {
    'train': 'train',
    'dev': 'validation',
    'valid': 'validation',
    'validation': 'validation',
    'test': 'test'
}


def _hf_split_name(split: str) -> str:
    mapped = _HF_SPLIT_MAP.get(split.lower())
    if mapped is None:
        raise ValueError(f'Unsupported split "{split}" for FIM episodic loader.')
    return mapped


def load_fim_sequences(source: str, split: str, limit: Optional[int] = None) -> List[SequenceDict]:
    """Load a HuggingFace EasyTPP split and normalise it into python lists."""
    split_mapped = _hf_split_name(split)
    if source.endswith('.json'):
        data = load_dataset('json', data_files={split_mapped: source}, split=split_mapped)
    elif source.startswith('easytpp'):
        data = load_dataset(source, split=split_mapped)
    else:
        # Allow passing a dataset repository without prefix
        data = load_dataset(f'easytpp/{source}', split=split_mapped)

    if limit is not None:
        data = data.select(range(min(limit, len(data))))

    raw = data[:]
    seq_len = raw.get('seq_len') or [len(seq) for seq in raw['time_since_start']]
    sequences: List[SequenceDict] = []
    for idx in range(len(raw['time_since_start'])):
        sequences.append(
            {
                'time_since_start': list(raw['time_since_start'][idx]),
                'time_since_last_event': list(raw['time_since_last_event'][idx]),
                'type_event': list(raw['type_event'][idx]),
                'seq_len': int(seq_len[idx])
            }
        )
    return sequences


@dataclass
class _EpisodeTensors:
    times: torch.Tensor
    types: torch.Tensor
    lengths: torch.Tensor


class FIMEpisodeDataset(Dataset):
    """Dataset that forms (context, inference) episodes required by FIM."""

    def __init__(
        self,
        *,
        context_sequences: List[SequenceDict],
        inference_sequences: List[SequenceDict],
        context_size: Optional[int],
        inference_size: Optional[int],
        max_num_events: Optional[int],
        split: str,
        sampling_strategy: str = 'sequential',
        episodes_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.context_sequences = context_sequences
        self.inference_sequences = inference_sequences
        self.context_size = min(len(context_sequences), context_size or len(context_sequences))
        self.inference_size = min(len(inference_sequences), inference_size or len(inference_sequences))
        self.max_num_events = max_num_events or self._infer_max_length()
        self.split = split.lower()
        self.mode = 'train' if self.split == 'train' else 'eval'
        self.sampling_strategy = sampling_strategy
        self.seed = seed
        self.num_marks = self._detect_num_marks()
        self._dt_stats = self._compute_dt_stats()
        self._fixed_context_indices = list(range(self.context_size))
        default_episodes = self._default_episode_count()
        self.episodes_per_epoch = episodes_per_epoch or default_episodes

    def _infer_max_length(self) -> int:
        lengths = [seq['seq_len'] for seq in (self.context_sequences + self.inference_sequences)]
        return max(lengths) if lengths else 1

    def _detect_num_marks(self) -> int:
        marks = set()
        for seq in self.context_sequences + self.inference_sequences:
            marks.update(int(mark) for mark in seq['type_event'])
        return max(len(marks), 1)

    def _compute_dt_stats(self) -> Tuple[float, float, float, float]:
        deltas = []
        for seq in self.inference_sequences:
            deltas.extend(seq['time_since_last_event'][1:self.max_num_events])
        if not deltas:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.array(deltas, dtype=np.float32)
        return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())

    def _default_episode_count(self) -> int:
        if not self.inference_sequences:
            return 0
        if self.mode == 'train':
            return max(len(self.inference_sequences) // max(1, self.inference_size), 1)
        return math.ceil(len(self.inference_sequences) / max(1, self.inference_size))

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        context_indices = self._select_context_indices(idx)
        inference_indices = self._select_inference_indices(idx)
        ctx = self._materialise(context_indices, self.context_sequences)
        inf = self._materialise(inference_indices, self.inference_sequences)
        eval_times = torch.zeros(self.inference_size, 1, dtype=torch.float32)
        return {
            'context_event_times': ctx.times,
            'context_event_types': ctx.types,
            'context_seq_lengths': ctx.lengths,
            'inference_event_times': inf.times,
            'inference_event_types': inf.types,
            'inference_seq_lengths': inf.lengths,
            'intensity_evaluation_times': eval_times,
            'num_marks': torch.tensor(self.num_marks, dtype=torch.long),
        }

    def _select_context_indices(self, idx: int) -> List[int]:
        if self.mode == 'train' and self.sampling_strategy == 'random':
            rng = random.Random(self.seed + idx)
            return self._draw_indices(rng, len(self.context_sequences), self.context_size)
        return self._fixed_context_indices

    def _select_inference_indices(self, idx: int) -> List[int]:
        total = len(self.inference_sequences)
        if total == 0:
            return []
        if self.mode == 'train':
            if self.sampling_strategy == 'random':
                rng = random.Random(self.seed * 997 + idx)
                return self._draw_indices(rng, total, self.inference_size)
            start = (idx * self.inference_size) % total
        else:
            start = (idx * self.inference_size) % total
        indices = []
        for offset in range(self.inference_size):
            indices.append((start + offset) % total)
        return indices

    @staticmethod
    def _draw_indices(rng: random.Random, pool_size: int, sample_size: int) -> List[int]:
        if pool_size == 0:
            return []
        if sample_size >= pool_size:
            return [i % pool_size for i in range(sample_size)]
        return rng.sample(range(pool_size), sample_size)

    def _materialise(self, indices: List[int], pool: List[SequenceDict]) -> _EpisodeTensors:
        num_paths = len(indices)
        times = torch.zeros(num_paths, self.max_num_events, dtype=torch.float32)
        types = torch.zeros(num_paths, self.max_num_events, dtype=torch.long)
        lengths = torch.zeros(num_paths, dtype=torch.long)
        for row, seq_idx in enumerate(indices):
            seq = pool[seq_idx]
            trunc = min(int(seq['seq_len']), self.max_num_events)
            if trunc == 0:
                trunc = 1
            lengths[row] = trunc
            times[row, :trunc] = torch.tensor(seq['time_since_start'][:trunc], dtype=torch.float32)
            types[row, :trunc] = torch.tensor(seq['type_event'][:trunc], dtype=torch.long)
        return _EpisodeTensors(times=times, types=types, lengths=lengths)

    def get_dt_stats(self) -> Tuple[float, float, float, float]:
        return self._dt_stats


class FIMEpisodeCollator:
    """Collate function that converts episodic samples into BatchEncoding."""

    def __init__(self, num_marks: int):
        self.num_marks = num_marks

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> BatchEncoding:
        batch_size = len(batch)
        context_times = torch.stack([item['context_event_times'] for item in batch], dim=0).unsqueeze(-1)
        context_types = torch.stack([item['context_event_types'] for item in batch], dim=0).unsqueeze(-1)
        context_lengths = torch.stack([item['context_seq_lengths'] for item in batch], dim=0)
        inference_times = torch.stack([item['inference_event_times'] for item in batch], dim=0).unsqueeze(-1)
        inference_types = torch.stack([item['inference_event_types'] for item in batch], dim=0).unsqueeze(-1)
        inference_lengths = torch.stack([item['inference_seq_lengths'] for item in batch], dim=0)
        eval_times = torch.stack([item['intensity_evaluation_times'] for item in batch], dim=0)
        num_marks = torch.full((batch_size,), self.num_marks, dtype=torch.long)

        data = OrderedDict(
            [
                ('context_event_times', context_times),
                ('context_event_types', context_types),
                ('context_seq_lengths', context_lengths),
                ('inference_event_times', inference_times),
                ('inference_event_types', inference_types),
                ('inference_seq_lengths', inference_lengths),
                ('intensity_evaluation_times', eval_times),
                ('num_marks', num_marks),
            ]
        )
        return BatchEncoding(data)


class FIMEpisodeDataLoader:
    """Data loader that produces episodic batches for FIM inside EasyTPP."""

    def __init__(self, data_config, model_specs=None, **kwargs):
        self.data_config = data_config
        self.model_specs = model_specs or {}
        self.batch_size = kwargs.get('batch_size', 1)
        specs = getattr(data_config, 'data_specs', None)
        self.context_size = self._get_param('fim_context_size', specs, default=None)
        self.inference_size = self._get_param('fim_inference_size', specs, default=None)
        self.max_num_events = self._get_param('fim_max_num_events', specs, default=100)
        self.sampling_strategy = self._get_param('fim_sampling_strategy', specs, default='sequential')
        self.episodes_per_epoch = self._get_param('fim_episodes_per_epoch', specs, default=None)
        self.seed = kwargs.get('seed', 42)

    def _get_param(self, key: str, specs, default):
        if self.model_specs and key in self.model_specs:
            return self.model_specs[key]
        if specs is not None and hasattr(specs, key):
            value = getattr(specs, key)
            if value is not None:
                return value
        return default

    def _load_context_sequences(self, limit=None):
        return load_fim_sequences(self.data_config.train_dir, 'train', limit)

    def _load_inference_sequences(self, split: str, limit=None):
        source = self.data_config.get_data_dir(split)
        return load_fim_sequences(source, split, limit)

    def _make_loader(self, split: str, shuffle: bool):
        context_sequences = self._load_context_sequences(limit=self.context_size)
        inference_sequences = self._load_inference_sequences(split, limit=self.inference_size)
        dataset = FIMEpisodeDataset(
            context_sequences=context_sequences,
            inference_sequences=inference_sequences,
            context_size=self.context_size,
            inference_size=self.inference_size,
            max_num_events=self.max_num_events,
            split=split,
            sampling_strategy=self.sampling_strategy,
            episodes_per_epoch=self.episodes_per_epoch,
            seed=self.seed,
        )
        collator = FIMEpisodeCollator(num_marks=dataset.num_marks)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=collator)

    def train_loader(self, **_):
        return self._make_loader(split='train', shuffle=True)

    def valid_loader(self, **_):
        return self._make_loader(split='dev', shuffle=False)

    def test_loader(self, **_):
        return self._make_loader(split='test', shuffle=False)

