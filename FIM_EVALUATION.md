# FIM Evaluation & Benchmarking Guide

This document explains how FIM was integrated into EasyTPP, how to select datasets and models, how to run the benchmark (zero-shot and finetuned FIM vs baselines), and where outputs are written.

## Integration summary
- FIM model wrapper: `easy_tpp/model/torch_model/torch_fim.py`
- FIM zero-shot adapter: `easy_tpp/benchmark/fim_adapter.py`
- Episodic data loader for FIM: `easy_tpp/preprocess/fim_episode.py`
- Benchmark driver: `examples/run_fim_benchmark.py`
- Multi-dataset configs:
  - Model/data definitions: `examples/configs/fim_multi.yaml`
  - Benchmark manifest (datasets, modes): `examples/configs/fim_benchmark_multi.yaml`

## How datasets are selected
- `examples/configs/fim_benchmark_multi.yaml` lists datasets under `datasets:`. Each entry points to:
  - `baseline_config`: currently `examples/configs/fim_multi.yaml`
  - `baseline_dataset_id`: dataset key in `fim_multi.yaml`’s `data:` section (e.g., `amazon`, `taxi`, `retweet`, `taobao`, `stackoverflow`)
  - `fim_train_experiment_id` / `fim_eval_experiment_id`: FIM experiments in `fim_multi.yaml`
  - `fim_modes`: `zero_shot`, `finetune`, or both
- To add a dataset:
  1) Add its data block (paths/specs) plus FIM/NHP/THP experiments to `examples/configs/fim_multi.yaml`.
  2) Add a dataset entry to `examples/configs/fim_benchmark_multi.yaml` pointing to those experiment IDs.

## How models are selected
- Baselines are discovered from `baseline_config` and filtered by `model_whitelist`. Current whitelist: `NHP, THP`.
- To add a baseline model: define train/eval experiments for that dataset in `fim_multi.yaml` and include the model id in `model_whitelist`.
- FIM runs in two modes per dataset (if enabled in `fim_modes`):
  - `zero_shot`: evaluate the checkpoint without finetuning (uses train split as context).
  - `finetune`: train on the train split, then evaluate on test/valid.
- The FIM checkpoint is set in `fim.checkpoint_path` within `examples/configs/fim_benchmark_multi.yaml`.

## Resource and batching notes
- FIM context_size is capped by `min(2000, num_train_sequences)` automatically. Per-dataset context/inference/max events are set in `fim_multi.yaml`.
- Amazon FIM uses a reduced batch size (2) to avoid GPU OOM; other datasets use the default batch size in the trainer settings.
- Script uses GPU 0 by default (set in trainer configs).

## How to run
From repo root:
```
PYTHONPATH=. ./.venv/bin/python examples/run_fim_benchmark.py \
  --config examples/configs/fim_benchmark_multi.yaml \
  --output benchmark_results.csv
```

## Outputs
- Aggregate CSV: `benchmark_results.csv` (path passed via `--output`).
- Per-run configs/logs/checkpoints: under `./checkpoints/<run_id>/`.
- Console summary prints each row; full stdout is in your terminal log (Cursor terminal files under `.cursor/projects/.../terminals/`).

## Extending or changing settings quickly
- Adjust batch sizes/epochs/model specs per dataset/model in `examples/configs/fim_multi.yaml`.
- Add/remove datasets or FIM modes in `examples/configs/fim_benchmark_multi.yaml`.
- Switch baselines by editing `model_whitelist` in the dataset entries (e.g., add `SAHP`, `FullyNN`, etc., after defining their experiments).
