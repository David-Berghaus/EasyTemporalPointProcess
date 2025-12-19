from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from easy_tpp.benchmark import FIMAdapterConfig, evaluate_fim_checkpoint
from easy_tpp.config_factory import RunnerConfig
from easy_tpp.runner import Runner


@dataclass
class BaselineSpec:
    label: str
    config_path: Path
    train_experiment_id: str
    eval_experiment_id: Optional[str]
    model_id: str


@dataclass
class TrainResult:
    checkpoint: Path
    fallback_metrics: Optional[Dict[str, float]]


_TRAIN_CACHE: Dict[Tuple[Path, str], TrainResult] = {}
_DATASET_SIZE_CACHE: Dict[Tuple[str, str], int] = {}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run EasyTPP baselines and a FIM checkpoint side-by-side.")
    parser.add_argument("--config", type=Path, required=True, help="Benchmark YAML configuration.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.csv"),
        help="Path to write the aggregated CSV table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    datasets = cfg.get("datasets", [])
    if not datasets:
        raise ValueError("Benchmark config must define at least one dataset entry.")

    fim_cfg = cfg.get("fim")
    if not fim_cfg:
        raise ValueError("Benchmark config must include a 'fim' section with checkpoint details.")
    fim_config_path = Path(fim_cfg["config"])
    fim_checkpoint = fim_cfg["checkpoint_path"]
    fim_train_id_default = fim_cfg.get("train_experiment_id")
    fim_eval_id_default = fim_cfg.get("eval_experiment_id")
    fim_label_default = fim_cfg.get("label", "FIM-Hawkes (EasyTPP)")
    fim_model_specs_default = fim_cfg.get("model_specs", {})

    rows: List[Dict] = []

    for dataset_entry in datasets:
        dataset_id = dataset_entry["dataset_id"]
        dataset_label = dataset_entry.get("name", dataset_id)
        fim_model_specs = {**fim_model_specs_default, **dataset_entry.get("fim_model_specs", {})}
        fim_model_specs.setdefault("fim_checkpoint_path", fim_checkpoint)
        fim_modes = dataset_entry.get("fim_modes", ["finetune"])
        data_specs = _load_data_specs(dataset_entry)
        train_len = _dataset_length(dataset_id, split="train")
        test_len = _dataset_length(dataset_id, split="test")
        context_cap = min(2000, train_len) if train_len is not None else 2000
        inference_cap = data_specs.get("fim_inference_size") or context_cap
        if test_len is not None:
            inference_cap = min(inference_cap, test_len)
        # Build per-dataset FIM specs with caps
        fim_model_specs_local = {
            **fim_model_specs,
            "fim_context_size": context_cap,
            "fim_inference_size": inference_cap,
            "fim_max_num_events": data_specs.get("fim_max_num_events"),
        }

        print(f"\n=== Dataset: {dataset_id} ===")
        baseline_specs = _resolve_baselines(dataset_entry)
        baseline_rows = _evaluate_baselines(baseline_specs)
        rows.extend(_annotate_rows(dataset_label, baseline_rows))

        fim_train_exp = dataset_entry.get("fim_train_experiment_id", fim_train_id_default)
        fim_eval_exp = dataset_entry.get("fim_eval_experiment_id", fim_eval_id_default)

        if "zero_shot" in fim_modes:
            zero_metrics = _run_fim_zero_shot(
                checkpoint_path=fim_checkpoint,
                dataset_id=dataset_id,
                data_specs=data_specs,
                model_specs=fim_model_specs_local,
                label=dataset_entry.get("fim_zero_shot_label", f"{fim_label_default} (zero-shot)"),
            )
            rows.append({**zero_metrics, "dataset": dataset_label})
            _print_single_row(rows[-1])

        if "finetune" in fim_modes:
            fim_row = _run_fim_pipeline(
                fim_config=fim_config_path,
                train_experiment_id=fim_train_exp,
                eval_experiment_id=fim_eval_exp,
                model_specs=fim_model_specs_local,
                label=dataset_entry.get("fim_label", fim_label_default),
            )
            rows.append({**fim_row, "dataset": dataset_label})
            _print_single_row(rows[-1])

    if rows:
        _write_csv(args.output, rows)
        _print_summary(rows)
        print(f"\nWrote benchmark table to {args.output.resolve()}")
    else:
        print("No results recorded. Check benchmark configuration.")


def _evaluate_baselines(specs: List[BaselineSpec]) -> List[Dict]:
    result_rows = []
    for spec in specs:
        train_info = _ensure_trained(spec)
        metrics = _run_evaluation(spec, train_info)
        result_rows.append(_metrics_to_row(spec.label, metrics))
    return result_rows


def _metrics_to_row(label: str, metrics: Dict[str, float]) -> Dict[str, float]:
    rmse = metrics.get("rmse")
    acc = metrics.get("acc")
    loglike = metrics.get("loglike")
    num_events = metrics.get("num_events")
    # Some models may not emit full metrics; fall back gracefully.
    if loglike is None:
        raise RuntimeError(f"Missing loglike metric for baseline {label}. Got keys: {metrics.keys()}")
    if acc is None:
        acc = 0.0
    if rmse is None:
        rmse = float("nan")
    type_error = 100.0 * (1.0 - acc) if acc is not None else float("nan")
    return {
        "model": label,
        "rmse": float(rmse),
        "type_error": float(type_error),
        "loglike": float(loglike),
        "num_events": int(num_events) if num_events is not None else None,
        "duration_seconds": None,
    }


def _annotate_rows(dataset_id: str, rows: List[Dict]) -> List[Dict]:
    return [{**row, "dataset": dataset_id} for row in rows]


def _write_csv(path: Path, rows: List[Dict]) -> None:
    fieldnames = ["dataset", "model", "rmse", "type_error", "loglike", "num_events", "duration_seconds"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _print_single_row(row: Dict) -> None:
    print(
        f"{row['dataset']:>15s} | {row['model']:<20s} | "
        f"RMSE: {row['rmse']:.4f} | Type Err (%): {row['type_error']:.2f} | "
        f"LogLike: {row['loglike']:.4f}"
    )


def _print_summary(rows: List[Dict]) -> None:
    print("\n=== Benchmark Summary ===")
    for row in rows:
        _print_single_row(row)


def _ensure_trained(spec: BaselineSpec) -> TrainResult:
    cache_key = (spec.config_path.resolve(), spec.train_experiment_id)
    if cache_key in _TRAIN_CACHE:
        return _TRAIN_CACHE[cache_key]

    print(
        f"→ Training baseline {spec.label} via {spec.config_path} "
        f"[{spec.train_experiment_id}]"
    )
    runner_cfg = RunnerConfig.build_from_yaml_file(
        spec.config_path, experiment_id=spec.train_experiment_id
    )
    runner = Runner.build_from_config(runner_cfg)
    runner.train()
    saved_dir = runner.get_model_dir()
    test_loader = runner._data_loader.test_loader()  # pylint: disable=protected-access
    fallback_metrics = None
    if test_loader is not None:
        fallback_metrics = runner._evaluate_model(test_loader)  # pylint: disable=protected-access

    _TRAIN_CACHE[cache_key] = TrainResult(checkpoint=Path(saved_dir), fallback_metrics=fallback_metrics)
    return _TRAIN_CACHE[cache_key]


def _run_evaluation(spec: BaselineSpec, train_info: TrainResult) -> Dict[str, float]:
    checkpoint_path = train_info.checkpoint
    if spec.eval_experiment_id is None:
        if not train_info.fallback_metrics:
            raise RuntimeError(
                f"No evaluation experiment or fallback metrics for baseline {spec.label}."
            )
        return train_info.fallback_metrics

    print(
        f"→ Evaluating baseline {spec.label} via {spec.config_path} "
        f"[{spec.eval_experiment_id}]"
    )
    runner_cfg = RunnerConfig.build_from_yaml_file(
        spec.config_path, experiment_id=spec.eval_experiment_id
    )
    runner_cfg.model_config.pretrained_model_dir = str(checkpoint_path)
    runner = Runner.build_from_config(runner_cfg)
    test_loader = runner._data_loader.test_loader()  # pylint: disable=protected-access
    return runner._evaluate_model(test_loader)  # pylint: disable=protected-access


def _resolve_baselines(dataset_entry: Dict) -> List[BaselineSpec]:
    manual_entries = dataset_entry.get("baselines")
    if manual_entries:
        return [_spec_from_manual_entry(entry) for entry in manual_entries]

    config_path = dataset_entry.get("baseline_config")
    if not config_path:
        raise ValueError(
            "Each dataset must define either 'baselines' or 'baseline_config' for automatic discovery."
        )
    baseline_dataset_id = dataset_entry.get("baseline_dataset_id", dataset_entry["dataset_id"])
    model_whitelist = dataset_entry.get("model_whitelist")
    return _discover_baselines(Path(config_path), baseline_dataset_id, model_whitelist)


def _run_fim_zero_shot(
    checkpoint_path: str,
    dataset_id: str,
    data_specs: Dict,
    model_specs: Dict,
    label: str,
) -> Dict[str, float]:
    context_size = data_specs.get("fim_context_size")
    inference_size = data_specs.get("fim_inference_size")
    max_events = data_specs.get("fim_max_num_events")
    num_integration_points = model_specs.get("fim_num_integration_points", 5000)
    metrics = evaluate_fim_checkpoint(
        FIMAdapterConfig(
            checkpoint_path=checkpoint_path,
            dataset_id=dataset_id,
            context_size=context_size,
            inference_size=inference_size,
            max_num_events=max_events,
            sampling_method=model_specs.get("fim_sampling_method"),
            num_integration_points=num_integration_points,
        )
    )
    return {
        "model": label,
        "rmse": metrics["rmse"],
        "type_error": metrics["type_error"],
        "loglike": metrics["loglike"],
        "num_events": metrics["num_events"],
        "duration_seconds": metrics["duration_seconds"],
    }


def _load_data_specs(dataset_entry: Dict) -> Dict:
    cfg_path = Path(dataset_entry["baseline_config"])
    cfg = yaml.safe_load(cfg_path.read_text())
    ds_id = dataset_entry.get("baseline_dataset_id", dataset_entry["dataset_id"])
    return cfg["data"][ds_id]["data_specs"]


def _dataset_length(dataset_id: str, split: str) -> Optional[int]:
    key = (dataset_id, split)
    if key in _DATASET_SIZE_CACHE:
        return _DATASET_SIZE_CACHE[key]
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    ds = load_dataset(f"easytpp/{dataset_id}", split=split)
    _DATASET_SIZE_CACHE[key] = len(ds)
    return len(ds)


def _spec_from_manual_entry(entry: Dict) -> BaselineSpec:
    config_path = Path(entry["config"])
    eval_id = entry.get("experiment_id")
    train_id = entry.get("train_experiment_id")
    if train_id is None:
        train_id = _infer_train_experiment(config_path, eval_id)
    label = entry.get("label", eval_id or train_id)
    model_id = entry.get("model_id", label)
    return BaselineSpec(label=label, config_path=config_path, train_experiment_id=train_id, eval_experiment_id=eval_id, model_id=model_id)


def _infer_train_experiment(config_path: Path, eval_id: Optional[str]) -> str:
    if not eval_id:
        raise ValueError("Must provide 'train_experiment_id' when 'experiment_id' is absent.")
    candidate = None
    if eval_id.endswith("_eval"):
        candidate = eval_id[:-5] + "_train"
    elif eval_id.endswith("_valid"):
        candidate = eval_id[:-6] + "_train"
    if candidate is None:
        raise ValueError(
            f"Cannot infer train experiment for {eval_id}; please set 'train_experiment_id'."
        )
    data = yaml.safe_load(config_path.read_text())
    if candidate not in data:
        raise ValueError(
            f"Could not find inferred train experiment '{candidate}' in {config_path}. "
            "Please specify 'train_experiment_id' explicitly."
        )
    return candidate


def _discover_baselines(
    config_path: Path, dataset_id: str, model_whitelist: Optional[List[str]]
) -> List[BaselineSpec]:
    config_data = yaml.safe_load(config_path.read_text())
    experiments = {
        key: value
        for key, value in config_data.items()
        if isinstance(value, dict) and key not in {"data", "pipeline_config_id"}
    }
    grouped: Dict[str, Dict[str, str]] = {}
    for exp_id, exp_cfg in experiments.items():
        base_cfg = exp_cfg.get("base_config", {})
        exp_dataset = base_cfg.get("dataset_id")
        stage = str(base_cfg.get("stage", "")).lower()
        model_id = base_cfg.get("model_id")
        if model_id is None or stage not in {"train", "eval"}:
            continue
        if exp_dataset != dataset_id:
            continue
        # Avoid double-running the external FIM model as a baseline; it is handled separately.
        if model_id == "FIMHawkesModel":
            continue
        grouped.setdefault(model_id, {})[stage] = exp_id

    specs: List[BaselineSpec] = []
    for model_id, stages in grouped.items():
        if model_whitelist and model_id not in model_whitelist:
            continue
        train_id = stages.get("train")
        if not train_id:
            continue
        eval_id = stages.get("eval")
        label = f"{model_id} (EasyTPP)"
        specs.append(
            BaselineSpec(
                label=label,
                config_path=config_path,
                train_experiment_id=train_id,
                eval_experiment_id=eval_id,
                model_id=model_id,
            )
        )

    if not specs:
        raise ValueError(
            f"No baseline experiments discovered for dataset '{dataset_id}' in {config_path}."
        )
    return specs


def _run_fim_pipeline(
    fim_config: Path,
    train_experiment_id: str,
    eval_experiment_id: Optional[str],
    model_specs: Dict,
    label: str,
) -> Dict[str, float]:
    if not train_experiment_id:
        raise ValueError("FIM configuration must provide 'train_experiment_id'.")

    cache_key = (fim_config.resolve(), train_experiment_id)
    if cache_key not in _TRAIN_CACHE:
        print(f"→ Training FIM via {fim_config} [{train_experiment_id}]")
        runner_cfg = RunnerConfig.build_from_yaml_file(fim_config, experiment_id=train_experiment_id)
        runner_cfg.model_config.model_specs.update(model_specs or {})
        runner = Runner.build_from_config(runner_cfg)
        runner.train()
        saved_dir = runner.get_model_dir()
        test_loader = runner._data_loader.test_loader()  # pylint: disable=protected-access
        fallback_metrics = runner._evaluate_model(test_loader)  # pylint: disable=protected-access
        _TRAIN_CACHE[cache_key] = TrainResult(checkpoint=Path(saved_dir), fallback_metrics=fallback_metrics)

    train_info = _TRAIN_CACHE[cache_key]
    if eval_experiment_id:
        print(f"→ Evaluating FIM via {fim_config} [{eval_experiment_id}]")
        runner_cfg = RunnerConfig.build_from_yaml_file(fim_config, experiment_id=eval_experiment_id)
        runner_cfg.model_config.pretrained_model_dir = str(train_info.checkpoint)
        runner_cfg.model_config.model_specs.update(model_specs or {})
        runner = Runner.build_from_config(runner_cfg)
        test_loader = runner._data_loader.test_loader()  # pylint: disable=protected-access
        metrics = runner._evaluate_model(test_loader)  # pylint: disable=protected-access
    else:
        if not train_info.fallback_metrics:
            raise RuntimeError("No evaluation metrics available for FIM.")
        metrics = train_info.fallback_metrics

    return _metrics_to_row(label, metrics)

if __name__ == "__main__":
    main()

