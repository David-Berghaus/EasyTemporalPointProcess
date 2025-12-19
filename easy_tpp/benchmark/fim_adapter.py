from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Locate the external FIM repository so we can import the model definition.
# ---------------------------------------------------------------------------
DEFAULT_FIM_REPO = "/cephfs/users/berghaus/FoundationModels/FIM"
_fim_repo_root = Path(os.environ.get("FIM_REPO_ROOT", DEFAULT_FIM_REPO)).resolve()
_fim_src = _fim_repo_root / "src"

if not _fim_src.exists():
    raise ImportError(
        "FIM source directory not found. "
        "Set the FIM_REPO_ROOT environment variable or update DEFAULT_FIM_REPO."
    )

if str(_fim_src) not in sys.path:
    sys.path.insert(0, str(_fim_src))

from fim.models.hawkes import FIMHawkes  # type: ignore  # pylint: disable=wrong-import-position


@dataclass
class FIMAdapterConfig:
    """Configuration describing how to run a FIM checkpoint on an EasyTPP dataset."""

    checkpoint_path: str
    dataset_id: str
    context_size: Optional[int] = None
    inference_size: Optional[int] = None
    max_num_events: Optional[int] = 100
    sampling_method: Optional[str] = None
    num_integration_points: int = 5000


class _ContextBatch:
    """Helper container holding tensors for context sequences."""

    def __init__(self, time_seqs: torch.Tensor, type_seqs: torch.Tensor, seq_len: torch.Tensor):
        self.time_seqs = time_seqs
        self.type_seqs = type_seqs
        self.seq_len = seq_len
        max_len = time_seqs.size(1)
        positions = torch.arange(max_len, device=time_seqs.device).unsqueeze(0)
        self.seq_non_pad_mask = positions.expand(seq_len.size(0), -1) < seq_len.unsqueeze(1)


def evaluate_fim_checkpoint(cfg: FIMAdapterConfig) -> Dict[str, float]:
    """Evaluate a FIM checkpoint under the EasyTPP next-event metric recipe."""

    adapter = _FIMAdapter(cfg.checkpoint_path, cfg.sampling_method)
    return adapter.evaluate(
        dataset_id=cfg.dataset_id,
        context_size=cfg.context_size,
        inference_size=cfg.inference_size,
        max_num_events=cfg.max_num_events,
        num_integration_points=cfg.num_integration_points,
    )


class _FIMAdapter:
    """Thin wrapper that loads a FIM checkpoint and reproduces EasyTPP metrics."""

    def __init__(self, checkpoint_path: str, sampling_method: Optional[str]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FIMHawkes.load_model(Path(checkpoint_path))
        self.model.to(self.device)
        self.model.eval()
        # Stabilise sampler defaults
        if sampling_method in ("thinning", "inverse_transform"):
            self.model.event_sampler.sampling_method = sampling_method
        self.model.event_sampler.num_samples_boundary = max(50, self.model.event_sampler.num_samples_boundary)
        self.model.event_sampler.over_sample_rate = 5.0

    def evaluate(
        self,
        dataset_id: str,
        context_size: Optional[int],
        inference_size: Optional[int],
        max_num_events: Optional[int],
        num_integration_points: int,
    ) -> Dict[str, float]:
        start = time.time()
        dataset_id = dataset_id if dataset_id.startswith("easytpp/") else f"easytpp/{dataset_id}"
        context_raw = _load_dataset_slice(dataset_id, "train", context_size)
        inference_raw = _load_dataset_slice(dataset_id, "test", inference_size)
        context_raw = _truncate_batch(context_raw, max_num_events)
        inference_raw = _truncate_batch(inference_raw, max_num_events)

        num_marks = _detect_num_marks(context_raw, inference_raw)
        context_batch = self._build_context_batch(context_raw)
        self._configure_sampler(context_raw)

        with torch.no_grad():
            precomputed_ctx = {
                "context_event_times": context_batch.time_seqs.unsqueeze(0).unsqueeze(-1).to(self.device),
                "context_event_types": context_batch.type_seqs.unsqueeze(0).unsqueeze(-1).to(self.device),
                "context_seq_lengths": context_batch.seq_non_pad_mask.sum(dim=1).unsqueeze(0).to(self.device),
            }
            enhanced_context = self.model.encode_context(precomputed_ctx)

        totals = {
            "sq_err": 0.0,
            "correct": 0,
            "events": 0,
            "nll": 0.0,
        }

        for seq_idx in range(len(inference_raw["time_since_start"])):
            inference_item = _make_inference_item(inference_raw, seq_idx, self.device)
            pred = _predict_sequence(
                model=self.model,
                inference_sequence=inference_item,
                context_batch=context_batch,
                device=self.device,
                precomputed_enhanced_context=enhanced_context,
                num_marks=num_marks,
            )

            true_dtimes = inference_item["time_delta_seqs"].cpu()[:, 1:]
            true_types = inference_item["type_seqs"].cpu()[:, 1:]
            seq_len = inference_item["seq_len"].item()
            mask = torch.zeros_like(true_types, dtype=torch.bool)
            if seq_len > 1:
                mask[0, : seq_len - 1] = True

            events_in_seq = mask.sum().item()
            if events_in_seq == 0:
                continue

            totals["sq_err"] += torch.sum((pred["dtimes"][mask] - true_dtimes[mask]) ** 2).item()
            totals["correct"] += torch.sum(pred["types"][mask] == true_types[mask]).item()
            totals["events"] += events_in_seq

            nll = _compute_model_nll(
                model=self.model,
                inference_sequence=inference_item,
                context_batch=context_batch,
                device=self.device,
                precomputed_enhanced_context=enhanced_context,
                num_marks=num_marks,
                num_integration_points=num_integration_points,
            )
            totals["nll"] += nll

        if totals["events"] == 0:
            raise RuntimeError("No evaluable events found in inference split.")

        rmse = (totals["sq_err"] / totals["events"]) ** 0.5
        acc = totals["correct"] / totals["events"]
        loglike = -totals["nll"] / max(len(inference_raw["seq_len"]), 1)

        return {
            "dataset": dataset_id,
            "rmse": rmse,
            "type_error": 100.0 * (1.0 - acc),
            "loglike": loglike,
            "num_events": totals["events"],
            "duration_seconds": time.time() - start,
        }

    def _configure_sampler(self, context_raw: Dict[str, List[List[float]]]) -> None:
        max_delta = 1.0
        if context_raw["time_since_last_event"]:
            max_delta = max(max(seq) for seq in context_raw["time_since_last_event"])
        self.model.event_sampler.dtime_max = float(max_delta * 1.2)

    def _build_context_batch(self, context_raw: Dict[str, List[List[float]]]) -> _ContextBatch:
        num_sequences = len(context_raw["time_since_start"])
        if num_sequences == 0:
            raise ValueError("Context set is empty; provide at least one training sequence.")
        max_len = max(len(seq) for seq in context_raw["time_since_start"])
        times = []
        types = []
        lengths = []
        for idx in range(num_sequences):
            pad_len = max_len - len(context_raw["time_since_start"][idx])
            times.append(context_raw["time_since_start"][idx] + [0.0] * pad_len)
            types.append(context_raw["type_event"][idx] + [0] * pad_len)
            lengths.append(context_raw["seq_len"][idx])

        time_tensor = torch.tensor(times, dtype=torch.float32, device=self.device)
        type_tensor = torch.tensor(types, dtype=torch.long, device=self.device)
        len_tensor = torch.tensor(lengths, dtype=torch.long, device=self.device)
        return _ContextBatch(time_tensor, type_tensor, len_tensor)


def _load_dataset_slice(dataset_id: str, split: str, size: Optional[int]) -> Dict[str, List[List[float]]]:
    split_name = "validation" if split == "dev" else split
    dataset = load_dataset(dataset_id, split=split_name)
    length = len(dataset)
    effective_size = length if size is None else min(size, length)
    data = dataset[:effective_size]
    if "seq_len" not in data:
        data["seq_len"] = [len(seq) for seq in data["time_since_start"]]
    return data


def _truncate_batch(batch: Dict[str, List[List[float]]], limit: Optional[int]) -> Dict[str, List[List[float]]]:
    if limit is None or limit < 0:
        return batch
    truncated = {
        "time_since_start": [],
        "time_since_last_event": [],
        "type_event": [],
        "seq_len": [],
    }
    for times, deltas, types, length in zip(
        batch["time_since_start"],
        batch["time_since_last_event"],
        batch["type_event"],
        batch["seq_len"],
    ):
        trunc = min(length, limit)
        truncated["time_since_start"].append(times[:trunc])
        truncated["time_since_last_event"].append(deltas[:trunc])
        truncated["type_event"].append(types[:trunc])
        truncated["seq_len"].append(trunc)
    return truncated


def _detect_num_marks(context_data: Dict[str, List[List[float]]], inference_data: Dict[str, List[List[float]]]) -> int:
    marks = set()
    for seq in context_data["type_event"]:
        marks.update(seq)
    for seq in inference_data["type_event"]:
        marks.update(seq)
    return max(len(marks), 1)


def _make_inference_item(
    inference_raw: Dict[str, List[List[float]]],
    idx: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    seq_len = inference_raw["seq_len"][idx]
    times = torch.tensor([inference_raw["time_since_start"][idx]], dtype=torch.float32, device=device)
    deltas = torch.tensor([inference_raw["time_since_last_event"][idx]], dtype=torch.float32, device=device)
    types = torch.tensor([inference_raw["type_event"][idx]], dtype=torch.long, device=device)
    seq_len_tensor = torch.tensor([seq_len], dtype=torch.long, device=device)
    positions = torch.arange(times.size(1), device=device).unsqueeze(0)
    mask = positions < seq_len_tensor.unsqueeze(1)
    return {
        "time_seqs": times,
        "time_delta_seqs": deltas,
        "type_seqs": types,
        "seq_len": seq_len_tensor,
        "seq_non_pad_mask": mask,
    }


def _predict_sequence(
    model,
    inference_sequence,
    context_batch: _ContextBatch,
    device: torch.device,
    precomputed_enhanced_context: torch.Tensor,
    num_marks: int,
) -> Dict[str, torch.Tensor]:
    seq_len = inference_sequence["seq_len"].item()
    if seq_len <= 1:
        zeros = torch.zeros_like(inference_sequence["time_delta_seqs"][:, 1:])
        return {"dtimes": zeros, "types": zeros.long()}

    dtime_preds: List[float] = []
    type_preds: List[int] = []
    context_times = context_batch.time_seqs.unsqueeze(0).unsqueeze(-1).to(device)
    context_types = context_batch.type_seqs.unsqueeze(0).unsqueeze(-1).to(device)
    context_lengths = context_batch.seq_non_pad_mask.sum(dim=1).unsqueeze(0).to(device)

    for prefix_len in range(1, seq_len):
        inf_times = inference_sequence["time_seqs"][0, :prefix_len].view(1, 1, prefix_len, 1).to(device)
        inf_types = inference_sequence["type_seqs"][0, :prefix_len].to(dtype=torch.long).view(1, 1, prefix_len, 1).to(device)
        inf_lengths = torch.tensor([[prefix_len]], device=device)
        x = {
            "context_event_times": context_times,
            "context_event_types": context_types,
            "context_seq_lengths": context_lengths,
            "inference_event_times": inf_times,
            "inference_event_types": inf_types,
            "inference_seq_lengths": inf_lengths,
            "intensity_evaluation_times": torch.zeros(1, 1, 1, device=device),
            "precomputed_enhanced_context": precomputed_enhanced_context,
            "num_marks": torch.tensor([num_marks], device=device),
        }
        _orig_nll = getattr(model, "_nll_loss", None)
        try:
            if _orig_nll is not None:
                model._nll_loss = lambda *_, **__: torch.tensor(0.0, device=device)
            model_out = model.forward(x)
        finally:
            if _orig_nll is not None:
                model._nll_loss = _orig_nll

        intensity_obj = model_out["intensity_function"]
        hist_times = x["inference_event_times"].squeeze(0).squeeze(-1)
        hist_dtimes = torch.zeros_like(hist_times)
        hist_dtimes[:, 1:] = hist_times[:, 1:] - hist_times[:, :-1]
        hist_types = x["inference_event_types"].squeeze(0).squeeze(-1)

        def intensity_fn(query_times, _hist_ignore):
            per_mark = intensity_obj.evaluate(query_times)
            return per_mark.permute(0, 2, 3, 1)

        sampler = model.event_sampler
        if getattr(sampler, "sampling_method", "thinning") == "inverse_transform":
            accepted_dtimes, weights = sampler.draw_next_time_one_step_inverse_transform(
                intensity_obj, compute_last_step_only=True
            )
        else:
            accepted_dtimes, weights = sampler.draw_next_time_one_step(
                time_seq=hist_times,
                time_delta_seq=hist_dtimes,
                event_seq=hist_types,
                intensity_fn=intensity_fn,
                compute_last_step_only=True,
            )

        t_last = hist_times[:, -1:].unsqueeze(-1)
        delta_samples = torch.clamp(accepted_dtimes - t_last, min=0.0)
        dtime_pred = torch.sum(delta_samples * weights, dim=-1).squeeze().item()

        intensities = intensity_obj.evaluate(accepted_dtimes)
        total_intensity = intensities.sum(dim=1, keepdim=True)
        probs = intensities / (total_intensity + 1e-9)
        expected_probs = torch.sum(probs * weights.unsqueeze(1), dim=-1)
        type_pred = torch.argmax(expected_probs.squeeze()).item()

        dtime_preds.append(dtime_pred)
        type_preds.append(type_pred)

    max_len = inference_sequence["time_delta_seqs"].size(1) - 1
    dtime_tensor = torch.tensor(dtime_preds + [0.0] * (max_len - len(dtime_preds)))
    type_tensor = torch.tensor(type_preds + [0] * (max_len - len(type_preds)))
    return {"dtimes": dtime_tensor.unsqueeze(0), "types": type_tensor.unsqueeze(0)}


def _compute_model_nll(
    model,
    inference_sequence,
    context_batch: _ContextBatch,
    device: torch.device,
    precomputed_enhanced_context: torch.Tensor,
    num_marks: int,
    num_integration_points: int,
) -> float:
    context_times = context_batch.time_seqs.unsqueeze(0).unsqueeze(-1).to(device)
    context_types = context_batch.type_seqs.unsqueeze(0).unsqueeze(-1).to(device)
    context_lengths = context_batch.seq_non_pad_mask.sum(dim=1).unsqueeze(0).to(device)

    inf_times = inference_sequence["time_seqs"].unsqueeze(0).unsqueeze(-1).to(device)
    inf_types = inference_sequence["type_seqs"].unsqueeze(0).unsqueeze(-1).to(device)
    inf_lengths = inference_sequence["seq_len"].unsqueeze(0).to(device)
    x = {
        "context_event_times": context_times,
        "context_event_types": context_types,
        "context_seq_lengths": context_lengths,
        "inference_event_times": inf_times,
        "inference_event_types": inf_types,
        "inference_seq_lengths": inf_lengths,
        "intensity_evaluation_times": torch.zeros(1, 1, 1, device=device),
        "precomputed_enhanced_context": precomputed_enhanced_context,
        "num_marks": torch.tensor([num_marks], device=device),
    }
    with torch.no_grad():
        model_out = model.forward(x)
    intensity_obj = model_out["intensity_function"]
    if isinstance(model.config.nll, dict):
        model.config.nll["num_integration_points"] = num_integration_points
    event_times_for_nll = x["inference_event_times_norm"] if model.normalize_times else x["inference_event_times"]
    nll = model._nll_loss(
        intensity_fn=intensity_obj,
        event_times=event_times_for_nll.squeeze(-1),
        event_types=inf_types.squeeze(-1),
        seq_lengths=inf_lengths,
    )
    return float(nll.item())







