from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn

from easy_tpp.benchmark.fim_adapter import _ContextBatch, _predict_sequence
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.preprocess.event_tokenizer import BatchEncoding

DEFAULT_FIM_REPO = "/cephfs/users/berghaus/FoundationModels/FIM"
FIM_REPO_ROOT = Path(os.environ.get("FIM_REPO_ROOT", DEFAULT_FIM_REPO)).expanduser().resolve()
FIM_SRC = FIM_REPO_ROOT / "src"

if not FIM_SRC.exists():
    raise ImportError(
        f"FIM source directory not found at {FIM_SRC}. "
        "Set FIM_REPO_ROOT to the root of the FIM repository."
    )

if str(FIM_SRC) not in sys.path:
    sys.path.insert(0, str(FIM_SRC))

from fim.models.hawkes import FIMHawkes  # type: ignore  # pylint: disable=wrong-import-position


class FIMHawkesModel(TorchBaseModel):
    """Wrap the external FIM Hawkes model so it can run inside EasyTPP."""

    CUSTOM_BATCH_FORMAT = True

    def __init__(self, model_config):
        super().__init__(model_config)
        self.model_specs = model_config.model_specs or {}
        checkpoint_path = self.model_specs.get("fim_checkpoint_path")
        if checkpoint_path is None:
            raise ValueError("`fim_checkpoint_path` must be provided in model_specs for FIMHawkesModel.")

        self.device = torch.device("cuda" if model_config.gpu >= 0 and torch.cuda.is_available() else "cpu")
        self.fim_model = FIMHawkes.load_model(Path(checkpoint_path))
        self.fim_model.to(self.device)
        self.event_sampler = self.fim_model.event_sampler
        sampling_method = self.model_specs.get("fim_sampling_method")
        if sampling_method:
            self.event_sampler.sampling_method = sampling_method
        self.num_marks = self.fim_model.config.max_num_marks
        self.num_integration_points = self.model_specs.get("fim_num_integration_points", 5000)

        # Freeze TorchBaseModel-specific embeddings as they are unused.
        for module in [getattr(self, attr, None) for attr in ("layer_type_emb",)]:
            if isinstance(module, nn.Module):
                module.requires_grad_(False)

    def loglike_loss(self, batch: BatchEncoding) -> Tuple[torch.Tensor, int]:
        batch_inputs = self._clone_batch(batch)
        fim_out = self.fim_model.forward(batch_inputs)
        if isinstance(self.fim_model.config.nll, dict):
            self.fim_model.config.nll["num_integration_points"] = self.num_integration_points
        event_times = (
            batch_inputs["inference_event_times_norm"]
            if self.fim_model.normalize_times
            else batch_inputs["inference_event_times"]
        )
        nll = self.fim_model._nll_loss(  # pylint: disable=protected-access
            intensity_fn=fim_out["intensity_function"],
            event_times=event_times.squeeze(-1),
            event_types=batch_inputs["inference_event_types"].squeeze(-1),
            seq_lengths=batch_inputs["inference_seq_lengths"],
        )
        num_events = torch.clamp(batch_inputs["inference_seq_lengths"] - 1, min=0).sum().item()
        num_events = max(int(num_events), 1)
        return nll, num_events

    def predict_one_step_at_every_event(
        self, batch: BatchEncoding
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self._clone_batch(batch)
        with torch.no_grad():
            enhanced_context = self.fim_model.encode_context(
                {
                    "context_event_times": inputs["context_event_times"],
                    "context_event_types": inputs["context_event_types"],
                    "context_seq_lengths": inputs["context_seq_lengths"],
                }
            )

        B, num_inf, max_len, _ = inputs["inference_event_times"].shape
        pred_steps = max(max_len - 1, 1)
        pred_d_list = []
        pred_t_list = []
        label_d_list = []
        label_t_list = []
        mask_list = []

        for b in range(B):
            context_batch = _ContextBatch(
                time_seqs=inputs["context_event_times"][b].squeeze(-1),
                type_seqs=inputs["context_event_types"][b].squeeze(-1),
                seq_len=inputs["context_seq_lengths"][b],
            )
            for p in range(num_inf):
                seq_len = int(inputs["inference_seq_lengths"][b, p].item())
                inference_item = self._build_inference_item(inputs, b, p, seq_len)
                preds = _predict_sequence(
                    model=self.fim_model,
                    inference_sequence=inference_item,
                    context_batch=context_batch,
                    device=self.device,
                    precomputed_enhanced_context=enhanced_context[b : b + 1],
                    num_marks=self.num_marks,
                )
                pred_d, pred_t = self._pad_prediction(preds, pred_steps, seq_len)
                true_d, true_t, mask = self._extract_labels(inputs, b, p, pred_steps, seq_len)
                pred_d_list.append(pred_d)
                pred_t_list.append(pred_t)
                label_d_list.append(true_d)
                label_t_list.append(true_t)
                mask_list.append(mask)

        return (
            torch.stack(pred_d_list, dim=0),
            torch.stack(pred_t_list, dim=0),
            torch.stack(label_d_list, dim=0),
            torch.stack(label_t_list, dim=0),
            torch.stack(mask_list, dim=0),
        )

    def predict_multi_step_since_last_event(self, batch: BatchEncoding):
        pred_d, pred_t, label_d, label_t, _ = self.predict_one_step_at_every_event(batch)
        return pred_d, pred_t, label_d, label_t

    def _clone_batch(self, batch: BatchEncoding) -> Dict[str, torch.Tensor]:
        if isinstance(batch, BatchEncoding):
            return {k: v.to(self.device) for k, v in batch.items()}
        return batch

    def _build_inference_item(self, inputs, batch_idx: int, path_idx: int, seq_len: int):
        times = inputs["inference_event_times"][batch_idx, path_idx, :seq_len, 0]
        deltas = torch.zeros_like(times)
        if seq_len > 1:
            deltas[1:] = times[1:] - times[:-1]
        types = inputs["inference_event_types"][batch_idx, path_idx, :seq_len, 0].long()
        mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)
        seq_tensor = torch.tensor([seq_len], device=self.device)
        return {
            "time_seqs": times.unsqueeze(0),
            "time_delta_seqs": deltas.unsqueeze(0),
            "type_seqs": types.unsqueeze(0),
            "seq_len": seq_tensor,
            "seq_non_pad_mask": mask.unsqueeze(0),
        }

    def _pad_prediction(self, preds: Dict[str, torch.Tensor], max_len: int, seq_len: int):
        dtime_row = torch.zeros(max_len, device=self.device)
        type_row = torch.zeros(max_len, dtype=torch.long, device=self.device)
        effective = max(seq_len - 1, 0)
        if effective > 0:
            dtime_row[:effective] = preds["dtimes"][0, :effective]
            type_row[:effective] = preds["types"][0, :effective]
        return dtime_row, type_row

    def _extract_labels(self, inputs, batch_idx: int, path_idx: int, max_len: int, seq_len: int):
        label_d = torch.zeros(max_len, device=self.device)
        label_t = torch.zeros(max_len, dtype=torch.long, device=self.device)
        mask = torch.zeros(max_len, dtype=torch.bool, device=self.device)
        if seq_len > 1:
            times = inputs["inference_event_times"][batch_idx, path_idx, :seq_len, 0]
            label_d[: seq_len - 1] = times[1:] - times[:-1]
            label_t[: seq_len - 1] = inputs["inference_event_types"][batch_idx, path_idx, 1:seq_len, 0]
            mask[: seq_len - 1] = True
        return label_d, label_t, mask

