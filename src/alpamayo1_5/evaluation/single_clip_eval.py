# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-clip minADE evaluation for Alpamayo 1.5.

Provides a structured CLI entry point and a reusable ``SingleClipEvaluator`` class
that runs the full load → infer → score pipeline and writes a JSON result file.

Example usage::

    python src/alpamayo1_5/evaluation/single_clip_eval.py \\
        --clip_id 030c760c-ae38-49aa-9ad8-f5650a545d26 \\
        --num_traj_samples 16

"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alpamayo1_5 import helper
from alpamayo1_5.evaluation.minade import MinADECalculator
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

logger = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class EvalConfig:
    """Configuration for :class:`SingleClipEvaluator`.

    Attributes:
        clip_id: PhysicalAI-AV clip ID to evaluate.
        t0_us: Evaluation timestamp in microseconds.
        model_name: HuggingFace model identifier or local path.
        device: Target device (``"cuda"`` or ``"cpu"``).
        dtype: Model floating-point precision.
        attn_implementation: Attention backend override.  Pass ``"sdpa"`` when
            ``flash-attn`` is not available.
        num_traj_samples: Number of trajectory samples drawn per inference call.
        top_p: Nucleus sampling probability for VLM decoding.
        temperature: Sampling temperature for VLM decoding.
        max_generation_length: Maximum number of tokens for VLM generation.
        seed: Random seed for reproducible sampling.
        maybe_stream: Whether to stream data from HuggingFace if not cached locally.
        output: Path to write the JSON result file.
    """

    clip_id: str = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    t0_us: int = 5_100_000
    model_name: str = "nvidia/Alpamayo-1.5-10B"
    device: str = "cuda"
    dtype: str = "bfloat16"
    attn_implementation: str | None = None
    num_traj_samples: int = 1
    top_p: float = 0.98
    temperature: float = 0.6
    max_generation_length: int = 256
    seed: int = 42
    maybe_stream: bool = True
    output: str = field(default="outputs/single_clip_eval.json")


class SingleClipEvaluator:
    """Evaluates Alpamayo 1.5 trajectory prediction on a single dataset clip.

    Runs the full inference pipeline and reports minADE against the
    ground-truth future ego trajectory.

    Example::

        cfg = EvalConfig(clip_id="030c760c-ae38-49aa-9ad8-f5650a545d26")
        result = SingleClipEvaluator(cfg).run()
        print(f"minADE: {result['minADE']:.4f} m")
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.torch_dtype = _DTYPE_MAP[config.dtype]

    def run(self) -> dict[str, Any]:
        """Execute the full evaluation pipeline.

        Returns:
            A dict containing ``clip_id``, ``t0_us``, ``num_traj_samples``,
            ``minADE``, ``all_ADE``, and (when available) ``cot``.
        """
        cfg = self.config

        print(f"[1/4] Loading dataset: clip_id={cfg.clip_id}, t0_us={cfg.t0_us}")
        data = load_physical_aiavdataset(
            clip_id=cfg.clip_id,
            t0_us=cfg.t0_us,
            maybe_stream=cfg.maybe_stream,
        )

        print("[2/4] Loading model...")
        model, processor = self._load_model_and_processor()

        print("[3/4] Running inference...")
        pred_xyz, extra = self._infer(model, processor, data)

        print("[4/4] Calculating minADE...")
        result = MinADECalculator().calculate(
            pred_xyz=pred_xyz,
            gt_future_xyz=data["ego_future_xyz"],
        )

        cot_text = _extract_cot(extra)
        output = {
            "clip_id": cfg.clip_id,
            "t0_us": cfg.t0_us,
            "num_traj_samples": cfg.num_traj_samples,
            "minADE": result.min_ade,
            "all_ADE": result.all_ade,
            "cot": cot_text,
        }

        _save_result(output, Path(cfg.output))
        _print_summary(output, pred_xyz)

        return output

    def _load_model_and_processor(self) -> tuple[Alpamayo1_5, Any]:
        kwargs: dict[str, Any] = {"dtype": self.torch_dtype}
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation

        model = Alpamayo1_5.from_pretrained(self.config.model_name, **kwargs).to(self.device)
        model.eval()
        processor = helper.get_processor(model.tokenizer)
        return model, processor

    @torch.inference_mode()
    def _infer(
        self,
        model: Alpamayo1_5,
        processor: Any,
        data: dict[str, Any],
    ) -> tuple[torch.Tensor, Any]:
        cfg = self.config
        frames = data["image_frames"].flatten(0, 1)
        messages = helper.create_message(
            frames=frames,
            camera_indices=data["camera_indices"],
        )

        tokenized_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = helper.to_device(
            {
                "tokenized_data": tokenized_inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            },
            self.device,
        )

        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.torch_dtype,
            enabled=self.device.type == "cuda",
        ):
            pred_xyz, _pred_rot, extra = (
                model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=cfg.top_p,
                    temperature=cfg.temperature,
                    num_traj_samples=cfg.num_traj_samples,
                    max_generation_length=cfg.max_generation_length,
                    return_extra=True,
                )
            )

        return pred_xyz, extra


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_cot(extra: Any) -> Any:
    if not isinstance(extra, dict) or "cot" not in extra:
        return None
    try:
        return extra["cot"][0]
    except Exception:
        return str(extra["cot"])


def _print_summary(output: dict[str, Any], pred_xyz: torch.Tensor) -> None:
    print("\n========== Evaluation Result ==========")
    print(f"clip_id:          {output['clip_id']}")
    print(f"t0_us:            {output['t0_us']}")
    print(f"num_traj_samples: {output['num_traj_samples']}")
    print(f"minADE:           {output['minADE']:.6f} m")
    print(f"all_ADE:          {output['all_ADE']}")

    _print_trajectory(pred_xyz, traj_index=0)

    if output.get("cot"):
        print("\n========== Chain-of-Causation ==========")
        print(output["cot"])


def _print_trajectory(pred_xyz: torch.Tensor, traj_index: int = 0) -> None:
    """Print XYZ waypoints for a single predicted trajectory.

    Args:
        pred_xyz: Model output tensor.  Accepted shapes:
            ``(B, G, K, T, 3)``, ``(B, K, T, 3)``, ``(K, T, 3)``, ``(T, 3)``.
        traj_index: Index of the trajectory sample to print.
    """
    pred = pred_xyz.detach().float().cpu()

    if pred.ndim == 5:
        traj = pred[0, 0, traj_index]
    elif pred.ndim == 4:
        traj = pred[0, traj_index]
    elif pred.ndim == 3:
        traj = pred[traj_index]
    elif pred.ndim == 2:
        traj = pred
    else:
        raise ValueError(f"Unsupported pred_xyz shape: {tuple(pred.shape)}")

    print(f"\n========== Predicted Trajectory (index={traj_index}) ==========")
    for i, point in enumerate(traj):
        x, y = float(point[0]), float(point[1])
        z = float(point[2]) if point.shape[0] > 2 else 0.0
        print(f"  [{i:02d}] x={x:8.3f}  y={y:8.3f}  z={z:8.3f}")


def _make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    return obj


def _save_result(output: dict[str, Any], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(_make_json_serializable(output), f, ensure_ascii=False, indent=2)
    print(f"Result saved to: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-clip minADE evaluation for Alpamayo 1.5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clip_id",
        type=str,
        default="030c760c-ae38-49aa-9ad8-f5650a545d26",
        help="PhysicalAI-AV clip ID.",
    )
    parser.add_argument(
        "--t0_us",
        type=int,
        default=5_100_000,
        help="Evaluation timestamp in microseconds.",
    )
    parser.add_argument("--model_name", type=str, default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "sdpa", "flash_attention_2", "eager"],
        help="Attention backend.  Use sdpa if flash-attn is unavailable.",
    )
    parser.add_argument(
        "--num_traj_samples",
        type=int,
        default=1,
        help="Number of sampled trajectories per inference call.",
    )
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_generation_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_stream",
        action="store_true",
        help="Disable HuggingFace dataset streaming.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/single_clip_eval.json",
        help="Path to write the JSON result file.",
    )
    return parser


def main() -> None:
    """Run single-clip minADE evaluation from the command line."""
    args = _build_parser().parse_args()
    cfg = EvalConfig(
        clip_id=args.clip_id,
        t0_us=args.t0_us,
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        num_traj_samples=args.num_traj_samples,
        top_p=args.top_p,
        temperature=args.temperature,
        max_generation_length=args.max_generation_length,
        seed=args.seed,
        maybe_stream=not args.no_stream,
        output=args.output,
    )
    SingleClipEvaluator(cfg).run()


if __name__ == "__main__":
    main()
