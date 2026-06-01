# Evaluation

This subpackage provides utilities for evaluating Alpamayo 1.5 trajectory
prediction accuracy.

## Contents

| File | Description |
|------|-------------|
| `minade.py` | `MinADECalculator` — computes minADE over the XY plane for multi-sample predictions |
| `single_clip_eval.py` | `SingleClipEvaluator` + CLI — runs the full load → infer → score pipeline on a single clip |

## Metric: minADE

**Average Displacement Error (ADE)** for sample *k*:

```
ADE_k = mean_t || pred_xy_k[t] - gt_xy[t] ||_2
```

**minADE** selects the best sample across *K* predicted trajectories:

```
minADE = min_k ADE_k
```

Only the XY plane is used (Z is ignored), consistent with the inline check
in `test_inference.py`.

## Quick Start

### Single-clip evaluation (CLI)

```bash
python src/alpamayo1_5/evaluation/single_clip_eval.py
```

This runs on the default clip (`030c760c-ae38-49aa-9ad8-f5650a545d26`,
`t0_us=5_100_000`) and writes results to `outputs/single_clip_eval.json`.

#### Common options

```bash
python src/alpamayo1_5/evaluation/single_clip_eval.py \
    --clip_id 030c760c-ae38-49aa-9ad8-f5650a545d26 \
    --t0_us 5100000 \
    --num_traj_samples 16 \
    --attn_implementation sdpa \
    --output outputs/my_run.json
```

| Flag | Default | Notes |
|------|---------|-------|
| `--clip_id` | `030c760c-ae38-49aa-9ad8-f5650a545d26` | PhysicalAI-AV clip ID |
| `--t0_us` | `5100000` | Evaluation timestamp (µs) |
| `--model_name` | `nvidia/Alpamayo-1.5-10B` | HF model ID or local path |
| `--num_traj_samples` | `1` | Trajectory samples per call |
| `--dtype` | `bfloat16` | `bfloat16 \| float16 \| float32` |
| `--attn_implementation` | *(auto)* | Use `sdpa` if `flash-attn` is unavailable |
| `--seed` | `42` | CUDA RNG seed for reproducibility |
| `--no_stream` | *(streaming on)* | Disable HuggingFace dataset streaming |
| `--output` | `outputs/single_clip_eval.json` | Result file path |

### Python API

```python
from alpamayo1_5.evaluation import EvalConfig, SingleClipEvaluator

cfg = EvalConfig(
    clip_id="030c760c-ae38-49aa-9ad8-f5650a545d26",
    num_traj_samples=16,
    seed=42,
)
result = SingleClipEvaluator(cfg).run()
print(f"minADE: {result['minADE']:.4f} m")
```

### Using `MinADECalculator` directly

```python
from alpamayo1_5.evaluation import MinADECalculator

calculator = MinADECalculator()
result = calculator.calculate(pred_xyz=pred_xyz, gt_future_xyz=data["ego_future_xyz"])
print(result.min_ade, result.all_ade)
```

## Example output

Console:

```
[1/4] Loading dataset: clip_id=030c760c-ae38-49aa-9ad8-f5650a545d26, t0_us=5100000
[2/4] Loading model...
[3/4] Running inference...
[4/4] Calculating minADE...
Result saved to: outputs/single_clip_eval.json

========== Evaluation Result ==========
clip_id:          030c760c-ae38-49aa-9ad8-f5650a545d26
t0_us:            5100000
num_traj_samples: 1
minADE:           0.373767 m
all_ADE:          [0.373767...]

========== Predicted Trajectory (index=0) ==========
  [00] x=   0.000  y=   0.000  z=   0.000
  [01] x=   0.412  y=   0.023  z=  -0.001
  ...

========== Chain-of-Causation ==========
Nudge to the left to clear the construction equipment blocking the right side of our lane
```

`outputs/single_clip_eval.json`:

```json
{
  "clip_id": "030c760c-ae38-49aa-9ad8-f5650a545d26",
  "t0_us": 5100000,
  "num_traj_samples": 1,
  "minADE": 0.3737667798995972,
  "all_ADE": [0.3737667798995972],
  "cot": [["Nudge to the left to clear the construction equipment ..."]]
}
```

## Prerequisites

Follow the [top-level README](../../../../README.md) to set up the environment
and authenticate with HuggingFace before running evaluation.
