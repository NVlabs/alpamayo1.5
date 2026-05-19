<div align="center">

# 🏔️ Alpamayo 1.5

### Supercharging Autonomous Driving with Interactive, Steerable Reasoning

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Alpamayo--1.5--10B-blue)](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

</div>

## Updates

- [May 2026] SFT and RL post-training scripts are available in [Alpamayo Recipes](https://github.com/NVlabs/alpamayo-recipes): [Alpamayo 1.5 SFT](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_5_sft) and [Alpamayo 1.x RL post-training](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_x_rl).

**📖 Please read the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) first!**
The model card contains comprehensive details on model architecture, inputs/outputs, licensing, and tested hardware configurations. This GitHub README focuses on setup, usage, and frequently asked questions.

## Prerequisites

- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit 12.x** with `nvcc` (required to compile `flash-attn` from source). If you don't have it, see [Troubleshooting](#flash-attention-issues) for a fallback using PyTorch's built-in SDPA.
- **Python 3.12**

### Hardware requirements

| Configuration                                           | VRAM   |
| ------------------------------------------------------- | ------ |
| Single-sample inference (`num_traj_samples=1`)          | ~24 GB |
| Multi-sample inference (`num_traj_samples=16`)          | ~40 GB |
| Multi-sample inference with CFG (`num_traj_samples=16`) | ~60 GB |

Measured on an NVIDIA H100 80GB GPU.

## Getting Started

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Set up the environment

```bash
uv venv a1_5_venv
source a1_5_venv/bin/activate
uv sync --active
```

> **Note:** If `uv sync` fails on `flash-attn`, see [Troubleshooting](#flash-attention-issues) below.

### 3. Authenticate with HuggingFace

The model and dataset require access to gated resources. Request access here:

- 🤗 [PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- 🤗 [Alpamayo-1.5-10B Model](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

Then authenticate:

```bash
hf auth login
```

Get your token at: https://huggingface.co/settings/tokens

> **Note:** The `physical_ai_av` package (auto-installed via dependencies) streams data from the HuggingFace dataset. You must have accepted the dataset access request above before running inference.

## Running Inference

### Test script

NOTE: This script will download both some example data (relatively small) and the model weights (22 GB).
The latter can be particularly slow depending on network bandwidth.
For reference, it takes around 2.5 minutes on a 100 MB/s wired connection.

```bash
python src/alpamayo1_5/test_inference.py
```

In case you would like to obtain more trajectories and reasoning traces, please feel free to increase
the `num_traj_samples` argument in the script.

### Interactive notebooks

We provide notebooks that demonstrate the different capabilities of Alpamayo 1.5 under `notebooks/`, including standard model inference, incorporating navigation guidance, modifying the number of cameras, and visual question answering.

### Inference methods

Alpamayo 1.5 provides two inference methods:

- **`sample_trajectories_from_data_with_vlm_rollout`** -- Full pipeline: the VLM generates chain-of-causation reasoning, then a diffusion expert produces trajectory predictions conditioned on the VLM's hidden states. This is the primary inference method used by the test script and most notebooks.

- **`generate_text`** -- Text-only generation for visual question answering (VQA). Returns extracted text fields.

## Fine-tuning and Post-training Recipes

SFT and RL post-training scripts are maintained in [Alpamayo Recipes](https://github.com/NVlabs/alpamayo-recipes):

- [Alpamayo 1.5 SFT](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_5_sft)
- [Alpamayo 1.x RL post-training](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_x_rl), including Alpamayo 1.5

## Project Structure

```
alpamayo_1.5_release/
├── notebooks/
│   ├── inference.ipynb                  # Standard model inference
│   ├── inference_cam_num.ipynb          # Inference with different camera counts
│   ├── inference_nav.ipynb              # Inference with navigation guidance
│   └── inference_vqa.ipynb              # Visual question answering
├── src/
│   └── alpamayo1_5/
│       ├── action_space/
│       │   └── ...                      # Action space definitions
│       ├── diffusion/
│       │   └── ...                      # Diffusion model components
│       ├── geometry/
│       │   └── ...                      # Geometry utilities and modules
│       ├── models/
│       │   ├── ...                      # Model components and utils functions
│       ├── __init__.py                  # Package marker
│       ├── config.py                    # Model and experiment configuration
│       ├── helper.py                    # Utility functions
│       ├── load_physical_aiavdataset.py # Dataset loader
│       ├── test_inference.py            # Inference test script
├── pyproject.toml                       # Project dependencies
└── uv.lock                              # Locked dependency versions
```

## Troubleshooting

### Flash Attention issues

The model uses Flash Attention 2 by default. `flash-attn` requires CUDA Toolkit (specifically `nvcc`) at build time. If you see build errors during `uv sync`:

**Option A: Install without flash-attn and use SDPA fallback**

```bash
uv sync --active --no-install-package flash-attn
```

Then load the model with PyTorch's built-in scaled dot-product attention:

```python
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

model = Alpamayo1_5.from_pretrained(
    "nvidia/Alpamayo-1.5-10B",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda")
```

**Option B: Install CUDA Toolkit, then retry**

Install CUDA Toolkit 12.x (e.g., via your package manager or [NVIDIA's install guide](https://developer.nvidia.com/cuda-downloads)), ensure `nvcc` is on your PATH, then re-run:

```bash
uv sync --active
```

## Frequently Asked Questions (FAQ)

<details>
<summary><strong>How does Alpamayo 1.5 relate to Alpamayo 1?</strong></summary>

Alpamayo 1.5 expands upon the architecture released in Alpamayo 1 and fully realizes what is described in our paper [*"Alpamayo 1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail
"*](https://arxiv.org/abs/2511.00088). Specifically:

| Feature                                 | Description                                                      | Alpamayo 1             | Alpamayo 1.5       |
| --------------------------------------- | ---------------------------------------------------------------- | ---------------------- | ------------------ |
| **Chain-of-Causation (CoC) reasoning**  | Hybrid auto-labeling with human in the loop for reasoning traces | ✅ Included            | ✅ Included        |
| **Vision-Language-Action architecture** | Cosmos-Reason backbone + action expert                           | ✅ Included            | ✅ Included        |
| **Trajectory prediction**               | 6.4s horizon, 64 waypoints at 10 Hz                              | ✅ Supported           | ✅ Supported       |
| **RL post-training**                    | Reinforcement learning for reasoning/action consistency          | ❌ Not RL post-trained | ✅ RL post-trained |
| **Navigation conditioning**             | Explicit navigation inputs                                       | ❌ Not supported       | ✅ Supported       |
| **General VQA**                         | Supports visual question answering                               | ❌ Not supported       | ✅ Supported       |
| **Flexible multi-camera support**       | Supports a variable number of input cameras                      | ❌ Not supported       | ✅ Supported       |

</details>

<details>
<summary><strong>Does Alpamayo 1.5 accept navigation inputs?</strong></summary>

Yes! Please see `notebooks/inference_nav.ipynb` for examples.

</details>

<details>
<summary><strong>Does Alpamayo 1.5 support general VQA?</strong></summary>

Yes! Please see `notebooks/inference_vqa.ipynb` for examples.

</details>

<details>
<summary><strong>Was Alpamayo 1.5 post-trained with Reinforcement Learning (RL)?</strong></summary>

Yes! Alpamayo 1.5 has undergone RL post-training, achieving improvements in reasoning quality and reasoning-trajectory alignment as a result.

</details>

<details>
<summary><strong>Does Alpamayo 1.5 accept different numbers of cameras?</strong></summary>

Yes! Please see `notebooks/inference_cam_num.ipynb` for examples. Note that model accuracy may degrade with fewer cameras, the magnitude of which will depend on the specific scenario. For instance, it is expected that Alpamayo 1.5 would struggle to see cross-traffic in a right turn if only provided a front-facing camera.

</details>

<details>
<summary><strong>What are the minimum GPU requirements?</strong></summary>

You need an NVIDIA GPU with at least **24 GB VRAM** for inference. Tested configurations include RTX 3090, A100, H100, and B200. Running on GPUs with less memory (e.g., 16 GB) will likely result in CUDA out-of-memory errors. Please refer to our [hardware requirements](#hardware-requirements) for more information.

</details>

<details>
<summary><strong>Can I use this model in production / commercial applications?</strong></summary>

No. The model weights are released under a **non-commercial license**. This release is intended for research, experimentation, and evaluation purposes only. See the [License](#license) section and the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) for details.

</details>

## License

Apache License 2.0 - see [LICENSE](./LICENSE) for details.

## Disclaimer

Alpamayo 1.5 is a pre-trained reasoning model designed to accelerate research and development in the autonomous vehicle (AV) domain. It is intended to serve as a foundation for a range of AV-related use cases-from instantiating an end-to-end backbone for autonomous driving to enabling reasoning-based auto-labeling tools. In short, it should be viewed as a building block for developing customized AV applications.

Important notes:

- Alpamayo 1.5 is provided solely for research, experimentation, and evaluation purposes.
- Alpamayo 1.5 is not a fully fledged driving stack. Among other limitations, it lacks access to critical real-world sensor inputs, does not incorporate required diverse and redundant safety mechanisms, and has not undergone automotive-grade validation for deployment.

By using this model, you acknowledge that it is a research tool intended to support scientific inquiry, benchmarking, and exploration—not a substitute for a certified AV stack. The developers and contributors disclaim any responsibility or liability for the use of the model or its outputs.

## Citation

If you use Alpamayo 1.5 in your research, please cite:

```bibtex
@article{nvidia2025alpamayo,
      title={{Alpamayo-R1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
      author={NVIDIA and Yan Wang and Wenjie Luo and Junjie Bai and Yulong Cao and Tong Che and Ke Chen and Yuxiao Chen and Jenna Diamond and Yifan Ding and Wenhao Ding and Liang Feng and Greg Heinrich and Jack Huang and Peter Karkus and Boyi Li and Pinyi Li and Tsung-Yi Lin and Dongran Liu and Ming-Yu Liu and Langechuan Liu and Zhijian Liu and Jason Lu and Yunxiang Mao and Pavlo Molchanov and Lindsey Pavao and Zhenghao Peng and Mike Ranzinger and Ed Schmerling and Shida Shen and Yunfei Shi and Sarah Tariq and Ran Tian and Tilman Wekel and Xinshuo Weng and Tianjun Xiao and Eric Yang and Xiaodong Yang and Yurong You and Xiaohui Zeng and Wenyuan Zhang and Boris Ivanovic and Marco Pavone},
      year={2025},
      journal={arXiv preprint arXiv:2511.00088},
}
```
