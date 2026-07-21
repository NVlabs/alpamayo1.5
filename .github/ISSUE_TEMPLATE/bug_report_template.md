---
name: Bug report
about: Create a bug report to help us improve Alpamayo
title: "[BUG]"
labels: "? - Needs Triage, bug"
assignees: 'yesfandiari'

---

**Describe the bug**
A clear and concise description of what the bug is.

**Steps/Code to reproduce bug**
Follow this guide http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports to craft a minimal bug report. This helps us reproduce the issue and resolve it more quickly.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment overview (please complete the following information)**
 - Deployment: [local from source (uv), Slurm, or Cloud (specify provider)]
 - Install method: `uv venv` + `uv sync` — paste `uv --version` and Python version (3.12 expected)
 - flash-attn: compiled from source (CUDA Toolkit 12.x + `nvcc`) or PyTorch SDPA fallback?
 - Model checkpoint: [e.g. nvidia/Alpamayo-1.5-10B]; HuggingFace gated access granted? (yes/no)
 - Dataset: PhysicalAI-Autonomous-Vehicles access granted? (yes/no)

**Environment details**
 - Hardware: GPU type(s) and VRAM (~24 GB single-sample, ~40 GB multi-sample, ~60 GB with CFG; H100 80GB tested), number of GPUs
 - Operating System
 - CUDA Toolkit / NVIDIA driver version (from `nvcc --version` and `nvidia-smi`)
 - Inference settings: `num_traj_samples`, CFG on/off, and script/notebook used (`src/alpamayo1_5/test_inference.py` or a notebook under `notebooks/`)

**Additional context**
Add any other context about the problem here.
