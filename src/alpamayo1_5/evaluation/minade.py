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

"""Minimum Average Displacement Error (minADE) computation for trajectory evaluation."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MinADEResult:
    """Result of a minADE computation.

    Attributes:
        min_ade: Minimum ADE across all predicted trajectory samples, in meters.
        all_ade: Per-sample ADE values, in meters.
    """

    min_ade: float
    all_ade: list[float]


class MinADECalculator:
    """Computes minADE over the XY plane for multi-sample trajectory predictions.

    The Average Displacement Error (ADE) for sample ``k`` is::

        ADE_k = mean_t || pred_xy_k[t] - gt_xy[t] ||_2

    The minimum is then taken across all ``K`` samples::

        minADE = min_k ADE_k
    """

    def calculate(
        self,
        pred_xyz: torch.Tensor,
        gt_future_xyz: torch.Tensor,
    ) -> MinADEResult:
        """Compute minADE between predicted and ground-truth trajectories.

        Args:
            pred_xyz: Predicted trajectory tensor.  Accepted shapes:
                ``(B, G, K, T, 3)``, ``(B, K, T, 3)``, or ``(K, T, 3)``.
            gt_future_xyz: Ground-truth future trajectory, shape ``(1, 1, T, 3)``.

        Returns:
            MinADEResult with ``min_ade`` and per-sample ``all_ade``.
        """
        pred_xy = self._extract_pred_xy(pred_xyz)  # (K, T, 2)
        gt_xy = self._extract_gt_xy(gt_future_xyz)  # (T, 2)

        diff = np.linalg.norm(pred_xy - gt_xy[None, :, :], axis=-1)  # (K, T)
        ade_per_sample = diff.mean(axis=-1)  # (K,)

        return MinADEResult(
            min_ade=float(ade_per_sample.min()),
            all_ade=[float(x) for x in ade_per_sample.tolist()],
        )

    def _extract_gt_xy(self, gt_future_xyz: torch.Tensor) -> np.ndarray:
        """Extract XY coordinates from the ground-truth tensor.

        Args:
            gt_future_xyz: Shape ``(1, 1, T, 3)``.

        Returns:
            Array of shape ``(T, 2)``.
        """
        gt = gt_future_xyz.detach().cpu().numpy()
        assert gt.ndim == 4, f"{gt.ndim=}, expected (1, 1, T, 3)"
        return gt[0, 0, :, :2]

    def _extract_pred_xy(self, pred_xyz: torch.Tensor) -> np.ndarray:
        """Extract XY coordinates from the model output tensor.

        Handles the shapes produced by
        ``sample_trajectories_from_data_with_vlm_rollout``:

        * ``(B, G, K, T, 3)`` — standard multi-sample output
        * ``(B, K, T, 3)`` — without trajectory-group dimension
        * ``(K, T, 3)`` — unbatched

        Args:
            pred_xyz: Model output trajectory tensor.

        Returns:
            Array of shape ``(K, T, 2)``.
        """
        pred = pred_xyz.detach().cpu().numpy()

        if pred.ndim == 5:
            return pred[0, 0, :, :, :2]
        if pred.ndim == 4:
            return pred[0, :, :, :2]
        if pred.ndim == 3:
            return pred[:, :, :2]

        raise ValueError(f"Unsupported pred_xyz shape: {pred.shape}")
