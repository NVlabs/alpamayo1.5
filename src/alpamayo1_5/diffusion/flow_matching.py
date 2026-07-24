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

from typing import Literal

import torch
from alpamayo1_5.diffusion.base import BaseDiffusion, StepFn


class FlowMatching(BaseDiffusion):
    """Flow Matching model.

    References:
    Flow Matching for Generative Modeling
        https://arxiv.org/pdf/2210.02747
    Guided Flows for Generative Modeling and Decision Making
        https://arxiv.org/pdf/2311.13443
    """

    def __init__(
        self,
        int_method: Literal["euler"] = "euler",
        num_inference_steps: int = 10,
        inference_guidance_weight: float = 1.0,
        *args,
        **kwargs,
    ):
        """Initialize the FlowMatching model.

        Args:
            int_method: The integration method used in inference.
            num_inference_steps: The number of inference steps.
            inference_guidance_weight: The weight of the guidance during inference.
        """
        super().__init__(*args, **kwargs)
        self.int_method = int_method
        self.num_inference_steps = num_inference_steps
        self.inference_guidance_weight = inference_guidance_weight

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        step_fn: StepFn,
        unguided_step_fn: StepFn | None = None,
        device: torch.device = torch.device("cpu"),
        return_all_steps: bool = False,
        inference_step: int | None = None,
        int_method: Literal["euler"] | None = None,
        use_classifier_free_guidance: bool | None = None,
        inference_guidance_weight: float | None = None,
        temperature: float = 1.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sample data from the model.

        Args:
            batch_size: The batch size.
            step_fn: The denoising step function that takes a noisy x and a
                timestep t and returns either a denoised x, a vector field or noise depending on
                the prediction type of the diffusion model. (assumed to be with guidance if the
                diffusion model uses classifier free guidance)
            unguided_step_fn: The denoising step function. (assumed to be without guidance)
            device: The device to use.
            return_all_steps: Whether to return all steps.
            inference_step: The number of inference steps. (override self.num_inference_steps)
            int_method: The integration method used in inference. (override self.int_method)
            use_classifier_free_guidance: Whether to use classifier free guidance.
            inference_guidance_weight: The weight of the guidance during inference. (override self.inference_guidance_weight)
            temperature: The temperature for controlling the initial noise. Note that using
                temperature < 1.0 will result in a more stable sampling with less diversity.
            generator: The generator used for initial noise. A list of
                generators (one per batch row) draws each row from its own
                generator, enabling batched per-row deterministic sampling.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                The final sampled tensor [B, *x_dims] if return_all_steps is False,
                otherwise a tuple of all sampled tensors [B, T, *x_dims] and the time steps [T].
        """
        int_method = int_method or self.int_method
        inference_step = inference_step or self.num_inference_steps
        if use_classifier_free_guidance is None:
            use_classifier_free_guidance = self.use_classifier_free_guidance
        if inference_guidance_weight is None:
            inference_guidance_weight = self.inference_guidance_weight
        if use_classifier_free_guidance and unguided_step_fn is None:
            raise ValueError("unguided_step_fn is required when using classifier free guidance")
        if int_method == "euler":
            return self._euler(
                batch_size=batch_size,
                step_fn=step_fn,
                unguided_step_fn=unguided_step_fn,
                device=device,
                return_all_steps=return_all_steps,
                inference_step=inference_step,
                inference_guidance_weight=inference_guidance_weight,
                use_classifier_free_guidance=use_classifier_free_guidance,
                temperature=temperature,
                generator=generator,
            )
        else:
            raise ValueError(f"Invalid integration method: {int_method}")

    @staticmethod
    def _guided_v(
        step_fn: StepFn,
        x: torch.Tensor,
        t: torch.Tensor,
        unguided_step_fn: StepFn,
        inference_guidance_weight: float,
    ) -> torch.Tensor:
        """Guided v for flow matching.

        eq 6 in https://arxiv.org/pdf/2311.13443
        Guided Flows for Generative Modeling and Decision Making

        Args:
            step_fn: The denoising step function. (assumed to be with guidance)
            x: The input tensor.
            t: The timestep.
            unguided_step_fn: The denoising step function. (assumed to be without guidance)
            inference_guidance_weight: The weight of the guidance during inference.
        """
        guided_v = step_fn(x=x, t=t)
        unguided_v = unguided_step_fn(x=x, t=t)
        return (1 - inference_guidance_weight) * unguided_v + inference_guidance_weight * guided_v

    def _euler(
        self,
        batch_size: int,
        step_fn: StepFn,
        unguided_step_fn: StepFn | None = None,
        device: torch.device = torch.device("cpu"),
        return_all_steps: bool = False,
        inference_step: int | None = None,
        inference_guidance_weight: float | None = None,
        use_classifier_free_guidance: bool | None = None,
        temperature: float = 1.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Euler integration for flow matching.

        Args:
            batch_size: The batch size.
            step_fn: The denoising step function that takes a noisy x and a
                timestep t and returns either a denoised x, a vector field or noise depending on
                the prediction type of the diffusion model. (assumed to be with guidance if the
                diffusion model uses classifier free guidance)
            unguided_step_fn: The denoising step function. (assumed to be without guidance)
            device: The device to use.
            return_all_steps: Whether to return all steps.
            inference_step: The inference step.
            inference_guidance_weight: The weight of the guidance during inference.
            use_classifier_free_guidance: Whether to use classifier free guidance.
            temperature: The temperature for controlling the initial noise. Note that using
                temperature < 1.0 will result in a more stable sampling with less diversity.
            generator: The generator used for initial noise. A list of
                generators (one per batch row) draws each row from its own
                generator, enabling batched per-row deterministic sampling.
        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                The final sampled tensor [B, *x_dims] if return_all_steps is False,
                otherwise a tuple of all sampled tensors [B, T, *x_dims] and the time steps [T].
        """
        x = self._initial_noise(batch_size, device=device, generator=generator) * temperature
        time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
        n_dim = len(self.x_dims)
        if return_all_steps:
            all_steps = [x]

        for i in range(inference_step):
            dt = time_steps[i + 1] - time_steps[i]
            dt = dt.view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            t_start = time_steps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            if use_classifier_free_guidance:
                v = self._guided_v(
                    step_fn=step_fn,
                    x=x,
                    t=t_start,
                    unguided_step_fn=unguided_step_fn,
                    inference_guidance_weight=inference_guidance_weight,
                )
            else:
                v = step_fn(x=x, t=t_start)
            x = x + dt * v
            if return_all_steps:
                all_steps.append(x)
        if return_all_steps:
            return torch.stack(all_steps, dim=1), time_steps
        return x

    def _initial_noise(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> torch.Tensor:
        """Draw the initial Gaussian noise [batch_size, *x_dims].

        A single generator (or None) seeds the whole batch at once. A list of
        generators draws each batch row from its own generator, enabling
        batched per-row deterministic sampling.
        """
        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"generator list length {len(generator)} must equal batch_size {batch_size}"
                )
            return torch.cat(
                [torch.randn(1, *self.x_dims, device=device, generator=g) for g in generator],
                dim=0,
            )
        return torch.randn(batch_size, *self.x_dims, device=device, generator=generator)
