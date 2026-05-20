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

"""Base Reasoning VLA model implementation for Alpamayo 1.5 release."""

import copy
import logging
from typing import Any

import einops
import hydra.utils as hyu
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from alpamayo1_5.models.token_utils import extract_text_tokens

logger = logging.getLogger(__name__)

# Constants
IGNORE_INDEX = -100
TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}
SPECIAL_TOKENS_KEYS = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "_padding_0",
    "image_end",
    "traj_history_start",
    "_padding_1",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "_padding_2",
    "_padding_3",
    "traj_future_start",
    "_padding_4",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "_padding_5",
    "_padding_6",
    "_padding_7",
    "_padding_8",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]
SPECIAL_TOKENS = {k: "<|" + k + "|>" for k in SPECIAL_TOKENS_KEYS}


def _recursive_setattr(obj: Any, attr: str, value: Any) -> None:
    """Recursively set attribute on object and all its children."""
    setattr(obj, attr, value)
    for child in getattr(obj, "children", lambda: [])():
        _recursive_setattr(child, attr, value)


def replace_pad_token(input_ids: torch.Tensor, new_ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Replace pad tokens in input_ids with new token values."""
    mask = input_ids == pad_idx
    return input_ids.masked_scatter(mask, new_ids)


def tokenize_history_trajectory(
    tokenizer: Any, traj_data: dict[str, Any], start_idx: int = 0
) -> torch.Tensor:
    """Tokenize the history trajectory with prefix shape of (B, n_traj, ...).

    Args:
        tokenizer: Trajectory tokenizer with encode method
        traj_data: dict containing "ego_history_xyz" and "ego_history_rot"
        start_idx: start of token index of the history trajectory tokens

    Returns:
        torch.Tensor: [B, n_traj * tokens_per_history_traj]
    """
    assert "ego_history_xyz" in traj_data
    assert traj_data["ego_history_xyz"].ndim == 4, "ego_history_xyz must be 4D of [B, n_traj, T, 3]"

    B = traj_data["ego_history_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)

    hist_idx = (
        tokenizer.encode(
            hist_xyz=hist_xyz[:, :1],
            hist_rot=hist_rot[:, :1],
            fut_xyz=hist_xyz,  # note hist_xyz is passed to fut_xyz as it's encoding history.
            fut_rot=hist_rot,
        )
        + start_idx
    )  # [B*n_traj, tokens_per_history_traj]
    hist_idx = einops.rearrange(hist_idx, "(b n_traj) n -> b (n_traj n)", b=B)

    return hist_idx


class TrajectoryFusionMixin:
    """Mixin class providing trajectory fusion functionality."""

    def _validate_mixin_requirements(self, require_future: bool = False) -> dict[str, Any]:
        """Validate that all required mixin attributes are present."""
        hist_traj_tokenizer = getattr(self, "hist_traj_tokenizer", None)
        if hist_traj_tokenizer is None:
            raise AttributeError("TrajectoryFusionMixin requires 'hist_traj_tokenizer' attribute")

        hist_token_start_idx = getattr(self, "hist_token_start_idx", None)
        if hist_token_start_idx is None:
            raise AttributeError("TrajectoryFusionMixin requires 'hist_token_start_idx' attribute")

        config = getattr(self, "config", None)
        if config is None or not hasattr(config, "traj_token_ids"):
            raise AttributeError("TrajectoryFusionMixin requires 'config' with 'traj_token_ids'")

        result = {
            "hist_traj_tokenizer": hist_traj_tokenizer,
            "hist_token_start_idx": hist_token_start_idx,
            "config": config,
        }

        if require_future:
            traj_tokenizer = getattr(self, "traj_tokenizer", None)
            if traj_tokenizer is None:
                raise AttributeError("Requires 'traj_tokenizer' attribute for future trajectories")

            future_token_start_idx = getattr(self, "future_token_start_idx", None)
            if future_token_start_idx is None:
                raise AttributeError(
                    "Requires 'future_token_start_idx' attribute for future trajectories"
                )

            result.update(
                {
                    "traj_tokenizer": traj_tokenizer,
                    "future_token_start_idx": future_token_start_idx,
                }
            )

        return result

    def fuse_traj_tokens(
        self, input_ids: torch.Tensor, traj_data: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """Fuse the trajectory tokens into the input ids.

        Args:
            input_ids: [B, n_token]
            traj_data: dict containing ego_history_xyz, ego_history_rot, etc.

        Returns:
            input_ids: [B, n_token] with trajectory tokens fused
        """
        if (
            traj_data is None
            or traj_data.get("ego_history_xyz") is None
            or traj_data.get("ego_history_rot") is None
        ):
            return input_ids

        has_future = "ego_future_xyz" in traj_data and traj_data["ego_future_xyz"] is not None
        attrs = self._validate_mixin_requirements(require_future=has_future)

        hist_idx = tokenize_history_trajectory(
            attrs["hist_traj_tokenizer"], traj_data, attrs["hist_token_start_idx"]
        )
        input_ids = replace_pad_token(
            input_ids, hist_idx, attrs["config"].traj_token_ids["history"]
        )

        return input_ids


class ReasoningVLAConfig(PretrainedConfig):
    """Configuration for the ReasoningVLA model."""

    model_type = "alpamayo_reasoning_vla"

    def __init__(
        self,
        vlm_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        vlm_backend: str = "qwenvl3",
        traj_tokenizer_cfg: dict[str, Any] | None = None,
        hist_traj_tokenizer_cfg: dict[str, Any] | None = None,
        traj_vocab_size: int = 768,
        tokens_per_history_traj: int = 16,
        tokens_per_future_traj: int = 64,
        model_dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        add_special_tokens: bool = False,
        **kwargs: Any,
    ) -> None:
        if attn_implementation is None:
            attn_implementation = "flash_attention_2"
        kwargs["attn_implementation"] = attn_implementation
        super().__init__(**kwargs)

        self.vlm_name_or_path = vlm_name_or_path
        self.vlm_backend = vlm_backend.lower()
        self.model_dtype = model_dtype
        self.attn_implementation = attn_implementation

        self.traj_tokenizer_cfg = traj_tokenizer_cfg
        self.hist_traj_tokenizer_cfg = hist_traj_tokenizer_cfg

        self.traj_vocab_size = traj_vocab_size
        self.tokens_per_history_traj = tokens_per_history_traj
        self.tokens_per_future_traj = tokens_per_future_traj
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.add_special_tokens = add_special_tokens

        # Initialize VLM-specific configurations
        self._initialize_vlm_config()

    def _initialize_vlm_config(self) -> None:
        """Initialize VLM-specific configuration based on backend type."""
        if self.vlm_name_or_path is None:
            return

        processor = self._build_processor()
        self.vocab_size = len(processor.tokenizer)
        self.traj_token_start_idx = processor.tokenizer.traj_token_start_idx
        self.traj_token_ids = processor.tokenizer.traj_token_ids

    def _build_processor(self) -> AutoProcessor:
        """Build the processor with trajectory tokens."""
        processor_kwargs = {}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels

        processor = AutoProcessor.from_pretrained(self.vlm_name_or_path, **processor_kwargs)
        tokenizer = processor.tokenizer

        # Add traj tokens to the tokenizer
        if self.traj_vocab_size is not None:
            discrete_tokens = [f"<i{v}>" for v in range(self.traj_vocab_size)]
            num_new_tokens = tokenizer.add_tokens(discrete_tokens)
            assert len(discrete_tokens) == num_new_tokens
            tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
            tokenizer.traj_token_end_idx = tokenizer.convert_tokens_to_ids(
                f"<i{self.traj_vocab_size - 1}>"
            )

        if self.add_special_tokens:
            special_tokens = list(SPECIAL_TOKENS.values())
            tokenizer.add_tokens(special_tokens, special_tokens=True)
        else:
            tokenizer.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)

        tokenizer.traj_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v) for k, v in TRAJ_TOKEN.items()
        }

        return processor


class ReasoningVLA(PreTrainedModel, TrajectoryFusionMixin):
    """Reasoning Vision-Language-Action model."""

    config_class: type[ReasoningVLAConfig] = ReasoningVLAConfig
    base_model_prefix: str = "vlm"
    _supports_sdpa: bool = True
    _supports_flash_attn: bool = True

    def __init__(
        self,
        config: ReasoningVLAConfig,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
        print_param_count: bool = True,
    ) -> None:
        super().__init__(config)

        if pretrained_modules is not None:
            for module in pretrained_modules.values():
                if not isinstance(module, torch.nn.Module):
                    continue
                _recursive_setattr(module, "_is_hf_initialized", True)
        else:
            pretrained_modules = {}

        # Initialize VLM backbone
        self._initialize_vlm_backbone(config, pretrained_modules, original_vocab_size)

        # Initialize trajectory tokenizers
        self._initialize_trajectory_tokenizers(config, pretrained_modules)

        # Build tokenizer
        self.tokenizer = self._build_tokenizer(config)
        self.special_token_ids = {
            k: self.tokenizer.convert_tokens_to_ids(v) for k, v in SPECIAL_TOKENS.items()
        }

        # Log parameter count
        if print_param_count:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

    def _build_tokenizer(self, config: ReasoningVLAConfig) -> Any:
        """Build tokenizer with trajectory tokens."""
        processor_kwargs = {}
        if config.min_pixels is not None:
            processor_kwargs["min_pixels"] = config.min_pixels
        if config.max_pixels is not None:
            processor_kwargs["max_pixels"] = config.max_pixels

        processor = AutoProcessor.from_pretrained(config.vlm_name_or_path, **processor_kwargs)
        tokenizer = processor.tokenizer

        if config.traj_vocab_size is not None:
            discrete_tokens = [f"<i{v}>" for v in range(config.traj_vocab_size)]
            tokenizer.add_tokens(discrete_tokens)
            tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")

        if config.add_special_tokens:
            tokenizer.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)
        else:
            tokenizer.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)

        tokenizer.traj_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v) for k, v in TRAJ_TOKEN.items()
        }

        return tokenizer

    def _initialize_vlm_backbone(
        self,
        config: ReasoningVLAConfig,
        pretrained_modules: dict[str, Any],
        original_vocab_size: int | None = None,
    ) -> None:
        """Initialize the VLM backbone based on configuration."""
        if "vlm" in pretrained_modules:
            self.vlm = pretrained_modules["vlm"]
            self.original_vocab_size = original_vocab_size
        else:
            self._initialize_qwenvl3_vlm(config)

    def _initialize_qwenvl3_vlm(self, config: ReasoningVLAConfig) -> None:
        """Initialize Qwen3-VL VLM backbone.

        Qwen3-VL uses Qwen3VLForConditionalGeneration from transformers.
        See: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
        """
        vlm_config = Qwen3VLConfig.from_pretrained(
            config.vlm_name_or_path,
            dtype=config.model_dtype,
            attn_implementation=config.attn_implementation,
        )
        self.original_vocab_size = vlm_config.text_config.vocab_size
        vlm_config.text_config.vocab_size = config.vocab_size
        vlm_config.vocab_size = config.vocab_size
        self.vlm = Qwen3VLForConditionalGeneration(vlm_config)

    def _initialize_trajectory_tokenizers(
        self, config: ReasoningVLAConfig, pretrained_modules: dict[str, Any]
    ) -> None:
        """Initialize trajectory tokenizers."""
        if "traj_tokenizer" in pretrained_modules:
            self.traj_tokenizer = pretrained_modules["traj_tokenizer"]
        elif config.traj_tokenizer_cfg is not None:
            self.traj_tokenizer = hyu.instantiate(config.traj_tokenizer_cfg, load_weights=False)
        else:
            self.traj_tokenizer = None

        self.future_token_start_idx = self.hist_token_start_idx = config.traj_token_start_idx

        if config.hist_traj_tokenizer_cfg is not None:
            self.hist_traj_tokenizer = hyu.instantiate(config.hist_traj_tokenizer_cfg)
            if self.traj_tokenizer is not None:
                self.hist_token_start_idx += self.traj_tokenizer.vocab_size
        else:
            self.hist_traj_tokenizer = self.traj_tokenizer

    @classmethod
    def from_pretrained_submodules(
        cls,
        config: ReasoningVLAConfig,
    ) -> "ReasoningVLA":
        """Load submodules with pretrained submodules and initialize the model."""
        pretrained_modules = {}

        # Load VLM
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            config.vlm_name_or_path,
            dtype=config.model_dtype,
            attn_implementation=config.attn_implementation,
        )

        original_vocab_size = vlm.config.text_config.vocab_size
        vlm.resize_token_embeddings(config.vocab_size)
        vlm.config.text_config.vocab_size = config.vocab_size
        vlm.config.vocab_size = config.vocab_size
        pretrained_modules["vlm"] = vlm

        if config.traj_tokenizer_cfg is not None:
            traj_tokenizer = hyu.instantiate(config.traj_tokenizer_cfg)
            pretrained_modules["traj_tokenizer"] = traj_tokenizer

        return cls(
            config,
            pretrained_modules=pretrained_modules,
            original_vocab_size=original_vocab_size,
        )

    def get_output_embeddings(self) -> torch.nn.Module:
        """Get the output embeddings of the model."""
        return self.vlm.get_output_embeddings()

    def get_input_embeddings(self) -> torch.nn.Module:
        """Get the input embeddings of the model."""
        return self.vlm.language_model.embed_tokens

    def tie_weights(self) -> None:
        """Delegate weight tying to the nested VLM model."""
        if hasattr(self.vlm, "tie_weights"):
            self.vlm.tie_weights()

    def generate_text(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_samples: int = 1,
        max_generation_length: int = 256,
    ) -> dict[str, np.ndarray]:
        """Generate text responses from tokenized inputs.

        This method runs autoregressive VLM generation and extracts structured text outputs from the generated token sequence.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_samples: Number of response samples to generate.
            max_generation_length: Maximum number of new tokens to generate.

        Returns:
            Dict mapping text-field names to arrays of extracted strings,
            each with shape ``[B, num_samples]``. Keys include ``"cot"``,
            ``"meta_action"``, and ``"answer"``.
        """
        data = copy.deepcopy(data)
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")

        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        generated = self.vlm.generate(
            input_ids=input_ids, **tokenized_data, generation_config=generation_config
        )
        generated_tokens = generated.sequences[:, input_ids.shape[1] :]

        extra = extract_text_tokens(self.tokenizer, generated_tokens)
        for key in extra:
            extra[key] = np.array(extra[key]).reshape([input_ids.shape[0], num_samples])
        return extra
