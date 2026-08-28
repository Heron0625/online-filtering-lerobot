#!/usr/bin/env python

from typing import Any

import torch

from lerobot.policies.pi05.processor_pi05 import (
    Pi05PrepareStateTokenizerProcessorStep,
    make_pi05_pre_post_processors,
)
from lerobot.processor import PolicyProcessorPipeline

from .configuration_online_filtering_pi05 import OnlineFilteringPI05Config


def make_online_filtering_pi05_pre_post_processors(
    config: OnlineFilteringPI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """Reuse PI0.5 normalization/tokenization with the initial state in the prompt."""
    return make_pi05_pre_post_processors(config, dataset_stats)


def reconcile_online_filtering_pi05_pretrained_processors(
    config: OnlineFilteringPI05Config,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Prevent a stock PI0.5 processor from tokenizing a future state frame."""
    state_steps = [
        step for step in preprocessor.steps if isinstance(step, Pi05PrepareStateTokenizerProcessorStep)
    ]
    if len(state_steps) != 1:
        raise ValueError(
            "online_filtering_pi05 requires exactly one PI0.5 state tokenizer processor step"
        )
    state_steps[0].temporal_index = config.prompt_state_temporal_index
    return preprocessor, postprocessor
