from typing import Any

import torch

from lerobot.policies.xvla.processor_xvla import (
    make_xvla_pre_post_processors,
    reconcile_xvla_pretrained_processors,
)
from lerobot.processor import PolicyProcessorPipeline

from .configuration_online_filtering_xvla import OnlineFilteringXVLAConfig


def make_online_filtering_xvla_pre_post_processors(
    config: OnlineFilteringXVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    return make_xvla_pre_post_processors(config, dataset_stats)


def reconcile_online_filtering_xvla_pretrained_processors(
    config: OnlineFilteringXVLAConfig,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Apply X-VLA checkpoint compatibility while retaining temporal axes."""
    return reconcile_xvla_pretrained_processors(config, preprocessor, postprocessor)
