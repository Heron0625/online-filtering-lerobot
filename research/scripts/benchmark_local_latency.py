#!/usr/bin/env python
"""Offline, random-weight full-size PI05/online-flow latency benchmark. No robot I/O."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["pi05", "online_flow"], required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    features = {
        f"observation.images.{name}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        for name in ["head", "chest", "right", "left"]
    }
    features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(16,))
    common = dict(
        input_features=features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        device="cuda",
        dtype="bfloat16",
        compile_model=False,
    )
    print(f"Initializing full-size {args.model}, random weights", flush=True)
    if args.model == "pi05":
        policy = PI05Policy(PI05Config(**common))
    else:
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        from lerobot.policies.online_filtering_pi05.configuration_online_filtering_pi05 import (
            OnlineFilteringPI05Config,
        )
        from lerobot.policies.online_filtering_pi05.modeling_online_filtering_pi05 import (
            OnlineFilteringPI05Policy,
        )
        from lerobot.policies.pi05.online_filtering import DINOv3PatchEncoder

        encoder = DINOv3PatchEncoder(DINOv3ViTModel(DINOv3ViTConfig(num_register_tokens=4)))
        config = OnlineFilteringPI05Config(
            **common, training_stage="online_flow", fast_num_tokens=64, fast_spatial_tokens=True
        )
        policy = OnlineFilteringPI05Policy(config, fast_encoder=encoder)
    policy.eval()
    batch = {key: torch.rand(1, 3, 480, 640, device="cuda") for key in features if key != OBS_STATE}
    batch[OBS_STATE] = torch.rand(1, 16, device="cuda") * 2 - 1
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(100, 10000, (1, 200), device="cuda")
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.arange(200, device="cuda")[None] < 100
    result = dict(
        model=args.model,
        weights="random initialization, full architecture",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        python=platform.python_version(),
        cameras=4,
        input_resolution=[480, 640],
        model_resolution=[224, 224],
        batch_size=1,
        state_dim=16,
        action_dim=16,
        token_tensor_length=200,
        valid_tokens=100,
        chunk_size=50,
        denoise_steps=10,
        compile=False,
        warmup=args.warmup,
        iterations=args.iterations,
        parameter_count=sum(p.numel() for p in policy.parameters()),
        scope="policy computation including image preprocessing; inputs already on GPU; excludes tokenizer, camera, IPC, robot",
    )

    def measure(name, fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for i in range(args.iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
            if isinstance(output, torch.Tensor) and not torch.isfinite(output).all():
                raise RuntimeError(f"nonfinite output in {name}")
            print(f"{name}: {i + 1}/{args.iterations}: {samples[-1]:.2f} ms", flush=True)
        result[name] = dict(
            mean_ms=float(np.mean(samples)),
            p50_ms=float(np.median(samples)),
            p95_ms=float(np.percentile(samples, 95)),
            min_ms=min(samples),
            max_ms=max(samples),
            samples_ms=samples,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    with torch.inference_mode():
        measure("chunk_no_new_observations", lambda: policy.predict_action_chunk(batch))
        if args.model == "online_flow":
            measure("fast_observation_encoding", lambda: policy._make_single_fast_tokens(batch, delay=1))

            def update_chunk():
                def callback(step, num_steps, generation):
                    if step == num_steps // 2:
                        packet = policy.encode_fast_observation_packet(
                            batch,
                            capture_step=1,
                            capture_timestamp=time.monotonic(),
                            generation=generation,
                            executed_prefix=torch.zeros(1, 1, 16, device="cuda"),
                        )
                        policy.push_fast_observation(packet)

                return policy.predict_action_chunk(batch, observation_callback=callback)

            measure("chunk_one_synchronous_new_observation", update_chunk)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
