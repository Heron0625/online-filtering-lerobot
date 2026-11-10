"""Audit saved XVLA online models on matched real-data particles and fast ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.online_filtering_xvla.configuration_online_filtering_xvla import (
    OnlineFilteringXVLAConfig,
)
from lerobot.policies.online_filtering_xvla.modeling_online_filtering_xvla import OnlineFilteringXVLAPolicy
from lerobot.utils.constants import OBS_STATE


def main():
    """Compare native parity and fast-condition dependence without updating weights."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = OnlineFilteringXVLAConfig.from_pretrained(args.checkpoint)
    cfg.device = "cuda"
    cfg.dinov3_model_id = "/data/yhr/pretrain/dinov3-vits16-pretrain-lvd1689m"
    root = Path("/data/yhr/pretrain/xvla_official/libero_lerobot_xvla_controller_v2_masked")
    meta = LeRobotDatasetMetadata("lerobot/libero", root=root)
    ds = LeRobotDataset(
        "lerobot/libero",
        root=root,
        delta_timestamps=resolve_delta_timestamps(cfg, meta),
        video_backend="pyav",
        return_uint8=True,
    )
    samples, indices = [], []
    for i in range(8):
        index = int((i + 0.5) * len(ds) / 8)
        for _ in range(200):
            sample = ds[index]
            if not sample["action_is_pad"].any() and sample[cfg.action_validity_feature].all():
                break
            index += 50
        samples.append(sample)
        indices.append(index)
    pre, _ = make_pre_post_processors(
        cfg,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {
                "tokenizer_name": "/data/yhr/pretrain/online_filtering_baselines/bart-large-tokenizer"
            },
        },
    )
    batch = pre(default_collate(samples))
    model = OnlineFilteringXVLAPolicy.from_pretrained(args.checkpoint, config=cfg).eval()
    result = {"checkpoint": str(args.checkpoint), "indices": indices, "samples": len(samples), "cases": []}
    with torch.inference_mode():
        assert model._observation_validity(batch).all()
        targets = model._prepare_action_targets(batch)
        slow = model._slow_inputs(batch)
        batch_size = targets.shape[0]
        zero = torch.zeros(batch_size, device="cuda", dtype=torch.long)
        fast0 = model._make_fast_tokens(batch, zero)
        # Disable only learned adapter outputs and compare the reimplemented transformer path.
        saved = {k: v.output_proj.state_dict() for k, v in model.model.fast_adapters.items()}
        saved = {k: {n: t.clone() for n, t in d.items()} for k, d in saved.items()}
        for v in model.model.fast_adapters.values():
            v.output_proj.weight.zero_()
            v.output_proj.bias.zero_()
        torch.manual_seed(20261101)
        noise = torch.randn_like(targets)
        t = torch.full((batch_size,), 0.5, device="cuda")
        p = noise * 0.5 + targets * 0.5
        online = model.model.predict_endpoint(
            slow_condition=slow[0],
            domain_id=slow[2],
            base_proprio=slow[1],
            old_fast_tokens=fast0,
            new_fast_tokens=fast0,
            old_delay=zero,
            new_delay=zero,
            particle=p,
            flow_time=t,
        )
        prop, particle = model.model.action_space.preprocess(slow[1], p)
        native = model.model.transformer(
            domain_id=slow[2],
            proprio=prop,
            action_with_noise=particle,
            t=t,
            **{k: v for k, v in slow[0].items() if k != "primary_visual_tokens"},
        )
        result["zero_adapter_native_max_abs"] = (online - native).abs().max().item()
        torch.testing.assert_close(online, native, atol=1e-5, rtol=1e-5)
        for k, v in model.model.fast_adapters.items():
            v.output_proj.load_state_dict(saved[k])
        for query in [0, 5, 9]:
            new = torch.full_like(zero, query * 25 // 9)
            old = torch.full_like(zero, max(query - 1, 0) * 25 // 9)
            t = torch.full((batch_size,), 1 - query / 10, device="cuda")
            p = t[:, None, None] * noise + (1 - t[:, None, None]) * targets
            p = torch.where((torch.arange(50, device="cuda")[None] < new[:, None])[..., None], targets, p)
            correct = None
            for mode in [
                "correct",
                "stale_all",
                "shuffled_images",
                "stale_images",
                "stale_proprio",
                "no_adapters",
            ]:
                changed = dict(batch)
                if mode == "shuffled_images":
                    for key in model.fast_image_features:
                        changed[key] = batch[key].roll(1, dims=0)
                if mode in ["stale_images", "stale_all"]:
                    for key in model.fast_image_features:
                        changed[key] = batch[key][:, :1].expand_as(batch[key])
                if mode in ["stale_proprio", "stale_all"]:
                    changed[OBS_STATE] = batch[OBS_STATE][:, :1].expand_as(batch[OBS_STATE])
                if mode == "no_adapters":
                    for v in model.model.fast_adapters.values():
                        v.output_proj.weight.zero_()
                        v.output_proj.bias.zero_()
                pred = model._predict(changed, p, t, old, new, slow=slow, drop=False)
                if mode == "no_adapters":
                    for k, v in model.model.fast_adapters.items():
                        v.output_proj.load_state_dict(saved[k])
                if correct is None:
                    correct = pred
                loss, losses = model._masked_action_loss(pred, targets, new, batch)
                mask = torch.arange(50, device="cuda")[None] >= new[:, None]
                result["cases"].append(
                    {
                        "query": query,
                        "old": int(old[0]),
                        "new": int(new[0]),
                        "mode": mode,
                        "loss": loss.item(),
                        **{k: v.item() for k, v in losses.items()},
                        "suffix_prediction_delta_rms": (pred[mask] - correct[mask])
                        .square()
                        .mean()
                        .sqrt()
                        .item(),
                    }
                )
        # Current visual dropout masks action-transformer keys AFTER multimodal fusion.
        images, image_masks = model._prepare_images_at(batch, zero)
        changed_images = images.roll(1, dims=0)
        enc2 = model.model.encode_slow_condition(
            batch["observation.language.tokens"], changed_images, image_masks
        )
        n = slow[0]["primary_visual_tokens"]
        result["fused_language_image_dependency_rms"] = (
            (slow[0]["vlm_features"][:, n:] - enc2["vlm_features"][:, n:]).square().mean().sqrt().item()
        )
        full_drop = torch.ones(batch_size, device="cuda", dtype=torch.bool)
        masked1 = model.model.encode_slow_condition(
            batch["observation.language.tokens"], images, image_masks, visual_drop_mask=full_drop
        )
        masked2 = model.model.encode_slow_condition(
            batch["observation.language.tokens"], changed_images, image_masks, visual_drop_mask=full_drop
        )
        result["prefusion_dropout_language_max_abs"] = (
            masked1["vlm_features"][:, n:] - masked2["vlm_features"][:, n:]
        ).abs().max().item()
        torch.testing.assert_close(masked1["vlm_features"][:, n:], masked2["vlm_features"][:, n:])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
