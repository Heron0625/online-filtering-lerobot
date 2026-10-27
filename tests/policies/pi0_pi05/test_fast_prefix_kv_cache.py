import torch
from torch import nn
from transformers import GemmaConfig

from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.policies.pi05.modeling_pi05 import (
    compute_layer_complete,
    forward_cached_expert_with_fast_adapters,
)
from lerobot.policies.pi05.online_filtering import FastObservationAdapter
from lerobot.policies.pi_gemma import PiGemmaModel, layernorm_forward


def test_fast_adapter_cached_prefix_matches_joint_loop() -> None:
    torch.manual_seed(7)
    config = GemmaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=10,
    )
    config.use_adarms = False
    config.adarms_cond_dim = None
    config._attn_implementation = "eager"  # noqa: SLF001
    prefix_model = PiGemmaModel(config).eval()
    expert_model = PiGemmaModel(config).eval()
    adapters = nn.ModuleDict({"0": FastObservationAdapter(64, 8)})
    nn.init.normal_(adapters["0"].output_proj.weight, std=0.01)

    prefix = torch.randn(1, 5, 64)
    suffix = torch.randn(1, 3, 64)
    fast = torch.randn(1, 4, 64)
    pad_mask = torch.ones(1, 8, dtype=torch.bool)
    autoregressive = torch.tensor([[0] * 5 + [1] * 3], dtype=torch.bool)
    position_ids = torch.cumsum(pad_mask, dim=1) - 1
    attention = prepare_attention_masks_4d(make_att_2d_masks(pad_mask, autoregressive))
    delay = torch.ones(1, dtype=torch.long)
    flow_time = torch.full((1,), 0.6)

    with torch.no_grad():
        joint = [prefix, suffix]
        for index, layers in enumerate(zip(prefix_model.layers, expert_model.layers, strict=True)):
            joint = compute_layer_complete(
                joint,
                attention,
                position_ids,
                [None, None],
                layers,
                prefix_model.rotary_emb,
                    fast_adapter=adapters["0"] if index == 0 else None,
                fast_tokens=fast,
                fast_delays=delay,
                fast_flow_times=flow_time,
            )
        expected, _ = layernorm_forward(expert_model.norm, joint[1], None)

        cache = prefix_model(
            inputs_embeds=prefix,
            attention_mask=attention[:, :, :5, :5],
            position_ids=position_ids[:, :5],
            use_cache=True,
        ).past_key_values
        actual = forward_cached_expert_with_fast_adapters(
            expert_model,
            suffix,
            attention_mask=attention[:, :, -3:, :],
            position_ids=position_ids[:, -3:],
            past_key_values=cache,
            adarms_cond=None,
            fast_adapters=adapters,
            fast_tokens=fast,
            fast_delays=delay,
            fast_flow_times=flow_time,
            fast_observation_ages=None,
        )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

