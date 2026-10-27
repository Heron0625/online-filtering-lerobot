"""Evaluate a saved updater without training, using real dataset preprocessing."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.online_filtering_pi05.configuration_online_filtering_pi05 import OnlineFilteringPI05Config
from lerobot.policies.online_filtering_pi05.modeling_online_filtering_pi05 import OnlineFilteringPI05Policy
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.policies.pi05.online_filtering import hard_commit_action_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-root', default='/data/zhoukai/dataset/libero')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(271828)
    config = OnlineFilteringPI05Config.from_pretrained(args.checkpoint)
    config.device = 'cuda'
    from lerobot.policies.pi05.online_filtering import DINOv3PatchEncoder
    encoder = DINOv3PatchEncoder.from_pretrained('/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m')
    policy = OnlineFilteringPI05Policy.from_pretrained(args.checkpoint, config=config, fast_encoder=encoder).eval()
    pre, post = make_pre_post_processors(config, pretrained_path=args.checkpoint,
        preprocessor_overrides={'tokenizer_processor':{'tokenizer_name':'/data/zhoukai/pretrain/paligemma-tokenizer'}})
    dataset = LeRobotDataset('lerobot/libero', root=args.dataset_root, video_backend='pyav',
        delta_timestamps={**{k: [i / 10 for i in range(11)] for k in [*config.image_features, OBS_STATE]},
                          ACTION: [i / 10 for i in range(50)]}) if args.dataset_root else None
    def simulation_batch(group):
        from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation, close_envs
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.utils import NEW_ROLLOUT_OPTION
        suite = ['libero_spatial','libero_object','libero_goal','libero_10'][group]
        cfg = LiberoEnv(task=suite, task_ids=[0], max_parallel_tasks=1)
        envs = make_env(cfg, n_envs=1, use_async_envs=False)
        env = envs[suite][0]
        ep, ap = make_env_pre_post_processors(cfg, config)
        samples = []
        try:
            for row in range(4):
                raw, _ = env.reset(seed=[271828+group*10+row], options={NEW_ROLLOUT_OPTION:True})
                def process(raw):
                    obs = preprocess_observation(raw)
                    obs['task'] = list(env.call('task_description'))
                    return pre(ep(obs))
                obs = process(raw)
                with torch.no_grad():
                    plan = policy.predict_action_chunk(obs)
                    physical = post(plan).squeeze(0).cpu()
                history = [obs]
                for step in range(10):
                    a = ap({ACTION:physical[step].unsqueeze(0)})[ACTION]
                    raw, _, _, _, _ = env.step(a.cpu().numpy())
                    history.append(process(raw))
                sample = {key: torch.stack([h[key][0] for h in history]) for key in [*config.image_features,OBS_STATE]}
                sample[ACTION] = plan[0]
                for key in [OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK]:
                    sample[key] = obs[key][0]
                samples.append(sample)
        finally:
            close_envs(envs)
        return default_collate(samples)
    results = []
    checks = {}
    grad_batch = None
    for group in range(4):
        indices = [int((group * 4 + i + 0.5) * (len(dataset) - 60) / 16) for i in range(4)] if dataset else list(range(group*4,group*4+4))
        batch = pre(default_collate([dataset[i] for i in indices])) if dataset else simulation_batch(group)
        grad_batch = batch
        with torch.no_grad():
            initial = policy._initial_slow_batch(batch)
            images, image_masks = policy._preprocess_images(initial)
            actions = policy.prepare_action(batch)
            old_d = torch.tensor([0, 1, 2, 0], device='cuda')
            new_d = torch.tensor([2, 5, 10, 8], device='cuda')
            steps = torch.tensor([1, 3, 5, 9], device='cuda')
            valid = policy._observation_validity(batch)
            if not valid.gather(1, new_d[:, None]).all():
                raise RuntimeError('Selected observation is padded; choose another sample')
            old_fast = policy._make_fast_tokens(batch, old_d)
            new_fast = policy._make_fast_tokens(batch, new_d)
            # Hold delay embedding and commitment fixed; change only image/proprio.
            stale_batch = dict(batch)
            for key in [*config.image_features, OBS_STATE]:
                stale_batch[key] = batch[key].clone()
                rows = torch.arange(4, device=batch[key].device)
                stale_batch[key][rows, new_d] = batch[key][rows, old_d]
            stale_at_new_delay = policy._make_fast_tokens(stale_batch, new_d)
            noise = policy.model.sample_noise(actions.shape, actions.device)
            def teachers(a, b, fa, fb):
                return policy._integrate_paired_teacher_particles(images=images, img_masks=image_masks,
                    tokens=batch[OBS_LANGUAGE_TOKENS], masks=batch[OBS_LANGUAGE_ATTENTION_MASK],
                    base_noise=noise, executed_actions=actions, switch_steps=steps,
                    old_delays=a, new_delays=b, old_fast_tokens=fa, new_fast_tokens=fb)
            old, target = teachers(old_d, new_d, old_fast, new_fast)
            observation_old, observation_new = teachers(new_d, new_d, stale_at_new_delay, new_fast)
            ia, ib = teachers(new_d, new_d, new_fast, new_fast)
            checks['same_condition_teacher_max_error'] = max(checks.get('same_condition_teacher_max_error', 0), (ia-ib).abs().max().item())
            prefix, prefix_valid, _ = policy.model.embed_prefix(images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK])
            slow = policy.posterior_slow_resampler(prefix.float(), prefix_valid)
            def update(x, a, b, fa, fb):
                cond = policy._make_posterior_condition_tokens(slow, fa, fb)
                return policy.posterior_updater(x.float(), old_delays=a, new_delays=b,
                    flow_times=1-steps.float()/10, condition_tokens=cond, executed_actions=actions.float())
            pred = update(old, old_d, new_d, old_fast, new_fast)
            common = hard_commit_action_prefix(old, actions, new_d)
            def velocity(x, fast, step):
                time = 1 - step.float() / config.num_inference_steps
                positions = torch.arange(x.shape[1], device=x.device)[None, :]
                model_time = torch.where(positions < new_d[:, None], 0., time[:, None])
                return policy.model.predict_velocity(images, image_masks,
                    batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
                    x, model_time, fast_adapters=policy.model.fast_adapters,
                    fast_tokens=fast, fast_delays=new_d)
            v_old = velocity(common, stale_at_new_delay, steps)
            v_new = velocity(common, new_fast, steps)
            def finish(x, fast):
                x = hard_commit_action_prefix(x, actions, new_d)
                for step in range(int(steps.min()), config.num_inference_steps):
                    current = torch.maximum(steps, torch.full_like(steps, step))
                    active = step >= steps
                    v = velocity(x, fast, current)
                    x = torch.where(active[:, None, None], x-v/config.num_inference_steps, x)
                    x = hard_commit_action_prefix(x, actions, new_d)
                return x
            endpoint_old = finish(common, stale_at_new_delay)
            endpoint_new = finish(common, new_fast)
            endpoint_student = finish(pred, new_fast)
            endpoint_teacher = finish(target, new_fast)
            stale = update(old, old_d, new_d, old_fast, old_fast)
            shuffled = update(old, old_d, new_d, old_fast, new_fast.roll(1, 0))
            identity = update(ia, new_d, new_d, new_fast, new_fast)
            dim = config.output_features[ACTION].shape[0]
            positions = torch.arange(50, device='cuda')[None, :]
            suffix = positions >= new_d[:, None]
            if 'action_is_pad' in batch:
                suffix &= ~batch['action_is_pad'].bool()
            for i in range(4):
                mask = suffix[i]
                delta = (target-old)[i, mask, :dim].float().flatten()
                correction = (pred-old)[i, mask, :dim].float().flatten()
                mse = lambda x,y: (x[i,mask,:dim].float()-y[i,mask,:dim].float()).square().mean().item()
                base = mse(old,target)
                result = dict(index=indices[i], old_delay=old_d[i].item(), new_delay=new_d[i].item(), switch_step=steps[i].item(),
                    teacher_delta_mse=base, endpoint_mse=mse(pred,target), identity_endpoint_mse=mse(identity,ia),
                    stale_condition_endpoint_mse=mse(stale,target), shuffled_condition_endpoint_mse=mse(shuffled,target),
                    delta_cosine=torch.nn.functional.cosine_similarity(delta,correction,dim=0).item(),
                    relative_error=mse(pred,target)/max(base,1e-12), student_delta_mse=mse(pred,old))
                result.update(observation_only_particle_mse=mse(observation_old, observation_new),
                    same_particle_observation_velocity_mse=mse(v_old, v_new),
                    same_particle_observation_endpoint_mse=mse(endpoint_old, endpoint_new),
                    no_updater_final_teacher_mse=mse(endpoint_new, endpoint_teacher),
                    updater_final_teacher_mse=mse(endpoint_student, endpoint_teacher),
                    observation_velocity_mse_per_dim=(v_old[i,mask,:dim].float()-v_new[i,mask,:dim].float()).square().mean(0).tolist(),
                    observation_endpoint_mse_per_dim=(endpoint_old[i,mask,:dim].float()-endpoint_new[i,mask,:dim].float()).square().mean(0).tolist())
                results.append(result)
            committed = positions < new_d[:,None]
            checks['committed_prefix_max_error'] = max(checks.get('committed_prefix_max_error',0), (pred-actions.float())[committed].abs().max().item())
            print(json.dumps({'group':group,'samples':results[-4:]}), flush=True)
    policy.train()
    loss, metrics = policy(grad_batch)
    loss.backward()
    checks['trainable_parameters'] = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    checks['frozen_teacher_gradients'] = [n for n,p in policy.named_parameters() if not p.requires_grad and p.grad is not None]
    checks['gradient_modules'] = {prefix:sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) for n,p in policy.named_parameters() if n.startswith(prefix)) for prefix in ['posterior_updater.', 'posterior_fast_proj.', 'posterior_slow_resampler.', 'posterior_condition_roles']}
    total = sum(r['teacher_delta_mse'] for r in results)
    aggregate = {key:sum(r[key] for r in results)/len(results) for key in ['teacher_delta_mse','endpoint_mse','student_delta_mse','identity_endpoint_mse','stale_condition_endpoint_mse','shuffled_condition_endpoint_mse','delta_cosine']}
    aggregate['weighted_relative_error'] = sum(r['endpoint_mse'] for r in results)/max(total,1e-12)
    aggregate['samples_improved'] = sum(r['relative_error'] < 1 for r in results)
    for key in ['observation_only_particle_mse', 'same_particle_observation_velocity_mse',
                'same_particle_observation_endpoint_mse', 'no_updater_final_teacher_mse',
                'updater_final_teacher_mse']:
        aggregate[key] = sum(r[key] for r in results)/len(results)
    report = dict(checkpoint=args.checkpoint, samples=results, checks=checks, aggregate=aggregate,
                  note='16 simulated observation pairs, task 0 of each suite; prefixes are actions actually executed from an old plan. Not a task-success evaluation. DINO architecture reconstructed from checkpoint shapes and installed library defaults.' if not dataset else 'Dataset samples may have been seen during training; this is not a held-out generalization test.', gradient_batch_metrics=metrics)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({'checks':checks,'aggregate':aggregate}),flush=True)


if __name__ == '__main__':
    main()
