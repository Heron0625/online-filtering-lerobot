"""Checkpoint component checks; synthetic inputs do not establish policy efficacy."""
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from lerobot.policies.pi05.online_filtering import PosteriorResidualUpdater


def main():
    root = Path('/data/zhoukai/robotmine/ckpt/online_filtering_pi05_clean_causal')
    checkpoint = root / 'stage_b_online_teacher/checkpoints/003000/pretrained_model'
    config = json.loads((checkpoint / 'config.json').read_text())
    torch.manual_seed(1234)
    updater = PosteriorResidualUpdater(action_dim=config['max_action_dim'], hidden_dim=config['posterior_hidden_dim'],
        num_heads=config['posterior_num_heads'], num_layers=config['posterior_num_layers'], max_horizon=config['chunk_size']).cuda().eval()
    with safe_open(str(checkpoint/'model.safetensors'), framework='pt') as f:
        updater.load_state_dict({k.removeprefix('posterior_updater.'): f.get_tensor(k) for k in f.keys() if k.startswith('posterior_updater.')},strict=True)
    x = torch.randn(4,50,32,device='cuda')
    executed = torch.randn_like(x)
    old = torch.tensor([0,1,2,0],device='cuda')
    new = torch.tensor([2,5,10,8],device='cuda')
    time = torch.tensor([.9,.7,.5,.1],device='cuda')
    condition = torch.randn(4,55,256,device='cuda',requires_grad=True)
    pred = updater(x,old_delays=old,new_delays=new,flow_times=time,condition_tokens=condition,executed_actions=executed)
    mask = torch.arange(50,device='cuda')[None,:] >= new[:,None]
    pred[mask].square().mean().backward()
    changed = condition.detach().clone()
    # Preserve slow and old blocks, alter new and innovation blocks.
    changed[:,21:] += torch.randn_like(changed[:,21:])
    with torch.no_grad():
        alt = updater(x,old_delays=old,new_delays=new,flow_times=time,condition_tokens=changed,executed_actions=executed)
    checks = dict(device=str(pred.device), finite_output=bool(torch.isfinite(pred).all()),
        prefix_max_error=(pred-executed)[~mask].abs().max().item(),
        new_condition_gradient_norm=condition.grad[:,21:].norm().item(),
        new_condition_output_change_mse=(pred-alt)[mask].square().mean().item(),
        updater_gradient_tensors=sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in updater.parameters()))
    log = root/'stage_b_online_teacher_train_20260916_131954.log'
    rows = []
    for line in log.read_text(errors='ignore').replace('\r','\n').splitlines():
        if 'teacher_delta_mse:' not in line:
            continue
        rows.append({k:float(v) for k,v in re.findall(r'([a-z_]+):(-?[0-9.]+(?:e[-+][0-9]+)?)',line)})
    keys=['posterior_endpoint_loss','teacher_delta_mse','student_delta_mse','delta_cosine_similarity','bridge_gain']
    last = rows[-10:]
    tail = {k:sum(r[k] for r in last)/len(last) for k in keys}
    tail['endpoint_vs_identity_ratio'] = tail['posterior_endpoint_loss']/tail['teacher_delta_mse']
    tail['student_vs_teacher_delta_ratio'] = tail['student_delta_mse']/tail['teacher_delta_mse']
    report = dict(checkpoint=str(checkpoint),synthetic_gpu_checks=checks,last_10_training_log_windows=tail,
        last_training_log_window={k:rows[-1][k] for k in keys},
        limitation='Real observation validation blocked: dataset cache, DINO cache and LIBERO assets missing; synthetic checks establish wiring only. Training log metrics are rounded and averaged.')
    out=Path('/data/zhoukai/robotmine/eval/stage_b_validation/component_report.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
