"""같은 seed의 parent와 후보를 평가하고 조건부 승격을 판단한다."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .config import StandingCfg
from .evaluate import evaluate
from .compare_standing import ActionTrace, action_summary, aggregate, policy_probe

SEEDS=(2026,2027,2028)


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def safe(summary):
    return all(row['survival_rate']==1 and row['self_contact_env_steps']==0 and
        row['unexpected_contact_env_steps']==0 and row['saturation_fraction']<.01 for row in summary['per_seed'])


def correct_feedback(summary):
    local=summary.get('policy_probe',{}).get('local_feedback')
    feedback=summary['actions']['feedback']
    if not local:
        return False
    roll=feedback.get('bounded_roll',feedback['applied_roll'])['error_correlation']
    pitch=feedback.get('bounded_pitch',feedback['applied_pitch'])['error_correlation']
    return bool(roll is not None and pitch is not None and roll>0 and pitch<0 and
        local['roll']['position_per_degree']['mean']>1e-5 and local['pitch']['position_per_degree']['mean']<-1e-5)


def classify(candidate,parent,zero,version='v3'):
    metrics=candidate['aggregate'];old=parent['aggregate']
    seed_improved=[dict(seed=row['seed'],peak=row['peak_orientation_error']<baseline['peak_orientation_error'],
        integrated=row['integrated_orientation_error']<baseline['integrated_orientation_error'])
        for row,baseline in zip(candidate['per_seed'],zero['per_seed'])]
    safety=safe(candidate) and metrics['survival_rate']>=old['survival_rate'] and metrics['saturation_fraction']<=old['saturation_fraction']+1e-6
    success=safety and all(row['peak'] and row['integrated'] for row in seed_improved)
    peak_reduction=1-metrics['peak_orientation_error']/old['peak_orientation_error']
    integrated_reduction=1-metrics['integrated_orientation_error']/old['integrated_orientation_error']
    boundary=candidate['actions']['boundary_applied_fraction'];old_boundary=parent['actions']['boundary_applied_fraction']
    bounded=candidate['actions']['bounded_99_fraction'];old_bounded=parent['actions']['bounded_99_fraction']
    feedback=correct_feedback(candidate) and not correct_feedback(parent)
    boundary_improvement=boundary<old_boundary-1e-4 or bounded<old_bounded-1e-4
    bias=float(np.abs(candidate['actions']['applied']['mean']).mean()) if 'applied' in candidate['actions'] else None
    old_bias=float(np.abs(parent['actions']['applied']['mean']).mean()) if 'applied' in parent['actions'] else None
    bias_improvement=bias is not None and old_bias is not None and bias<old_bias-1e-4
    if version in ('v5','v6','v7'):
        distribution=candidate['policy_probe']['distribution']
        bias=float(np.abs(distribution['latent_mean']['mean']).mean())
        old_distribution=parent.get('policy_probe',{}).get('distribution')
        old_mean=(old_distribution['latent_mean']['mean'] if old_distribution else parent['actions']['policy_raw']['mean'])
        old_bias=float(np.abs(old_mean).mean())
        bias_improvement=bias<old_bias-1e-4
        boundary=candidate['actions']['policy_boundary_fraction']
        old_boundary=parent['actions']['policy_boundary_fraction']
        boundary_improvement=boundary<old_boundary-1e-4
    recovery=(metrics['push_recovery_rate'] or 0)>(old['push_recovery_rate'] or 0)+1e-6
    common_parent_return=candidate.get('common_parent_return',old['mean_return'])
    aligned=metrics['mean_return']>common_parent_return+1e-4 and peak_reduction>=.15 and integrated_reduction>=.15
    collapsed=correct_feedback(parent) and not correct_feedback(candidate)
    severe_boundary=boundary>.9 and boundary>old_boundary+.15
    input_ok=True
    if version in ('v4','v5','v6','v7'):
        previous=candidate['policy_probe']['observations']['previous_action']
        input_ok=(max(previous['normalized']['max'])<=1.000001 and min(previous['normalized']['min'])>=-1.000001)
    success=success and input_ok
    partial=(safety and input_ok and peak_reduction>=.15 and integrated_reduction>=.15 and
        (feedback or boundary_improvement or recovery or (bias_improvement if version in ('v4','v5','v6','v7') else aligned)) and not collapsed and not severe_boundary)
    status='SUCCESS' if success else ('PARTIAL' if partial else 'FAIL')
    return dict(status=status,safe=safety,per_seed_zero_improved=seed_improved,
        peak_reduction=peak_reduction,integrated_reduction=integrated_reduction,
        feedback_improved=feedback,boundary_improved=boundary_improvement,recovery_improved=recovery,
        return_alignment_improved=aligned,feedback_collapsed=collapsed,severe_boundary=severe_boundary,
        persistent_bias_improved=bias_improvement,
        mean_absolute_applied_bias=(float(np.abs(candidate['actions']['applied']['mean']).mean())
            if 'applied' in candidate['actions'] else None),
        mean_absolute_latent_bias=bias if version in ('v5','v6','v7') else None,semantic_input_ok=input_ok,
        thresholds=dict(partial_reduction=.15,saturation_fraction=.01,severe_boundary=.9))


def common_return(summary,scale,pose_weight=None):
    from .config import REWARD_WEIGHTS
    old_pose=summary['profile'].get('pose_reward_weight',REWARD_WEIGHTS['pose'])
    if summary['profile']['orientation_reward_scale']==scale and (pose_weight is None or pose_weight==old_pose):
        return summary['aggregate']['mean_return']
    with np.load(summary['trace']) as data:
        rpy=np.deg2rad(data['error'])+np.array(summary['per_seed'][0]['reference_orientation'][:2])
        roll,pitch=rpy[...,0],rpy[...,1]
        gravity=np.stack((np.sin(pitch),-np.sin(roll)*np.cos(pitch),-np.cos(roll)*np.cos(pitch)),-1)
        ref=np.array(summary['per_seed'][0]['reference_orientation'][:2]);r,p=ref
        gref=np.array([np.sin(p),-np.sin(r)*np.cos(p),-np.cos(r)*np.cos(p)])
        orientation=.03*np.exp(-np.sum((gravity-gref)**2,-1)/scale**2)*np.clip((gravity*gref).sum(-1),0,1)
        reward=(data['reward'].astype(float) if summary['profile']['orientation_reward_scale']==scale
            else data['reward'].astype(float)-data['terms'][...,0]+orientation)
        if pose_weight is not None and pose_weight!=old_pose:
            if old_pose==0:raise ValueError('pose weight0의 trace에서 원시 pose score를 복원할 수 없습니다.')
            reward+=(pose_weight/old_pose-1)*data['terms'][...,2]
        return float((reward*data['valid']).sum(0).mean())


def measure(cfg,output,label,checkpoint=None):
    rows=[];traces=[]
    actual_cfg=(replace(cfg,action_basis='joint',standing_basis=None,
        action_mapping='clip' if cfg.action_mapping=='identity' else cfg.action_mapping)
        if checkpoint is None else cfg)
    for seed in SEEDS:
        recorder=ActionTrace()
        row=evaluate(replace(actual_cfg,num_envs=16,seed=seed,balanced_push_directions=True),checkpoint=checkpoint,
            random=label=='random',scripted=label=='scripted',step_callback=recorder)
        rows.append(row);traces.append(recorder.arrays())
    data={key:np.concatenate([trace[key] for trace in traces],axis=1) for key in traces[0]}
    path=output/f'{label}.npz';np.savez_compressed(path,**data)
    summary=dict(per_seed=rows,aggregate=aggregate(rows),actions=action_summary(data,cfg.action_mapping),trace=str(path),
        profile=dict(action_mapping=actual_cfg.action_mapping,orientation_reward_scale=actual_cfg.orientation_reward_scale,
            pose_reward_weight=actual_cfg.pose_reward_weight,action_basis=actual_cfg.action_basis,
            previous_action_normalization=actual_cfg.previous_action_normalization))
    if checkpoint:
        summary['checkpoint']=str(checkpoint)
        summary['policy_probe']=policy_probe(checkpoint,data,cfg.action_mapping)
    zero_path=output/'zero.npz'
    if zero_path.exists() and label!='zero':
        with np.load(zero_path) as zero:
            if not np.array_equal(zero['raw'][0],data['raw'][0]) or not np.array_equal(zero['push_start'][0],data['push_start'][0]):
                raise RuntimeError('동일 평가의 초기 관측·외란 시점이 다릅니다.')
    print(f"평가 완료: {label}, 최대={summary['aggregate']['peak_orientation_error']:.6f}°, 누적={summary['aggregate']['integrated_orientation_error']:.6f}",flush=True)
    return summary


def evaluate_candidate(directory,output,name,checkpoints,parent_path=None,cfg=None):
    output.mkdir(parents=True,exist_ok=True)
    if cfg is None:
        cfg=StandingCfg(**json.loads((directory/'environment.json').read_text())['config'])
    path=output/'summary.json'
    result=json.loads(path.read_text()) if path.exists() else dict(name=name,baselines={},checkpoints={})
    for label in ('zero','random','scripted'):
        if label not in result['baselines']:
            result['baselines'][label]=measure(cfg,output,label)
            save(path,result)
    parent=json.loads(parent_path.read_text()) if parent_path else None
    for iteration in checkpoints:
        key=str(iteration)
        if key in result['checkpoints']:
            continue
        summary=measure(cfg,output,f'checkpoint_{iteration}',directory/f'checkpoint_{iteration}.pt')
        if parent:
            reference=parent['checkpoints'].get(key,parent['best'])
            summary['common_parent_return']=common_return(reference,cfg.orientation_reward_scale,cfg.pose_reward_weight)
            summary['classification']=classify(summary,reference,result['baselines']['zero'],name[:2] if name.startswith(('v4','v5','v6','v7')) else 'v3')
        result['checkpoints'][key]=summary;save(path,result)
    trained=[(int(i),row) for i,row in result['checkpoints'].items() if int(i)>0 and safe(row)]
    successes=[(i,row) for i,row in trained if parent and row['classification']['status']=='SUCCESS']
    choices=successes or trained
    zero=result['baselines']['zero']['aggregate']
    def rank(item):
        metrics=item[1]['aggregate']
        return (metrics['integrated_orientation_error']/zero['integrated_orientation_error'],
            metrics['peak_orientation_error']/zero['peak_orientation_error'],
            metrics['recovery_seconds'] if metrics['recovery_seconds'] is not None else float('inf'),
            not correct_feedback(item[1]),item[1]['actions']['boundary_applied_fraction'])
    if choices:
        best_iteration,best=min(choices,key=rank)
    else:
        best_iteration,best=min(((int(i),row) for i,row in result['checkpoints'].items() if int(i)>0),key=rank)
    result.update(best_checkpoint=best_iteration if choices else None,best_eligible=bool(choices),
                  diagnostic_checkpoint=best_iteration,best=best,parent=str(parent_path) if parent_path else None)
    if parent:
        reference=parent['checkpoints'].get(str(best_iteration),parent['best']) if name.startswith(('v4','v5','v6','v7')) else parent['best']
        best['common_parent_return']=common_return(reference,cfg.orientation_reward_scale,cfg.pose_reward_weight)
        decision=classify(best,reference,result['baselines']['zero'],name[:2] if name.startswith(('v4','v5','v6','v7')) else 'v3')
        result['decision']=decision
        save(output/'decision.json',decision)
    save(path,result)
    return result


def main():
    parser=argparse.ArgumentParser(description='Standing 후보의 동일 seed 조건부 평가')
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--name',default='parent')
    parser.add_argument('--checkpoints',type=int,nargs='+',default=[25])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baselines-only',action='store_true')
    parser.add_argument('--action-mapping',choices=['clip','tanh'],default='clip')
    parser.add_argument('--reward-scale',type=float,default=.05)
    args=parser.parse_args()
    if args.baselines_only:
        args.output.mkdir(parents=True,exist_ok=True)
        cfg=StandingCfg(stage=2,v2_stage='A',episode_seconds=10.,smoothing_seconds=.15,
            action_mapping=args.action_mapping,orientation_reward_scale=args.reward_scale)
        result=dict(name=args.name,baselines={label:measure(cfg,args.output,label) for label in ('zero','random','scripted')},checkpoints={})
        save(args.output/'summary.json',result)
    else:
        if args.candidate is None:
            parser.error('checkpoint 평가에는 candidate 경로가 필요합니다.')
        evaluate_candidate(args.candidate,args.output,args.name,args.checkpoints,args.parent)


if __name__=='__main__':
    main()
