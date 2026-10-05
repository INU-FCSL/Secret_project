"""같은 상태·저장 경로에서 자세 보상 scale 두 값만 비교한다."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .config import REWARD_WEIGHTS
from .control_authority import AuthorityProbe, orientation, patterns
from .reference import gravity_from_rpy
from .rewards import standing_rewards

TERMS=tuple(REWARD_WEIGHTS)
SCALES=(.05,.01)


def gravity_tensor(rpy):
    roll,pitch=rpy[...,0],rpy[...,1]
    return torch.stack((pitch.sin(),-roll.sin()*pitch.cos(),-roll.cos()*pitch.cos()),-1)


def sensitivity(probe):
    rows=[]
    ref=torch.tensor(gravity_from_rpy(probe.eq[:3]),dtype=torch.float64)
    zeros=torch.zeros(1,12,dtype=torch.float64)
    for axis in (0,1):
        for error in (0,.05,.1,.2,.3,.5,.75,1,2):
            rpy=probe.eq[:3].copy();rpy[axis]+=math.radians(error)
            values={}
            for scale in SCALES:
                reward,terms=standing_rewards(torch.tensor([gravity_from_rpy(rpy)]),torch.tensor([probe.eq[3]]),
                    torch.tensor((probe.eq_joints-probe.default)[None]),zeros,zeros,
                    torch.tensor(probe.eq_torque[None,:12]),reference_gravity=ref,
                    reference_height=float(probe.eq[3]),orientation_reward_scale=scale)
                values[str(scale)]=dict(orientation_score=float(terms['upright']/REWARD_WEIGHTS['upright']),
                    weighted_orientation_step=float(terms['upright']*.02),total_step=float(reward*.02))
            rows.append(dict(axis=('roll','pitch')[axis],error_degrees=error,values=values))
    return rows


def rescore(data,reference):
    """외란 이후 자세와 저장된 나머지 보상항을 그대로 사용한다."""
    error=torch.from_numpy(data['error']).double().deg2rad()
    rpy=error+torch.tensor(reference[:2],dtype=torch.float64)
    ref=gravity_tensor(torch.tensor(reference[:2],dtype=torch.float64))
    gravity=gravity_tensor(rpy)
    d2=(gravity-ref).square().sum(-1)
    dot=(gravity*ref).sum(-1).clamp(0,1)
    original=data['terms'].astype(np.float64)
    valid=data['valid']
    outputs={}
    for scale in SCALES:
        orientation=(REWARD_WEIGHTS['upright']*torch.exp(-d2/scale**2)*dot*.02).numpy()
        terms=original.copy();terms[...,0]=orientation
        outputs[str(scale)]=dict(components=dict(zip(TERMS,(terms*valid[...,None]).sum(0).mean(0).tolist())),
            total_return=float(((data['reward'].astype(float)+orientation-original[...,0])*valid).sum(0).mean()),
            orientation_reconstruction_max_error=float(np.abs(orientation-original[...,0])[valid].max()))
    if outputs['0.05']['orientation_reconstruction_max_error']>1e-7:
        raise RuntimeError('저장된 기존 자세 보상의 재구성 오차가 큽니다.')
    return outputs


def stored_trajectories(directory):
    summary=json.loads((directory/'comparison.json').read_text())
    output={}
    for name in ('zero','scripted','v3a_0','v3a_25','v3a_100'):
        path=directory/f'{name}.npz'
        with np.load(path) as archive:
            data={k:archive[k] for k in archive.files}
        reference=summary[name]['per_seed'][0]['reference_orientation']
        output[name]=dict(values=rescore(data,reference),source=str(path),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),valid_steps=int(data['valid'].sum()))
    contrast={}
    for scale in map(str,SCALES):
        zero=output['zero']['values'][scale]['total_return']
        contrast[scale]={name:row['values'][scale]['total_return']-zero for name,row in output.items()}
    return dict(controllers=output,contrast_to_zero=contrast,physics_replayed=False)


def finite_horizon(probe):
    """적분 상태·필터를 복원하고 같은 궤적을 두 보상으로 평가한다."""
    import mujoco
    m,d=probe.model,probe.data
    columns=[]
    for name in ('roll_knee','pitch_knee'):
        plus=probe.run(action=.5*patterns()[name],seconds=4.)
        minus=probe.run(action=-.5*patterns()[name],seconds=4.)
        columns.append(np.array(plus['final_delta'][:2])-np.array(minus['final_delta'][:2]))
    inverse=np.linalg.inv(np.array(columns).T)
    basis=np.stack([patterns()[name] for name in ('roll_knee','pitch_knee')],axis=1)
    rows=[]
    for axis in (0,1):
        for sign in (-1,1):
            mujoco.mj_setState(m,d,probe.state,probe.state_spec)
            d.xfrc_applied[:]=0
            for _ in range(150):
                d.xfrc_applied[probe.base,axis]=sign*.5
                mujoco.mj_step(m,d)
            state=np.empty_like(probe.state)
            mujoco.mj_getState(m,d,state,probe.state_spec)
            state_hash=hashlib.sha256(state.tobytes()).hexdigest()
            initial_error=np.rad2deg(orientation(d.qpos[3:7])[:2]-probe.eq[:2]).tolist()
            for name in ('zero','scripted','wrong'):
                mujoco.mj_setState(m,d,state,probe.state_spec);mujoco.mj_forward(m,d)
                applied=np.zeros(12);totals={str(s):0. for s in SCALES};values={}
                for step in range(50):
                    error=orientation(d.qpos[3:7])[:2]-probe.eq[:2]
                    coefficients=np.rad2deg(-2*error-.2*d.qvel[3:5])@inverse.T
                    corrective=np.clip(np.clip(coefficients,-1,1)@basis.T,-1,1)
                    request=np.zeros(12) if name=='zero' else corrective*(1 if name=='scripted' else -1)
                    delta=-math.expm1(-.02/.15)*(request-applied);applied+=delta
                    d.ctrl[:12]=probe.default+probe.scale*applied;d.ctrl[12:]=0
                    for substep in range(10):
                        d.xfrc_applied[:]=0
                        if .3+(step*10+substep)*.002<.5:
                            d.xfrc_applied[probe.base,axis]=sign*.5
                        mujoco.mj_step(m,d)
                    rpy=orientation(d.qpos[3:7])
                    for scale in SCALES:
                        reward,_=standing_rewards(torch.tensor([gravity_from_rpy(rpy)]),torch.tensor([d.qpos[2]]),
                            torch.tensor((d.qpos[probe.qids]-probe.default)[None]),torch.tensor(d.qvel[probe.vids][None]),
                            torch.tensor(delta[None]),torch.tensor(d.actuator_force[None,:12]),
                            reference_gravity=torch.tensor(gravity_from_rpy(probe.eq[:3])),reference_height=float(probe.eq[3]),
                            orientation_reward_scale=scale)
                        totals[str(scale)]+=.99**step*float(reward)*.02
                    if step+1 in (1,5,10,25,50):
                        values[str(step+1)]=dict(totals)
                rows.append(dict(axis=axis,sign=sign,controller=name,initial_state_sha256=state_hash,
                                 initial_error=initial_error,horizons=values))
    ordering=[]
    for axis in (0,1):
        for sign in (-1,1):
            group={row['controller']:row for row in rows if row['axis']==axis and row['sign']==sign}
            for horizon in ('1','5','10','25','50'):
                values={name:row['horizons'][horizon]['0.01'] for name,row in group.items()}
                ordering.append(dict(axis=axis,sign=sign,horizon=int(horizon),values=values,
                    correct_order=values['scripted']>values['zero']>values['wrong']))
    return dict(rows=rows,ordering=ordering,all_correct=all(row['correct_order'] for row in ordering))


def main():
    parser=argparse.ArgumentParser(description='V3-B 자세 보상의 학습 전 검증')
    parser.add_argument('--trajectories',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    probe=AuthorityProbe()
    results=dict(sensitivity=sensitivity(probe),rescoring=stored_trajectories(args.trajectories),finite_horizon=finite_horizon(probe))
    (args.output/'reward_validation.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    print(f"보상 검증 저장: {args.output}, 유한 시간 순서 통과={results['finite_horizon']['all_correct']}",flush=True)
    if not results['finite_horizon']['all_correct']:
        raise RuntimeError('유한 시간 보상 순서가 조건을 충족하지 못했습니다.')


if __name__=='__main__':
    main()
