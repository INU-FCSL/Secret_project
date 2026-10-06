"""물리를 보존하고 여러 상태·응답 시간에서 기립 명령의 유효 방향을 측정한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from .control_authority import AuthorityProbe, orientation, patterns
from .action_diagnostics import mode_decomposition

HORIZONS=(.2,.5,1.,2.)
GATE=dict(orientation_energy=.95,max_dominant_dimensions=4,scripted_effective=.70,ppo_weak=.30)


def spectrum(matrix):
    _,s,vh=np.linalg.svd(matrix,full_matrices=False)
    fraction=s**2/np.sum(s**2)
    return dict(singular_values=s.tolist(),rank_relative_1_percent=int((s>.01*s[0]).sum()),
        dimensions_95=int(np.searchsorted(np.cumsum(fraction),.95)+1),energy_fraction=fraction.tolist(),
        directions=vh.tolist())


def response(probe,state,command):
    m,d=probe.model,probe.data
    mujoco.mj_setState(m,d,state,probe.state_spec);mujoco.mj_forward(m,d)
    previous=np.zeros(12);alpha=1-np.exp(-probe.control_dt/.15)
    values=[];peak=0.;self_steps=0;abnormal_steps=0;saturation=0;foot_loss=0
    warning_before=np.array([w.number for w in d.warning])
    for step in range(round(max(HORIZONS)/probe.control_dt)):
        previous+=alpha*(command-previous)
        d.ctrl[:12]=probe.default+probe.scale*previous;d.ctrl[12:]=0
        for _ in range(probe.decimation):
            mujoco.mj_step(m,d)
            feet,pairs,abnormal=probe.contacts()
            self_steps+=bool(pairs);abnormal_steps+=abnormal;foot_loss+=feet<4
            torque=abs(d.actuator_force[:12]);peak=max(peak,float(torque.max()))
            saturation+=int((torque>=.52-1e-6).sum())
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError('Jacobian 시험에서 유한하지 않은 물리 상태가 발생했습니다.')
        t=(step+1)*probe.control_dt
        if any(abs(t-h)<1e-8 for h in HORIZONS):
            values.append(np.r_[np.rad2deg(orientation(d.qpos[3:7])[:2]),1000*d.qpos[2]])
    return np.array(values),dict(peak_torque=peak,self_contact_samples=self_steps,abnormal_ground_samples=abnormal_steps,
        saturation_samples=saturation,foot_loss_samples=foot_loss,
        warning_delta=(np.array([w.number for w in d.warning])-warning_before).tolist())


def states(probe):
    result={'equilibrium':probe.state.copy()}
    m,d=probe.model,probe.data
    for axis in (0,1):
        for sign in (-1,1):
            mujoco.mj_setState(m,d,probe.state,probe.state_spec);mujoco.mj_forward(m,d)
            d.xfrc_applied[probe.base,axis]=sign*.5
            for _ in range(round(.25/probe.dt)):
                mujoco.mj_step(m,d)
            d.xfrc_applied[:]=0
            state=np.empty_like(probe.state);mujoco.mj_getState(m,d,state,probe.state_spec)
            result[f'push_{axis}_{sign}']=state
    return result


def projected_energy(trace,directions):
    result={}
    with np.load(trace) as data:
        for phase,mask in [('all',data['valid']),('after_push',data['after'])]:
            result[phase]={}
            for key in ('requested','bounded','applied'):
                values=data[key][mask].astype(float)
                total=float((values**2).sum(-1).mean())
                coeff=values@directions.T
                effective=float((coeff**2).sum(-1).mean())
                result[phase][key]=dict(total_energy=total,effective_energy=effective,
                    effective_fraction=effective/total if total else 0.,weak_fraction=1-effective/total if total else 0.,
                    mean=values.mean(0).tolist(),modes=mode_decomposition(values))
    return result


def analyze(output,epsilon=.05):
    output.mkdir(parents=True,exist_ok=True)
    probe=AuthorityProbe();initial=states(probe)
    jacobians={};safety=[]
    for label,state in initial.items():
        jac=np.empty((len(HORIZONS),3,12))
        for joint in range(12):
            command=np.zeros(12);command[joint]=epsilon
            positive,p_safe=response(probe,state,command)
            negative,n_safe=response(probe,state,-command)
            jac[:,:,joint]=(positive-negative)/(2*epsilon)
            safety.extend([dict(state=label,joint=joint,sign=1,**p_safe),dict(state=label,joint=joint,sign=-1,**n_safe)])
        jacobians[label]=jac
        print('Jacobian 측정 완료:',label,flush=True)
    stacked=np.concatenate([jac[:,:2].reshape(-1,12) for jac in jacobians.values()])
    svd=spectrum(stacked);rank=svd['dimensions_95'];directions=np.array(svd['directions'][:rank])
    energies={name:projected_energy(path,directions) for name,path in {
        'scripted':'/tmp/microdog_v4/v4a/evaluation/scripted.npz',
        'v3e75':'/tmp/microdog_v3cde/v3e/evaluation/checkpoint_75.npz',
        'v4a25':'/tmp/microdog_v4/v4a/evaluation/checkpoint_25.npz'}.items()}
    applied={name:value['after_push']['applied'] for name,value in energies.items()}
    safe=all(not r['self_contact_samples'] and not r['abnormal_ground_samples'] and not r['saturation_samples']
        and not any(r['warning_delta']) for r in safety)
    conditions=dict(low_dimension=rank<=GATE['max_dominant_dimensions'],
        scripted_majority=applied['scripted']['effective_fraction']>=GATE['scripted_effective'],
        ppo_weak_energy=all(applied[name]['weak_fraction']>=GATE['ppo_weak'] for name in ('v3e75','v4a25')),
        safe_local_measurement=safe)
    result=dict(horizons=list(HORIZONS),epsilon=epsilon,units=['degree/action','degree/action','mm/action'],
        jacobians={name:jac.tolist() for name,jac in jacobians.items()},stacked_orientation=svd,
        per_state={name:[spectrum(jac[i,:2]) for i in range(len(HORIZONS))] for name,jac in jacobians.items()},
        energies=energies,safety=safety,gate=dict(thresholds=GATE,conditions=conditions,run_v4b=all(conditions.values())),
        equilibrium=probe.equilibrium(),
        note='2×12 행렬의 rank 상한 2는 출력 차원에 따른다. 여러 상태·시간을 쌓은 결과와 실제 명령 에너지를 함께 판단한다. null은 측정한 국소 자세 반응의 약한 방향이며 모든 물리 효과가 없다는 뜻은 아니다.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    np.savez_compressed(output/'jacobians.npz',stacked=stacked,**jacobians)
    print('Subspace 판정:',result['gate'],flush=True)
    return result


def design_basis(output):
    """측정된 1초 응답으로 두 차등 모드의 제어 권한과 안전한 결합 범위를 맞춘다."""
    summary=json.loads((output/'summary.json').read_text())
    if not summary['gate']['run_v4b']:
        raise RuntimeError('reduced basis 실행 조건을 충족하지 않았습니다.')
    jac=np.array(summary['jacobians']['equilibrium'])[2,:2]
    modes=patterns();pitch=modes['pitch_knee'];roll=modes['roll_knee']
    gp=abs(float((jac@pitch)[1]));gr=abs(float((jac@roll)[0]))
    ps,rs=gr/(gp+gr),gp/(gp+gr)
    matrix=np.stack((ps*pitch,rs*roll),axis=-1)
    assert np.max(np.abs(matrix).sum(-1))<=1.000001
    probe=AuthorityProbe();initial=states(probe);tests=[]
    commands=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    for label,state in initial.items():
        baseline,_=response(probe,state,np.zeros(12))
        for command in commands:
            values,safety=response(probe,state,matrix@command)
            tests.append(dict(state=label,basis_command=command,response=(values-baseline).tolist(),**safety))
    safe=all(not r['self_contact_samples'] and not r['abnormal_ground_samples'] and not r['saturation_samples']
        and not r['foot_loss_samples'] and not any(r['warning_delta']) for r in tests)
    result=dict(names=['pitch_knee','roll_knee'],dimension=2,matrix=matrix.tolist(),scales=[ps,rs],
        measured_gains=[gp,gr],equalized_local_gain=ps*gp,horizons=list(HORIZONS),tests=tests,safe=safe,
        previous_action='12차원 실제 smoothed joint action을 identity로 관측한다.')
    (output/'basis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if not safe:
        raise RuntimeError('basis의 단일축 또는 결합 안전 시험을 통과하지 못했습니다.')
    print('2차원 basis 안전 검증 완료:',result['scales'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description='기립 action의 물리 Jacobian과 명령 에너지 분석')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--design-basis',action='store_true')
    args=parser.parse_args()
    if args.design_basis:
        design_basis(args.output)
    else:
        analyze(args.output)


if __name__=='__main__':
    main()
