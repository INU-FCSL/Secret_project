"""학습 없이 기립 제어 권한과 외란 응답을 측정하는 CPU MuJoCo 진단."""
import argparse
import json
import math
import hashlib
from pathlib import Path
import numpy as np
import mujoco
from .config import MODEL_PATH, ACTION_SCALE, LEG_JOINT_NAMES


def orientation(q):
    w, x, y, z = q
    return np.array([math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                     math.asin(np.clip(2*(w*y-z*x), -1, 1)),
                     math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))])


class AuthorityProbe:
    """XML과 기본 RL 설정을 보존하며 독립적인 상태에서 각 시험을 시작한다."""
    def __init__(self):
        self.model = m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = d = mujoco.MjData(m)
        self.key = m.key('neutral_standing').id
        self.base = m.body('base').id
        self.floor = m.geom('floor').id
        self.feet = np.array([next(g for g in range(m.ngeom)
            if m.geom_bodyid[g] == m.body(leg+'_lower').id
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
            for leg in ('fl', 'fr', 'rl', 'rr')])
        joint_ids = m.actuator_trnid[:12, 0]
        assert tuple(m.joint(int(j)).name for j in joint_ids) == LEG_JOINT_NAMES
        self.qids, self.vids = m.jnt_qposadr[joint_ids], m.jnt_dofadr[joint_ids]
        self.default = m.key_ctrl[self.key, :12].copy()
        self.scale = np.array(ACTION_SCALE)
        self.dt = float(m.opt.timestep)
        self.decimation = 10
        self.control_dt = self.dt*self.decimation
        mujoco.mj_resetDataKeyframe(m, d, self.key)
        tail = []
        for i in range(round(35/self.dt)):
            mujoco.mj_step(m, d)
            if i >= round(30/self.dt) and i % self.decimation == 0:
                tail.append(np.r_[orientation(d.qpos[3:7]), d.qpos[2]])
        tail = np.array(tail)
        self.eq = np.r_[orientation(d.qpos[3:7]), d.qpos[2]]
        self.eq_torque = d.actuator_force.copy()
        self.eq_joints = d.qpos[self.qids].copy()
        self.eq_all_joints = d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]].copy()
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mujoco.mj_stateSize(m, self.state_spec))
        mujoco.mj_getState(m, d, self.state, self.state_spec)
        self.noise = dict(std=tail.std(0).tolist(), peak_to_peak=np.ptp(tail, axis=0).tolist())
        self.foot_jacobian = []
        for foot in self.feet:
            jac = np.zeros((3, m.nv))
            mujoco.mj_jacGeom(m, d, jac, None, int(foot))
            self.foot_jacobian.append(jac[:, self.vids].tolist())

    def equilibrium(self):
        return dict(height=float(self.eq[3]), rpy_degrees=np.rad2deg(self.eq[:3]).tolist(),
                    joint_names=list(LEG_JOINT_NAMES), joint_position_degrees=np.rad2deg(self.eq_joints).tolist(),
                    actuator_torque=self.eq_torque.tolist(), noise=self.noise,
                    all_joint_names=[self.model.joint(int(j)).name for j in self.model.actuator_trnid[:,0]],
                    all_joint_position_degrees=np.rad2deg(self.eq_all_joints).tolist(),
                    model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                    foot_position_jacobian=self.foot_jacobian, timestep=self.dt,
                    mujoco_version=mujoco.__version__)

    def contacts(self):
        feet = set()
        pairs = {}
        abnormal = False
        for c in self.data.contact:
            if c.dist > 0:
                continue
            a, b = map(int, c.geom)
            if self.floor in (a,b):
                other = b if a == self.floor else a
                if other in self.feet:
                    feet.add(other)
                else:
                    abnormal = True
            else:
                pair = tuple(sorted((a,b)))
                pairs[pair] = max(pairs.get(pair,0.), -float(c.dist))
        return len(feet), pairs, abnormal

    def run(self, *, tau=.75, action=None, disturbance=None, seconds=5., controller=None,
            stop_on_self=False, seed=42):
        m,d = self.model,self.data
        mujoco.mj_setState(m,d,self.state,self.state_spec)
        mujoco.mj_forward(m,d)
        perturb = disturbance or {}
        kind = perturb.get('kind','none')
        onset = float(perturb.get('onset',1.)) if kind == 'force' else 0.
        end = onset + float(perturb.get('duration',0.)) if kind == 'force' else 0.
        axis = int(perturb.get('axis',0))
        value = float(perturb.get('value',0.))
        if kind == 'orientation':
            angle = math.radians(value)
            half = np.zeros(3); half[axis] = angle/2
            perturb_q = np.r_[math.cos(angle/2), np.sin(half)]
            out = np.empty(4)
            mujoco.mju_mulQuat(out, perturb_q, d.qpos[3:7].copy())
            d.qpos[3:7] = out
            mujoco.mj_forward(m,d)
            bottom = np.min(d.geom_xpos[self.feet,2]-m.geom_size[self.feet,0])
            d.qpos[2] -= bottom + 1e-6
        elif kind == 'angular_velocity':
            d.qvel[3+axis] += value
        elif kind == 'joint':
            d.qpos[self.qids] += np.asarray(perturb['offset_degrees'])*math.pi/180
        mujoco.mj_forward(m,d)
        initial = dict(rpy=orientation(d.qpos[3:7]).tolist(), height=float(d.qpos[2]),
                       joint_position=d.qpos[self.qids].tolist(), angular_velocity=d.qvel[3:6].tolist())
        filtered = np.zeros(12)
        rng = np.random.default_rng(seed)
        records = []
        torque_peak = 0.
        saturation = self_steps = foot_loss = abnormal_steps = 0
        min_height = float('inf')
        contact_events = []
        active_pairs = {}
        terminated = False
        warn_before = [int(w.number) for w in d.warning]
        for control in range(round(seconds/self.control_dt)):
            t = control*self.control_dt
            if controller is not None:
                request = np.asarray(controller(t, orientation(d.qpos[3:7]), d.qvel[3:6].copy()))
            elif isinstance(action,str) and action == 'random':
                request = rng.uniform(-1,1,12)
            else:
                request = np.zeros(12) if action is None else np.asarray(action)
            request = np.clip(request,-1,1)
            alpha = 1. if tau == 0 else -math.expm1(-self.control_dt/tau)
            filtered += alpha*(request-filtered)
            d.ctrl[:12] = np.clip(self.default+self.scale*filtered,m.actuator_ctrlrange[:12,0],m.actuator_ctrlrange[:12,1])
            d.ctrl[12:] = 0
            peak = np.zeros(12)
            step_self = False
            for substep in range(self.decimation):
                ts = t + substep*self.dt
                d.xfrc_applied[:] = 0
                if kind == 'force' and onset <= ts < end:
                    d.xfrc_applied[self.base,axis] = value
                mujoco.mj_step(m,d)
                peak = np.maximum(peak,abs(d.actuator_force[:12]))
                saturation += np.count_nonzero(abs(d.actuator_force[:12]) >= .52-1e-6)
                count,pairs,abnormal = self.contacts()
                foot_loss += count != 4
                abnormal_steps += abnormal
                step_self |= bool(pairs)
                for pair,depth in pairs.items():
                    if pair not in active_pairs:
                        active_pairs[pair] = dict(pair=list(pair),start=ts,duration=0.,peak_depth=0.,depths=[])
                    event = active_pairs[pair]
                    event['duration'] += self.dt
                    event['peak_depth'] = max(event['peak_depth'],depth)
                    event['depths'].append(depth)
                for pair in list(active_pairs):
                    if pair not in pairs:
                        contact_events.append(active_pairs.pop(pair))
                finite = np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.qacc).all()
                rpy = orientation(d.qpos[3:7]) if finite else np.full(3,np.nan)
                if not finite or abnormal or d.qpos[2]<.10 or np.linalg.norm(rpy[:2])>math.pi/4 or (stop_on_self and pairs):
                    terminated = True
                    break
            min_height = min(min_height,float(d.qpos[2]))
            self_steps += step_self
            torque_peak = max(torque_peak,float(peak.max()))
            records.append(dict(time=t+(substep+1)*self.dt,rpy=rpy.tolist(),height=float(d.qpos[2]),
                angular_velocity=d.qvel[3:6].tolist(),target=d.ctrl[:12].tolist(),
                joint_position=d.qpos[self.qids].tolist(),torque=d.actuator_force[:12].tolist(),
                peak_torque=peak.tolist(),feet=count,self_contact=step_self,
                linear_velocity=d.qvel[:3].tolist(),xy=d.qpos[:2].tolist(),applied_action=filtered.tolist()))
            if terminated:
                break
        contact_events.extend(active_pairs.values())
        for event in contact_events:
            event['bodies'] = [m.body(int(m.geom_bodyid[g])).name for g in event['pair']]
        a=np.array([[*r['rpy'][:2],r['height']] for r in records])
        time=np.array([r['time'] for r in records])
        eq=self.eq[[0,1,3]]
        snapshots={str(t):records[np.argmin(abs(time-t))] for t in [.1,.2,.5,1.,2.] if time[-1]>=t-1e-8}
        metrics={}
        for label,target in [('level',np.array([0.,0.,eq[2]])),('equilibrium',eq)]:
            errors=a-target
            post=time>=onset
            stable_feet=np.array([r['feet']==4 for r in records])
            recovery={}
            for threshold in [.6,.2,.1]:
                good=(np.max(abs(errors[:,:2]),axis=1)<math.radians(threshold)) & (abs(errors[:,2])<.005) & stable_feet
                good &= time>=end
                streak=0;rec=None
                for i,g in enumerate(good):
                    streak=streak+1 if g else 0
                    if streak>=10:
                        rec=max(0.,time[i]-end-.18)
                        break
                recovery[str(threshold)]=rec
            peak_tilt=float(np.rad2deg(np.linalg.norm(errors[post,:2],axis=1)).max())
            if kind != 'force':
                peak_tilt=max(peak_tilt,float(np.rad2deg(np.linalg.norm(np.array(initial['rpy'][:2])-target[:2]))))
            metrics[label]=dict(peak_tilt_degrees=peak_tilt,
                integrated_tilt_degree_seconds=float(np.rad2deg(np.linalg.norm(errors[post,:2],axis=1)).sum()*self.control_dt),
                final_error_degrees=np.rad2deg(errors[-1,:2]).tolist(),recovery_seconds=recovery)
        elapsed_phys=sum(round(r['time']/self.dt)-round((records[i-1]['time'] if i else 0)/self.dt) for i,r in enumerate(records))
        return dict(tau=tau,disturbance=disturbance,terminated=terminated,peak_torque=torque_peak,
            saturation_samples=int(saturation),self_contact_control_steps=int(self_steps),
            foot_loss_fraction=foot_loss/elapsed_phys,unexpected_ground_samples=abnormal_steps,
            min_height=min_height,warning_delta=[int(w.number)-b for w,b in zip(d.warning,warn_before)],
            final_delta=np.r_[np.rad2deg(a[-1,:2]-eq[:2]),1000*(a[-1,2]-eq[2])].tolist(),
            initial=initial,snapshots=snapshots,records=records,contact_events=contact_events,metrics=metrics)


def patterns():
    result={}
    front=np.array([1,1,-1,-1]);left=np.array([1,-1,1,-1])
    for name,sign in [('pitch',front),('roll',left)]:
        for joint,column in [('hip',1),('knee',2)]:
            a=np.zeros((4,3));a[:,column]=sign;result[name+'_'+joint]=a.ravel()
        a=np.zeros((4,3));a[:,1]=-.5*sign;a[:,2]=sign
        result[name+'_combo']=a.ravel()
    a=np.zeros((4,3));a[:,0]=1;result['hip_roll_common']=a.ravel().copy()
    a[:,0]=left;result['hip_roll_differential']=a.ravel()
    return result


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def run_authority(probe, output):
    singles=[]
    for i,name in enumerate(LEG_JOINT_NAMES):
        for sign in [-1,1]:
            action=np.zeros(12);action[i]=sign
            r=probe.run(action=action,seconds=4.)
            r.update(joint=name,sign=sign);singles.append(r)
    save(output/'single_actions.json',singles)
    combos=[]
    for name,action in patterns().items():
        for magnitude in [-1.,-.5,.5,1.]:
            r=probe.run(action=action*magnitude,seconds=4.)
            r.update(pattern=name,magnitude=magnitude,action=(action*magnitude).tolist());combos.append(r)
    save(output/'patterns.json',combos)
    print(f'action 응답 저장 완료: {output}')


def run_smoothing(probe, output):
    p=output
    pat=patterns()
    ablations=[]
    for tau in [.75,.30,.15,.05]:
     cases=[('zero',None,None),('random','random',None)]
     for name in ['pitch_knee','roll_knee','pitch_combo','roll_combo']:
      for sign in [-1,1]:cases.append((name+str(sign),pat[name]*sign,None))
     for sign in [-1,1]:cases.append(('all'+str(sign),np.ones(12)*sign,None))
     for axis in [0,1]:
      for sign in [-1,1]:cases.append((f'push_{axis}_{sign}',None,dict(kind='force',axis=axis,value=sign,duration=.15)))
     unsafe=False
     for label,action,dist in cases:
      r=probe.run(tau=tau,action=action,disturbance=dist,seconds=4.,stop_on_self=True);r['case']=label;ablations.append(r)
      if r['self_contact_control_steps'] or r['terminated'] or r['saturation_samples'] or any(r['warning_delta']):unsafe=True;break
     save(p/'smoothing.json',ablations)
     print('필터 안전성 검사',tau,'제외' if unsafe else '통과',flush=True)

def run_disturbances(probe, output):
    p=output
    pat=patterns()
    sweeps=[]
    for duration in [.15,.3,.5,1.]:
     for force in [.25,.5,.75,1.,1.25,1.5]:
      for axis in [0,1]:
       for sign in [-1,1]:
        r=probe.run(disturbance=dict(kind='force',axis=axis,value=force*sign,duration=duration),seconds=5.)
        r.update(category='force',force=force,duration=duration,axis=axis,sign=sign);sweeps.append(r)
    for kind,values in [('orientation',[.5,1.,2.,3.]),('angular_velocity',[.02,.05,.1,.2])]:
     for value in values:
      for axis in [0,1]:
       for sign in [-1,1]:
        r=probe.run(disturbance=dict(kind=kind,axis=axis,value=value*sign),seconds=5.)
        r.update(category=kind,value=value,axis=axis,sign=sign);sweeps.append(r)
    for name in ['pitch_combo','roll_combo']:
     for magnitude in [.25,.5,1.]:
      for sign in [-1,1]:
       offset=pat[name]*np.array([.1,.5,.5]*4)*magnitude*sign
       r=probe.run(disturbance=dict(kind='joint',offset_degrees=offset.tolist()),seconds=5.)
       r.update(category='joint',pattern=name,magnitude=magnitude,sign=sign);sweeps.append(r)
    save(p/'disturbances.json',sweeps)
    print('외란 개별 측정 완료',len(sweeps),flush=True)

def run_controllers(probe, output):
    p=output
    pat=patterns()
    combos=json.loads((p/'patterns.json').read_text())
    basis_names=['roll_knee','pitch_knee'];basis=np.stack([pat[n] for n in basis_names],axis=1)
    J=[]
    for n in basis_names:
     plus=next(x for x in combos if x['pattern']==n and x['magnitude']==.5)
     minus=next(x for x in combos if x['pattern']==n and x['magnitude']==-.5)
     J.append(np.array(plus['final_delta'][:2])-np.array(minus['final_delta'][:2]))
    J=np.array(J).T;inv=np.linalg.inv(J)
    save(p/'action_basis.json',dict(pattern_names=basis_names,action_basis=basis.tolist(),response_degrees=J.tolist(),singular_values=np.linalg.svd(J)[1].tolist()))
    rows=[]
    for tau in [.75,.3,.15,.05]:
     for goal in ['equilibrium','level']:
      target=probe.eq[:2] if goal=='equilibrium' else np.zeros(2)
      feedforward=inv@np.rad2deg(target-probe.eq[:2])
      for kp,kd in [(.5,.1),(1.,.1),(2.,.2)]:
       def controller(t,rpy,omega):
        command=feedforward+inv@np.rad2deg(kp*(target-rpy[:2])-kd*omega[:2])
        return basis@np.clip(command,-1,1)
       for axis in [0,1]:
        for sign in [-1,1]:
         dist=dict(kind='force',axis=axis,value=sign*.75,duration=.5)
         r=probe.run(tau=tau,disturbance=dist,controller=controller,seconds=5.)
         r.update(goal=goal,kp=kp,kd=kd,axis=axis,sign=sign);rows.append(r)
     save(p/'controllers.json',rows)
     print('필터별 제어 비교 완료',tau,flush=True)
    save(p/'controllers.json',rows)

def run_validation(probe, output):
    p=output
    pat=patterns()
    b=json.loads((p/'action_basis.json').read_text());basis=np.array(b['action_basis']);inv=np.linalg.inv(np.array(b['response_degrees']))
    rows=[]
    for kind in ['force','orientation','angular_velocity','joint']:
     cases=[]
     if kind=='force':
      for value in [.5,.75,1.]:
       for duration in [.15,.3,.5,1.]:
        for axis in [0,1]:
         for sign in [-1,1]:cases.append(dict(kind=kind,value=value*sign,duration=duration,axis=axis))
     elif kind in ['orientation','angular_velocity']:
      values=[1.,2.] if kind=='orientation' else [.05,.1]
      for value in values:
       for axis in [0,1]:
        for sign in [-1,1]:cases.append(dict(kind=kind,value=value*sign,axis=axis))
     else:
      for name,action in patterns().items():
       if name not in ['roll_combo','pitch_combo']:continue
       for sign in [-1,1]:cases.append(dict(kind=kind,offset_degrees=(action*np.array([.1,.5,.5]*4)*sign).tolist()))
     for case in cases:
      for tau in [.75,.3,.15,.05]:
       for label in ['zero','scripted']:
        def controller(t,rpy,omega):
         command=inv@np.rad2deg(2*(probe.eq[:2]-rpy[:2])-.2*omega[:2])
         return basis@np.clip(command,-1,1)
        r=probe.run(tau=tau,disturbance=case,controller=controller if label=='scripted' else None,seconds=5.)
        r.update(policy=label,category=kind);rows.append(r)
     save(p/'validation.json',rows)
     print('독립 외란 비교 완료',kind,flush=True)
    save(p/'validation.json',rows)

def main():
    parser=argparse.ArgumentParser(description='MicroDog 기립 제어 권한의 물리 진단')
    parser.add_argument('--output',type=Path,default=Path('/tmp/microdog_authority'))
    parser.add_argument('--suite',choices=['all','authority','smoothing','disturbances','controllers','validation'],default='authority')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    probe=AuthorityProbe()
    save(args.output/'equilibrium.json',probe.equilibrium())
    if args.suite in ('all','authority'):
        run_authority(probe,args.output)
    if args.suite in ('all','smoothing'):
        run_smoothing(probe,args.output)
    if args.suite in ('all','disturbances'):
        run_disturbances(probe,args.output)
    if args.suite in ('all','controllers'):
        if not (args.output/'patterns.json').exists():
            run_authority(probe,args.output)
        run_controllers(probe,args.output)
    if args.suite in ('all','validation'):
        if not (args.output/'action_basis.json').exists():
            run_authority(probe,args.output)
            run_controllers(probe,args.output)
        run_validation(probe,args.output)

if __name__=='__main__':
    main()
