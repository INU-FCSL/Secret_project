"""XML을 보존하는 mjlab Simulation 기반 GPU 기립 환경."""
import math
from contextlib import contextmanager
import numpy as np
import mujoco
import torch
import warp as wp
from tensordict import TensorDict
from rsl_rl.env import VecEnv
from mjlab.sim.sim import Simulation, SimulationCfg, MujocoCfg
from .config import StandingCfg, MODEL_PATH, LEG_JOINT_NAMES, ACTION_SCALE
from .rewards import standing_rewards


@wp.kernel
def _contact_events(ncon: wp.array(dtype=wp.int32), geom: wp.array(dtype=wp.vec2i),
                    world: wp.array(dtype=wp.int32), dist: wp.array(dtype=float),
                    feet: wp.array(dtype=wp.int32), flags: wp.array2d(dtype=wp.int32)):
    i = wp.tid()
    if i < ncon[0] and dist[i] <= 0.0:
        pair = geom[i]
        if pair[0] != 0 and pair[1] != 0:
            wp.atomic_max(flags, world[i], 0, 1)
        else:
            other = pair[0]
            if pair[0] == 0:
                other = pair[1]
            is_foot = False
            for j in range(4):
                if other == feet[j]:
                    is_foot = True
            if not is_foot:
                wp.atomic_max(flags, world[i], 1, 1)


class StandingEnv(VecEnv):
    num_actions = 12
    num_obs = 42

    def __init__(self, cfg=StandingCfg()):
        self.cfg = cfg
        self.device = cfg.device
        self.num_envs = cfg.num_envs
        if not self.device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('이 환경은 CUDA와 MuJoCo Warp가 필요합니다.')
        torch.manual_seed(cfg.seed)
        wp.init()
        self._wp_stream = wp.get_stream(self.device)
        self._cuda_stream = torch.cuda.ExternalStream(self._wp_stream.cuda_stream, device=self.device)
        self.mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        m = self.mj_model
        if (m.nq, m.nv, m.nu) != (24, 23, 17):
            raise ValueError('검증된 MicroDog 모델 차원이 아닙니다.')
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'neutral_standing')
        if key < 0:
            raise ValueError('neutral_standing keyframe이 필요합니다.')
        names = tuple(m.joint(int(j)).name for j in m.actuator_trnid[:12, 0])
        if names != LEG_JOINT_NAMES:
            raise ValueError('다리 actuator 순서가 예상과 다릅니다.')
        # Simulation 설정은 XML의 solver 설정과 같은 값을 사용한다.
        opt = m.opt
        simcfg = SimulationCfg(nconmax=128, njmax=512, mujoco=MujocoCfg(
            timestep=cfg.timestep, integrator='implicitfast', solver='newton',
            cone='pyramidal', iterations=int(opt.iterations), tolerance=float(opt.tolerance),
            ls_iterations=int(opt.ls_iterations), ls_tolerance=float(opt.ls_tolerance),
            ccd_iterations=int(opt.ccd_iterations), gravity=tuple(opt.gravity),
        ))
        with self._stream():
            self.sim = Simulation(self.num_envs, simcfg, m, self.device)
        self.qpos = wp.to_torch(self.sim.wp_data.qpos)
        self.qvel = wp.to_torch(self.sim.wp_data.qvel)
        self.qacc = wp.to_torch(self.sim.wp_data.qacc)
        self.ctrl = wp.to_torch(self.sim.wp_data.ctrl)
        self.torque = wp.to_torch(self.sim.wp_data.actuator_force)
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(np.array(x, copy=True), dtype=dtype, device=self.device)
        self.qpos_ids = tensor(m.jnt_qposadr[m.actuator_trnid[:12, 0]], torch.long)
        self.qvel_ids = tensor(m.jnt_dofadr[m.actuator_trnid[:12, 0]], torch.long)
        self.initial_qpos = tensor(m.key_qpos[key])
        self.initial_ctrl = tensor(m.key_ctrl[key])
        self.default_joint_position = self.initial_ctrl[:12]
        self.action_scale = tensor(ACTION_SCALE)
        joint_limits = m.jnt_range[m.actuator_trnid[:12, 0]]
        limits = np.stack([np.maximum(joint_limits[:, 0], m.actuator_ctrlrange[:12, 0]),
                           np.minimum(joint_limits[:, 1], m.actuator_ctrlrange[:12, 1])], axis=1)
        self.target_low = tensor(limits[:, 0])
        self.target_high = tensor(limits[:, 1])
        self.previous_actions = torch.zeros((self.num_envs, 12), device=self.device)
        self.requested_actions = torch.zeros_like(self.previous_actions)
        self.joint_targets = self.initial_ctrl[:12].expand(self.num_envs, -1).clone()
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_episode_length = math.ceil(cfg.episode_seconds / cfg.step_dt)
        self._feet = torch.tensor([next(g for g in range(m.ngeom)
            if m.geom_bodyid[g] == m.body(leg + '_lower').id
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
            for leg in ('fl', 'fr', 'rl', 'rr')], device=self.device)
        self._feet_wp = wp.from_torch(self._feet.to(torch.int32), dtype=wp.int32)
        with wp.ScopedDevice(self.device):
            self._events_wp = wp.zeros((self.num_envs, 2), dtype=wp.int32)
        self._events = wp.to_torch(self._events_wp)
        self.reset()

    @contextmanager
    def _stream(self):
        current = torch.cuda.current_stream(self.device)
        self._cuda_stream.wait_stream(current)
        with wp.ScopedStream(self._wp_stream), torch.cuda.stream(self._cuda_stream):
            yield
        current.wait_stream(self._cuda_stream)

    def reset(self, env_ids=None):
        ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None
               else torch.as_tensor(env_ids, dtype=torch.long, device=self.device))
        with self._stream():
            self.sim.reset(ids)
            self.qpos[ids] = self.initial_qpos
            self.qvel[ids] = 0
            self.ctrl[ids] = self.initial_ctrl
            self.previous_actions[ids] = 0
            self.requested_actions[ids] = 0
            self.joint_targets[ids] = self.default_joint_position
            self.episode_length_buf[ids] = 0
            self.sim.forward()
        return self.get_observations()

    def projected_gravity(self):
        # freejoint quaternion은 wxyz이며 회전행렬의 세 번째 행이 중력 투영에 대응한다.
        q = self.qpos[:, 3:7]
        w, x, y, z = q.unbind(-1)
        return torch.stack((-2*(x*z-w*y), -2*(y*z+w*x), -(1-2*(x*x+y*y))), -1)

    def get_observations(self):
        obs = torch.cat((self.projected_gravity(), self.qvel[:, 3:6],
            self.qpos[:, self.qpos_ids] - self.default_joint_position,
            self.qvel[:, self.qvel_ids], self.previous_actions), -1)
        return TensorDict({'actor': obs, 'critic': obs.clone()}, batch_size=[self.num_envs])

    def contact_status(self):
        contact = self.sim.wp_data.contact
        geom = wp.to_torch(contact.geom).long()
        world = wp.to_torch(contact.worldid).long()
        dist = wp.to_torch(contact.dist)
        ncon = wp.to_torch(self.sim.wp_data.nacon)
        valid = (torch.arange(len(dist), device=self.device) < ncon[0]) & (dist <= 0)
        # 비활성 배열 슬롯은 world 번호를 안전한 값으로 바꾼 뒤 0을 누적한다.
        world = world.clamp(0, self.num_envs - 1)
        floor = (geom[:, 0] == 0) | (geom[:, 1] == 0)
        other = torch.where(geom[:, 0] == 0, geom[:, 1], geom[:, 0])
        counts = torch.zeros((self.num_envs, 4), device=self.device)
        for i in range(4):
            counts[:, i].scatter_add_(0, world, (valid & floor & (other == self._feet[i])).float())
        self_count = torch.zeros(self.num_envs, device=self.device)
        self_count.scatter_add_(0, world, (valid & ~floor).float())
        abnormal = torch.zeros(self.num_envs, device=self.device)
        foot_geom = (other[:, None] == self._feet[None]).any(-1)
        abnormal.scatter_add_(0, world, (valid & floor & ~foot_geom).float())
        return counts > 0, self_count, abnormal > 0

    def step(self, actions):
        if tuple(actions.shape) != (self.num_envs, 12):
            raise ValueError('action shape은 (환경 수, 12)여야 합니다.')
        actions = actions.to(device=self.device, dtype=torch.float32).detach()
        if not torch.isfinite(actions).all():
            raise ValueError('action에 NaN 또는 무한값이 있습니다.')
        requested = actions.clamp(-1, 1)
        delta = requested - self.previous_actions
        self.requested_actions.copy_(requested)
        # 0.75초 시정수의 필터. 관측에 실제 적용 action을 포함해 필터 상태를 노출한다.
        alpha = -math.expm1(-self.cfg.step_dt / self.cfg.smoothing_seconds)
        self.previous_actions.add_(alpha * delta)
        applied_delta = alpha * delta
        self.joint_targets.copy_(torch.clamp(self.default_joint_position +
            self.action_scale * self.previous_actions, self.target_low, self.target_high))
        peak = torch.zeros_like(self.joint_targets)
        saturation = torch.zeros_like(self.joint_targets, dtype=torch.long)
        with self._stream():
            self._events.zero_()
            self.ctrl[:, :12] = self.joint_targets
            self.ctrl[:, 12:] = 0
            for _ in range(self.cfg.decimation):
                self.sim.step()
                peak = torch.maximum(peak, self.torque[:, :12].abs())
                saturation += self.torque[:, :12].abs() >= .52 - 1e-6
                c = self.sim.wp_data.contact
                wp.launch(_contact_events, dim=c.dist.shape[0], inputs=[
                    self.sim.wp_data.nacon, c.geom, c.worldid, c.dist,
                    self._feet_wp, self._events_wp], device=self.device)
            self.sim.forward()
        self.episode_length_buf += 1
        finite = (torch.isfinite(self.qpos).all(-1) & torch.isfinite(self.qvel).all(-1)
                  & torch.isfinite(self.qacc).all(-1) & torch.isfinite(self.torque).all(-1))
        gravity = self.projected_gravity()
        feet, self_contacts, abnormal = self.contact_status()
        self_contacts = torch.maximum(self_contacts, self._events[:, 0].float())
        abnormal = abnormal | self._events[:, 1].bool()
        reward, terms = standing_rewards(gravity, self.qpos[:, 2],
            self.qpos[:, self.qpos_ids] - self.default_joint_position,
            self.qvel[:, self.qvel_ids], applied_delta, self.torque[:, :12])
        reward = torch.where(finite, reward * self.cfg.step_dt, torch.zeros_like(reward))
        reasons = dict(invalid=~finite, low_height=self.qpos[:, 2] < self.cfg.min_height,
            tilt=gravity[:, 2] > -math.cos(math.radians(self.cfg.max_tilt_degrees)),
            abnormal_ground=abnormal)
        terminated = torch.stack(list(reasons.values())).any(0)
        timeout = self.episode_length_buf >= self.max_episode_length
        done = terminated | timeout
        terminal_obs = self.get_observations().clone()
        diagnostics = dict(height=self.qpos[:, 2].clone(), gravity=gravity.clone(),
            feet=feet.clone(), self_contacts=self_contacts.clone(), peak_torque=peak,
            saturation_steps=saturation, targets=self.joint_targets.clone(),
            applied_actions=self.previous_actions.clone(), reasons=reasons)
        extras = {'time_outs': timeout & ~terminated, 'diagnostics': diagnostics,
                  'terminal_observation': terminal_obs,
                  'log': {f'Reward/{k}': torch.nan_to_num(v).mean() for k, v in terms.items()}}
        if done.any():
            self.reset(torch.where(done)[0])
        return self.get_observations(), reward, done.long(), extras

    def close(self):
        torch.cuda.synchronize(self.device)
