"""auto-reset 전 terminal 관측으로 시간 제한 bootstrap을 계산한다."""
import torch
from rsl_rl.algorithms import PPO


class TerminalBootstrapPPO(PPO):
    def process_env_step(self, obs, rewards, dones, extras):
        timeout=extras.get('time_outs')
        if timeout is None or not timeout.any():
            return super().process_env_step(obs,rewards,dones,extras)
        if 'terminal_observation' not in extras:
            raise ValueError('timeout bootstrap에는 auto-reset 전 terminal 관측이 필요합니다.')
        timeout=timeout.bool()
        terminated=extras.get('diagnostics',{}).get('terminated')
        if terminated is not None:
            timeout=timeout & ~terminated.bool()
        with torch.no_grad():
            terminal_value=self.critic(extras['terminal_observation']).squeeze(-1)
            corrected=rewards+torch.where(timeout,self.gamma*terminal_value,0)
        # native V(s_t) 보정을 제거하고 정확한 V(terminal)를 한 번만 더한다.
        forwarded=dict(extras,time_outs=torch.zeros_like(timeout))
        return super().process_env_step(obs,corrected,dones,forwarded)
