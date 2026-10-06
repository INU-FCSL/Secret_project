"""native PPO의 loss gradient에 약한 actor mean penalty만 추가한다."""
import math
from rsl_rl.algorithms import PPO


class MeanRegularizedPPO(PPO):
    def __init__(self, *args, mean_regularization=0., **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(mean_regularization) or mean_regularization<0:
            raise ValueError('mean regularization coefficient는 유한한 음수 아닌 값이어야 합니다.')
        if self.symmetry or self.rnd or self.is_multi_gpu or self.actor.is_recurrent:
            raise ValueError('이 진단은 단일 GPU MLP PPO 전용입니다.')
        self.mean_regularization=mean_regularization

    def update(self):
        if self.mean_regularization==0:
            return super().update()
        original=self.optimizer.zero_grad
        penalties=[]
        def zero_grad(*args, **kwargs):
            original(*args, **kwargs)
            penalty=self.mean_regularization*self.actor.distribution.mean.square().mean()
            # native update의 zero_grad 직후, loss.backward 직전에 호출된다.
            # ∇(L_PPO+c·mean(μ²))=∇L_PPO+∇penalty를 정확히 누적하고
            # native PPO가 결합 gradient를 clipping·Adam에 전달하게 한다.
            # graph는 이어지는 native loss.backward가 사용할 수 있게 유지한다.
            penalty.backward(retain_graph=True)
            penalties.append(float(penalty.detach()))
        self.optimizer.zero_grad=zero_grad
        try:
            result=super().update()
        finally:
            self.optimizer.zero_grad=original
        result['mean_regularization']=sum(penalties)/len(penalties)
        return result
