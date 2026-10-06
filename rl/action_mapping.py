"""정책의 latent action을 안전한 controller 명령으로 변환한다."""
import torch


def bounded_action(raw, mode='clip'):
    if mode=='clip':
        return raw.clamp(-1,1)
    if mode=='tanh':
        return raw.tanh()
    if mode=='identity':
        if not torch.isfinite(raw).all() or (raw.abs()>1).any():
            raise ValueError('bounded policy action은 유한한 [-1,1] 범위여야 합니다.')
        return raw
    raise ValueError('action mapping은 clip, tanh 또는 identity여야 합니다.')


def equivalent_raw(command, mode='clip'):
    """scripted의 기존 bounded 명령을 같은 의미의 latent 입력으로 변환한다."""
    if mode in ('clip','identity'):
        return command
    if mode=='tanh':
        return torch.atanh(command.clamp(-1+1e-6,1-1e-6))
    raise ValueError('action mapping은 clip, tanh 또는 identity여야 합니다.')
