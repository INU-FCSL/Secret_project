"""다리 순서 FL/FR/RL/RR의 명령을 직교 대칭 모드로 분해한다."""
import numpy as np

LEG_MODES=np.array([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],dtype=float)/2
MODE_NAMES=('common','front_rear','left_right','diagonal')


def stats(values,axis=0):
    return dict(mean=values.mean(axis).tolist(),std=values.std(axis).tolist(),
        min=values.min(axis).tolist(),max=values.max(axis).tolist(),count=len(values))


def mode_decomposition(actions):
    values=np.asarray(actions,dtype=float).reshape(-1,4,3)
    coefficients=np.einsum('ml,nlj->nmj',LEG_MODES,values)
    total=float((values**2).sum((1,2)).mean())
    result={}
    for i,name in enumerate(MODE_NAMES):
        energy=float((coefficients[:,i]**2).sum(-1).mean())
        result[name]=dict(coefficients=stats(coefficients[:,i],axis=0),energy=energy,
            energy_fraction=energy/total if total else 0.,
            mean_bias_energy=float((coefficients[:,i].mean(0)**2).sum()))
    return dict(total_energy=total,modes=result,energy_preserved=bool(np.isclose(
        sum(row['energy'] for row in result.values()),total,atol=1e-10)),
        joint_types=['hip_roll','hip_pitch','knee'])
