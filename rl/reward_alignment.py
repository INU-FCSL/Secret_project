"""동일 trajectory의 물리 순위와 보상항·pose 재점수화를 비교한다."""
import argparse
import json
from pathlib import Path
import numpy as np
from .config import REWARD_WEIGHTS

TERMS=('upright','height','pose','joint_velocity','action_rate','effort')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def ranks(values):
    values=np.asarray(values);order=np.argsort(values,kind='stable');result=np.empty(len(values),float)
    start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[order[end]]==values[order[start]]:end+=1
        result[order[start:end]]=(start+end-1)/2+1;start=end
    return result


def correlation(x,y,rank=False):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if rank:x,y=ranks(x),ranks(y)
    return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and x.std()>0 and y.std()>0 else None


def sources(base=Path('/tmp/microdog_v6')):
    fix=json.loads((base/'fix/evaluation/summary.json').read_text())
    ref=json.loads(Path('/tmp/microdog_v5/reference/summary.json').read_text())
    result={k:fix['baselines'][k] for k in ('zero','scripted')}
    for i in range(6):result[f'imitation_{i}']=json.loads((base/f'imitation/evaluation/checkpoint_{i}/summary.json').read_text())
    result['v3e75']=ref['v3e75'];return result


def audit(output):
    rows={};arrays={};first=None
    for name,summary in sources().items():
        data=dict(np.load(summary['trace']));arrays[name]=data
        if first is None:first=data
        assert np.array_equal(data['raw'][0],first['raw'][0]),name
        assert np.array_equal(data['push_start'],first['push_start']),name
        time=np.arange(len(data['reward']))[:,None]*.02;start=data['push_start']
        masks=dict(pre=data['valid']&(time<start),during=data['valid']&(time>=start)&(time<start+.5),
            recovery=data['valid']&(time>=start+.5)&(time<start+1.5),
            steady=data['valid']&(time>=start+1.5),episode=data['valid'])
        segments={}
        for phase,mask in masks.items():
            component=(data['terms'].astype(float)*mask[...,None]).sum(0)
            segments[phase]=dict(components=dict(zip(TERMS,component.mean(0).tolist())),
                total=float(component.sum(-1).mean()),
                per_episode_components=component.tolist(),
                stored_reward_float64=float((data['reward'].astype(float)*mask).sum(0).mean()))
        metric=summary['aggregate']
        rows[name]=dict(segments=segments,physical={k:metric[k] for k in
            ('integrated_orientation_error','peak_orientation_error','recovery_seconds')},trace=summary['trace'])
    names=list(rows);physical=sorted(names,key=lambda n:tuple(rows[n]['physical'][k] for k in
        ('integrated_orientation_error','peak_orientation_error','recovery_seconds')))
    current=float(REWARD_WEIGHTS['pose']);candidates=(current,current*.5,current*.25,0.)
    rescored={}
    for weight in candidates:
        values={n:rows[n]['segments']['episode']['total']+
            (weight/current-1)*rows[n]['segments']['episode']['components']['pose'] for n in names}
        after={n:sum(rows[n]['segments'][s]['total']+(weight/current-1)*rows[n]['segments'][s]['components']['pose']
            for s in ('during','recovery','steady')) for n in names}
        worse=[f'imitation_{i}' for i in range(1,6)]
        checks=dict(good_above_worse=all(values['scripted']>values[n] and values['imitation_0']>values[n] for n in worse),
            deterioration_0_to_5=values['imitation_5']<values['imitation_0'],
            better_than_zero=all(values[n]>values['zero'] for n in ['scripted']+[f'imitation_{i}' for i in range(6)]),
            pose_retained=weight>0)
        rescored[str(weight)]=dict(weight=weight,return_value=values,post_disturbance_return=after,
            return_ranking=sorted(names,key=lambda n:-values[n]),checks=checks,
            spearman_integrated=correlation(list(values.values()),[-rows[n]['physical']['integrated_orientation_error'] for n in names],True),
            spearman_peak=correlation(list(values.values()),[-rows[n]['physical']['peak_orientation_error'] for n in names],True))
    eligible=[v['weight'] for v in rescored.values() if all(v['checks'].values())]
    selected=max(eligible) if eligible else None
    changes=[]
    for i in range(1,6):
        old,new=rows[f'imitation_{i-1}'],rows[f'imitation_{i}']
        changes.append(dict(from_update=i-1,to_update=i,
            components={k:new['segments']['episode']['components'][k]-old['segments']['episode']['components'][k] for k in TERMS},
            total=new['segments']['episode']['total']-old['segments']['episode']['total'],
            physical={k:new['physical'][k]-old['physical'][k] for k in old['physical']}))
    full_change={k:rows['imitation_5']['segments']['episode']['components'][k]-rows['imitation_0']['segments']['episode']['components'][k] for k in TERMS}
    result=dict(current_pose_weight=current,trajectories=rows,physical_ranking=physical,
        return_ranking=rescored[str(current)]['return_ranking'],counterfactual=rescored,selected_pose_weight=selected,
        adjacent_updates=changes,update_0_to_5_components=full_change,
        reward_conflict=full_change['upright']<0 and full_change['pose']>-full_change['upright'],
        matched_initial_state_and_disturbance=True,
        note='보상항을 float64로 합산했다. 전체 episode와 외란 이후 지표를 구분하고 저장된 물리를 재실행하지 않았다.')
    save(output/'reward_audit.json',result)
    print('보상 순위·재점수화 완료:',output,'현재 pose weight',current,'선정 후보',selected,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description='Standing 물리 목표와 보상 순위 검증')
    parser.add_argument('--output',type=Path,default=Path('/tmp/microdog_v7/reward'));args=parser.parse_args()
    audit(args.output)


if __name__=='__main__':main()
