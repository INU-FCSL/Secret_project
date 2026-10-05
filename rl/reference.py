"""모델별 자연 평형 보정과 목표 자세의 중력 투영."""
from functools import lru_cache
import hashlib
import math
import numpy as np
import mujoco
from .config import MODEL_PATH


def gravity_from_rpy(rpy):
    roll, pitch, _ = rpy
    return (math.sin(pitch), -math.sin(roll)*math.cos(pitch),
            -math.cos(roll)*math.cos(pitch))


@lru_cache(maxsize=8)
def _calibrate(path, digest):
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('neutral_standing').id)
    for _ in range(round(35/model.opt.timestep)):
        mujoco.mj_step(model, data)
    w,x,y,z = data.qpos[3:7]
    rpy = (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
           math.asin(np.clip(2*(w*y-z*x),-1,1)),
           math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    if not np.isfinite(data.qpos).all() or any(warning.number for warning in data.warning):
        raise RuntimeError('자연 평형 보정 중 유효하지 않은 상태가 발생했습니다.')
    return dict(orientation=rpy, height=float(data.qpos[2]), model_sha256=digest,
                mujoco_version=mujoco.__version__, calibration_seconds=35.)


def calibrate_reference():
    return dict(_calibrate(str(MODEL_PATH), hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()))
