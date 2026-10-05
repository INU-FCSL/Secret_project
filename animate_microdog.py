import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


MODEL = Path(__file__).with_name("microdog.xml")

model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)


def set_joint(name, angle_deg):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )

    qpos_addr = model.jnt_qposadr[joint_id]

    data.qpos[qpos_addr] = np.deg2rad(angle_deg)


with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.cam.lookat[:] = [0.0, 0.0, 0.12]
    viewer.cam.distance = 0.55
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -18

    start_time = time.time()

    while viewer.is_running():

        t = time.time() - start_time

        # 천천히 -20 ~ +20 deg 반복
        swing = 20.0 * np.sin(2.0 * np.pi * 0.25 * t)

        # Hip roll
        set_joint("fl_hip_roll_joint", swing)
        set_joint("fr_hip_roll_joint", -swing)
        set_joint("rl_hip_roll_joint", swing)
        set_joint("rr_hip_roll_joint", -swing)

        # Hip pitch
        set_joint("fl_hip_pitch_joint", -20 + swing)
        set_joint("fr_hip_pitch_joint", -20 + swing)

        set_joint("rl_hip_pitch_joint", 20 - swing)
        set_joint("rr_hip_pitch_joint", 20 - swing)

        # Knee
        knee = 45 + 15 * np.sin(
            2.0 * np.pi * 0.25 * t
        )

        set_joint("fl_knee_joint", knee)
        set_joint("fr_knee_joint", knee)
        set_joint("rl_knee_joint", knee)
        set_joint("rr_knee_joint", knee)

        # 변경된 qpos로 geometry 다시 계산
        mujoco.mj_forward(model, data)

        viewer.sync()

        time.sleep(1 / 60)
