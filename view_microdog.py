from pathlib import Path
import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL = Path(__file__).with_name("microdog.xml")
model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)

# qpos = [base xyz + quaternion] + 17 joint positions
# We only set joint positions by joint name to avoid ordering mistakes.
pose_deg = {
    # FL / FR front legs
    "fl_hip_roll_joint":  8,  "fl_hip_pitch_joint": -18, "fl_knee_joint": 38,
    "fr_hip_roll_joint": -8,  "fr_hip_pitch_joint": -18, "fr_knee_joint": 38,
    # RL / RR rear legs
    "rl_hip_roll_joint":  8,  "rl_hip_pitch_joint":  18, "rl_knee_joint": 38,
    "rr_hip_roll_joint": -8,  "rr_hip_pitch_joint":  18, "rr_knee_joint": 38,
    # head / tail
    "head_yaw_joint": 0, "head_pitch_joint": 0, "head_roll_joint": 0,
    "tail_yaw_joint": 0, "tail_pitch_joint": -10,
}

for name, deg in pose_deg.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    adr = model.jnt_qposadr[jid]
    data.qpos[adr] = np.deg2rad(deg)

mujoco.mj_forward(model, data)

# PREVIEW MODE:
# Do not step physics; simply display the MicroDog pose without falling/shaking.
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.0, 0.0, 0.12]
    viewer.cam.distance = 0.55
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -18

    while viewer.is_running():
        viewer.sync()
        time.sleep(1/60)
