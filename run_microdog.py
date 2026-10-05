from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL = Path(__file__).with_name("microdog.xml")
model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)

targets_deg = np.array([
     8, -18, 38,   # FL
    -8, -18, 38,   # FR
     8,  18, 38,   # RL
    -8,  18, 38,   # RR
     0,   0,  0,   # head
     0, -10        # tail
], dtype=float)

data.ctrl[:] = np.deg2rad(targets_deg)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.0, 0.0, 0.12]
    viewer.cam.distance = 0.55
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -18

    while viewer.is_running():
        t0 = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        dt = model.opt.timestep - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)
