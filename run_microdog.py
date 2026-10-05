from pathlib import Path
import time
import mujoco
import mujoco.viewer

MODEL = Path(__file__).with_name("microdog.xml")
model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)

# XML에 정의한 기립 자세와 같은 목표각으로 시작한다.
standing_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_KEY, "neutral_standing"
)
if standing_id < 0:
    raise ValueError("XML에 neutral_standing keyframe이 없습니다.")
mujoco.mj_resetDataKeyframe(model, data, standing_id)
mujoco.mj_forward(model, data)

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
