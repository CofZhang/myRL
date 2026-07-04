import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

XML_PATH = Path("/home/zhz/Desktop/myRL/resources/robots/zof/xml/zof_deploy_from_urdf.xml")

DEFAULT_Q = np.array([
    -0.1, 1.1, -1.5,
     0.1, 1.1, -1.5,
    -0.1, 1.3, -1.5,
     0.1, 1.3, -1.5,
], dtype=np.float64)

FOOT_GEOMS = {
    "FL": 12,
    "FR": 22,
    "RL": 32,
    "RR": 42,
}


def geom_name(model, gid):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    if name is not None:
        return name

    body_id = model.geom_bodyid[gid]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return f"<unnamed geom on {body_name}>"


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # 先把 base 放在世界原点，只用来计算 default_q 下脚相对 base 的高度
    data.qpos[0:3] = np.array([0.0, 0.0, 0.0])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[7:19] = DEFAULT_Q
    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)

    print("Foot sphere positions with base_z = 0:")
    bottom_z_list = []

    for leg, gid in FOOT_GEOMS.items():
        center = data.geom_xpos[gid].copy()
        radius = model.geom_size[gid, 0]
        bottom_z = center[2] - radius
        bottom_z_list.append(bottom_z)

        print(
            f"{leg}: "
            f"center=({center[0]: .4f}, {center[1]: .4f}, {center[2]: .4f}) "
            f"radius={radius:.5f} "
            f"bottom_z={bottom_z: .4f}"
        )

    min_bottom_z = min(bottom_z_list)
    max_bottom_z = max(bottom_z_list)

    print()
    print(f"min foot bottom_z: {min_bottom_z:.4f}")
    print(f"max foot bottom_z: {max_bottom_z:.4f}")
    print(f"foot height spread: {max_bottom_z - min_bottom_z:.4f}")

    base_z_needed = -min_bottom_z + 0.005
    print(f"suggested base_z for reset: {base_z_needed:.4f}")

    print()
    print("Simulating 1 second with zero torque...")

    data.qpos[0:3] = np.array([0.0, 0.0, base_z_needed])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[7:19] = DEFAULT_Q
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    steps = int(1.0 / dt)

    for _ in range(steps):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)

    print("After 1 second zero torque:")
    print("base position:", data.qpos[0:3])
    print("base quat:", data.qpos[3:7])
    print("joint q:", data.qpos[7:19])
    print("joint dq:", data.qvel[6:18])
    print("has nan qpos:", np.isnan(data.qpos).any())
    print("has nan qvel:", np.isnan(data.qvel).any())
    print("ncon:", data.ncon)

    print()
    print("Simulating 2 seconds with PD holding default_q...")

    kp = 40.0
    kd = 1.0

    torque_limits = np.array([
        23.7, 23.7, 35.55,
        23.7, 23.7, 35.55,
        23.7, 23.7, 35.55,
        23.7, 23.7, 35.55,
    ], dtype=np.float64)

    data.qpos[0:3] = np.array([0.0, 0.0, base_z_needed])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[7:19] = DEFAULT_Q
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    mujoco.mj_forward(model, data)

    steps = int(2.0 / dt)

    max_abs_tau = 0.0

    for _ in range(steps):
        q = data.qpos[7:19].copy()
        dq = data.qvel[6:18].copy()

        tau = kp * (DEFAULT_Q - q) - kd * dq
        tau = np.clip(tau, -torque_limits, torque_limits)

        max_abs_tau = max(max_abs_tau, np.max(np.abs(tau)))

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

    print("After 2 seconds PD stand:")
    print("kp:", kp, "kd:", kd)
    print("base position:", data.qpos[0:3])
    print("base quat:", data.qpos[3:7])
    print("joint q:", data.qpos[7:19])
    print("joint dq:", data.qvel[6:18])
    print("max abs torque:", max_abs_tau)
    print("has nan qpos:", np.isnan(data.qpos).any())
    print("has nan qvel:", np.isnan(data.qvel).any())
    print("ncon:", data.ncon)

    print()
    print("Contacts after PD stand:")
    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = contact.geom1
        g2 = contact.geom2

        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)

        print(
            f"{i:2d}: "
            f"{geom_name(model, g1)} <-> {geom_name(model, g2)} "
            f"normal_force={force[0]: .3f}"
        )

    print()
    print("Opening viewer: pure PD holding default_q. Close the window to exit.")

    # 重新回到 default_q 初始姿态，避免从前面测试后的状态开始看
    data.qpos[0:3] = np.array([0.0, 0.0, base_z_needed])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[7:19] = DEFAULT_Q
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 140

        while viewer.is_running():
            q = data.qpos[7:19].copy()
            dq = data.qvel[6:18].copy()

            tau = kp * (DEFAULT_Q - q) - kd * dq
            tau = np.clip(tau, -torque_limits, torque_limits)

            data.ctrl[:] = tau

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
