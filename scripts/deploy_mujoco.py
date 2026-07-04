"""
deploy_mujoco.py — MuJoCo 可视化部署验证程序
=============================================
不依赖 Isaac Gym / legged_gym / rsl_rl。
直接加载 policy.pt，在 MuJoCo 里运行并弹出可视化窗口。

用法：
    conda activate /home/zhz/pan1/leggedgym
    cd /home/zhz/Desktop/myRL
    python scripts/deploy_mujoco.py              # 带可视化窗口（默认）
    python scripts/deploy_mujoco.py --headless   # 无窗口，只打印数字
    python scripts/deploy_mujoco.py --cmd_vx 0.3 # 改变前进速度
"""

import argparse
import os

import mujoco
import mujoco.viewer
import numpy as np
import torch

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH    = os.path.join(REPO_ROOT, "resources", "robots", "zof", "xml", "zof_deploy_from_urdf.xml")
DEFAULT_POLICY_PATH = os.path.join(
    REPO_ROOT,
    "resources", "policies", "zof_flat",
    "Jul04_19-37-52_ppo_v7_stand_corrected_flat",
    "checkpoint_1500", "policy.pt",
)

# ---------------------------------------------------------------------------
# 训练超参（必须和 zof_config.py 完全一致）
# ---------------------------------------------------------------------------
SIM_DT     = 0.005   # MuJoCo 物理步长（秒）
DECIMATION = 4       # 每 4 个物理步 policy 出一次动作 → 50 Hz
POLICY_DT  = SIM_DT * DECIMATION   # 0.02 s

KP = 40.0    # PD 位置增益 (Nm/rad)
KD =  1.0    # PD 速度增益 (Nm·s/rad)
TORQUE_LIMITS = np.array([
    23.7, 23.7, 35.55,
    23.7, 23.7, 35.55,
    23.7, 23.7, 35.55,
    23.7, 23.7, 35.55,
], dtype=np.float64)

CMD_VX_DEFAULT = 0.5   # 默认前向速度指令 m/s
INIT_PITCH = -0.05    # rad, 让默认关节角下四个脚尽量同时接触地面

# ---------------------------------------------------------------------------
# 关节名（训练顺序：FL FR RL RR，每腿 hip/thigh/calf）
# 这是 policy 输出的 12 维动作对应的关节顺序
# ---------------------------------------------------------------------------
JOINT_NAMES_TRAIN_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",   # 0,1,2
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",   # 3,4,5
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",   # 6,7,8
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",   # 9,10,11
]

FOOT_BODY_NAMES = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]

# ---------------------------------------------------------------------------
# 默认关节角（训练顺序）
# ---------------------------------------------------------------------------
DEFAULT_Q_TRAIN = np.array([
    -0.1,  1.1, -1.5,   # FL
     0.1,  1.1, -1.5,   # FR
    -0.1,  1.3, -1.5,   # RL
     0.1,  1.3, -1.5,   # RR
], dtype=np.float64)

# ---------------------------------------------------------------------------
# 残差缩放（训练顺序）
# ---------------------------------------------------------------------------
HIP_IDS   = [0, 3, 6,  9]
THIGH_IDS = [1, 4, 7, 10]
CALF_IDS  = [2, 5, 8, 11]

RES_SCALE = np.zeros(12, dtype=np.float64)
for i in HIP_IDS:
    RES_SCALE[i] = 0.06
for i in THIGH_IDS + CALF_IDS:
    RES_SCALE[i] = 0.16

# ---------------------------------------------------------------------------
# 观测缩放
# ---------------------------------------------------------------------------
OBS_ANG_VEL = 0.25
OBS_DOF_POS = 1.0
OBS_DOF_VEL = 0.05
CMD_LIN_VEL_SCALE = 2.0
CMD_ANG_VEL_SCALE = 0.25

# ---------------------------------------------------------------------------
# IK 步态参数
# ---------------------------------------------------------------------------
GAIT_FREQ       = 2.2
DUTY_FACTOR     = 0.58
PHASE_OFFSETS   = np.array([0.0, 0.5, 0.5, 0.0])  # FL FR RL RR
THIGH_LEN       = 0.220
CALF_LEN        = 0.219
SWING_HEIGHT    = 0.055
STEP_LEN_BASE   = 0.055
STEP_LEN_CMD_GAIN = 0.050
MIN_STEP_LEN    = 0.045
MAX_STEP_LEN    = 0.105


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def fk_xz(thigh, calf):
    x =  THIGH_LEN * np.sin(thigh) + CALF_LEN * np.sin(thigh + calf)
    z = -THIGH_LEN * np.cos(thigh) - CALF_LEN * np.cos(thigh + calf)
    return x, z


def ik_xz(x, z):
    eps = 1e-6
    r = np.sqrt(x*x + z*z)
    r = np.clip(r, abs(THIGH_LEN - CALF_LEN) + 1e-4, THIGH_LEN + CALF_LEN - 1e-4)
    cos_calf = (r*r - THIGH_LEN**2 - CALF_LEN**2) / (2 * THIGH_LEN * CALF_LEN)
    cos_calf = np.clip(cos_calf, -1+eps, 1-eps)
    calf  = -np.arccos(cos_calf)
    cos_a = (THIGH_LEN**2 + r*r - CALF_LEN**2) / (2 * THIGH_LEN * r)
    cos_a = np.clip(cos_a, -1+eps, 1-eps)
    thigh = np.arctan2(x, -z) + np.arccos(cos_a)
    return thigh, calf


def compute_gait_reference(policy_step: int, cmd_vx: float, cmd_wz: float = 0.0) -> np.ndarray:
    """返回 IK 步态基准关节角，训练顺序 (12,)"""
    t = policy_step * POLICY_DT
    phase = np.mod(t * GAIT_FREQ + PHASE_OFFSETS, 1.0)

    sl = np.clip(STEP_LEN_BASE + STEP_LEN_CMD_GAIN * max(cmd_vx, 0.0),
                 MIN_STEP_LEN, MAX_STEP_LEN)
    swing = phase >= DUTY_FACTOR

    stance_s = np.clip(phase / DUTY_FACTOR, 0.0, 1.0)
    swing_s  = np.clip((phase - DUTY_FACTOR) / (1.0 - DUTY_FACTOR), 0.0, 1.0)
    swing_curve = swing_s**2 * (3.0 - 2.0 * swing_s)

    x_delta = np.where(swing, -0.5*sl + sl*swing_curve,  0.5*sl - sl*stance_s)
    z_delta = np.where(swing, SWING_HEIGHT * 4.0 * swing_s * (1.0 - swing_s), 0.0)

    thigh_def = DEFAULT_Q_TRAIN[THIGH_IDS]
    calf_def  = DEFAULT_Q_TRAIN[CALF_IDS]
    def_fx, def_fz = fk_xz(thigh_def, calf_def)

    thigh, calf = ik_xz(def_fx + x_delta, def_fz + z_delta)

    target = DEFAULT_Q_TRAIN.copy()
    for i, (ti, ci) in enumerate(zip(THIGH_IDS, CALF_IDS)):
        target[ti] = thigh[i]
        target[ci] = calf[i]
    return target


def quat_rotate_inverse_wxyz(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """world → body 坐标系旋转，四元数格式 (w,x,y,z)"""
    w, x, y, z = q_wxyz
    t = 2.0 * np.array([y*v[2] - z*v[1],
                         z*v[0] - x*v[2],
                         x*v[1] - y*v[0]])
    return v - w * t + np.cross([x, y, z], t)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="不开窗口，只打印数字")
    p.add_argument("--steps", type=int, default=0, help="物理步数（0=无限）")
    p.add_argument("--policy_path", type=str, default=DEFAULT_POLICY_PATH, help="TorchScript policy.pt 路径")
    p.add_argument("--cmd_vx", type=float, default=CMD_VX_DEFAULT, help="W 前进速度指令 m/s")
    p.add_argument("--turn_wz", type=float, default=0.5, help="A/D 原地转向角速度指令 rad/s")
    p.add_argument("--headless_mode", choices=["stand", "forward", "left", "right"], default="forward", help="headless 自动运行的命令模式")
    p.add_argument("--kp", type=float, default=KP, help="部署 PD 位置增益")
    p.add_argument("--kd", type=float, default=KD, help="部署 PD 速度增益")
    return p.parse_args()


def build_index_maps(model):
    """
    按关节名从 MuJoCo model 里查出：
      - q_idx[i]  : qpos 里第 i 个训练关节的位置索引（qposadr）
      - v_idx[i]  : qvel 里第 i 个训练关节的速度索引（dofadr）
      - act_idx[i]: ctrl 里第 i 个训练关节对应的力矩 actuator 索引
    这样完全不依赖硬编码的顺序。
    """
    q_idx   = np.zeros(12, dtype=np.int32)
    v_idx   = np.zeros(12, dtype=np.int32)
    act_idx = np.zeros(12, dtype=np.int32)

    for i, jname in enumerate(JOINT_NAMES_TRAIN_ORDER):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError(f"关节 '{jname}' 在 XML 里找不到")
        q_idx[i] = model.jnt_qposadr[jid]
        v_idx[i] = model.jnt_dofadr[jid]

        # 找名字和关节名相同的 motor actuator
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        if aid < 0:
            raise RuntimeError(f"力矩 actuator '{jname}' 在 XML 里找不到")
        act_idx[i] = aid

    return q_idx, v_idx, act_idx


def find_foot_sphere_geoms(model):
    """Find the four small foot contact spheres under the calf bodies."""
    foot_geom_ids = []

    for body_name in FOOT_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise RuntimeError(f"足端 body {body_name!r} 在 XML 里找不到")

        candidates = []
        for geom_id in range(model.ngeom):
            is_same_body = model.geom_bodyid[geom_id] == body_id
            is_sphere = model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
            can_contact = model.geom_contype[geom_id] != 0
            if is_same_body and is_sphere and can_contact:
                candidates.append(geom_id)

        if not candidates:
            raise RuntimeError(f"足端 body {body_name!r} 下没有可接触 sphere geom")

        foot_geom_ids.append(candidates[-1])

    return np.array(foot_geom_ids, dtype=np.int32)


def set_default_pose_on_ground(model, data, q_idx, foot_geom_ids, clearance=0.004):
    """Set default pose and place the feet just above the floor."""
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[3] = np.cos(INIT_PITCH / 2.0)
    data.qpos[5] = np.sin(INIT_PITCH / 2.0)

    for i, qi in enumerate(q_idx):
        data.qpos[qi] = DEFAULT_Q_TRAIN[i]

    data.qpos[2] = 0.42
    mujoco.mj_forward(model, data)

    foot_bottom_z = np.min(
        data.geom_xpos[foot_geom_ids, 2] - model.geom_size[foot_geom_ids, 0]
    )
    data.qpos[2] += clearance - foot_bottom_z
    mujoco.mj_forward(model, data)

    return data.qpos[2]




def sensor_slice(model, sensor_name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        return None
    start = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return slice(start, start + dim)

def make_step_fn(model, data, policy, q_idx, v_idx, act_idx, command_state, kp, kd):
    """
    返回一个闭包，每次调用执行一个 policy step（含 DECIMATION 个物理步）。
    使用闭包是为了让 headless 和 viewer 两条路复用同一段控制逻辑。
    command_state: {"vx": float, "stand": bool} —— 运行时可被键盘回调修改
    """
    last_action  = np.zeros(12)
    target_q_hold = DEFAULT_Q_TRAIN.copy()
    policy_step  = [0]
    phys_step    = [0]
    imu_quat_slice = sensor_slice(model, "imu_quat")
    imu_gyro_slice = sensor_slice(model, "imu_gyro")

    def step():
        # 每 DECIMATION 个物理步运行一次策略，只更新 target_q_hold
        if phys_step[0] % DECIMATION == 0:
            # 从 command_state 读当前指令
            cmd_vx_now = command_state["vx"]
            cmd_wz_now = command_state["wz"]
            stand_now  = command_state["stand"]

            # 读关节状态（训练顺序）
            q  = np.array([data.qpos[qi] for qi in q_idx])
            dq = np.array([data.qvel[vi] for vi in v_idx])

            # 躯干状态：优先使用 MuJoCo IMU sensor，更接近真机部署协议。
            if imu_quat_slice is not None and imu_gyro_slice is not None:
                base_quat = data.sensordata[imu_quat_slice].copy()
                ang_vel_body = data.sensordata[imu_gyro_slice].copy()
            else:
                base_quat = data.qpos[3:7].copy()   # wxyz
                ang_vel_world = data.qvel[3:6].copy()
                ang_vel_body = quat_rotate_inverse_wxyz(base_quat, ang_vel_world)
            proj_grav = quat_rotate_inverse_wxyz(base_quat, np.array([0.0, 0.0, -1.0]))

            cmd_scaled = np.array([
                cmd_vx_now * CMD_LIN_VEL_SCALE,
                0.0,
                cmd_wz_now * CMD_ANG_VEL_SCALE,
            ])

            # 构造 45 维观测
            obs = np.concatenate([
                ang_vel_body * OBS_ANG_VEL,
                proj_grav,
                cmd_scaled,
                (q - DEFAULT_Q_TRAIN) * OBS_DOF_POS,
                dq * OBS_DOF_VEL,
                last_action,
            ]).astype(np.float32)

            # 策略推理 + IK 基准 → 只更新 target_q_hold
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

            if stand_now:
                # 站立命令也交给 policy，只是 reference 切回 default_q
                ref_q = DEFAULT_Q_TRAIN.copy()
            else:
                ref_q = compute_gait_reference(policy_step[0], cmd_vx_now, cmd_wz_now)

            last_action[:] = action

            target_q_new = ref_q + action * RES_SCALE

            if stand_now:
                # S 站立：限速过渡到 DEFAULT_Q，避免瞬间跳变
                max_target_rate = 2.0  # rad/s
                max_delta = max_target_rate * POLICY_DT
                delta = np.clip(
                    target_q_new - target_q_hold,
                    -max_delta,
                    max_delta,
                )
                target_q_hold[:] = target_q_hold + delta
            else:
                # W 走路：直接跟踪 policy 输出
                target_q_hold[:] = target_q_new
            policy_step[0] += 1

        # 每个物理步都重新读 q/dq 并重算 PD（200 Hz 反馈）
        q  = np.array([data.qpos[qi] for qi in q_idx])
        dq = np.array([data.qvel[vi] for vi in v_idx])
        torques = kp * (target_q_hold - q) - kd * dq
        torques = np.clip(torques, -TORQUE_LIMITS, TORQUE_LIMITS)

        # 把力矩发到对应的 actuator
        for i in range(12):
            data.ctrl[act_idx[i]] = torques[i]

        mujoco.mj_step(model, data)
        phys_step[0] += 1

        # 每 100 步打印一次
        if phys_step[0] % 100 == 0:
            h = data.qpos[2]
            vx = data.qvel[0]
            wz = data.qvel[5]
            cmd_vx = command_state["vx"]
            cmd_wz = command_state["wz"]
            mode = "stand" if command_state["stand"] else "move"
            print(
                f"  phys={phys_step[0]:5d}  mode={mode:5s} "
                f"cmd=({cmd_vx:.2f},{cmd_wz:.2f}) "
                f"height={h:.3f} m  vx={vx:.3f} m/s  wz={wz:.3f} rad/s"
            )

        return phys_step[0]

    return step


def run(args):
    # ------------------------------------------------------------------
    # 1. 加载 MuJoCo 模型
    # ------------------------------------------------------------------
    print(f"[deploy] 加载模型: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    print(f"  nu={model.nu}  nq={model.nq}  nv={model.nv}")

    # ------------------------------------------------------------------
    # 2. 建索引映射（按名字查，不依赖顺序）
    # ------------------------------------------------------------------
    q_idx, v_idx, act_idx = build_index_maps(model)
    print(f"  act_idx (训练顺序→ctrl索引): {act_idx.tolist()}")

    # ------------------------------------------------------------------
    # 3. 初始姿态
    # ------------------------------------------------------------------
    mujoco.mj_resetData(model, data)
    foot_geom_ids = find_foot_sphere_geoms(model)
    init_height = set_default_pose_on_ground(model, data, q_idx, foot_geom_ids)
    print(f"  init_height={init_height:.3f} m  init_pitch={INIT_PITCH:.3f} rad  foot_geom_ids={foot_geom_ids.tolist()}")

    # ------------------------------------------------------------------
    # 4. 加载策略
    # ------------------------------------------------------------------
    print(f"[deploy] 加载策略: {args.policy_path}")
    policy = torch.jit.load(args.policy_path, map_location="cpu")
    policy.eval()

    # ------------------------------------------------------------------
    # 5. 运行
    # ------------------------------------------------------------------
    print(f"  deploy_pd: kp={args.kp:.2f}  kd={args.kd:.2f}")

    command_state = {
        "vx": 0.0,
        "wz": 0.0,
        "stand": True,
    }

    def set_command(vx, wz, stand):
        command_state["vx"] = vx
        command_state["wz"] = wz
        command_state["stand"] = stand

    def key_callback(keycode):
        if keycode == ord("W"):
            set_command(args.cmd_vx, 0.0, False)
            print(f"[key] W: walk forward vx={args.cmd_vx}")
        elif keycode == ord("A"):
            set_command(0.0, args.turn_wz, False)
            print(f"[key] A: turn left wz={args.turn_wz}")
        elif keycode == ord("D"):
            set_command(0.0, -args.turn_wz, False)
            print(f"[key] D: turn right wz={-args.turn_wz}")
        elif keycode == ord("S"):
            set_command(0.0, 0.0, True)
            print("[key] S: stand")

    step_fn = make_step_fn(model, data, policy, q_idx, v_idx, act_idx, command_state, args.kp, args.kd)
    max_steps = args.steps if args.steps > 0 else 10**9

    if args.headless:
        if args.headless_mode == "stand":
            set_command(0.0, 0.0, True)
        elif args.headless_mode == "forward":
            set_command(args.cmd_vx, 0.0, False)
        elif args.headless_mode == "left":
            set_command(0.0, args.turn_wz, False)
        elif args.headless_mode == "right":
            set_command(0.0, -args.turn_wz, False)

        print(
            f"[deploy] headless，mode={args.headless_mode}，"
            f"cmd=({command_state['vx']:.2f},{command_state['wz']:.2f})，"
            f"运行 {args.steps or '∞'} 物理步 ..."
        )
        phys = 0
        while phys < max_steps:
            phys = step_fn()
        print("[deploy] 完成")

    else:
        print("[deploy] 启动 MuJoCo viewer，关闭窗口或 Ctrl+C 退出")
        print(f"[deploy] 键盘: W=前进(vx={args.cmd_vx})  A=左转(wz={args.turn_wz})  D=右转(wz={-args.turn_wz})  S=站立")
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.cam.distance  = 2.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth   = 160

            phys = 0
            while viewer.is_running() and phys < max_steps:
                phys = step_fn()
                viewer.sync()

        print("[deploy] viewer 关闭")


if __name__ == "__main__":
    args = parse_args()
    run(args)
