# zof-rl：四足机器人 IK 步态基准 + RL 残差混合控制

本项目实现了一套以 **解析 IK 步态基准 + 强化学习残差偏移** 为核心的四足机器人运动控制框架，从 Isaac Gym PPO 训练、TorchScript 策略导出，到 MuJoCo 闭环验证与真机部署协议，形成完整、可对齐、可复现的 sim-to-real 链路。

与"端到端策略直接输出关节目标 / 力矩"的常见做法不同，本项目让策略网络只学一件难事——**在解析步态基准之上做修正**，从而把探索空间从 12 维绝对关节角压到小幅残差，让训练更快、更稳、更易迁移到真机。

---

## 一、核心创新：IK 步态基准 + RL 残差混合控制

策略网络不输出绝对关节角，也不输出力矩，而是输出 12 维残差：

```
target_q = ik_gait_reference(t, cmd) + action * residual_scale
torque   = Kp * (target_q - q) - Kd * dq
```

其中残差按关节类型差异化缩放，体现解剖学先验：

| 关节类型 | residual_scale | 含义 |
|---|---|---|
| hip（侧摆） | 0.06 | 侧摆幅度需小，避免横向摆腿过大 |
| thigh / calf（矢状面） | 0.16 | 矢状面允许更大修正，承担步态微调 |

`ik_gait_reference` 在训练与部署端被**同一份代码原样复现**（见 [zof_gym/envs/zof_robot.py](zof_gym/envs/zof_robot.py) 与 [scripts/deploy_mujoco.py](scripts/deploy_mujoco.py)），保证 sim 与 real 的控制协议物理一致。

**为何这样设计**：
- 模型驱动的步态生成器提供"会走"的先验，策略只学"走得更稳"的残差，探索空间缩小一个数量级；
- 残差有界，初期随机策略不会让机器人乱飞乱撞，训练早期存活率显著提升；
- 策略网络不需要从零学步态相位，可以直接利用相位锁定的足端轨迹；
- 部署时即便策略推理出现小幅异常，IK 基准仍是一个安全的回退姿态。

---

## 二、闭环解析 IK + Bezier 摆动曲线足端轨迹生成器

在 [zof_gym/envs/zof_robot.py](zof_gym/envs/zof_robot.py) 中实现了完全向量化、可微化的足端轨迹生成：

- **矢状面闭环 IK**：仅对 x-z 平面解析求逆，把 `cos`/`atan2` 严格 clamp 到数值安全区间，避免可达域边界奇异，训练中零 NaN；
- **Bezier 摆动曲线**：摆动相采用 `s²(3-2s)` smoothstep + 抛物线高度 `4s(1-s)·swing_height`，足端轨迹是连续可导的贝塞尔形状，而非折线，落地冲击小；
- **指令自适应步长**：`step_length = clamp(base + gain·vx, min, max)`，步长随指令速度连续伸缩，让策略无需自己学"走快点就把腿迈大"；
- **对角小跑步态相位**：`phase_offsets = [0.0, 0.5, 0.5, 0.0]`，FL/RR 与 FR/RL 配对，duty=0.58 偏向有支撑占空比，提升稳定性；
- **站立门控**：当指令低于阈值时，`x_delta`/`z_delta` 归零，reference 退化为 `default_q`，无需切换策略即可原地站立。

---

## 三、模式感知的双模奖励设计（行走 / 站立）

通过 `_stand_command_mask()` 在线区分行走与站立两种模式，分别施加针对性奖励项：

**行走模式**：
- `tracking_lin_vel` / `tracking_ang_vel`：速度跟踪主奖励
- `foot_bezier_tracking`：足端 3D 位置跟踪 IK 目标轨迹，加权 `[1.0, 0.25, 1.0]`（横向容差最大）
- `contact_schedule`：实际接触模式与相位期望接触的 L1 一致性
- `foot_lateral_deviation`：容差式惩罚（仅当横向偏移 > 0.035 m 才计入），避免奖励项在正常摆动时被"白扣"
- `long_air_time`：仅惩罚超过 0.55 s 的过长摆动相

**站立模式**（仅当 stand mask 为真才生效）：
- `stand_base_motion`：机身 xy 线速度与 yaw 角速度归零
- `stand_orientation_flat`：`projected_gravity` 的 xy 分量归零
- `stand_joint_posture`：关节回正到 default
- `stand_foot_stance`：四足端保持默认支撑构型
- `stand_foot_contact`：四脚同时保持接触

更关键的是训练端**显式采样站立命令**：`stand_probability = 0.25` 让 25% 的环境采样到 `[0,0,0]` 指令，使策略真正见过"原地站立"任务，而非依赖部署端硬切 `default_q`。

---

## 四、解剖学先验驱动的关节专属奖励塑形

奖励项不采用统一 `Σ torque²` 的粗放方式，而是按关节在运动学中的角色分别塑形：

| 奖励项 | 作用关节 | 设计动机 |
|---|---|---|
| `hip_deviation` | 4 个 hip | 侧摆关节应贴近默认角，避免横摆过大 |
| `hip_abduction_limit` | 4 个 hip | **容差式**：仅当 `\|q- default\| > 0.12 rad` 才惩罚，正常小调整不被扣分 |
| `calf_too_straight` | 4 个 calf | 仅惩罚 calf 角 > -1.0 的过伸状态，避免膝盖反张 |
| `calf_too_straight` 的阈值 1.0 | — | 来自对 zof URDF 几何的实际分析 |
| `ik_residual` | 12 维 action | `\Σ action²`，正则化策略输出幅度，鼓励"少修正" |

**容差式惩罚**是本项目奖励设计的一个反复出现的范式——只在状态越过容忍带时才计入惩罚，避免正常运动被无差别扣分导致策略保守化。

---

## 五、双频率部署架构：hold target_q，not torque

这是真机部署中最容易被忽视、却直接决定成败的工程创新。

**错误做法**（项目早期版本）：
```
每 DECIMATION=4 步：
  policy 推理 → target_q → torque
中间 3 步：
  复用旧 torque          ← 致命
```

**正确做法**（当前 [scripts/deploy_mujoco.py](scripts/deploy_mujoco.py)）：
```
每 DECIMATION=4 步（50 Hz）：
  policy 推理 → target_q_hold = ref_q + action * res_scale
每个物理步（200 Hz）：
  重新读 q/dq
  torque = Kp * (target_q_hold - q) - Kd * dq
  torque = clip(torque, ±limits)
```

**核心原则**：低频更新目标，高频更新反馈。接触地面时 dq 变化极快，若 torque 只 50 Hz 更新，PD 阻尼项严重滞后，机器人会出现抽搐、乱蹬、视觉上"被弹飞"。这一发现使部署效果从"4.5 秒后倒地"变为"稳定行走"。

---

## 六、Sim-to-Real 部署契约的严格对齐

本项目把"训练协议"和"部署协议"视为同一份契约，在多个维度强制对齐：

### 1. 关节顺序显式定义，不信任加载默认值

训练标准顺序：`FL → FR → RL → RR`，每腿 `hip → thigh → calf`。在 MuJoCo 部署中通过 `mj_name2id` 按名字建立 `q_idx / v_idx / act_idx` 映射表，**完全不依赖硬编码索引**，避免不同物理引擎加载顺序差异导致的错位。

### 2. Observation 只用真机可获得的信息

45 维观测严格排除 `base_lin_vel`（真机无准确测量源），只用：
- IMU 角速度（body frame）+ projected gravity
- 指令 vx, vy, wz
- 关节位置误差（相对 default）、关节速度、上次动作

### 3. MuJoCo 部署使用 IMU sensor 而非 ground truth

部署脚本优先读取 MuJoCo 的 `imu_quat` / `imu_gyro` sensor，**故意不读 `qvel[3:6]` ground truth**，让 MuJoCo 闭环更接近真机感知协议，提前暴露 IMU 坐标系、四元数顺序等部署陷阱。

### 4. S 刹车不关闭 policy

关键部署经验：`S` 不能写成 `action = 0, ref_q = default_q`——这会让运动中的机器人瞬间失去策略残差、被 PD 硬拉回 default 而前扑。正确做法是 `S` 仅提供 `[0,0,0]` 指令，policy 仍输出残差协助主动刹车，并对 `target_q` 施加 2 rad/s 的变化率限幅做站立过渡。

---

## 七、基于行为测试的 checkpoint 选择

不选取最后一个 checkpoint。通过 MuJoCo headless 回归测试套件量化评估各 checkpoint：

```bash
python scripts/deploy_mujoco.py --headless --headless_mode stand   --steps 1000
python scripts/deploy_mujoco.py --headless --headless_mode forward --steps 1000 --cmd_vx 0.5
python scripts/deploy_mujoco.py --headless --headless_mode left    --steps 1000 --turn_wz 0.5
python scripts/deploy_mujoco.py --headless --headless_mode right   --steps 1000 --turn_wz 0.5
```

四个模式的通过判据（高度范围、速度收敛区间）已固化。最终 v7 实验选取 `checkpoint_1500` 而非更晚的 checkpoint——因为后期 episode length 出现崩塌迹象，行为测试比训练曲线更能反映真实可用性。

**当前 baseline（2026-07-04 冻结）**：

| 模式 | 状态 | 关键指标 |
|---|---|---|
| stand | pass | height 0.299–0.301 m，wz 收敛至 -0.002 rad/s |
| forward | pass | vx 0.38–0.49 m/s（指令 0.5），轻微 yaw 漂移 |
| left turn | pass | wz +0.30 ~ +0.58 rad/s |
| right turn | pass | wz -0.19 ~ -0.59 rad/s |

---

## 八、工程结构

```
zof_gym/envs/
  zof_config.py       # 任务配置：观测/动作/奖励/gait 参数
  zof_robot.py        # IK 步态生成 + 残差 PD 控制 + 自定义奖励
scripts/
  train.py            # PPO 训练入口（基于 rsl_rl）
  export_policy.py    # 导出 TorchScript actor
  deploy_mujoco.py    # MuJoCo 可视化 / headless 部署验证
  check_*.py          # 模型/策略/姿态/资源检查脚本
resources/
  robots/zof/         # URDF + MuJoCo XML（含 deploy 版本）
  policies/zof_flat/  # 导出的 policy.pt
notes/                # 设计笔记与调试记录（每个设计决策都有据可查）
```

设计笔记目录 [notes/](notes/) 完整记录了每个设计决策的动机、备选方案与失败教训，是项目方法论的载体，而非附属文档。

---

## 九、设计哲学

1. **混合优于纯学习**：把可解析的部分（步态、IK）交给模型，把难解析的部分（地形适应、姿态修正）交给 RL。
2. **容差优于硬惩罚**：奖励项只在状态越过容忍带时才计入，避免策略过度保守。
3. **协议优于代码**：训练—导出—部署是同一份契约的三次实现，而非三套独立程序。
4. **行为测试优于训练曲线**：checkpoint 的选取以闭环行为为准，不以 reward 数值为准。
5. **工程先于真机**：每个部署陷阱（PD 频率、S 刹车、关节顺序）都在 MuJoCo 中提前发现并固化。

---

## 后续路线

- 斜坡 / 复杂地形：当前 `stand_orientation_flat` 仅适合平地，将引入 terrain normal 对齐的站立奖励；
- MoE 多专家：在当前 IK + residual 基础上扩展多专家门控；
- 真机部署：协议已就绪（见 [notes/real_robot_deploy_protocol_plan.md](notes/real_robot_deploy_protocol_plan.md)），待硬件就绪。
