# MuJoCo WASD Baseline 冻结记录

日期：2026-07-04

## Baseline 目的

本 baseline 用于冻结当前已经通过 MuJoCo 验收的平地 WASD 部署版本。

当前目标不是复杂地形、斜坡、盲走或 MoE/NP3O，而是确认下面这条链路已经稳定：

```text
Isaac Gym PPO 训练
-> policy.pt 导出
-> MuJoCo XML
-> 45 维 observation 构造
-> IK gait reference + policy residual
-> target_q
-> 高频 PD torque
-> MuJoCo WASD 控制
```

## 使用文件

### Policy

```text
resources/policies/zof_flat/Jul04_19-37-52_ppo_v7_stand_corrected_flat/checkpoint_1500/policy.pt
```

来源 checkpoint：

```text
runs/zof_flat/Jul04_19-37-52_ppo_v7_stand_corrected_flat/model_1500.pt
```

### MuJoCo XML

```text
resources/robots/zof/xml/zof_deploy_from_urdf.xml
```

这个 XML 来自 URDF 对齐版本，不再使用旧的手写错误 XML。

### 部署脚本

```text
scripts/deploy_mujoco.py
```

### 部署 PD

```text
kp = 40.0
kd = 1.0
```

### 控制频率

```text
MuJoCo physics dt = 0.005 s
physics frequency = 200 Hz
policy decimation = 4
policy frequency = 50 Hz
```

关键原则：

```text
policy / target_q 低频更新
PD torque 每个物理步高频重算
hold target_q, not torque
```

## WASD 命令语义

当前 `deploy_mujoco.py` 中：

```text
W: vx = args.cmd_vx, wz = 0
A: vx = 0, wz = +args.turn_wz
D: vx = 0, wz = -args.turn_wz
S: vx = 0, wz = 0, stand=True
```

注意：

```text
A/D 是原地转向，不是左右平移。
```

部署脚本会把 command 写入 observation：

```text
obs[6] = cmd_vx * CMD_LIN_VEL_SCALE
obs[7] = 0
obs[8] = cmd_wz * CMD_ANG_VEL_SCALE
```

## 重要部署结论

### S 刹车不能关闭 policy

错误逻辑：

```python
if stand_now:
    action = np.zeros(12)
    ref_q = DEFAULT_Q_TRAIN.copy()
```

这个逻辑会让运动中的机器人突然失去 policy residual，只靠 default_q + PD 硬拉回站姿，容易前扑摔倒。

正确逻辑：

```text
S 只是 command = [0, 0, 0]
policy 仍然参与输出 residual
reference 切到 default_q
target_q = default_q + policy_residual
```

原因：

```text
训练协议是 reference + policy residual。
部署协议必须保持一致。
```

## Headless 回归测试

测试命令均在：

```bash
cd /home/zhz/Desktop/myRL
```

### 1. Stand

命令：

```bash
python scripts/deploy_mujoco.py --headless --headless_mode stand --steps 1000
```

结果摘要：

```text
height: 0.299 ~ 0.301 m
vx:     接近 0
wz:     从 -0.026 收敛到约 -0.002 rad/s
```

结论：

```text
通过。站立稳定。
```

### 2. Forward

命令：

```bash
python scripts/deploy_mujoco.py --headless --headless_mode forward --steps 1000 --cmd_vx 0.5
```

结果摘要：

```text
height: 0.279 ~ 0.299 m
vx:     约 0.38 ~ 0.49 m/s
wz:     有轻微漂移，约 -0.09 ~ 0.22 rad/s
```

结论：

```text
通过。前进稳定，速度接近 0.5 m/s，但存在轻微 yaw 漂移。
```

### 3. Left Turn

命令：

```bash
python scripts/deploy_mujoco.py --headless --headless_mode left --steps 1000 --turn_wz 0.5
```

结果摘要：

```text
height: 0.278 ~ 0.294 m
vx:     约 -0.02 ~ 0.10 m/s
wz:     约 +0.30 ~ +0.58 rad/s
```

结论：

```text
通过。左转稳定。
```

### 4. Right Turn

命令：

```bash
python scripts/deploy_mujoco.py --headless --headless_mode right --steps 1000 --turn_wz 0.5
```

结果摘要：

```text
height: 0.283 ~ 0.296 m
vx:     约 -0.08 ~ 0.09 m/s
wz:     约 -0.19 ~ -0.59 rad/s
```

结论：

```text
通过。右转稳定。
```

## Baseline 结论

当前 MuJoCo 平地 WASD baseline 通过：

```text
stand   pass
forward pass
left    pass
right   pass
```

可以进入下一阶段：

```text
真机部署前协议整理
```

## 当前已知不足

1. `forward` 有轻微 yaw 漂移。
2. A/D 只支持原地转向，不支持横移。
3. 当前奖励中的 `stand_orientation_flat` 只适合平地，不适合斜坡。
4. MuJoCo 仍然不等于真机，真机部署前必须检查关节方向、零点、力矩限制和安全逻辑。

