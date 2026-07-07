# MuJoCo WASD Baseline v7.1-stop 封版记录

日期：2026-07-06

## 封版目的

本记录冻结当前已经通过 MuJoCo 平地 WASD 验收的部署 baseline。

它的定位是：

```text
平地 MuJoCo 部署回归基准
```

不是复杂地形、斜坡、盲走、MoE 或 NP3O 的起点。

## 固定文件

Policy:

```text
/home/zhz/Desktop/myRL/resources/policies/zof_flat/Jul04_19-37-52_ppo_v7_stand_corrected_flat/checkpoint_1500/policy.pt
```

对应训练 checkpoint:

```text
/home/zhz/Desktop/myRL/runs/zof_flat/Jul04_19-37-52_ppo_v7_stand_corrected_flat/model_1500.pt
```

MuJoCo XML:

```text
/home/zhz/Desktop/myRL/resources/robots/zof/xml/zof_deploy_from_urdf.xml
```

部署脚本:

```text
/home/zhz/Desktop/myRL/scripts/deploy_mujoco.py
```

## 固定部署协议

Observation 45 维：

```text
0-2:   base angular velocity, body frame
3-5:   projected gravity
6-8:   command vx, vy, wz
9-20:  joint position error = q - default_q
21-32: joint velocity
33-44: last action
```

Action 12 维：

```text
IK gait reference 上的 residual
```

目标关节角：

```text
target_q = reference_q + action * residual_scale
```

Residual scale:

```text
hip = 0.06
thigh/calf = 0.16
```

训练关节顺序：

```text
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
```

Default q:

```text
[-0.1, 1.1, -1.5,
  0.1, 1.1, -1.5,
 -0.1, 1.3, -1.5,
  0.1, 1.3, -1.5]
```

## 固定控制参数

MuJoCo:

```text
physics dt = 0.005 s
physics frequency = 200 Hz
policy decimation = 4
policy frequency = 50 Hz
```

PD:

```text
kp = 40.0
kd = 1.0
```

关键控制结构：

```text
policy / target_q 50 Hz 更新
PD torque 每个 MuJoCo 物理步 200 Hz 重新计算
hold target_q, not torque
```

S 刹车部署参数：

```text
stand target_q rate limit = 1.5 rad/s
```

这个限速只作用于 `stand=True` 时 target_q 的过渡。S 刹车仍然继续运行 policy residual。

## WASD 语义

```text
W: vx = args.cmd_vx, wz = 0
A: vx = 0, wz = +args.turn_wz
D: vx = 0, wz = -args.turn_wz
S: vx = 0, wz = 0, stand=True
```

A/D 是原地转向，不是横移。

## Headless 回归测试

测试目录：

```bash
cd /home/zhz/Desktop/myRL
```

### Stand

```bash
python scripts/deploy_mujoco.py --headless --steps 1000 --headless_mode stand
```

验收：

```text
稳定站立，不倒，不持续滑动。
```

### Forward short

```bash
python scripts/deploy_mujoco.py --headless --steps 1000 --headless_mode forward --cmd_vx 0.5
```

实测摘要：

```text
height = 0.279 ~ 0.299 m
vx     = 0.38 ~ 0.49 m/s
wz     = 约 -0.09 ~ +0.22 rad/s
```

结论：

```text
短时前进稳定，但有轻微 yaw 摆动。
```

### Left / Right

```bash
python scripts/deploy_mujoco.py --headless --steps 1000 --headless_mode left --turn_wz 0.5
python scripts/deploy_mujoco.py --headless --steps 1000 --headless_mode right --turn_wz 0.5
```

验收：

```text
左转、右转均稳定。
```

### Stop regression

```bash
python scripts/deploy_mujoco.py --headless --steps 1600 --headless_mode stop --cmd_vx 0.5
```

行为：

```text
前 800 物理步: forward, cmd=(0.5, 0.0)
后 800 物理步: stand, cmd=(0.0, 0.0)
```

实测摘要：

```text
step 800:  height=0.294 m, vx=0.420 m/s, wz=-0.090 rad/s
step 900:  height=0.301 m, vx=0.013 m/s, wz=-0.096 rad/s
step 1600: height=0.299 m, vx=0.000 m/s, wz=-0.008 rad/s
```

结论：

```text
通过。S 刹车在约 0.5 s 内把 vx 收到接近 0，高度稳定，无持续 yaw。
```

## 长时 Forward 诊断

命令：

```bash
python scripts/deploy_mujoco.py --headless --steps 5000 --headless_mode forward --cmd_vx 0.5
```

实测摘要：

```text
0-1000 steps:    vx 约 0.38 ~ 0.49 m/s
2000 steps:      vx 约 0.36 m/s
3000 steps:      vx 约 0.35 m/s
4000 steps:      vx 约 0.29 m/s
5000 steps:      vx 约 0.22 m/s
height:          约 0.278 ~ 0.299 m，未塌陷
wz:              周期摆动，峰值约 +0.26 / -0.12 rad/s
```

注意：

```text
5000 MuJoCo 物理步 = 25 s
当前训练 episode_length_s = 20 s
```

结论：

```text
长时前进不会摔，但速度保持不足，25 s 内 vx 从约 0.4 m/s 衰减到约 0.2 m/s。
```

## 封版结论

v7.1-stop 通过：

```text
stand: stable
forward short: stable
left/right: stable
stop: pass
```

v7.1-stop 不解决：

```text
长时 forward 掉速
直行 yaw 摆动
更自然的运动到站立过渡
真机动力学差异
```

因此，v7.1-stop 是可靠的 MuJoCo 部署基准，但不是最终训练策略。

