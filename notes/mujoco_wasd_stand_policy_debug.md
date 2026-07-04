# MuJoCo WASD 与站立刹车调试记录

日期：2026-07-04

## 背景

在 `ppo_v6_keyboard_flat` 后，机器人已经能在 Isaac Gym 中完成：

```text
站立
前进
左转
右转
```

之后为了改善“运动后切回站立时姿态不端正、前倾、腿斜”的问题，又训练了：

```text
runs/zof_flat/Jul04_19-37-52_ppo_v7_stand_corrected_flat/model_1500.pt
```

并导出到：

```text
resources/policies/zof_flat/Jul04_19-37-52_ppo_v7_stand_corrected_flat/checkpoint_1500/policy.pt
```

## 训练端修改

### 1. 命令空间

`zof_config.py` 中命令范围改为：

```python
lin_vel_x = [0.0, 0.8]
lin_vel_y = [0.0, 0.0]
ang_vel_yaw = [-0.8, 0.8]
```

含义：

```text
vx 支持从站立到前进
vy 暂时不训练横移
wz 支持左转和右转
```

### 2. 显式站立命令采样

加入：

```python
stand_probability = 0.25
```

并在 `ZofRobot._resample_commands()` 中让一部分环境采样到：

```text
vx = 0
vy = 0
wz = 0
```

目的：让 policy 真正见过站立命令，而不是只靠部署端硬切 default_q。

### 3. stand mask

在 `zof_robot.py` 中加入：

```python
def _stand_command_mask(self):
    return (
        (torch.abs(self.commands[:, 0]) < 0.05)
        & (torch.abs(self.commands[:, 1]) < 0.05)
        & (torch.abs(self.commands[:, 2]) < 0.10)
    )
```

这个 mask 用于判断当前是否是站立命令。

### 4. 站立时关闭 gait reference 踏步

在 `_compute_reference_targets()` 中，当 stand mask 为 true 时：

```python
x_delta = 0
z_delta = 0
```

效果：

```text
站立命令下 reference = default_q
运动命令下继续使用 trot reference
```

### 5. 站立矫正奖励

新增站立专用奖励：

```text
stand_base_motion
stand_orientation_flat
stand_joint_posture
stand_foot_stance
stand_foot_contact
```

它们只在站立命令下生效。

目标：

```text
身体速度接近 0
yaw 角速度接近 0
平地上身体 roll/pitch 接近水平
关节回到 default_q 附近
足端支撑形状接近默认站姿
四脚尽量稳定接触
```

当前权重：

```python
stand_base_motion = -2.0
stand_orientation_flat = -3.0
stand_joint_posture = -0.8
stand_foot_stance = -2.0
stand_foot_contact = -0.5
```

## 斜坡说明

当前奖励 `stand_orientation_flat` 只适合平地，因为它惩罚：

```text
projected_gravity[:, :2]
```

也就是希望 base 在世界坐标下水平。

斜坡上不能简单要求身体水平。斜坡稳定站立更合理的目标是：

```text
机器人 base 的 up 方向对齐地形法向 terrain normal
```

所以以后进入斜坡/复杂地形时，应增加基于 terrain normal 的站立姿态奖励，而不是继续使用平地水平奖励。

## MuJoCo 部署修改

### 1. 切换到新 policy

`deploy_mujoco.py` 中 `POLICY_PATH` 改为：

```text
resources/policies/zof_flat/Jul04_19-37-52_ppo_v7_stand_corrected_flat/checkpoint_1500/policy.pt
```

### 2. command_state 支持 yaw command

从：

```python
command_state = {
    "vx": 0.0,
    "stand": True,
}
```

扩展为：

```python
command_state = {
    "vx": 0.0,
    "wz": 0.0,
    "stand": True,
}
```

### 3. obs 中传入 wz

原来 MuJoCo 部署中 command obs 写死为：

```python
cmd_scaled = np.array([
    cmd_vx_now * CMD_LIN_VEL_SCALE,
    0.0,
    0.0,
])
```

这会导致 policy 永远看不到转向命令。

改为：

```python
cmd_scaled = np.array([
    cmd_vx_now * CMD_LIN_VEL_SCALE,
    0.0,
    cmd_wz_now * CMD_ANG_VEL_SCALE,
])
```

这样 A/D 才能通过 obs 传给 policy。

### 4. WASD 键盘语义

当前定义：

```text
W: vx = args.cmd_vx, wz = 0
A: vx = 0, wz = +0.5
D: vx = 0, wz = -0.5
S: vx = 0, wz = 0
```

注意：

```text
A/D 是原地转向，不是左右平移。
```

## 关键问题：S 刹车不能关闭 policy

最初 S 的逻辑是：

```python
if stand_now:
    action = np.zeros(12)
    ref_q = DEFAULT_Q_TRAIN.copy()
```

这个逻辑会导致：

```text
机器人正在走路时还有前向惯性
按 S 后 policy residual 被关闭
target_q 被强行拉回 default_q
机器人容易前扑或摔倒
```

这是错误的，因为训练协议是：

```text
target_q = reference + policy_residual
```

部署端不能在站立命令下突然变成：

```text
target_q = default_q + 0
```

正确逻辑是：

```python
with torch.no_grad():
    action = policy(obs)

if stand_now:
    ref_q = DEFAULT_Q_TRAIN.copy()
else:
    ref_q = compute_gait_reference(...)

 target_q = ref_q + action * RES_SCALE
```

也就是：

```text
S 不是关闭 policy
S 是给 policy 一个 [0, 0, 0] 的站立命令
```

policy 会根据当前身体速度、姿态、关节状态输出合适的 residual，辅助刹车和回正。

## 当前验证结论

在 MuJoCo 中：

```text
启动后站立正常
W 前进正常
A 左转正常
D 右转正常
W 后按 S 能稳定回到站立，不再明显前扑摔倒
```

## 后续建议

1. 保留当前 `ppo_v7` checkpoint 1500 作为平地 WASD 基线。
2. 不要用最后 checkpoint，训练后期 episode length 有掉崩迹象。
3. 后续优化 deploy 时，可以增加：

```text
--policy_path
--turn_wz
--cmd_vx
--stand_rate
```

4. 进入斜坡/复杂地形前，应重新设计 terrain normal 相关奖励，不要直接沿用平地水平站立奖励。
