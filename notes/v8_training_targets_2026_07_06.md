# v8 平地训练目标设计

日期：2026-07-06

## 总原则

v8 不改变部署协议。

必须保持：

```text
observation = 45 维
action = 12 维 IK reference residual
joint order = FL, FR, RL, RR / hip, thigh, calf
target_q = reference_q + action * residual_scale
residual scale = hip 0.06, thigh/calf 0.16
MuJoCo deploy PD = kp 40, kd 1
```

v8 的目标不是引入新算法，而是在当前 v7.1-stop baseline 上修正平地行为质量。

## v7.1-stop 暴露的问题

### 1. 长时 forward 掉速

5000 物理步 forward 测试中：

```text
cmd_vx = 0.5 m/s
前 5 s:  vx 约 0.38 ~ 0.49 m/s
25 s:    vx 约 0.22 m/s
height:  未塌陷
```

这说明策略不是失稳，而是长期速度保持能力不足。

优先级：

```text
最高
```

### 2. 直行 yaw 摆动

forward 测试中 `cmd_wz=0`，但 `wz` 会周期性摆动：

```text
wz 峰值约 +0.26 / -0.12 rad/s
```

这说明直行对称性还不够好。

优先级：

```text
中
```

### 3. S 刹车点头

部署侧用：

```text
stand target_q rate limit = 1.5 rad/s
```

已经让 stop regression 通过。训练侧仍可以让 policy 更自然地处理运动到零速的过渡。

优先级：

```text
中低
```

## v8 训练验收标准

### 必须通过

```bash
python scripts/deploy_mujoco.py --headless --steps 1000 --headless_mode stand
python scripts/deploy_mujoco.py --headless --steps 1000 --headless_mode forward --cmd_vx 0.5
python scripts/deploy_mujoco.py --headless --steps 1000 --headless_mode left --turn_wz 0.5
python scripts/deploy_mujoco.py --headless --steps 1000 --headless_mode right --turn_wz 0.5
python scripts/deploy_mujoco.py --headless --steps 1600 --headless_mode stop --cmd_vx 0.5
```

### 新增长时 forward 验收

```bash
python scripts/deploy_mujoco.py --headless --steps 5000 --headless_mode forward --cmd_vx 0.5
```

目标：

```text
后 1000 步平均 vx >= 0.35 m/s
height 不低于 0.27 m
abs(wz) 不持续大于 0.20 rad/s
不倒、不拖腿、不明显塌高
```

## 训练侧候选改动

### A. 延长 episode

当前：

```python
episode_length_s = 20
```

v8 候选：

```python
episode_length_s = 30
```

理由：

```text
长时 forward 测试是 25 s，而当前训练 episode 只有 20 s。
策略需要在训练里见过更长的连续行走。
```

风险：

```text
训练更慢；如果奖励设计不稳定，长 episode 可能放大坏 gait。
```

教学判断：

```text
建议作为 v8 的第一优先改动。
```

### B. 提高前进速度保持压力

当前：

```python
tracking_lin_vel = 3.0
tracking_sigma = 0.10
```

v8 候选：

```python
tracking_lin_vel = 3.5 或 4.0
tracking_sigma = 0.12
```

理由：

```text
速度掉到 0.2 m/s 仍然没摔，说明生存和姿态奖励可能比长期速度跟踪更容易满足。
略提高 tracking_lin_vel 可以让策略更重视持续前进。
```

不建议把 `tracking_sigma` 调得更小。太小会让训练早期梯度变窄，容易只学到保守动作。

风险：

```text
如果 tracking 过强，可能换来更大的身体摆动或脚滑。
```

### C. 增加零 yaw 直行约束

候选新增 reward：

```text
forward_yaw_rate
```

只在直行命令时生效：

```text
cmd_vx > 0.2
abs(cmd_wz) < 0.1
```

惩罚：

```text
base_ang_vel_z^2
```

建议权重：

```python
forward_yaw_rate = -0.2 到 -0.5
```

理由：

```text
不要全局惩罚 yaw，否则会伤害 A/D 原地转向。
只在直行命令下压 yaw，更符合当前问题。
```

风险：

```text
权重太大会让策略不愿意转向，或者用奇怪的脚步抵消 yaw。
```

### D. 增加速度下限惩罚

当前已有：

```python
stand_still_when_commanded = -2.0
```

它在：

```text
cmd_vx > 0.2 且 base_lin_vel_x < 0.15
```

时惩罚太慢。

v8 候选：

```text
把阈值从 0.15 提到 0.25
```

理由：

```text
v7.1 长时 forward 最终 vx 约 0.22，正好低于 0.25。
这个改动直接针对“走着走着变慢”。
```

风险：

```text
训练早期可能更难，因为机器人刚起步时也会被惩罚。
```

教学判断：

```text
可以做，但优先级低于 episode_length_s 和 tracking_lin_vel。
```

### E. 增加 command resampling 多样性

当前：

```python
resampling_time = 10.0
stand_probability = 0.25
```

候选：

```python
resampling_time = 6.0 或 8.0
stand_probability = 0.25 保持不变
```

理由：

```text
让训练中更常见“运动 -> 新命令 -> 零速/转向”的过渡。
```

风险：

```text
如果切换太频繁，可能削弱长时稳定走直。
```

教学判断：

```text
v8 第一版不建议先改 resampling_time。
先解决长时 forward，再增强切换。
```

## 推荐 v8 第一版最小改动

只做三个改动：

```python
episode_length_s = 30
tracking_lin_vel = 3.5
新增 forward_yaw_rate = -0.3
```

暂不改：

```text
observation
action
residual scale
IK gait reference
PD
stand_probability
resampling_time
terrain
网络结构
PPO 主要超参
```

理由：

```text
v8 的核心问题是长时速度保持和直行 yaw。
最小改动更容易判断因果。
```

## 训练命名建议

建议 run_name：

```text
ppo_v8_long_forward_yaw
```

目标 checkpoint 选择方式：

```text
不要默认选最后 checkpoint。
每 500 iteration 导出一次候选，优先用 MuJoCo 行为测试选 checkpoint。
```

建议候选：

```text
model_500.pt
model_1000.pt
model_1500.pt
model_2000.pt
```

如果后期 episode length 或行为变差，优先回退到中期 checkpoint。

## 暂不进入的方向

v8 不做：

```text
MoE
NP3O
复杂地形
斜坡
盲走
横移
真机部署代码
```

原因：

```text
当前最有价值的问题仍然是平地长时速度保持和直行稳定性。
```

