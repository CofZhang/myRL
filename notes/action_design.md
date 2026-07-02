# zof action v1 设计笔记

本文记录 zof 四足机器人第一版强化学习动作设计。第一版目标是先把平地稳定行走跑通，所以 action 不直接输出电机力矩，而是输出 12 个关节的位置目标偏移。

## action 总览

zof 是 12 自由度四足机器人：

```text
4 条腿 x 每条腿 3 个主动关节 = 12 个 action
```

zof action v1 一共 12 维：

| action 维度 | 关节名 | 含义 |
|---:|---|---|
| 0 | FL_hip_joint | 左前腿髋侧摆关节目标偏移 |
| 1 | FL_thigh_joint | 左前腿大腿关节目标偏移 |
| 2 | FL_calf_joint | 左前腿小腿关节目标偏移 |
| 3 | FR_hip_joint | 右前腿髋侧摆关节目标偏移 |
| 4 | FR_thigh_joint | 右前腿大腿关节目标偏移 |
| 5 | FR_calf_joint | 右前腿小腿关节目标偏移 |
| 6 | RL_hip_joint | 左后腿髋侧摆关节目标偏移 |
| 7 | RL_thigh_joint | 左后腿大腿关节目标偏移 |
| 8 | RL_calf_joint | 左后腿小腿关节目标偏移 |
| 9 | RR_hip_joint | 右后腿髋侧摆关节目标偏移 |
| 10 | RR_thigh_joint | 右后腿大腿关节目标偏移 |
| 11 | RR_calf_joint | 右后腿小腿关节目标偏移 |

注意：这个顺序以后必须在训练、导出、部署里保持一致。如果训练时 action 第 0 维控制 `FL_hip_joint`，部署时也必须让第 0 维控制 `FL_hip_joint`。

## action 不是力矩

第一版 action 不表示电机力矩。

策略网络输出的是：

```text
action: [12]
```

每个 action 表示对应关节相对默认站姿的目标角度偏移。

换句话说，策略不是直接说“给电机多大力”，而是说：

```text
这个关节目标角度应该比默认站姿大一点还是小一点
```

## 从 action 到目标关节角

第一版使用这个公式：

```text
target_q = default_q + action * action_scale
```

其中：

```text
target_q: 目标关节角
默认_q / default_q: 默认站立关节角
action: 策略网络输出
action_scale: 动作缩放系数，第一版先用 0.25
```

逐个关节写就是：

```text
target_q[i] = default_q[i] + action[i] * action_scale
```

如果：

```text
action_scale = 0.25
```

那么：

```text
action[i] = 0    -> target_q[i] = default_q[i]
action[i] = 1    -> target_q[i] = default_q[i] + 0.25 rad
action[i] = -1   -> target_q[i] = default_q[i] - 0.25 rad
```

所以 action 是一个归一化偏移，不是实际角度，也不是力矩。

## 默认站姿 default_q

`default_q` 是机器人自然站立时的 12 个关节角。

它的作用是给策略一个中心姿态。

策略不需要从零开始学每个关节的绝对角度，而是在默认站姿附近学习如何摆腿：

```text
站立中心姿态 + 小范围动作偏移 = 目标关节角
```

这会让训练更稳定，也更符合真实机器人的控制方式。

## PD 控制

得到 `target_q` 后，还不能直接发送给仿真或电机。我们需要用 PD 控制把目标关节角变成关节力矩。

PD 控制公式是：

```text
tau = Kp * (target_q - q) - Kd * dq
```

其中：

```text
tau: 输出力矩
Kp: 位置增益，可以理解成拉向目标位置的力度
Kd: 阻尼增益，可以理解成刹车力度
target_q: 目标关节角
q: 当前关节角
dq: 当前关节速度
```

白话理解：

```text
关节离目标越远，Kp 给的拉力越大
关节转得越快，Kd 给的刹车越大
```

举例：

```text
target_q = 1.0
q = 0.8
dq = 0.1
Kp = 20
Kd = 0.5
```

计算：

```text
tau = 20 * (1.0 - 0.8) - 0.5 * 0.1
    = 4.0 - 0.05
    = 3.95
```

说明电机会给一个正方向力矩，把关节往目标角度拉。

## 为什么第一版不用 torque action

另一种做法是让策略直接输出力矩：

```text
tau = action
```

这种方法自由度更高，但更难训练，也更危险。

早期训练时，策略动作基本是随机的。如果直接输出力矩，机器人可能会：

```text
乱抽动
关节撞限位
力矩剧烈抖动
仿真不稳定
真机非常危险
```

所以第一版不用 torque action。

第一版采用位置目标偏移 + PD 控制：

```text
策略负责决定腿往哪里摆
PD 控制负责稳定地把关节拉到目标位置
```

这更稳，也更适合从仿真迁移到真机。

## 为什么 action_scale 先用 0.25

`action_scale` 决定动作幅度。

如果太大：

```text
action_scale = 1.0
```

随机策略一开始就会让关节大幅摆动，机器人容易乱飞、摔倒、撞限位。

如果太小：

```text
action_scale = 0.05
```

腿摆不开，机器人可能学不会走路。

四足机器人第一版常用：

```text
action_scale = 0.25
```

这是一个比较稳的起点。以后如果发现步幅太小，可以略微增大；如果动作太剧烈，可以减小。

## 训练和部署必须一致

训练时如果使用：

```text
target_q = default_q + action * action_scale
```

部署时也必须使用同样的公式。

如果训练和部署不一致，比如训练时 `action_scale = 0.25`，部署时误用了 `0.5`，机器人动作会被放大一倍，策略表现可能完全失效。

所以这几个东西必须统一记录：

```text
关节顺序
默认关节角 default_q
action_scale
Kp
Kd
控制频率
策略推理频率
```

## zof action v1 设计原则

第一版 action 设计遵循这些原则：

```text
先稳定，不追求最强性能
不直接输出力矩
使用位置目标偏移
用 PD 控制提高稳定性
训练和部署公式保持一致
action_scale 从 0.25 起步
```

## 后续可能扩展

等第一版 PPO 能稳定走以后，可以考虑：

```text
不同关节使用不同 action_scale
髋侧摆关节使用更小 scale
加入 action rate penalty 减少抖动
尝试 torque action，但只在仿真中实验
MoE 中不同专家使用相同 action 接口
```

第一版不要急着做这些扩展。先跑通基础闭环。
