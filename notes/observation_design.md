# zof observation v1 设计笔记

本文记录 zof 四足机器人第一版强化学习观测设计。第一版目标是平地稳定行走和基础盲走，不使用相机、雷达、地形高度图，也不使用真实线速度估计。

## 观测总览

zof observation v1 一共 45 维：

| 维度范围 | 数量 | 名称 | 含义 |
|---|---:|---|---|
| 0-2 | 3 | base angular velocity | 机身角速度，通常来自 IMU 陀螺仪 |
| 3-5 | 3 | projected gravity | 重力方向投影到机身坐标系，用来表示身体姿态 |
| 6-8 | 3 | command vx, vy, wz | 用户希望机器人执行的速度命令 |
| 9-20 | 12 | joint position error | 当前关节角相对默认站姿的偏差 |
| 21-32 | 12 | joint velocity | 12 个关节速度 |
| 33-44 | 12 | last action | 上一次策略输出的动作 |

写成公式：

```text
obs = [
    base_ang_vel,                  # 3
    projected_gravity,             # 3
    command_vx_vy_wz,              # 3
    dof_pos - default_dof_pos,      # 12
    dof_vel,                       # 12
    last_action                    # 12
]

总维度 = 3 + 3 + 3 + 12 + 12 + 12 = 45
```

## 每一部分的作用

### 0-2: base angular velocity

`base angular velocity` 表示机身绕 x、y、z 轴旋转的速度。

它能告诉策略：

```text
身体是不是在前后翻
身体是不是在左右滚
身体是不是在原地旋转
```

如果机器人快摔了，角速度通常会变大。策略看到这个信息后，才有机会调整腿部动作来稳定身体。

### 3-5: projected gravity

`projected gravity` 是把世界坐标系里的重力方向投影到机器人身体坐标系里。

它能告诉策略：

```text
身体现在是正的还是歪的
身体前倾还是后仰
身体左倾还是右倾
```

它比直接给 roll、pitch、yaw 更常用，因为它连续、稳定，而且不容易受欧拉角奇异问题影响。

### 6-8: command vx, vy, wz

`command` 是希望机器人执行的运动命令：

```text
vx: 前进/后退速度
vy: 左右平移速度
wz: 原地转向角速度
```

强化学习策略不是只学一种固定动作，而是学一个“可控制”的行走策略。

如果不给 command，策略不知道用户想让它：

```text
站着不动
向前走
向后走
向左走
向右走
原地转圈
```

所以 command 必须放进 observation。

### 9-20: joint position error

`joint position error` 使用：

```text
joint_position_error = current_joint_position - default_joint_position
```

这里不给绝对关节角，而给“相对默认站姿的偏差”。

这样做的原因是：

```text
默认站姿是机器人最自然的中心姿态
策略只需要学习在这个姿态附近怎么摆腿
数值更小，更容易训练
部署时也方便把 action 变成目标关节角
```

第一版 action 也会采用类似形式：

```text
target_joint_position = default_joint_position + action * action_scale
```

所以 observation 里的关节位置也用相对默认站姿的形式，训练和部署更一致。

### 21-32: joint velocity

`joint velocity` 表示每个关节转得有多快。

它能告诉策略：

```text
腿正在向哪个方向摆
摆得快还是慢
关节是否出现剧烈抖动
```

如果没有关节速度，策略只知道当前关节在哪里，但不知道它正在往哪里运动。

### 33-44: last action

`last action` 是上一帧策略输出的 12 维动作。

它的作用是给策略一点“短期记忆”。

神经网络本身如果是普通 MLP，它只看当前这一帧 observation，不知道上一帧自己做了什么。加入 last action 后，策略可以知道：

```text
上一次腿往哪里摆了
现在要不要继续摆
动作变化是不是太突然
```

这有助于减少抖动，让步态更连续。

## 五个关键问题

### 1. 为什么不放 base linear velocity？

`base linear velocity` 是机身线速度，比如前进速度、左右速度、上下速度。

在 Isaac Gym 仿真里，这个量很容易拿到。但在真机上，它通常不好直接测量。

真机一般有 IMU 和关节编码器，但没有一个天然准确的 `base linear velocity`。如果训练时用了这个信息，而真机部署时没有，策略就会依赖一个部署端拿不到的输入。

所以第一版为了 sim-to-real，更稳妥地不放 base linear velocity。

简单说：

```text
训练时能拿到，不代表真机能拿到。
真机拿不到的信息，第一版不要喂给 actor。
```

### 2. 为什么不放地形高度？

地形高度图能告诉机器人前方地面高低，这对上台阶、坑洼地形很有帮助。

但是如果训练 actor 使用地形高度，部署时就必须有传感器提供类似信息，比如深度相机、雷达、足端探测等。

你的第一版目标是盲走，也就是只靠自身感觉走路：

```text
IMU
关节角
关节速度
上一帧动作
```

所以第一版不放地形高度。

后面要做更强的未知地形能力，可以有两条路线：

```text
盲走路线：继续不放地形高度，靠历史观测和域随机化增强适应性
感知路线：加入高度图、深度图或地形编码，但部署端也必须提供这些输入
```

### 3. 为什么要放 last action？

普通策略网络通常只看当前 observation。

如果不放 last action，它不知道自己上一帧让腿往哪里动了，动作可能更容易跳变。

加入 last action 后，策略能学到更平滑的控制规律，比如：

```text
上一帧腿已经向前摆，这一帧继续完成摆腿
上一帧动作太大，这一帧收一点
不要突然从一个极端动作跳到另一个极端动作
```

这对四足机器人很重要，因为真实电机和机械结构都不喜欢高频抖动。

### 4. joint position error 为什么是 q - default_q？

`q` 是当前关节角，`default_q` 是默认站立关节角。

使用：

```text
q - default_q
```

表示当前姿态偏离默认站姿多少。

这样有三个好处：

```text
数值更集中，神经网络更容易学
不同关节都围绕自己的默认角度表达
和 action 定义保持一致
```

如果直接使用绝对关节角，网络也能学，但输入分布更散，而且不如“相对默认站姿”直观。

### 5. command 为什么要放进 observation？

策略需要知道用户想让机器人做什么。

同一个机器人状态下，不同命令对应不同正确动作：

```text
command = [0, 0, 0]       应该站稳或慢走
command = [0.5, 0, 0]     应该向前走
command = [-0.5, 0, 0]    应该向后走
command = [0, 0.3, 0]     应该侧向走
command = [0, 0, 0.5]     应该原地左/右转
```

如果 observation 里没有 command，策略就无法区分这些任务，只能学出一种平均行为。

所以 command 是可控行走策略必须输入的信息。

## 第一版设计原则

zof observation v1 的原则是：

```text
只使用真机比较容易获得的信息
先不依赖外部感知
训练和部署输入保持一致
先把平地稳定行走跑通
再扩展复杂地形和 MoE
```

## 后续可能扩展

第一版 45 维观测跑通后，可以再考虑扩展：

```text
历史观测 history: 45 x N
足端接触状态 contact
电机延迟估计 motor delay
地形高度 height samples
深度相机或雷达特征
MoE gating 输入
```

但这些都应该在基础 PPO 策略稳定后再做。
