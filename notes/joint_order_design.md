# zof joint order v1 设计笔记

本文记录 zof 四足机器人第一版关节顺序设计。关节顺序是训练、导出、部署、真机控制能否对齐的核心问题。

## 为什么关节顺序重要

强化学习策略输出的是一个 12 维 action：

```text
action[0], action[1], ..., action[11]
```

神经网络本身不知道这些数字对应哪条腿、哪个关节。这个对应关系由我们人为定义。

如果训练时：

```text
action[0] -> FL_hip_joint
```

但部署时写成：

```text
action[0] -> RR_calf_joint
```

那策略输出就会发错关节，机器人一定乱动。

所以必须保证：

```text
训练、导出、部署、真机电机的关节顺序完全一致。
```

## zof 标准关节顺序 v1

zof 第一版统一使用下面这个标准顺序：

| 标准索引 | 关节名 | 腿 | 关节类型 |
|---:|---|---|---|
| 0 | FL_hip_joint | 左前腿 | 髋侧摆关节 |
| 1 | FL_thigh_joint | 左前腿 | 大腿关节 |
| 2 | FL_calf_joint | 左前腿 | 小腿关节 |
| 3 | FR_hip_joint | 右前腿 | 髋侧摆关节 |
| 4 | FR_thigh_joint | 右前腿 | 大腿关节 |
| 5 | FR_calf_joint | 右前腿 | 小腿关节 |
| 6 | RL_hip_joint | 左后腿 | 髋侧摆关节 |
| 7 | RL_thigh_joint | 左后腿 | 大腿关节 |
| 8 | RL_calf_joint | 左后腿 | 小腿关节 |
| 9 | RR_hip_joint | 右后腿 | 髋侧摆关节 |
| 10 | RR_thigh_joint | 右后腿 | 大腿关节 |
| 11 | RR_calf_joint | 右后腿 | 小腿关节 |

缩写含义：

```text
FL = Front Left   左前腿
FR = Front Right  右前腿
RL = Rear Left    左后腿
RR = Rear Right   右后腿
```

每条腿内部顺序固定为：

```text
hip -> thigh -> calf
```

整体顺序固定为：

```text
FL -> FR -> RL -> RR
```

## 所有数组都必须使用同一顺序

下面这些 12 维数组必须全部使用同一个标准顺序：

```text
joint_names
default_q
action
last_action
target_q
q
dq
Kp
Kd
torque
joint_position_error
joint_velocity
```

例如：

```text
index 0 永远表示 FL_hip_joint
index 1 永远表示 FL_thigh_joint
index 2 永远表示 FL_calf_joint
...
index 11 永远表示 RR_calf_joint
```

## observation 中的关节顺序

zof observation v1 有三段和关节顺序有关：

```text
9-20: joint position error
21-32: joint velocity
33-44: last action
```

它们也必须按标准关节顺序排列。

具体对应关系：

| 关节 | position error 维度 | velocity 维度 | last action 维度 |
|---|---:|---:|---:|
| FL_hip_joint | 9 | 21 | 33 |
| FL_thigh_joint | 10 | 22 | 34 |
| FL_calf_joint | 11 | 23 | 35 |
| FR_hip_joint | 12 | 24 | 36 |
| FR_thigh_joint | 13 | 25 | 37 |
| FR_calf_joint | 14 | 26 | 38 |
| RL_hip_joint | 15 | 27 | 39 |
| RL_thigh_joint | 16 | 28 | 40 |
| RL_calf_joint | 17 | 29 | 41 |
| RR_hip_joint | 18 | 30 | 42 |
| RR_thigh_joint | 19 | 31 | 43 |
| RR_calf_joint | 20 | 32 | 44 |

如果这三段顺序不一致，策略就会把某个关节的状态和另一个关节的动作混在一起。

## action 中的关节顺序

zof action v1 是 12 维：

```text
action[0]  -> FL_hip_joint
action[1]  -> FL_thigh_joint
action[2]  -> FL_calf_joint
action[3]  -> FR_hip_joint
action[4]  -> FR_thigh_joint
action[5]  -> FR_calf_joint
action[6]  -> RL_hip_joint
action[7]  -> RL_thigh_joint
action[8]  -> RL_calf_joint
action[9]  -> RR_hip_joint
action[10] -> RR_thigh_joint
action[11] -> RR_calf_joint
```

训练时、导出后、部署时都必须保持这个定义。

## default_q 的顺序

默认站姿 `default_q` 也必须是同样顺序：

```text
default_q = [
    FL_hip, FL_thigh, FL_calf,
    FR_hip, FR_thigh, FR_calf,
    RL_hip, RL_thigh, RL_calf,
    RR_hip, RR_thigh, RR_calf,
]
```

如果 `default_q` 顺序错了，机器人初始站姿和目标关节角都会错。

## Kp / Kd 的顺序

PD 控制参数也必须按标准顺序排列。

```text
Kp[0] -> FL_hip_joint
Kd[0] -> FL_hip_joint
Kp[1] -> FL_thigh_joint
Kd[1] -> FL_thigh_joint
...
Kp[11] -> RR_calf_joint
Kd[11] -> RR_calf_joint
```

如果 Kp / Kd 顺序错了，某些关节可能太软，某些关节可能太硬，训练和部署都会异常。

## 可能存在的多套顺序

一个完整四足项目里，通常会同时存在多套顺序：

```text
URDF joint 顺序
Isaac Gym DOF 顺序
训练 observation 顺序
训练 action 顺序
TorchScript 模型输出顺序
MuJoCo joint 顺序
真机电机顺序
```

这些顺序不一定天然一致。

所以不能靠猜，必须打印和确认。

## 为什么不能直接相信 URDF 顺序

URDF 文件里的 joint 出现顺序，不一定就是 Isaac Gym 加载后的 DOF 顺序。

原因可能包括：

```text
加载器按树结构遍历
fixed joint 被折叠
不同物理引擎处理顺序不同
URDF 文件本身顺序不符合标准顺序
```

所以训练前必须打印 Isaac Gym 的真实 DOF 顺序：

```text
gym.get_asset_dof_names(robot_asset)
```

然后确认它是否等于 zof 标准顺序。

## joint2motor 映射

真机部署时，电机通信协议通常使用：

```text
motor[0]
motor[1]
...
motor[11]
```

这个电机顺序不一定和训练顺序一样。

所以部署时可能需要一个映射：

```text
policy_joint_index -> motor_index
```

也可以叫：

```text
joint2motor_idx
```

例子：

```text
joint2motor_idx[0] = 3
```

意思是：

```text
策略里的第 0 个关节，要发送给真机 motor[3]
```

这个映射必须非常小心。映射错误在真机上很危险。

## 训练前必须做的检查

训练前要先做关节顺序检查表。

第一版检查表：

| 标准索引 | 标准关节名 | URDF 是否存在 | Isaac Gym DOF index | MuJoCo joint index | 真机 motor index |
|---:|---|---|---:|---:|---:|
| 0 | FL_hip_joint | 待检查 | 待打印 | 后续填写 | 后续填写 |
| 1 | FL_thigh_joint | 待检查 | 待打印 | 后续填写 | 后续填写 |
| 2 | FL_calf_joint | 待检查 | 待打印 | 后续填写 | 后续填写 |
| 3 | FR_hip_joint | 待检查 | 待打印 | 后续填写 | 后续填写 |
| 4 | FR_thigh_joint | 待检查 | 待打印 | 后续填写 | 后续填写 |
| 5 | FR_calf_joint | 待检查 | 待打印 | 后续填写 | 后续填写 |
| 6 | RL_hip_joint | 待检查 | 待打印 | 后续填写 | 后续填写 |
| 7 | RL_thigh_joint | 待检查 | 待打印 | 后续填写 | 后续填写 |
| 8 | RL_calf_joint | 待检查 | 待打印 | 后续填写 | 后续填写 |
| 9 | RR_hip_joint | 待检查 | 待打印 | 后续填写 | 后续填写 |
| 10 | RR_thigh_joint | 待检查 | 待打印 | 后续填写 | 后续填写 |
| 11 | RR_calf_joint | 待检查 | 待打印 | 后续填写 | 后续填写 |

后面写检查脚本时，要自动打印：

```text
Isaac Gym DOF names
body names
feet body names
joint limits
```

## zof joint order v1 设计原则

第一版关节顺序设计遵循：

```text
人为定义标准顺序
所有 12 维数组都遵守同一顺序
不要相信默认加载顺序，必须打印确认
训练和部署必须显式处理映射
真机 motor 顺序最后单独校验
```

关节顺序不是小细节，而是训练和部署能否成功对齐的核心。
