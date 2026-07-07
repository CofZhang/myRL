# 真机关节映射表

日期：2026-07-04

## 目的

真机部署前，必须确认真机 12 个电机与训练关节顺序完全一致。

训练和 MuJoCo 当前使用的关节顺序是：

```text
0  FL_hip
1  FL_thigh
2  FL_calf
3  FR_hip
4  FR_thigh
5  FR_calf
6  RL_hip
7  RL_thigh
8  RL_calf
9  RR_hip
10 RR_thigh
11 RR_calf
```

真机部署时，所有传感器读数和电机命令都必须映射到这个顺序。

如果顺序、正方向、零点任意一项错，policy 会表现为：

```text
腿乱蹬
站不稳
转向错误
一按 W/A/D 就摔倒
```

## 当前训练关节定义

### 默认关节角 default_q

训练顺序：

```text
[-0.1,  1.1, -1.5,
  0.1,  1.1, -1.5,
 -0.1,  1.3, -1.5,
  0.1,  1.3, -1.5]
```

对应：

| index | joint | default_q rad |
|---:|---|---:|
| 0 | FL_hip | -0.1 |
| 1 | FL_thigh | 1.1 |
| 2 | FL_calf | -1.5 |
| 3 | FR_hip | 0.1 |
| 4 | FR_thigh | 1.1 |
| 5 | FR_calf | -1.5 |
| 6 | RL_hip | -0.1 |
| 7 | RL_thigh | 1.3 |
| 8 | RL_calf | -1.5 |
| 9 | RR_hip | 0.1 |
| 10 | RR_thigh | 1.3 |
| 11 | RR_calf | -1.5 |

### Torque limits

训练和 MuJoCo 当前使用：

| joint type | limit Nm |
|---|---:|
| hip | 23.7 |
| thigh | 23.7 |
| calf | 35.55 |

注意：

```text
真机必须确认这些 limit 是关节侧力矩，还是电机侧力矩。
如果电机接口使用电流控制，需要换算电流限制。
```

## 真机映射表

下面表格必须在真机部署前逐项填写。

| train index | train joint | robot motor id | robot joint name | encoder zero offset rad | direction sign | joint lower rad | joint upper rad | torque/current limit | checked |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 0 | FL_hip | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| 1 | FL_thigh | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| 2 | FL_calf | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| 3 | FR_hip | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| 4 | FR_thigh | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| 5 | FR_calf | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| 6 | RL_hip | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| 7 | RL_thigh | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| 8 | RL_calf | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| 9 | RR_hip | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| 10 | RR_thigh | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| 11 | RR_calf | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |

字段含义：

```text
robot motor id:
  真机电机控制器中的 ID。

robot joint name:
  真机代码或硬件文档中的关节名。

encoder zero offset rad:
  真机编码器读数转换到训练 q 所需的零点偏移。

direction sign:
  +1 表示真机正方向与训练一致。
  -1 表示真机正方向与训练相反。

joint lower / upper:
  真机安全软件限位，不一定等同 URDF limit。

torque/current limit:
  真机实际发送命令时使用的安全限制。
```

## 编码器到训练 q 的转换

推荐统一成：

```python
q_train[i] = direction_sign[i] * (q_robot_raw[motor_id] - zero_offset[i])
```

速度同理：

```python
dq_train[i] = direction_sign[i] * dq_robot_raw[motor_id]
```

如果电机编码器在电机侧，而不是关节侧，需要除以减速比：

```python
q_joint = q_motor / gear_ratio
dq_joint = dq_motor / gear_ratio
```

方向和零点必须在关节侧统一后再进入 policy。

## 训练 target_q 到真机命令的转换

policy 输出后得到训练坐标系下的：

```text
target_q_train[12]
```

如果真机需要发送关节位置目标：

```python
target_q_robot[motor_id] = zero_offset[i] + direction_sign[i] * target_q_train[i]
```

如果真机需要发送力矩目标：

```python
tau_robot[motor_id] = direction_sign[i] * tau_train[i]
```

如果电机控制在电机侧，还需要乘/除减速比，具体取决于驱动器接口定义。

## 单关节方向测试

不要一开始跑 policy。

每个关节必须先做小幅方向测试：

```text
1. 机器人悬空或有支撑架
2. 只使能一个关节
3. 发送 +0.05 rad 小目标
4. 观察运动方向是否与训练定义一致
5. 发送 -0.05 rad 小目标
6. 填写 direction sign
```

建议测试顺序：

```text
FL_hip
FR_hip
RL_hip
RR_hip
FL_thigh
FR_thigh
RL_thigh
RR_thigh
FL_calf
FR_calf
RL_calf
RR_calf
```

原因：

```text
先测 hip 侧摆，幅度小，风险较低。
calf 可能导致脚端大幅移动，放后面测。
```

## default_q 静态检查

当 12 个关节映射都确认后，真机第一步不是跑 policy，而是发送：

```text
default_q
```

预期姿态：

```text
四条腿对称
身体接近水平
足端接近默认支撑位置
没有明显一条腿反向折叠
没有某条腿明显比其他腿高/低
```

如果 default_q 姿态不对，禁止继续 policy。

排查顺序：

```text
1. 关节顺序是否错
2. 零点是否错
3. 方向 sign 是否错
4. 单位是否是 rad
5. 是否忘记减速比
```

## 真机部署前通过条件

进入 policy 前，必须满足：

```text
[ ] 12 个 motor id 已填写
[ ] 12 个 zero offset 已填写
[ ] 12 个 direction sign 已填写
[ ] q_train 打印值和实际姿态一致
[ ] dq_train 静止时接近 0
[ ] default_q 能稳定摆出正确姿态
[ ] 低增益 PD 能保持 default_q
[ ] 急停功能可用
```

未满足以上条件，不要运行 policy。

