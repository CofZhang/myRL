# 真机部署安全检查清单

日期：2026-07-04

## 目的

真机部署前必须先建立安全边界。

MuJoCo 中的错误通常只是摔倒；真机中的错误可能导致：

```text
关节打限位
电机过流
腿部高速抽动
机器人摔倒
结构损坏
人员受伤
```

所以第一版真机部署必须遵守：

```text
先只读
再低增益
再站立
再小速度 policy
```

不要一开始直接运行完整 WASD policy。

## 总体原则

### 1. 默认安全

程序启动时默认：

```text
不发送电机力矩
不运行 policy
不推进 gait phase
command = [0, 0, 0]
```

必须由人工显式使能后才进入控制。

### 2. 急停优先

急停信号优先级最高。

一旦触发急停：

```text
停止 policy
停止 gait phase
target_q 停止更新
电机进入安全模式
记录最后状态
```

急停不能依赖 policy 或主循环正常运行。

### 3. 先低能量测试

第一版真机测试建议：

```text
机器人悬空或有支撑架
低 kp/kd
低速度命令
低 torque/current limit
随时可断电
```

## 必须具备的安全通道

### 1. 硬件急停

必须确认：

```text
[ ] 急停按钮能直接切断电机使能或动力电
[ ] 急停不依赖上位机程序
[ ] 急停触发后机器人不会继续执行上一条命令
[ ] 急停恢复需要人工确认
```

### 2. 软件急停

建议至少支持：

```text
[ ] 键盘急停
[ ] 遥控器急停
[ ] 网络/串口通信超时急停
[ ] 主循环异常急停
```

软件急停后：

```text
policy_enabled = False
motor_enabled = False 或 safe damping mode
command = [0, 0, 0]
```

## 姿态保护

如果出现以下情况，立即退出 policy：

```text
roll 绝对值过大
pitch 绝对值过大
projected_gravity 异常
IMU 数据 NaN
IMU 数据超时
```

建议第一版阈值保守：

```text
abs(roll)  > 0.6 rad  -> stop
abs(pitch) > 0.6 rad  -> stop
```

如果只使用 projected_gravity，可设置：

```text
projected_gravity[2] > -0.5 -> stop
```

含义：

```text
机器人身体已经明显偏离正常站立姿态。
```

## 关节保护

每个关节都需要软件限位。

部署前必须填写：

```text
joint lower limit
joint upper limit
velocity limit
torque/current limit
```

运行时检查：

```text
[ ] q 是否超过安全范围
[ ] dq 是否超过安全范围
[ ] target_q 是否超过安全范围
[ ] target_q 跳变是否过大
[ ] torque/current 是否超过 limit
```

如果任意一项异常：

```text
退出 policy
进入安全模式
打印异常关节 index/name/value
```

## Command 限幅

真机第一版不要使用 MuJoCo baseline 的最大命令。

建议第一版：

```text
vx_max = 0.2 m/s
wz_max = 0.2 rad/s
```

通过悬空和低速地面测试后，再逐步增加。

命令必须限幅：

```python
cmd_vx = clip(cmd_vx, 0.0, vx_max)
cmd_wz = clip(cmd_wz, -wz_max, wz_max)
```

不要允许外部输入直接进入 policy。

## target_q 保护

policy 输出后得到：

```text
target_q = reference_q + action * residual_scale
```

真机上必须检查：

```text
[ ] target_q 是否在安全关节范围内
[ ] target_q - q 是否过大
[ ] target_q 相对上一帧是否跳变过大
```

建议：

```text
max_target_delta_per_policy_step = 0.10 rad
```

站立到运动、运动到站立时可以更保守。

注意：

```text
target_q 限速过强会削弱步态。
但真机第一版宁可保守。
```

## Torque / Current 保护

如果使用 torque 模式：

```python
tau = kp * (target_q - q) - kd * dq
tau = clip(tau, -tau_limit, tau_limit)
```

必须确认：

```text
tau_limit 是关节侧还是电机侧
驱动器接口接收的是 Nm 还是电流
是否需要乘/除减速比
```

如果使用 current 模式，必须有：

```text
current_limit
temperature limit
driver fault monitoring
```

## 控制频率保护

当前 baseline：

```text
policy loop = 50 Hz
PD loop = 200 Hz
```

真机中必须检测循环超时：

```text
policy loop timeout
motor loop timeout
sensor timeout
communication timeout
```

如果控制周期异常：

```text
停止 policy
进入安全模式
```

不要在低频或不稳定频率下继续发旧 torque。

## 数据有效性检查

每一帧进入 policy 前检查：

```text
[ ] obs 中没有 NaN
[ ] obs 中没有 Inf
[ ] projected_gravity norm 接近 1
[ ] q/dq 范围合理
[ ] command 在限幅内
```

policy 输出后检查：

```text
[ ] action 没有 NaN/Inf
[ ] action 量级合理
[ ] target_q 量级合理
[ ] torque/current 量级合理
```

## 推荐真机测试顺序

### 阶段 0：无电机输出

```text
[ ] 只读 IMU
[ ] 只读 q/dq
[ ] 打印 observation
[ ] 跑 policy 但不发送命令
[ ] 打印 action/target_q/torque
```

### 阶段 1：单关节低增益测试

```text
[ ] 单关节 +0.05 rad
[ ] 单关节 -0.05 rad
[ ] 确认方向
[ ] 确认限位
```

### 阶段 2：全身 default_q

```text
[ ] 低 kp/kd 保持 default_q
[ ] 检查四腿姿态
[ ] 检查无异常发热/抖动
```

### 阶段 3：policy stand

```text
[ ] command = [0, 0, 0]
[ ] policy 输出 residual
[ ] target_q 接近 default_q
[ ] 保持站立
```

### 阶段 4：小速度前进

```text
[ ] vx = 0.1
[ ] vx = 0.2
[ ] 随时可急停
```

### 阶段 5：小角速度转向

```text
[ ] wz = +0.1
[ ] wz = -0.1
[ ] 再逐步增加到 0.2
```

## 进入真机 policy 的最低条件

必须全部满足：

```text
[ ] hardware e-stop 可用
[ ] software e-stop 可用
[ ] joint mapping 已确认
[ ] IMU mapping 已确认
[ ] default_q 低增益站立通过
[ ] obs 打印正确
[ ] policy action 量级正常
[ ] target_q 不越界
[ ] torque/current 不打满
[ ] 控制循环频率稳定
```

未全部满足时，不允许真机运行 policy。

