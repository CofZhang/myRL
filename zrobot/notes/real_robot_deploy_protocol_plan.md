# 真机部署前协议整理计划

日期：2026-07-04

## 目标

本阶段目标不是马上上真机跑，而是把从 MuJoCo 到真机之间的接口协议整理清楚。

最终真机部署必须复现训练和 MuJoCo 中的控制链路：

```text
IMU / 电机状态
-> 45 维 observation
-> policy.pt
-> 12 维 action residual
-> IK gait reference + residual
-> target_q
-> 高频 PD
-> 电机命令
```

核心原则：

```text
真机部署协议必须和训练协议一致。
```

## 当前训练 / 部署协议

### Observation 45 维

```text
0-2:   base angular velocity, body frame
3-5:   projected gravity, body frame
6-8:   command vx, vy, wz
9-20:  joint position error = q - default_q
21-32: joint velocity
33-44: last action
```

### Action 12 维

```text
action 是 IK gait reference 上的 residual
不是直接关节角
不是 torque
```

### 训练关节顺序

```text
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
```

### target_q

```text
target_q = reference_q + action * residual_scale
```

其中：

```text
hip residual scale = 0.06
thigh/calf residual scale = 0.16
```

### PD

当前 MuJoCo baseline：

```text
kp = 40.0
kd = 1.0
```

训练原始配置：

```text
kp = 20.0
kd = 0.5
```

真机上不能直接照搬，需要根据电机能力、减速比、控制周期和安全限幅重新验证。

### Torque limits

```text
hip/thigh = 23.7 Nm
calf = 35.55 Nm
```

真机上必须确认这些值是否对应电机侧、关节侧、减速后输出侧。

## 真机需要提供的数据

### 1. IMU

需要：

```text
base quaternion 或 roll/pitch/yaw
base angular velocity
```

必须确认：

```text
四元数顺序: wxyz 还是 xyzw
角速度坐标系: body frame 还是 world frame
单位: rad/s 还是 deg/s
IMU 安装方向是否和 base 坐标系一致
```

训练需要的是：

```text
base_ang_vel in body frame
projected_gravity in body frame
```

如果 IMU 角速度不是 body frame，必须先旋转。

### 2. 电机编码器

每个关节需要：

```text
joint position q
joint velocity dq
```

必须确认：

```text
关节顺序
零点定义
正方向
单位 rad / rad/s
是否需要从 motor side 换算到 joint side
是否有减速比
```

真机读到的顺序必须映射成训练顺序：

```text
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
```

## 真机输出命令

当前策略输出不能直接发给电机。

正确链路：

```text
policy action
-> residual = action * residual_scale
-> target_q = reference_q + residual
-> tau = kp * (target_q - q) - kd * dq
-> torque limit
-> 电机命令
```

如果真机电机接口不是 torque 模式，而是 position 模式，则需要重新定义：

```text
发送 target_q + 电机内部 PD
```

但这会改变控制器动力学，需要重新验证。

## 控制频率

当前 MuJoCo baseline：

```text
policy frequency = 50 Hz
PD frequency = 200 Hz
```

真机推荐结构：

```text
policy loop: 50 Hz
motor / PD loop: 200 Hz 或更高
```

重要原则：

```text
hold target_q, not torque
```

也就是说：

```text
policy 低频更新 target_q
底层高频读取当前 q/dq 并重新计算 torque
```

不要低频复用旧 torque。

## 安全逻辑

真机部署前必须实现以下安全保护。

### 1. 急停

必须有独立急停通道：

```text
键盘急停
遥控器急停
硬件急停
```

急停后：

```text
停止 policy
停止 gait phase 推进
电机进入安全模式
```

### 2. 姿态保护

如果出现：

```text
roll/pitch 过大
projected_gravity 异常
base height 估计异常
```

应立即退出 policy 控制。

### 3. 关节保护

检查：

```text
q 是否超过安全范围
dq 是否过大
target_q 是否跳变过大
torque 是否超过 limit
```

### 4. command 限幅

键盘或遥控命令必须限幅：

```text
vx: 0.0 ~ 0.5 或更保守
wz: -0.5 ~ 0.5 或更保守
```

初次真机不建议直接用训练最大速度。

### 5. target_q 变化率限制

从站立切换到运动、从运动切回站立时，需要限制 target_q 跳变。

但注意：

```text
行走中不要过度限制 target_q，否则步态会被削弱。
```

推荐：

```text
stand transition: 使用 target_q rate limit
normal walking: 尽量直接跟踪 policy target_q
```

## 真机部署前检查清单

### A. 静态硬件检查

```text
[ ] 12 个关节编号确认
[ ] 12 个关节零点确认
[ ] 12 个关节正方向确认
[ ] 关节限位确认
[ ] 电机力矩/电流限制确认
[ ] IMU 坐标系确认
[ ] IMU 四元数顺序确认
```

### B. 软件映射检查

```text
[ ] 真机 q 映射到训练 joint order
[ ] 真机 dq 映射到训练 joint order
[ ] 真机 IMU 映射到训练 obs
[ ] command scaling 与训练一致
[ ] residual scale 与训练一致
[ ] torque limit 与训练/硬件一致
```

### C. 不跑 policy 的测试

```text
[ ] 读取 q/dq/IMU 并打印
[ ] 电机零力矩模式安全
[ ] 低 kp/kd 保持 default_q
[ ] 逐个关节小幅正负运动，验证方向
[ ] 四腿 default_q 站立
```

### D. 跑 policy 前测试

```text
[ ] policy 输入 obs 打印
[ ] policy action 打印
[ ] action * residual_scale 后量级正常
[ ] target_q 在关节安全范围内
[ ] torque 不打满
```

### E. 初次 policy 测试

```text
[ ] 机器人悬空或支撑架测试
[ ] stand command only
[ ] W 小速度前进
[ ] S 回站立
[ ] A/D 小角速度转向
```

## 第一版真机部署建议

第一版真机不要追求完整 WASD。

建议顺序：

```text
1. 只读传感器，不发力矩
2. 纯 PD 保持 default_q
3. policy stand command
4. W 小速度前进，例如 vx=0.2
5. S 回站立
6. 小角速度 A/D
```

初始命令建议：

```text
vx = 0.2
wz = 0.2
```

不要一开始使用：

```text
vx = 0.5
wz = 0.5
```

## 后续文档

下一步应补充：

```text
real_robot_joint_mapping.md
real_robot_imu_mapping.md
real_robot_safety_checklist.md
real_robot_control_loop.md
```

