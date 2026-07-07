# 真机 IMU 映射与姿态观测协议

日期：2026-07-04

## 目的

真机部署时，policy 的 observation 前 6 维来自 IMU：

```text
0-2: base angular velocity, body frame
3-5: projected gravity, body frame
```

如果 IMU 坐标系、四元数顺序、角速度单位或旋转方向错，policy 会立刻误判身体姿态，表现为：

```text
站立时乱动
一启动就侧倒
前进/转向方向错误
S 刹车时前扑或后仰
```

所以 IMU 映射必须在跑 policy 前单独验证。

## 训练端需要的 IMU 量

训练端来自 legged_gym：

```python
self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
```

含义：

```text
base_ang_vel:
  base angular velocity expressed in robot body frame

projected_gravity:
  world gravity vector expressed in robot body frame
```

在水平站立时，期望：

```text
base_ang_vel ≈ [0, 0, 0]
projected_gravity ≈ [0, 0, -1]
```

## 真机 IMU 必须确认的字段

### 1. 四元数顺序

常见格式有两种：

```text
wxyz: [w, x, y, z]
xyzw: [x, y, z, w]
```

当前 MuJoCo / deploy 代码使用：

```text
wxyz
```

如果真机 IMU 输出是 `xyzw`，必须转换：

```python
q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
```

### 2. 四元数含义

必须确认 IMU 四元数表示的是：

```text
body -> world
```

还是：

```text
world -> body
```

训练/MuJoCo 中使用 `quat_rotate_inverse(base_quat, world_vector)` 得到 body frame 向量。

如果真机四元数方向相反，projected gravity 会错。

### 3. 角速度坐标系

必须确认 IMU gyro 输出的是：

```text
body frame angular velocity
```

还是：

```text
world frame angular velocity
```

policy 需要：

```text
body frame angular velocity
```

如果 IMU 已经输出 body frame gyro，直接使用。

如果 IMU 输出 world frame angular velocity，需要旋转到 body frame：

```python
ang_vel_body = quat_rotate_inverse_wxyz(q_wxyz, ang_vel_world)
```

### 4. 角速度单位

policy 需要：

```text
rad/s
```

如果 IMU 输出 `deg/s`：

```python
gyro_rad = gyro_deg * np.pi / 180.0
```

## projected_gravity 计算

世界重力方向定义为：

```python
gravity_world = np.array([0.0, 0.0, -1.0])
```

如果 `q_wxyz` 表示 body -> world，则：

```python
projected_gravity = quat_rotate_inverse_wxyz(q_wxyz, gravity_world)
```

部署代码中的函数：

```python
def quat_rotate_inverse_wxyz(q_wxyz, v):
    w, x, y, z = q_wxyz
    t = 2.0 * np.array([
        y * v[2] - z * v[1],
        z * v[0] - x * v[2],
        x * v[1] - y * v[0],
    ])
    return v - w * t + np.cross([x, y, z], t)
```

## 静态姿态验证

### 1. 水平站立

机器人水平站立不动时，应打印：

```text
base_ang_vel ≈ [0, 0, 0]
projected_gravity ≈ [0, 0, -1]
```

允许少量噪声，例如：

```text
projected_gravity x/y 在 ±0.05 内
projected_gravity z 接近 -1
```

### 2. 机器人向前俯仰

手动让机器人机身前倾一点。

预期：

```text
projected_gravity 的 x 分量会明显变化
```

具体正负号取决于 base 坐标定义，但必须和 MuJoCo / 训练定义一致。

验证方法：

```text
在 MuJoCo 中设置相同 pitch，打印 projected_gravity
真机做相同方向倾斜，比较符号是否一致
```

### 3. 机器人向左/右横滚

手动让机器人左倾或右倾。

预期：

```text
projected_gravity 的 y 分量会明显变化
```

同样要和 MuJoCo 符号一致。

### 4. 原地旋转 yaw

纯 yaw 旋转时，如果机身仍然水平：

```text
projected_gravity 应该仍接近 [0, 0, -1]
```

如果 yaw 一变，projected_gravity x/y 大幅变化，通常说明四元数方向或坐标系处理错了。

## IMU 到 observation 的构造

policy observation 前 6 维应为：

```python
obs[0:3] = base_ang_vel_body * OBS_ANG_VEL
obs[3:6] = projected_gravity
```

当前缩放：

```python
OBS_ANG_VEL = 0.25
```

注意：

```text
projected_gravity 不乘额外 scale。
```

## 真机 IMU 检查表

在跑 policy 前，必须填写：

| item | result |
|---|---|
| IMU model | TODO |
| quaternion order | TODO: wxyz / xyzw |
| quaternion meaning | TODO: body->world / world->body |
| gyro frame | TODO: body / world |
| gyro unit | TODO: rad/s / deg/s |
| IMU mounted frame aligned with base? | TODO |
| projected_gravity level pose | TODO |
| projected_gravity pitch test | TODO |
| projected_gravity roll test | TODO |
| yaw rotation does not corrupt gravity | TODO |

## 常见错误

### 错误 1：xyzw 当成 wxyz

现象：

```text
projected_gravity 完全不对
站立时 policy 以为机器人已经倒了
```

### 错误 2：角速度单位用 deg/s

现象：

```text
base_ang_vel 数值大约放大 57.3 倍
policy 输出剧烈动作
```

### 错误 3：world frame gyro 直接当 body frame

现象：

```text
机器人转向或倾斜时动作异常
平地静止时可能看不出来
```

### 错误 4：IMU 安装方向和 base 坐标不一致

现象：

```text
前倾被识别成侧倾
左倾被识别成前倾
转向时姿态估计混乱
```

解决：

```text
需要增加 IMU frame -> base frame 的固定旋转变换。
```

## 建议的最小 IMU 测试脚本

真机部署前应先写一个只打印 IMU 的脚本，不跑电机：

```text
scripts/check_real_imu.py
```

每 0.1 秒打印：

```text
raw quaternion
converted q_wxyz
raw gyro
gyro_rad_body
projected_gravity
```

通过这个脚本确认 IMU 协议后，再进入电机和 policy 测试。

