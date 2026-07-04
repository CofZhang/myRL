# MuJoCo 部署调试记录：PD 闭环更新频率问题

日期：2026-07-03

## 背景

训练端已经完成 `zof_flat` PPO，并导出了 TorchScript policy：

```text
resources/policies/zof_flat/Jul02_16-33-17_ppo_v5_ik_residual_bezier/checkpoint_3000/policy.pt
```

策略协议：

```text
obs: 45 维
action: 12 维
action 含义：IK gait reference 上的 RL residual，不是直接关节角，也不是力矩
```

部署端使用：

```text
resources/robots/zof/xml/zof_deploy_from_urdf.xml
scripts/deploy_mujoco.py
```

## 现象

最初在 MuJoCo 部署中出现：

```text
1. 机器人腿乱蹬
2. reset 后有时像飞天/被弹飞
3. 画面里小腿看起来缺失或时不时消失
4. cmd_vx=0.0 时机器人仍然乱动甚至冲出去
5. policy 部署效果明显差于 Isaac Gym 训练效果
```

后来确认：小腿“消失”主要是控制发散后视觉上看起来异常，并不是 XML 里真的缺少 calf/foot。

## 排查过程

### 1. 先检查 policy.pt

写了 `scripts/check_policy.py`，只加载 policy，不跑 MuJoCo。

确认结果：

```text
obs shape:    [1, 45] 或 [3, 45]
action shape: [1, 12] 或 [3, 12]
has nan: False
```

并测试了不同 `cmd_vx`：

```text
cmd_vx = 0.0, 0.3, 0.8 时 action 有明显变化
```

结论：

```text
policy.pt 本身能加载，输入输出维度正确，速度命令会影响 action。
```

### 2. 检查 action 尺度

将 policy 输出乘以 residual scale：

```text
hip scale = 0.06
thigh/calf scale = 0.16
```

例如 `cmd_vx=0.3` 时，最大 residual 约为：

```text
delta_q 约 0.2 rad 量级
```

再用：

```text
target_q = default_q + delta_q
```

得到的关节目标仍在合理范围内。

结论：

```text
policy 输出本身没有明显离谱，action scale 也不是主要问题。
```

### 3. 检查 MuJoCo joint / actuator 顺序

写了 `scripts/check_mujoco_model.py`，打印：

```text
nq = 19
nv = 18
nu = 12
```

解释：

```text
qpos = base position 3 + base quaternion 4 + 12 dof = 19
qvel = base linear velocity 3 + base angular velocity 3 + 12 dof = 18
```

确认 joint 顺序：

```text
qpos[7:19] =
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
```

确认 actuator 顺序：

```text
ctrl[0:12] =
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
```

结论：

```text
训练顺序、MuJoCo qpos/qvel 顺序、actuator 顺序一致。
```

### 4. 检查 reset 高度和脚底接触

写了 `scripts/check_mujoco_stand.py`，先设置 `default_q`，计算四个 foot sphere 的高度。

结果：

```text
FL/FR bottom_z = -0.3126
RL/RR bottom_z = -0.2858
foot height spread = 0.0267 m
suggested base_z = 0.3176 m
```

含义：

```text
base_z 约 0.31 m 时，脚底刚好接近地面。
前后脚高度差约 2.7 cm，和 default_q 前后腿 thigh 角不同有关。
```

### 5. 零力矩仿真

不加力矩仿真 1 秒。

结果：

```text
没有 NaN
没有飞天
机器人自然趴下
```

结论：

```text
reset 接触本身没有严重爆炸问题。
```

### 6. 纯 PD 站立

不加载 policy，只用 PD 保持 `default_q`。

测试参数：

```text
kp = 40
kd = 1
```

结果：

```text
base z 约 0.28 m
quat 接近 [1, 0, 0, 0]
max torque 约 7 Nm
ncon = 4
机器人能在 viewer 中稳稳站立
```

接触也确认是：

```text
floor <-> FL/FR/RL/RR calf body 上的 foot sphere/collision
```

结论：

```text
XML、reset、actuator、PD、collision 基础链路基本正确。
```

## 真正问题

原来的 `deploy_mujoco.py` 控制结构是：

```text
每 DECIMATION=4 个物理步：
  读取 q/dq
  构造 obs
  policy 推理
  计算 target_q
  计算 torque

中间 3 个物理步：
  复用旧 torque
```

这个问题很大。

PD torque 的公式是：

```text
torque = kp * (target_q - q) - kd * dq
```

其中 `q` 和 `dq` 每个物理步都会变化。尤其接触地面时，速度变化很快。如果 torque 只 50 Hz 更新一次，中间 3 个 200 Hz 物理步复用旧 torque，PD 的阻尼项就不能及时反馈，容易造成：

```text
1. 抽搐
2. 乱蹬
3. 趴地
4. 看起来像被弹飞
5. 视觉上腿部位置异常
```

正确理解：

```text
DECIMATION=4 表示 policy/action 低频更新，不表示 PD torque 也低频更新。
```

部署中应该是：

```text
policy / target_q: 50 Hz
PD torque:         200 Hz，每个物理步都根据当前 q/dq 重算
```

## 修正方式

将 deploy 控制结构改为：

```text
每 DECIMATION 个物理步：
  读取状态
  构造 obs
  policy 推理
  计算 ref_q
  target_q_hold = ref_q + action * residual_scale

每一个物理步：
  重新读取当前 q/dq
  torques = kp * (target_q_hold - q) - kd * dq
  clip torque
  写入 data.ctrl
  mujoco.mj_step
```

伪代码：

```python
if phys_step % DECIMATION == 0:
    obs = build_obs()
    action = policy(obs)
    ref_q = compute_gait_reference(policy_step, cmd_vx)
    target_q_hold[:] = ref_q + action * RES_SCALE
    policy_step += 1

q = current_joint_pos()
dq = current_joint_vel()
torque = kp * (target_q_hold - q) - kd * dq
torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)
data.ctrl[:] = torque
mujoco.mj_step(model, data)
```

核心原则：

```text
hold target_q，不要 hold torque。
```

## 验证结果

修正后，恢复真实 policy residual，运行：

```bash
python scripts/deploy_mujoco.py --headless --steps 1000 --cmd_vx 0.3 --kp 40 --kd 1.0
```

结果：

```text
phys=  100  height=0.298 m  vx=0.178 m/s
phys=  200  height=0.299 m  vx=0.340 m/s
phys=  300  height=0.300 m  vx=0.347 m/s
phys=  400  height=0.298 m  vx=0.303 m/s
phys=  500  height=0.292 m  vx=0.305 m/s
phys=  600  height=0.289 m  vx=0.276 m/s
phys=  700  height=0.295 m  vx=0.327 m/s
phys=  800  height=0.301 m  vx=0.347 m/s
phys=  900  height=0.303 m  vx=0.365 m/s
phys= 1000  height=0.296 m  vx=0.323 m/s
```

结论：

```text
机器人能稳定前进，速度接近 cmd_vx=0.3。
```

## 额外注意：cmd_vx=0.0

训练配置里：

```python
lin_vel_x = [0.3, 0.8]
```

所以 policy 没有训练过真正的 `cmd_vx=0.0` 原地站立。`cmd_vx=0.0` 是分布外输入，不应该作为主要部署效果判断标准。

如果需要原地站立，可以在部署端保留保护逻辑：

```python
if abs(cmd_vx) < 1e-6:
    ref_q = DEFAULT_Q_TRAIN.copy()
else:
    ref_q = compute_gait_reference(policy_step, cmd_vx)
```

但更根本的做法是在训练时加入低速/零速命令，并设计 stand still reward。

## 本次经验

1. 不要一开始就怀疑 policy 或 XML，要先拆开链路测试。
2. `policy.pt` 只是 `obs -> action` 的函数，不包含物理世界。
3. `action` 是 residual，不是关节角，也不是 torque。
4. MuJoCo 中 `nq=19, nv=18` 对 floating base 四足机器人是正常的。
5. reset 是否飞天，要先做零力矩测试和纯 PD 测试。
6. `DECIMATION` 表示高层 action 频率，不表示底层 PD 频率。
7. 部署时必须区分：

```text
高层策略频率
底层 PD 频率
物理仿真频率
```

8. 四足部署中最重要的原则之一：

```text
低频更新 target，高频更新 torque。
```
