# zof reset / termination v1 设计笔记

本文记录 zof 四足机器人第一版强化学习 reset / termination 设计。reset 决定一个 episode 什么时候结束，以及机器人什么时候回到初始状态重新开始训练。

## reset / termination 是什么

强化学习中，一次从开始到结束的过程叫一个 episode。

四足机器人训练时，一个 episode 可以理解成：

```text
机器人从初始站姿开始
  -> 执行动作
  -> 在仿真中走路
  -> 摔倒或超时
  -> episode 结束
  -> reset 回初始状态
  -> 开始下一轮
```

`termination` 表示当前 episode 结束。

`reset` 表示把机器人状态重置到初始状态，准备下一轮训练。

## 为什么 reset 很重要

reset 条件会强烈影响机器人学到什么。

如果 reset 太宽松：

```text
机器人趴地上滑也不结束
身体撞地也不结束
腿卡住了也不结束
```

策略可能会学到错误行为，比如趴着滑、蹭地走。

如果 reset 太严格：

```text
身体稍微晃一下就结束
腿部稍微碰地就结束
训练早期一动就失败
```

策略探索时间太短，也学不会走路。

所以 reset 要平衡：

```text
明显失败要结束
正常探索不要太早结束
```

## zof reset v1 目标

第一版只做平地稳定行走，所以 reset 设计要简单、清楚、稳定。

zof reset v1 推荐：

```text
reset = base_contact OR body_tilt_too_large OR timeout
```

也就是：

```text
机身碰地 -> 结束
身体倾斜太大 -> 结束
episode 超时 -> 结束
```

## 1. base 接触地面

正常行走时，只有脚应该接触地面。

如果 `base` 接触地面，通常说明机器人已经趴下或摔倒。

判断逻辑：

```text
base contact force > threshold
```

比如：

```text
threshold = 1.0 N
```

这个是最重要的失败条件之一。

推荐：

```text
base contact -> reset
```

## 2. 身体姿态倾斜过大

有时候 base 还没碰地，但机器人已经快翻了。

可以用 projected gravity 判断身体倾斜。

正常站立时，projected gravity 接近：

```text
[0, 0, -1]
```

如果身体前倾、后仰、侧翻，x/y 分量会变大。

一种判断方式：

```text
projected_gravity_x^2 + projected_gravity_y^2 > threshold
```

也可以用 roll / pitch 判断：

```text
abs(roll) > max_roll
abs(pitch) > max_pitch
```

第一版可以先理解成：

```text
roll 或 pitch 超过大约 45 度，就认为快摔倒
```

45 度大约是：

```text
0.8 rad
```

推荐：

```text
body tilt too large -> reset
```

## 3. episode timeout

即使机器人一直没摔，也不能让一个 episode 无限持续。

所以需要设置最大时长：

```text
episode_length_s = 20
```

如果超过 20 秒，就结束当前 episode，然后 reset。

注意：timeout 不等于失败。

它只是说明：

```text
这一轮已经跑够时间了，可以开始下一轮。
```

训练时要区分：

```text
摔倒 reset -> 失败
超时 reset -> 正常结束
```

## 4. 关节角限制

URDF 里有关节角度 limit。

如果关节角接近 limit，说明动作不太安全。

常见处理方式有两种：

```text
接近 limit -> 给 penalty
严重超过安全范围 -> reset
```

第一版可以先简单一些：

```text
joint near limit -> penalty
joint over hard safety limit -> 后面再加 reset
```

不要一开始把 joint reset 设计得太复杂。

## contact 分类

四足机器人不是所有接触地面都不好。

脚接触地面是正常的，身体接触地面是失败，大腿小腿碰地通常是不好的。

推荐分类：

| body 类型 | 处理方式 | 原因 |
|---|---|---|
| foot | normal | 脚本来就应该接触地面 |
| base | reset | 机身碰地基本说明摔倒 |
| hip | penalty | 髋部碰地不好，但早期不一定立刻结束 |
| thigh | penalty | 大腿碰地不好，但早期可允许探索 |
| calf | penalty | 小腿碰地不好，但早期可允许探索 |

第一版推荐：

```text
terminate_after_contacts_on = ["base"]
penalize_contacts_on = ["hip", "thigh", "calf"]
```

## 为什么 thigh/calf 不一碰就 reset

训练早期，机器人动作是随机的，经常会腿部碰地。

如果 thigh/calf 一碰地就 reset，episode 会非常短。

结果是：

```text
机器人还没来得及探索恢复动作，就被强制重置了。
```

所以第一版更合理的做法是：

```text
base 碰地 -> reset
thigh/calf 碰地 -> penalty
foot 碰地 -> normal
```

等机器人学会基本走路后，再考虑更严格的 contact 规则。

## 初始 reset 姿态

每次 reset 时，机器人应该回到一个合理初始状态。

基本包括：

```text
base position
base orientation
joint default positions
joint velocities = 0
base velocities = 0
```

第一版先使用固定初始姿态。

以后为了提高鲁棒性，可以加入随机化：

```text
初始关节角随机扰动
初始 base 位置随机扰动
初始 base 速度随机扰动
初始姿态小角度随机扰动
```

但第一版不要太复杂，先固定初始姿态，确保能学会基础行走。

## timeout 和 failure 的区别

这是一个很重要的概念。

`failure` 表示机器人失败了，比如：

```text
摔倒
机身碰地
姿态严重倾斜
```

`timeout` 表示当前 episode 到达最大时间，比如：

```text
已经走满 20 秒
```

timeout 不应该被当成摔倒惩罚。

如果把 timeout 当失败，会惩罚那些成功走满一整轮的机器人，这是错误的。

## zof reset v1 推荐表

| 条件 | 是否 reset | 是否算失败 | 说明 |
|---|---|---|---|
| base contact | 是 | 是 | 机身碰地，认为摔倒 |
| body tilt too large | 是 | 是 | 姿态严重倾斜，认为快摔倒 |
| timeout | 是 | 否 | 正常结束 episode |
| foot contact | 否 | 否 | 脚接触地面是正常行为 |
| hip contact | 否 | 否，给 penalty | 髋部碰地不好，但第一版不立刻结束 |
| thigh contact | 否 | 否，给 penalty | 大腿碰地不好，但第一版不立刻结束 |
| calf contact | 否 | 否，给 penalty | 小腿碰地不好，但第一版不立刻结束 |
| joint near limit | 否 | 否，给 penalty | 接近关节限位，扣分 |
| joint over hard safety limit | 后续添加 | 是 | 真机部署前需要严格保护 |

## 常见 reset 设计错误

### 1. reset 太宽松

结果：机器人趴着也能继续训练，可能学会趴地滑行。

### 2. reset 太严格

结果：训练早期 episode 太短，机器人还没探索就失败，学不到走路。

### 3. foot contact 也 reset

这是错误的。

脚接触地面是四足机器人行走的必要条件。

### 4. thigh/calf 一碰就 reset

训练早期太严格。

推荐第一版：

```text
thigh/calf contact -> penalty
```

不是立刻 reset。

### 5. timeout 当成 failure

timeout 是正常结束，不是摔倒。

如果把 timeout 当失败，会惩罚成功完成 episode 的机器人。

### 6. 初始姿态不合理

如果 reset 后机器人初始姿态本身就不稳，训练会很困难。

所以 default joint angles 和 base height 必须先调合理。

## zof reset v1 设计原则

第一版 reset 设计遵循：

```text
简单明确
base 碰地必 reset
脚接触地面正常
腿部碰地先 penalty，不急着 reset
timeout 是正常结束，不是失败
先固定初始状态，后面再做随机化
```

reset 的目标不是让机器人一犯错就结束，而是让训练知道什么是真正失败，同时保留足够探索空间。
