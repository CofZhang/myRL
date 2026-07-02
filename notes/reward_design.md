# zof reward v1 设计笔记

本文记录 zof 四足机器人第一版强化学习奖励设计。reward 是强化学习里最重要的部分之一，它决定机器人最终会学到什么行为。

## reward 是什么

强化学习里，reward 可以理解成：

```text
你怎么告诉机器人：什么行为是好的，什么行为是坏的。
```

策略网络一开始什么都不会，只会随机输出 action。训练过程中，它会不断尝试动作，然后根据 reward 判断哪些动作更值得保留。

如果 reward 设计合理，机器人会慢慢学会稳定行走。

如果 reward 设计错误，机器人可能会学到奇怪动作，比如：

```text
趴在地上滑
原地乱踢腿
站着不动
身体疯狂抖动
为了省力完全不走
为了速度直接摔倒
```

所以 reward 不是越多越好，而是要让每个奖励项都有明确目的。

## zof reward v1 目标

第一版只做平地稳定行走，不追求台阶、跳跃、复杂地形。

zof reward v1 的目标是：

```text
能站稳
能按 command 走
身体不要乱晃
动作不要太抖
不要用身体蹭地
不要太费力
```

第一版 reward 可以由这些项组成：

```text
reward =
    tracking_lin_vel
  + tracking_ang_vel
  + alive
  - lin_vel_z
  - ang_vel_xy
  - orientation
  - torques
  - dof_acc
  - action_rate
  - collision
```

其中有些是正奖励，有些是惩罚项。

## reward 项解释

### 1. tracking_lin_vel

作用：奖励机器人跟踪前进/侧向速度命令。

command 里有两个线速度命令：

```text
vx: 前进/后退速度
vy: 左右平移速度
```

机器人实际速度越接近命令，奖励越高。

常见公式：

```text
error = (command_vx - actual_vx)^2 + (command_vy - actual_vy)^2
reward = exp(-error / sigma)
```

这个 reward 是最核心的行走奖励。

它告诉机器人：

```text
用户让你往哪里走，你就尽量按那个速度走。
```

### 2. tracking_ang_vel

作用：奖励机器人跟踪转向速度命令。

command 里还有一个角速度命令：

```text
wz: 绕 z 轴转向角速度
```

实际转向速度越接近命令，奖励越高。

常见公式：

```text
error = (command_wz - actual_wz)^2
reward = exp(-error / sigma)
```

它告诉机器人：

```text
用户让你转，你就按要求转。
```

### 3. alive

作用：只要机器人没摔倒，每一步给一点小奖励。

例如：

```text
alive = 0.1
```

它鼓励机器人保持可控状态，不要轻易摔倒。

但 alive 不能太大。如果 alive 奖励太强，机器人可能会觉得：

```text
只要站着不动就能拿奖励，没必要走。
```

所以 alive 只能是辅助奖励，不能盖过 tracking 奖励。

### 4. lin_vel_z

作用：惩罚机身上下乱跳。

四足机器人平地走路时，身体可以有轻微上下起伏，但不应该像弹簧一样乱蹦。

常见惩罚：

```text
penalty = base_lin_vel_z^2
```

它告诉机器人：

```text
不要上下乱跳，身体高度要稳定。
```

### 5. ang_vel_xy

作用：惩罚身体绕 x/y 轴快速旋转。

机器人绕 z 轴转向是正常的，但绕 x/y 轴快速旋转通常表示身体在前后翻或左右滚。

常见惩罚：

```text
penalty = base_ang_vel_x^2 + base_ang_vel_y^2
```

它告诉机器人：

```text
不要前后翻，不要左右滚。
```

### 6. orientation

作用：惩罚身体姿态歪斜。

我们可以用 projected gravity 判断身体是否水平。

如果身体比较平，projected gravity 接近：

```text
[0, 0, -1]
```

如果身体倾斜，x/y 分量会变大。

常见惩罚：

```text
penalty = projected_gravity_x^2 + projected_gravity_y^2
```

它告诉机器人：

```text
身体尽量保持水平。
```

### 7. torques

作用：惩罚力矩过大。

常见惩罚：

```text
penalty = sum(torque^2)
```

它告诉机器人：

```text
不要用太大的力。
```

这个奖励项可以减少能耗，也能减少真机电机过热和机械冲击。

但 torques 惩罚不能太强。如果太强，机器人会不敢用力，最后走不动。

### 8. dof_acc

作用：惩罚关节加速度过大。

关节速度变化太快，说明动作很抖。

常见惩罚：

```text
penalty = sum((last_dof_vel - dof_vel)^2 / dt^2)
```

它告诉机器人：

```text
动作要平滑，不要乱抖。
```

### 9. action_rate

作用：惩罚 action 变化太快。

常见惩罚：

```text
penalty = sum((last_action - action)^2)
```

它告诉策略：

```text
不要上一帧一个动作，下一帧突然跳到另一个动作。
```

这个项对减少腿部高频抖动很有用。

但 action_rate 惩罚也不能太强，否则策略会过于保守，腿摆不开。

### 10. collision

作用：惩罚非足端部位撞地。

脚接触地面是正常的，但这些部位不应该接触地面：

```text
base
hip
thigh
calf
```

如果这些 body 的接触力超过阈值，就给惩罚。

它告诉机器人：

```text
只能用脚走，不要趴着、蹭着、撞着地面走。
```

## 推荐初始权重

zof reward v1 可以先用这些权重作为起点：

| reward 项 | 建议权重 | 说明 |
|---|---:|---|
| tracking_lin_vel | 1.0 | 主奖励，跟踪前进/侧向速度 |
| tracking_ang_vel | 0.5 | 跟踪转向速度 |
| alive | 0.1 | 没摔倒就给一点奖励 |
| lin_vel_z | -2.0 | 惩罚上下乱跳 |
| ang_vel_xy | -0.05 | 惩罚翻滚/俯仰角速度 |
| orientation | -1.0 | 惩罚身体歪斜 |
| torques | -0.00002 | 惩罚力矩过大 |
| dof_acc | -2.5e-7 | 惩罚关节加速度过大 |
| action_rate | -0.01 | 惩罚 action 变化太快 |
| collision | -1.0 | 惩罚身体/腿部撞地 |

这些权重不是最终答案，只是训练起点。

reward 权重一定要根据训练表现调整。

## 常见 reward 设计错误

### 1. tracking 奖励太弱

结果：机器人可能站着不动。

因为它发现：

```text
不动也不会摔，还能拿 alive reward。
```

### 2. alive 奖励太强

结果：机器人只想活着，不想移动。

alive 只能作为辅助奖励，不能超过跟踪速度奖励的重要性。

### 3. torque 惩罚太强

结果：机器人不敢用力，腿摆不开，走不动。

### 4. action_rate 惩罚太强

结果：动作太保守，策略不敢快速摆腿。

### 5. collision 惩罚太弱

结果：机器人可能学会趴着滑，或者用大腿、小腿蹭地。

### 6. command 范围太大

结果：一开始训练太难，机器人直接摔，学不到东西。

第一版 command 范围应该保守一些，例如：

```text
vx: [-0.5, 0.8]
vy: [-0.3, 0.3]
wz: [-0.5, 0.5]
```

等能走稳后，再逐渐扩大速度范围。

## 调 reward 时怎么看问题

训练时不要只看总 reward，还要看每个 reward 子项。

如果机器人站着不走：

```text
tracking_lin_vel 可能太弱
alive 可能太强
command 范围可能太小或采样有问题
```

如果机器人乱跳：

```text
lin_vel_z 惩罚可能太弱
orientation 惩罚可能太弱
Kp/Kd 或 action_scale 可能太大
```

如果机器人腿抖：

```text
action_rate 惩罚可能太弱
dof_acc 惩罚可能太弱
Kd 可能太小
```

如果机器人走不动：

```text
torques 惩罚可能太强
action_scale 可能太小
command 范围可能不合理
默认站姿可能不对
```

如果机器人趴着滑：

```text
collision 惩罚太弱
termination 条件太宽松
base 高度或姿态惩罚不足
```

## 后续复杂地形 reward 扩展

等平地稳定后，可以逐步加入更复杂的 reward：

```text
feet_air_time: 鼓励合理迈步节奏
feet_clearance: 鼓励摆腿时脚抬高
slip penalty: 惩罚脚接触地面时打滑
base_height: 约束身体高度
stumble penalty: 惩罚脚撞到竖直障碍
stand_still: 零速度命令时保持站稳
terrain curriculum: 从简单地形逐渐过渡到复杂地形
```

这些不要一开始全加。第一版先让平地能走。

## zof reward v1 设计原则

第一版 reward 设计遵循这些原则：

```text
先简单，后复杂
先平地，后地形
先稳定，后速度
先 PPO 基线，后 MoE
每个 reward 项都要知道自己在解决什么问题
训练和部署的观测/action 定义必须一致
```

如果某个 reward 项不知道为什么存在，就先不要加。
