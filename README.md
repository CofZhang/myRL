# Zof Flat - 基于Isaac Gym的四足机器人强化学习训练

基于 [legged_gym](https://github.com/leggedrobotics/legged_gym) 框架，使用 PPO 算法训练四足机器人（zof）在平地环境下行走。

## 项目简介

本项目实现了一个最小化的四足机器人强化学习训练流程，包括：
- 平地行走策略训练（PPO）
- 机器人URDF模型检查工具
- 训练启动与可视化回放脚本

### 机器人

- **名称**：Zof（四足机器人）
- **自由度**：12（每条腿3个关节：hip + thigh + calf）
- **腿部配置**：FL / FR / RL / RR（前左/前右/后左/后右）
- **模型格式**：URDF

## 项目结构

```
myRL/
├── scripts/
│   ├── check_zof_asset.py   # URDF模型检查工具
│   ├── train.py             # 训练启动器
│   └── play.py              # 策略回放可视化
├── zof_gym/
│   └── envs/
│       ├── __init__.py      # 任务注册
│       ├── zof_config.py    # 环境与PPO配置
│       └── zof_robot.py     # ZofRobot环境类
├── resources/
│   └── robots/zof/
│       ├── urdf/zof.urdf    # 机器人模型
│       └── meshes/          # 3D网格文件
├── notes/                   # 设计文档
└── docs/                    # 环境依赖文档
```

## 环境依赖

- **OS**：Linux（Ubuntu 20.04+）
- **Python**：3.8
- **PyTorch**：2.2.0+cu121
- **Isaac Gym**：Preview 4
- **legged_gym**：leggedrobotics 版本
- **rsl_rl**：配套 RL 算法库

详细安装步骤见 [docs/环境依赖文档.md](docs/环境依赖文档.md)。

## 快速开始

### 1. 检查机器人模型

```bash
python scripts/check_zof_asset.py
```

验证 URDF 能否正确加载，检查关节顺序、刚体名称、关节限位等。

### 2. 训练

```bash
# 无图形界面训练（推荐，速度快）
python scripts/train.py --task=zof_flat --headless --num_envs=1024 --max_iterations=1500

# 有图形界面训练
python scripts/train.py --task=zof_flat --num_envs=64 --max_iterations=1500
```

### 3. 回放

```bash
python scripts/play.py --task=zof_flat --num_envs=16 --load_run=<run目录名> --checkpoint=<轮数>
```

## 配置说明

所有训练参数在 `zof_gym/envs/zof_config.py` 中定义，分为两部分：

### ZofFlatCfg（环境配置）

| 配置类 | 说明 |
|--------|------|
| `env` | 环境参数（环境数、观测维度、回合时长） |
| `terrain` | 地形参数（平地、高度测量） |
| `init_state` | 初始姿态（站立高度、默认关节角） |
| `control` | 控制参数（PD增益、动作缩放） |
| `asset` | 机器人模型加载选项 |
| `normalization` | 观测归一化（缩放系数、裁剪） |
| `noise` | 训练噪声（模拟真机传感器） |
| `rewards` | 奖励权重（跟踪、惩罚项） |
| `commands` | 速度命令范围 |

### ZofFlatCfgPPO（训练配置）

| 配置类 | 说明 |
|--------|------|
| `policy` | 策略网络结构（Actor-Critic, [256,128,64]） |
| `algorithm` | PPO 超参数（学习率、clip、GAE） |
| `runner` | 训练流程（迭代数、保存间隔、恢复） |

## 观测向量（45维）

| 索引 | 内容 | 维度 |
|------|------|------|
| 0:3 | 基座角速度 | 3 |
| 3:6 | 重力方向投影 | 3 |
| 6:9 | 速度命令（vx, vy, wz） | 3 |
| 9:21 | 关节角度偏差 | 12 |
| 21:33 | 关节速度 | 12 |
| 33:45 | 上一步动作 | 12 |

## 训练效果

以 512 环境、5000 轮训练为例：

| 指标 | 值 |
|------|-----|
| Mean reward | ~12.5 |
| Episode length | ~850 步 |
| 线速度跟踪 | ~0.72 |
| 角速度跟踪 | ~0.41 |

## 设计文档

- [notes/action_design.md](notes/action_design.md) - 动作空间设计
- [notes/joint_order_design.md](notes/joint_order_design.md) - 关节顺序设计
- [notes/observation_design.md](notes/observation_design.md) - 观测空间设计
- [notes/reset_design.md](notes/reset_design.md) - 重置逻辑设计
- [notes/reward_design.md](notes/reward_design.md) - 奖励函数设计

## 技术栈

- [Isaac Gym](https://developer.nvidia.com/isaac-gym) - GPU加速物理仿真
- [legged_gym](https://github.com/leggedrobotics/legged_gym) - 腿式机器人环境框架
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) - 强化学习算法库（PPO）
- [PyTorch](https://pytorch.org/) - 深度学习框架
