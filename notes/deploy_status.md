# 部署验证状态

## 脚本
`scripts/deploy_isaac.py`

## 运行方法
```bash
conda activate /home/zhz/pan1/leggedgym
cd /home/zhz/Desktop/myRL
python scripts/deploy_isaac.py --steps 400 --headless    # 快速验证
python scripts/deploy_isaac.py --steps 2000 --headless   # 长跑
python scripts/deploy_isaac.py --steps 500 --viewer      # 带渲染（需 X11）
```

## 当前结果（2025-07-03，checkpoint_3000）
- 0–4.5 s：机器人站立并前进，base_vx 在 0–0.3 m/s 之间波动
- ~4.5 s：倒地（高度降到 0.06 m），之后趴在地上不动

## 倒地原因分析
1. 部署脚本没有 reset 逻辑，训练中摔倒会 reset，部署不会
2. 策略在 4–5 s 失稳——可能原因：
   a. episode_length_buf 计数方式与部署端 step_idx 有微小差异，导致 phase 偏移
   b. 训练时有随机命令和噪声，策略对固定命令的稳态还不够强
   c. 训练 checkpoint 3000 对应约 3000 * 24 env * 0.02 s ≈ 1440 s 总经验，不算很多

## 下一步改进
- [ ] 对比训练环境和部署脚本的 phase 计算（episode_length_buf 起始是 0 还是 1）
- [ ] 打印倒地前后的 projected_gravity 判断是哪个方向失稳
- [ ] 继续训练（checkpoint 5000/10000）再测
- [ ] 加 reset 逻辑，测试在 episode 边界行为

## 关键结论
部署闭环已通：policy.pt → obs → IK ref + residual → PD torque → Isaac Gym 能跑
第 6 步（部署验证程序）已完成基础版本。
