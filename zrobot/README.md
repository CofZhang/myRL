# zrobot Jetson 部署包

这是从开发电脑 `/home/zhz/Desktop/myRL` 中整理出来的 Jetson 轻量部署包。

## 当前内容

```text
policies/zof_wasd_policy.pt
config/zof_deploy_config.py
scripts/check_policy.py
notes/
```

## 第一步：在 Jetson 上测试 policy

```bash
cd ~/zrobot
python3 scripts/check_policy.py
```

正常输出应包含：

```text
action shape: (1, 12)
has nan: False
```

如果这一步失败，先解决 Jetson 的 Python / PyTorch / TorchScript 环境，不要连接电机。

## 后续顺序

```text
1. check_policy.py
2. check_real_imu.py
3. check_real_joints.py
4. check_real_pd_stand.py
5. deploy_real.py
```

真机 policy 前必须先阅读 notes 中的安全和映射文档。

