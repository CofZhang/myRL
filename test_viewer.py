"""
MuJoCo 机器人模型可视化
运行方式: python test_viewer.py
"""
import mujoco
import mujoco.viewer

# 机器人模型路径
xml_path = '/home/zhz/Desktop/qrobot/resources/robot/mei/xml/mei.xml'

# 加载模型
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("=" * 50)
print("MuJoCo 模型加载成功！")
print(f"nq (位置维度): {model.nq}")
print(f"nv (速度维度): {model.nv}")
print(f"nu (控制维度): {model.nu}")
print("=" * 50)
print("正在打开可视化窗口...")

# 启动可视化窗口
viewer = mujoco.viewer.launch_passive(model, data)

print("窗口已打开！")
print("按 Enter 退出...")

input()

viewer.close()
print("已退出")
