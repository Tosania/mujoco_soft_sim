import mujoco
import numpy as np

# 载入模型
model = mujoco.MjModel.from_xml_path("two_disks_uj.xml")
data = mujoco.MjData(model)

# 输出所有刚体名称
print("刚体数量:", model.nbody)
print("刚体名称列表:")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"  {i}: {name}")

# 更新仿真数据以计算位姿
mujoco.mj_forward(model, data)

# 输出每个刚体的世界坐标（质心位置）
print("\n刚体在世界坐标中的位置:")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    pos = data.xpos[i]  # 刚体在世界坐标系中的位置 (x, y, z)
    print(f"{name}: {pos}")
