# visualize_workspace.py
# 可视化 runs/softrod/workspace_points.npy

import numpy as np
import matplotlib.pyplot as plt

# === 修改这里为你的 npy 文件路径 ===
path = "source/workspace_points.npy"

# 读取数据
pts = np.load(path)
assert pts.ndim == 2 and pts.shape[1] == 3, f"文件格式错误: {pts.shape}"

# 绘制 3D 散点
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(f"Workspace Point Cloud  (N={len(pts)})")


# 让坐标轴比例一致
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([np.mean(x_limits) - plot_radius, np.mean(x_limits) + plot_radius])
    ax.set_ylim3d([np.mean(y_limits) - plot_radius, np.mean(y_limits) + plot_radius])
    ax.set_zlim3d([np.mean(z_limits) - plot_radius, np.mean(z_limits) + plot_radius])


set_axes_equal(ax)
plt.tight_layout()
plt.show()
