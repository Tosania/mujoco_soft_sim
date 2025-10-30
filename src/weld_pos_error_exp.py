# exp_ring_distance_dual_north.py
# 目标：同时拉动 len_north_1 和 len_north_2；只输出一张总图（10条曲线）
import os, time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from simulation import SoftRobot

# ====== 可调参数 ======
XML_PATH = "./xml/two_disks_uj.xml"
TARGET_ACTS = ["len_north_1", "len_north_2"]  # 改1：两根北向绳
BASE_LEN = 0.20
TARGET_LEN = 0.12
DT = 0.002
RAMP_STEPS = 1000
HOLD_STEPS = 1000
SHOW_VIEWER = True

RING_BODIES = [f"ring_{i}_body" for i in range(1, 11)]
RODB_BODIES = [
    "rodB_8",
    "rodB_16",
    "rodB_24",
    "rodB_32",
    "rodB_40",
    "rodB_48",
    "rodB_56",
    "rodB_64",
    "rodB_72",
    "rodB_last",
]


def main():
    os.makedirs("results/weld", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    pdf_path = f"results/weld/ring_distance_{stamp}.pdf"
    csv_path = f"results/weld/ring_distance_{stamp}.csv"

    robot = SoftRobot(XML_PATH)
    model, data = robot.model, robot.data
    model.opt.timestep = float(DT)

    # 找到两根目标执行器 id（改2：批量）
    act_ids_target = []
    for name in TARGET_ACTS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid == -1:
            raise RuntimeError(f"未找到执行器: {name}")
        act_ids_target.append(aid)
    act_ids_target = np.array(act_ids_target, dtype=np.int32)

    # ring/rodB id
    ring_ids, rodB_ids = [], []
    for rb, bb in zip(RING_BODIES, RODB_BODIES):
        r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rb)
        b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bb)
        if r_id == -1 or b_id == -1:
            raise RuntimeError(f"找不到 body：{rb} 或 {bb}")
        ring_ids.append(r_id)
        rodB_ids.append(b_id)
    ring_ids = np.array(ring_ids, dtype=np.int32)
    rodB_ids = np.array(rodB_ids, dtype=np.int32)

    # 其它执行器统一到 BASE_LEN（包含两根目标在内，接下来爬坡会覆盖）
    ALL_ACTS = [
        "len_north_1",
        "len_south_1",
        "len_east_1",
        "len_west_1",
        "len_north_2",
        "len_south_2",
        "len_east_2",
        "len_west_2",
    ]
    for name in ALL_ACTS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid != -1:
            data.ctrl[aid] = BASE_LEN
    mujoco.mj_forward(model, data)

    # 记录 rodB 初始世界坐标（距离基线）
    rodB_init_pos = data.xpos[rodB_ids].copy()

    total_steps = int(RAMP_STEPS + HOLD_STEPS)
    t_list = np.zeros(total_steps, dtype=np.float64)
    ring_pos = np.zeros((total_steps, 10, 3), dtype=np.float64)
    ring_dist = np.zeros((total_steps, 10), dtype=np.float64)

    ramp_vals = np.linspace(BASE_LEN, TARGET_LEN, RAMP_STEPS, dtype=np.float64)
    hold_vals = np.full(HOLD_STEPS, TARGET_LEN, dtype=np.float64)
    ctrl_traj = np.concatenate([ramp_vals, hold_vals])  # 两根用同一条爬坡

    def one_step(k, set_len):
        data.ctrl[act_ids_target] = float(set_len)
        mujoco.mj_step(model, data)
        t_list[k] = (k + 1) * DT

        # ring / rodB 的“当前世界坐标”
        ring_xyz = data.xpos[ring_ids]  # (10, 3)
        rodb_xyz = data.xpos[rodB_ids]  # (10, 3)

        ring_pos[k] = ring_xyz
        # --- ② 距离=同一时刻的 ring ↔ rodB（焊接对象） ---
        ring_dist[k] = np.linalg.norm(ring_xyz - rodb_xyz, axis=1)

    if SHOW_VIEWER:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for k, val in enumerate(ctrl_traj):
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    break
                one_step(k, val)
                viewer.sync()
    else:
        for k, val in enumerate(ctrl_traj):
            one_step(k, val)

    # 保存 CSV（保留，方便后处理）
    header = ["time_s"] + sum(
        [
            [f"ring{i+1}_x", f"ring{i+1}_y", f"ring{i+1}_z", f"ring{i+1}_dist"]
            for i in range(10)
        ],
        [],
    )
    table = np.zeros((total_steps, 1 + 4 * 10), dtype=np.float64)
    table[:, 0] = t_list
    for i in range(10):
        table[:, 1 + 4 * i : 1 + 4 * i + 3] = ring_pos[:, i, :]
        table[:, 1 + 4 * i + 3] = ring_dist[:, i]
    np.savetxt(csv_path, table, delimiter=",", header=",".join(header), comments="")
    print(f"CSV 已保存: {csv_path}")

    # 只输出一张图：10 条 ring 误差曲线集中在同一张
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure()
        for i in range(10):
            plt.plot(t_list, ring_dist[:, i], label=f"ring{i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance to initial rodB (m)")
        plt.title(
            "Rings distance to their initial welded rodB (north_1 & north_2 pulled)"
        )
        plt.legend(ncol=2, fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)
    print(f"✅ 实验完成！PDF 已保存：{pdf_path}")


if __name__ == "__main__":
    main()
