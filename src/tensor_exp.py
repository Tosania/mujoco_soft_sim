# exp_nine_plots.py
# 目标：逐步缩短一根绳子的目标长度，整个过程中“每一步采样”，并在结束后输出 9 张图：
# 1) 总弯曲角度 vs 时间
# 2-5) 四根 tendon 的张力 vs 时间（north/south/east/west）
# 6-9) 四根 tendon 的长度 vs 时间（north/south/east/west）
#
# 仅依赖：simulation.py（SoftRobot 类）, mujoco, numpy, matplotlib
# 运行：python exp_nine_plots.py

import os
import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from simulation import SoftRobot
from matplotlib.backends.backend_pdf import PdfPages


def main():
    XML_PATH = "single_disk.xml"
    ACT_LEN = "len_north_1"
    TARGET_TENDON = "tendon_north"
    BASE_LEN = 0.20
    DT = 0.002
    SWEEP_START = 0.20
    SWEEP_STOP = 0.15
    SWEEP_STEPS = 1000
    STEPS_PER_POINT = 25

    OUTDIR = "results"
    os.makedirs(OUTDIR, exist_ok=True)
    STAMP = time.strftime("%Y%m%d_%H%M%S")
    PDF_PATH = os.path.join(OUTDIR, f"experiment_results_{STAMP}.pdf")
    robot = SoftRobot(XML_PATH)
    model, data = robot.model, robot.data
    model.opt.timestep = float(DT)

    tendons = ["tendon_north", "tendon_south", "tendon_east", "tendon_west"]

    # 对应四个长度控制执行器（如有缺失，可自行调整或 try/except 跳过）
    base_len_actuators = []

    # 初始化其它三根到对称基准
    for name in base_len_actuators:
        try:
            robot.control(name, BASE_LEN)
        except Exception:
            pass

    tendon_ids = {}
    for tn in tendons:
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tn)
        if tid == -1:
            raise RuntimeError(f"未找到 tendon: {tn}")
        tendon_ids[tn] = tid

    # 为每条 tendon 找到所有“对其施力”的执行器 id（motor/position 都可能贡献 actuator_force）
    tendon_act_map = {tn: [] for tn in tendons}
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_TENDON):
            tgt_tid = int(model.actuator_trnid[a, 0])
            # 将执行器 a 归入对应 tendon
            for tn, tid in tendon_ids.items():
                if int(tid) == tgt_tid:
                    tendon_act_map[tn].append(a)

    # 生成设定点序列
    sweep = np.linspace(SWEEP_START, SWEEP_STOP, SWEEP_STEPS)

    # ======= 存储数组（每步采样）=======
    time_s = []  # 时间序列
    bend_deg = []  # 总弯曲角度

    tens_hist = {tn: [] for tn in tendons}  # 张力历史（绝对值）
    len_hist = {tn: [] for tn in tendons}  # 长度历史（data.ten_length）

    # ======= 度量函数 =======
    def measure_all(t_now):
        theta = float(robot.get_bending_angle(base_site="rod_root", tip_site="rod_tip"))
        bend_deg.append(theta)

        # 四根 tendon：长度与张力（张力取 get_tendon_force 的 tension 字段）
        for tn in tendons:
            L = float(robot.get_tendon_length(tn))
            T = float(robot.get_tendon_force(tn)["tension"])
            len_hist[tn].append(L)
            tens_hist[tn].append(T)

        time_s.append(float(t_now))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # ======= 预收敛到第一个设定点 =======
        # ======= 主扫描：设定 → 推进并每步采样 =======
        for setp in sweep[1:]:
            robot.control(ACT_LEN, float(setp))
            for _ in range(STEPS_PER_POINT):
                mujoco.mj_step(model, data)
                viewer.sync()
                t_now = (len(time_s) + 1) * DT
                measure_all(t_now)

    # ======= 生成 9 张图 =======
    with PdfPages(PDF_PATH) as pdf:
        # 1) 弯曲角 vs 时间
        fig = plt.figure()
        plt.plot(time_s, bend_deg)
        plt.xlabel("Time (s)")
        plt.ylabel("Bending angle (deg)")
        plt.title("Bending angle vs Time")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        plt.plot(time_s, tens_hist["tendon_north"])
        plt.xlabel("Time (s)")
        plt.ylabel("Tension (N)")
        plt.title(f"Tension vs Time")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        plt.plot(time_s, len_hist["tendon_north"])
        plt.xlabel("Time (s)")
        plt.ylabel("Tendon length (m)")
        plt.title(f"Length vs Time")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        plt.plot(bend_deg, len_hist["tendon_north"])
        plt.xlabel("Bending angle (deg)")
        plt.ylabel("Tendon length (m)")
        plt.title(f"Bending angle vs length — tendon_north")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        plt.plot(bend_deg, tens_hist["tendon_north"])
        plt.xlabel("Bending angle (deg)")
        plt.ylabel("Tension (N)")
        plt.title(f"Bending angle vs tension — tendon_north")
        pdf.savefig(fig)
        plt.close(fig)

        # 最后一页：简要统计
        fig = plt.figure()
        plt.axis("off")
        stats = (
            f"steps:{len(time_s)}\n"
            f"time:{len(time_s)*DT:.3f} s\n"
            f"output stamp:{STAMP}\n"
            f"PDF path:{PDF_PATH}"
        )
        plt.text(0.05, 0.8, stats, fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"✅ 实验完成！共采样 {len(time_s)} 步，结果已保存为：{PDF_PATH}")


if __name__ == "__main__":
    main()
