# manual_keyboard_direct.py
# 仅用 simulation.SoftRobot；零参数；直接键位映射：
#  1..8 -> 第1..8条 绳子往里缩(减小长度)
#  QWERTYUI -> 第1..8条 绳子往外放(增大长度)
#  + / - 调整步进；R 复位到中点；P 打印当前值

import re, threading, time
import numpy as np
import mujoco
import mujoco.viewer
import tkinter as tk
from simulation import SoftRobot

# ===== 默认配置（如需改，改这里即可） ==========================
XML_NAME = "two_disks_uj.xml"  # SoftRobot 会自行在 xml 目录下寻找
ACT_REGEX = r"^len_"  # 仅控制以 len_ 开头的执行器（绳长执行器）
RENDER_HZ = 60
SUBSTEPS = 1
INIT_STEP = 0.001  # 每次按键的长度改变量
# ============================================================


def enum_actuators(model, name_filter):
    pat = re.compile(name_filter) if name_filter else None
    items = []
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act_{a}"
        if (pat is None) or pat.search(name):
            lo, hi = float(model.actuator_ctrlrange[a, 0]), float(
                model.actuator_ctrlrange[a, 1]
            )
            items.append((a, name, (lo, hi)))
    items.sort(key=lambda x: (x[1], x[0]))  # 名称优先稳定排序
    return items


def sim_thread(xml_path, act_infos, shared, stop_flag, hz=60, substeps=1):
    robot = SoftRobot(xml_path)
    model, data = robot.model, robot.data

    # 初值置中
    for a, _, (lo, hi) in act_infos:
        data.ctrl[a] = (lo + hi) / 2.0
    mujoco.mj_forward(model, data)

    frame_dt = 1.0 / max(1, hz)
    with mujoco.viewer.launch_passive(model, data) as v:
        nxt = time.time()
        while not stop_flag["stop"]:
            if hasattr(v, "is_running") and not v.is_running():
                break

            # 应用共享控制
            with shared["lock"]:
                for a, _, (lo, hi) in act_infos:
                    data.ctrl[a] = float(np.clip(shared["ctrl"][a], lo, hi))

            for _ in range(max(1, int(substeps))):
                mujoco.mj_step(model, data)
            v.sync()

            nxt += frame_dt
            sleep_t = nxt - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                nxt = time.time()


def main():
    # 先用临时实例拿到 model 来枚举 actuator
    tmp = SoftRobot(XML_NAME)
    model = tmp.model
    act_infos = enum_actuators(model, ACT_REGEX)
    if not act_infos:
        all_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act_{a}"
            for a in range(model.nu)
        ]
        raise SystemExit(
            f"未匹配到任何执行器（ACT_REGEX={ACT_REGEX}）。可用：{all_names}"
        )

    # 映射：索引0..7 -> (缩键, 放键)
    # 1..8 缩；QWERTYUI 放
    shrink_keys = list("12345678")
    extend_keys = list("QWERTYUI")
    # 如果执行器少于8条，自动只取前 N 条
    n = min(len(act_infos), 8)
    shrink_keys = shrink_keys[:n]
    extend_keys = extend_keys[:n]

    # 共享状态
    shared = {"ctrl": {}, "step": float(INIT_STEP), "lock": threading.Lock()}
    for a, _, (lo, hi) in act_infos:
        shared["ctrl"][a] = (lo + hi) / 2.0

    stop_flag = {"stop": False}
    thd = threading.Thread(
        target=sim_thread,
        args=(XML_NAME, act_infos, shared, stop_flag, int(RENDER_HZ), int(SUBSTEPS)),
        daemon=True,
    )
    thd.start()

    # Tk 用于捕获键盘与显示状态提示
    root = tk.Tk()
    root.title("直接键控：1-8缩，QWERTYUI放（请保持此窗口为焦点）")
    info = tk.StringVar()

    def format_state():
        lines = []
        for i, (a, name, (lo, hi)) in enumerate(act_infos[:n]):
            val = shared["ctrl"][a]
            lines.append(
                f"[{i+1}/{extend_keys[i]}] {name}: {val:.6f}  [{lo:.4f},{hi:.4f}]"
            )
        return "步进(step): %.6f\n%s" % (shared["step"], "\n".join(lines))

    def refresh_label():
        with shared["lock"]:
            info.set(format_state())
        root.after(120, refresh_label)

    tk.Label(
        root, textvariable=info, justify="left", padx=8, pady=10, font=("Menlo", 10)
    ).pack()

    def bump_index(i, delta):
        with shared["lock"]:
            a, _, (lo, hi) = act_infos[i]
            shared["ctrl"][a] = float(np.clip(shared["ctrl"][a] + delta, lo, hi))

    def reset_mid():
        with shared["lock"]:
            for a, _, (lo, hi) in act_infos:
                shared["ctrl"][a] = (lo + hi) / 2.0

    def print_vals():
        with shared["lock"]:
            arr = [(name, shared["ctrl"][a]) for a, name, _ in act_infos[:n]]
        print("当前 ctrl：", ", ".join([f"{nm}={v:.6f}" for nm, v in arr]))

    def on_key(ev):
        k = ev.keysym
        # 步进调节
        if k in ("plus", "KP_Add", "equal"):
            with shared["lock"]:
                shared["step"] = min(1.0, shared["step"] * 1.5)
            return
        if k in ("minus", "KP_Subtract"):
            with shared["lock"]:
                shared["step"] = max(1e-6, shared["step"] / 1.5)
            return
        # 复位 / 打印
        if k in ("r", "R"):
            reset_mid()
            return
        if k in ("p", "P"):
            print_vals()
            return

        # 缩：1..8
        if k in shrink_keys:
            i = int(k) - 1
            if 0 <= i < n:
                bump_index(i, -shared["step"])
            return

        # 放：QWERTYUI（注意大小写）
        if k in extend_keys or k.lower() in [c.lower() for c in extend_keys]:
            # 转成索引
            key_upper = k.upper()
            if key_upper in extend_keys:
                i = extend_keys.index(key_upper)
                if 0 <= i < n:
                    bump_index(i, +shared["step"])
            return

    root.bind("<Key>", on_key)
    root.after(50, refresh_label)

    def on_close():
        stop_flag["stop"] = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    stop_flag["stop"] = True
    thd.join(timeout=2.0)


if __name__ == "__main__":
    main()
