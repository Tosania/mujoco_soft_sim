import time
import numpy as np
import mujoco
import mujoco.viewer
from pynput import keyboard

XML_PATH = "two_disks_uj.xml"

# ============== 1) 加载模型 ==============
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


# ============== 2) 工具函数：按名称找执行器 ==============
def get_actuator_id_try(names):
    """依次尝试一组候选名称，返回第一个存在的执行器 id；若都不存在则报错。"""
    for nm in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
        if aid != -1:
            return nm, aid
    raise RuntimeError(
        f"执行器未找到（尝试过：{names}）。请检查 XML 中的 actuator 命名。"
    )


# 期望的命名（两段各3根），并为第1段提供无后缀的回退名
NAME_CANDIDATES = {
    "left_1": ["pull_left_1", "pull_left"],
    "right_1": ["pull_right_1", "pull_right"],
    "top_1": ["pull_top_1", "pull_top"],
}

# 解析出每个逻辑通道的 (name, id) 与 ctrlrange 上限
ACTS = {}
CTRL_HI = {}
for key, cands in NAME_CANDIDATES.items():
    nm, aid = get_actuator_id_try(cands)
    ACTS[key] = (nm, aid)
    CTRL_HI[key] = model.actuator_ctrlrange[aid, 1]

# ============== 3) 键盘映射与状态 ==============
# 键 -> 逻辑通道
KEY_BIND = {
    "1": "left_1",
    "2": "right_1",
    "3": "top_1",
}
key_state = {k: False for k in KEY_BIND.keys()}


def reset_all():
    for _, aid in ACTS.values():
        data.ctrl[aid] = 0.0


def on_press(key):
    try:
        ch = key.char.lower()
        if ch in key_state:
            key_state[ch] = True
    except AttributeError:
        if key == keyboard.Key.space:
            for k in key_state:
                key_state[k] = False
            reset_all()


def on_release(key):
    try:
        ch = key.char.lower()
        if ch in key_state:
            key_state[ch] = False
    except AttributeError:
        pass


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# ============== 4) 指令发生器（带平滑/回落） ==============
# 每步增长/回落比例（相对上限），可按需求调
INC = 0.0008  # 按住键时增长速度
DEC = 0.0010  # 松开键时回落速度
ALPHA = 0.5  # 低通平滑因子（越大越跟手，越小越平滑）

# 各通道当前归一化目标（0~1）
u = {k: 0.0 for k in ACTS.keys()}


def step_controls():
    # 根据按键更新 u
    for key_char, logical in KEY_BIND.items():
        if key_state[key_char]:
            u[logical] = min(1.0, u[logical] + INC)
        else:
            u[logical] = max(0.0, u[logical] - DEC)

    # 低通并写入 data.ctrl
    for logical, (nm, aid) in ACTS.items():
        target = u[logical] * CTRL_HI[logical]
        data.ctrl[aid] = (1 - ALPHA) * data.ctrl[aid] + ALPHA * target


# ============== 5) 运行可视化循环 ==============
with mujoco.viewer.launch_passive(model, data) as viewer:
    last_t = time.time()
    while viewer.is_running():
        now = time.time()
        dt = now - last_t
        last_t = now

        step_controls()
        mujoco.mj_step(model, data)

        # HUD 显示当前各通道（方便调试）
        try:
            overlay_lines = [
                "[Segment-1] 1:left  2:right  3:top",
                f"  left_1 : {u['left_1']:.2f}  ({data.ctrl[ACTS['left_1'][1]]:6.2f}/{CTRL_HI['left_1']:g})",
                f"  right_1: {u['right_1']:.2f}  ({data.ctrl[ACTS['right_1'][1]]:6.2f}/{CTRL_HI['right_1']:g})",
                f"  top_1  : {u['top_1']:.2f}  ({data.ctrl[ACTS['top_1'][1]]:6.2f}/{CTRL_HI['top_1']:g})",
                "",
                "[SPACE] 全部清零",
            ]
            viewer.add_overlay(
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "Two-Segment Tendon Control",
                "\n".join(overlay_lines),
            )
        except Exception:
            # 某些版本没有 overlay 也无妨
            pass

        viewer.sync()
        time.sleep(0.001)
