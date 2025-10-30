# keyboard_control.py
# 用法: python keyboard_control.py
# 说明: 在终端里按数字键 1..8 触发对应电机的短脉冲力（见脚注）
# 依赖: mujoco, numpy, simulation.SoftRobot（请确保 PYTHONPATH 包含项目路径）

import sys
import time
import select
import termios
import tty
from pathlib import Path

import mujoco
import numpy as np
from simulation import SoftRobot

# ---------- 用户可调参数 ----------
XML_PATH = "./xml/two_disks_uj.xml"  # 如果你的 xml 在别处，改这里
FORCE = 20.0  # 按键触发时给电机的控制值（motor 的 ctrl）
DURATION_S = 0.12  # 每次按键持续时间（秒）
DT = 0.002  # 仿真步长（应与 xml 中 opt.timestep 一致）
RENDER_HZ = 60  # 渲染帧率

# ---------- 键 -> actuator 名称 映射 ----------
# 根据 two_disks_uj.xml 里的 actuator 名称 (mot_*_1 / mot_*_2)
KEY_MAP = {
    "1": "mot_east_1",
    "2": "mot_west_1",
    "3": "mot_north_1",
    "4": "mot_south_1",
    "5": "mot_east_2",
    "6": "mot_west_2",
    "7": "mot_north_2",
    "8": "mot_south_2",
    # 你可以在这里添加更多键（例如 q 放大/缩小 FORCE 等）
}


# ---------- 辅助：终端单字符非阻塞读取 (Unix) ----------
class NonBlockingInput:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_char(self, timeout=0.0):
        # timeout: 秒（float），0 表示立刻返回（非阻塞）
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            return sys.stdin.read(1)
        return None


def main():
    # 检查 xml 文件
    xml_p = Path(XML_PATH)
    if not xml_p.exists():
        print(f"[ERR] XML not found: {XML_PATH}")
        return

    robot = SoftRobot(XML_PATH)
    model, data = robot.model, robot.data

    # 把时间步写回 model（以脚本里的 DT 为准）
    model.opt.timestep = float(DT)
    steps_per_frame = max(1, int(round((1.0 / RENDER_HZ) / DT)))

    # 解析 KEY_MAP 中的 actuator id（若不存在则 warn）
    key_to_actid = {}
    for k, aname in KEY_MAP.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        if aid == -1:
            print(f"[WARN] actuator '{aname}' not found in model (key {k})")
        else:
            key_to_actid[k] = aid

    # 记录哪些 act_id 当前处于“激活脉冲中” -> 剩余步数
    pulse_remaining = {aid: 0 for aid in key_to_actid.values()}

    # 计算脉冲步数
    pulse_steps = max(1, int(round(DURATION_S / DT)))

    print("=== Keyboard control 启动 ===")
    print("按键映射（按下即触发短脉冲）:")
    for k, a in KEY_MAP.items():
        print(f"  {k}: {a}")
    print("按 Ctrl-C 退出。注意：需在终端中输入数字（非 viewer 窗口）。")

    # 打开 viewer（passive 模式）
    with mujoco.viewer.launch_passive(model, data) as v:
        # 进入非阻塞终端模式
        with NonBlockingInput():
            next_frame_t = time.time()
            try:
                while True:
                    # 读取一次终端输入（非阻塞）
                    ch = (
                        sys.stdin.read(1)
                        if select.select([sys.stdin], [], [], 0)[0]
                        else None
                    )
                    if ch:
                        if ch in key_to_actid:
                            aid = key_to_actid[ch]
                            pulse_remaining[aid] = pulse_steps
                            # 立刻施力（会在后续 mj_step 中生效）
                        elif ch == "\x03":  # Ctrl-C
                            raise KeyboardInterrupt
                        # 可在此扩展其它命令：比如调 FORCE、打印状态等

                    # 每帧推进若干仿真步
                    for _ in range(steps_per_frame):
                        # 在每个 mj_step 之前：设置 data.ctrl
                        # 对于处于脉冲期的 actuator，设置为 FORCE（正负可扩展）
                        for aid, rem in list(pulse_remaining.items()):
                            if rem > 0:
                                data.ctrl[aid] = FORCE
                                pulse_remaining[aid] = rem - 1
                            else:
                                # 恢复到 0（不施力）
                                data.ctrl[aid] = 0.0

                        mujoco.mj_step(model, data)

                    # 渲染并限速
                    v.sync()
                    # 简单限速
                    next_frame_t += 1.0 / max(1, RENDER_HZ)
                    sleep_t = next_frame_t - time.time()
                    if sleep_t > 0:
                        time.sleep(sleep_t)
            except KeyboardInterrupt:
                print("\n[INFO] 退出控制。")
            finally:
                # 退出前把所有控制置零
                for aid in key_to_actid.values():
                    data.ctrl[aid] = 0.0
                mujoco.mj_forward(model, data)


if __name__ == "__main__":
    main()
