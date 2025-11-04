# env_monitor_gui.py
# 作用：打开一个窗口，实时显示 SoftRobotReachEnv 的 obs / action / reward / distance / done 等信息
# 控制：Start/Pause/Step/Reset；支持随机动作自动跑，或手动输入 action（逗号分隔）
# 依赖：env_softrod.py（你的环境定义），gymnasium, numpy, mujoco, tkinter（标准库自带）

import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from env_softrod import (
    SoftRobotReachEnv,
    CTRL_LOW,
    CTRL_HIGH,
    DEFAULT_BASE_FORCE,
)  # 复用项目内常量

# —— 如果你的 xml 路径与默认不同可自行改这里 ——
XML_PATH = "./source/two_disks_uj.xml"  # 或者你项目里使用的路径


class EnvGUI:
    def __init__(self, master):
        self.master = master
        master.title("SoftRod Env Monitor")
        master.geometry("920x640")
        master.protocol("WM_DELETE_WINDOW", self.on_quit)

        # ==== 环境 ====
        self.env = SoftRobotReachEnv(xml_path=XML_PATH, render_mode="none")
        self.obs, self.info = self.env.reset(seed=0)
        self.last_action = np.full(
            self.env.action_space.shape, DEFAULT_BASE_FORCE, dtype=np.float32
        )
        self.step_count = 0
        self.running = False
        self.use_random = tk.BooleanVar(value=True)
        self.step_delay = tk.DoubleVar(value=0.02)  # 秒

        # 按你当前 _get_obs 的拼接顺序切片：bend(1), tip(3), lens(8), tens(8), goal(3)
        self.idx = {
            "bend": slice(0, 1),
            "tip": slice(1, 4),
            "lens": slice(4, 12),
            "tens": slice(12, 20),
            "goal": slice(20, 23),
        }

        # ==== 顶部控制条 ====
        top = ttk.Frame(master, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self.btn_start = ttk.Button(top, text="Start", command=self.on_start)
        self.btn_pause = ttk.Button(top, text="Pause", command=self.on_pause)
        self.btn_step = ttk.Button(top, text="Step", command=self.on_step)
        self.btn_reset = ttk.Button(top, text="Reset", command=self.on_reset)
        self.btn_quit = ttk.Button(top, text="Quit", command=self.on_quit)

        self.btn_start.pack(side=tk.LEFT, padx=4)
        self.btn_pause.pack(side=tk.LEFT, padx=4)
        self.btn_step.pack(side=tk.LEFT, padx=4)
        self.btn_reset.pack(side=tk.LEFT, padx=4)
        self.btn_quit.pack(side=tk.LEFT, padx=4)

        ttk.Checkbutton(top, text="随机动作", variable=self.use_random).pack(
            side=tk.LEFT, padx=12
        )
        ttk.Label(top, text="每步延迟(s):").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.step_delay, width=6).pack(side=tk.LEFT, padx=4)

        # ==== 手动动作输入 ====
        ctrl = ttk.LabelFrame(
            master, text="Action 输入（逗号分隔；会自动裁剪到合法区间）", padding=8
        )
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.act_str = tk.StringVar(
            value=",".join(
                [f"{DEFAULT_BASE_FORCE:.3f}"] * self.env.action_space.shape[0]
            )
        )
        ttk.Entry(ctrl, textvariable=self.act_str).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6
        )
        ttk.Button(ctrl, text="应用到下一步", command=self.on_apply_action).pack(
            side=tk.LEFT, padx=6
        )

        # ==== 监视信息区：左侧状态 / 右侧 obs 明细 ====
        body = ttk.Frame(master)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        # 左：状态/统计
        left = ttk.LabelFrame(body, text="状态", padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        self.lbl_shape = ttk.Label(left, text=self._shape_text(), justify=tk.LEFT)
        self.lbl_shape.pack(anchor="w")

        self.lbl_bounds = ttk.Label(left, text=self._bounds_text(), justify=tk.LEFT)
        self.lbl_bounds.pack(anchor="w", pady=(6, 0))

        self.lbl_runtime = ttk.Label(left, text="", justify=tk.LEFT)
        self.lbl_runtime.pack(anchor="w", pady=(6, 0))

        # 右：obs/action 明细
        right = ttk.LabelFrame(body, text="观测与动作", padding=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.text = tk.Text(right, height=28, wrap=tk.NONE)
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # 横竖滚动条
        yscroll = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.text.yview)
        xscroll = ttk.Scrollbar(right, orient=tk.HORIZONTAL, command=self.text.xview)
        self.text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        xscroll.pack(side=tk.BOTTOM, fill=tk.X)

        # 底部提示
        tip = ttk.Label(
            master,
            padding=6,
            text="提示：Start开始自动滚动（随机或手动动作）；Pause暂停；Step单步；Reset重置。可随时修改 Action 输入。",
        )
        tip.pack(side=tk.BOTTOM, fill=tk.X)

        # 首次渲染
        self._render_all()

    # ====== 界面文字生成 ======
    def _shape_text(self):
        return (
            f"obs dim: {self.env.observation_space.shape}\n"
            f"act dim: {self.env.action_space.shape}"
        )

    def _bounds_text(self):
        low = np.array(CTRL_LOW).ravel()
        high = np.array(CTRL_HIGH).ravel()
        return f"动作下界: {low}\n" f"动作上界: {high}"

    def _runtime_text(
        self, reward=0.0, distance=None, terminated=False, truncated=False
    ):
        lines = [
            f"steps: {self.step_count}",
            f"reward: {reward:.6f}",
        ]
        if distance is not None and np.isfinite(distance):
            lines.append(f"distance: {distance:.6f}")
        lines.append(f"terminated: {terminated}   truncated: {truncated}")
        return "\n".join(lines)

    def _format_vec(self, name, arr, fmt="{:.6f}"):
        arr = np.array(arr).ravel()
        s = ", ".join(fmt.format(x) for x in arr)
        return f"{name} [{len(arr)}]: [{s}]"

    def _render_all(self, reward=0.0, info=None, terminated=False, truncated=False):
        # 右侧文本
        self.text.delete("1.0", tk.END)

        # 根据切片拆 obs
        bend = self.obs[self.idx["bend"]]
        tip = self.obs[self.idx["tip"]]
        lens = self.obs[self.idx["lens"]]
        tens = self.obs[self.idx["tens"]]
        goal = self.obs[self.idx["goal"]]

        # info 里的 distance
        dist = None
        if info is not None and "distance" in info and np.isfinite(info["distance"]):
            dist = float(info["distance"])

        blocks = [
            self._format_vec("action", self.last_action),
            self._format_vec("obs.bend", bend),
            self._format_vec("obs.tip(xyz)", tip),
            self._format_vec("obs.lens[8]", lens),
            self._format_vec("obs.tens[8]", tens),
            self._format_vec("obs.goal(xyz)", goal),
        ]
        self.text.insert(tk.END, "\n".join(blocks) + "\n")

        # 左侧状态
        self.lbl_runtime.config(
            text=self._runtime_text(reward, dist, terminated, truncated)
        )

    # ====== 控制事件 ======
    def on_start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._run_loop, daemon=True).start()

    def on_pause(self):
        self.running = False

    def on_step(self):
        self._do_one_step()

    def on_reset(self):
        self.obs, self.info = self.env.reset()
        self.step_count = 0
        self._render_all(reward=0.0, info=self.info, terminated=False, truncated=False)

    def on_quit(self):
        self.running = False
        try:
            self.env.close()
        except Exception:
            pass
        self.master.destroy()

    def on_apply_action(self):
        try:
            vals = [
                float(x.strip())
                for x in self.act_str.get().split(",")
                if x.strip() != ""
            ]
            a = np.array(vals, dtype=np.float32)
            # 自动 reshape/裁剪
            if a.size != self.env.action_space.shape[0]:
                raise ValueError(
                    f"期望 {self.env.action_space.shape[0]} 维，但收到 {a.size} 维。"
                )
            a = np.clip(a, CTRL_LOW, CTRL_HIGH).astype(np.float32)
            self.last_action = a
            # 立即单步（方便看效果）
            self._do_one_step(custom_action=a)
        except Exception as e:
            messagebox.showerror("Action 解析错误", str(e))

    # ====== 主循环与单步 ======
    def _choose_action(self):
        if self.use_random.get():
            return self.env.action_space.sample().astype(np.float32)
        else:
            return self.last_action.astype(np.float32)

    def _do_one_step(self, custom_action=None):
        action = (
            self._choose_action()
            if custom_action is None
            else custom_action.astype(np.float32)
        )
        # 保存并执行
        self.last_action = action
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        self._render_all(
            reward=reward, info=info, terminated=terminated, truncated=truncated
        )
        if terminated or truncated:
            self.running = False  # 自动停
            # 你也可以选择自动 reset：self.on_reset()

    def _run_loop(self):
        while self.running:
            self._do_one_step()
            time.sleep(max(0.0, float(self.step_delay.get())))


def main():
    root = tk.Tk()
    import tkinter.font as tkfont

    def set_global_font(size=14):
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=size)
        tkfont.nametofont("TkTextFont").configure(size=size)
        tkfont.nametofont("TkFixedFont").configure(size=size)
        tkfont.nametofont("TkMenuFont").configure(size=size)
        tkfont.nametofont("TkHeadingFont").configure(size=size)
        tkfont.nametofont("TkTooltipFont").configure(size=size)

    set_global_font(size=14)
    app = EnvGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
