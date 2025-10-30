# bench_n_env_speed.py
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 禁用 Qt 的 X11
os.environ.setdefault("MPLBACKEND", "Agg")  # Matplotlib 非交互后端
os.environ.setdefault("MUJOCO_GL", "egl")
import time
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import psutil
import matplotlib.pyplot as plt

from trainer import SoftRodRL
from stable_baselines3.common.callbacks import BaseCallback


# -------------------- Resource Monitor --------------------
class ResourceMonitor:
    def __init__(self, interval_s: float = 0.2):
        self.proc = psutil.Process()
        self.interval = float(interval_s)
        self.samples_cpu, self.samples_rss = [], []
        self._stop = threading.Event()
        self._th = None

    def start(self):
        try:
            self.proc.cpu_percent(None)
        except Exception:
            pass
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)

    def _loop(self):
        while not self._stop.is_set():
            try:
                cpu = self.proc.cpu_percent(None)
                rss = self.proc.memory_info().rss
                self.samples_cpu.append(cpu)
                self.samples_rss.append(rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def summary(self) -> Dict[str, float]:
        cpu_avg = float(np.mean(self.samples_cpu)) if self.samples_cpu else 0.0
        cpu_max = float(np.max(self.samples_cpu)) if self.samples_cpu else 0.0
        rss_peak_gb = (
            float(np.max(self.samples_rss)) / (1024**3) if self.samples_rss else 0.0
        )
        return {"cpu_avg": cpu_avg, "cpu_max": cpu_max, "rss_peak_gb": rss_peak_gb}


# -------------------- Timing Callback --------------------
class TimingCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self._t_rollout_start = None
        self._t_prev_end = None
        self._ts_prev = 0
        self.sampling_sps_list = []
        self.overall_sps_list = []

    def _on_training_start(self):
        self._ts_prev = int(self.model.num_timesteps)
        self._t_prev_end = time.perf_counter()

    def _on_rollout_start(self):
        self._t_rollout_start = time.perf_counter()

    def _on_rollout_end(self):
        t_end = time.perf_counter()
        ts_now = int(self.model.num_timesteps)
        dts = ts_now - self._ts_prev
        if (
            dts > 0
            and self._t_rollout_start is not None
            and self._t_prev_end is not None
        ):
            t_sampling = t_end - self._t_rollout_start
            t_overall = t_end - self._t_prev_end
            if t_sampling > 0:
                self.sampling_sps_list.append(dts / t_sampling)
            if t_overall > 0:
                self.overall_sps_list.append(dts / t_overall)
        self._ts_prev = ts_now
        self._t_prev_end = t_end

    def _on_step(self):
        return True


# -------------------- Dataclass --------------------
@dataclass
class BenchResult:
    n_env: int
    steps: int
    wall_s: float
    steps_per_s: float
    cpu_avg: float
    cpu_max: float
    rss_peak_gb: float


# -------------------- Run One Benchmark --------------------
def run_one(n_env: int, *, total_timesteps: int, xml_path: str):
    agent = SoftRodRL(
        xml_path=xml_path,
        n_envs=n_env,
        total_timesteps=total_timesteps,
        eval_every_steps=int(1e12),  # 训练期不触发评估
        eval_episodes=1,
        render_mode="none",
        log_dir="runs/softrod_bench",
        device="cpu",
    )

    mon = ResourceMonitor(interval_s=0.2)
    mon.start()
    t0 = time.perf_counter()
    agent.train()  # 不再传 callback
    wall = time.perf_counter() - t0
    mon.stop()

    sps_e2e = (total_timesteps / wall) if wall > 1e-9 else 0.0
    summ = mon.summary()
    result = BenchResult(
        n_env=n_env,
        steps=int(total_timesteps),
        wall_s=float(wall),
        steps_per_s=float(sps_e2e),  # 图上用 e2e
        cpu_avg=summ["cpu_avg"],
        cpu_max=summ["cpu_max"],
        rss_peak_gb=summ["rss_peak_gb"],
    )
    extra = {
        "sps_sampling": float("nan"),
        "sps_overall": float("nan"),
        "sps_e2e": sps_e2e,
    }
    return result, extra


# -------------------- Plot --------------------
def plot_results(results: List[BenchResult], outdir: str):
    os.makedirs(outdir, exist_ok=True)
    results = sorted(results, key=lambda r: r.n_env)
    xs = [r.n_env for r in results]
    th = [r.steps_per_s for r in results]
    cpu = [r.cpu_avg for r in results]
    mem = [r.rss_peak_gb for r in results]

    plt.figure()
    plt.plot(xs, th, marker="o")
    plt.xlabel("n_env")
    plt.ylabel("steps / second (overall)")
    plt.title("Training Throughput vs n_env")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "throughput_vs_n_env.png"), dpi=160)
    plt.close()

    plt.figure()
    plt.plot(xs, cpu, marker="o")
    plt.xlabel("n_env")
    plt.ylabel("CPU usage avg (%)")
    plt.title("CPU Avg vs n_env")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "cpu_avg_vs_n_env.png"), dpi=160)
    plt.close()

    plt.figure()
    plt.plot(xs, mem, marker="o")
    plt.xlabel("n_env")
    plt.ylabel("Peak RSS (GB)")
    plt.title("Peak Memory vs n_env")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "mem_peak_vs_n_env.png"), dpi=160)
    plt.close()


# -------------------- Main --------------------
def main():
    N_ENVS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    TOTAL_TIMESTEPS = 4000
    XML_PATH = "./xml/two_disks_uj.xml"

    results = []
    for n in N_ENVS:
        try:
            print(f"\n[RUN] n_env={n} ...")
            r, extra = run_one(n, total_timesteps=TOTAL_TIMESTEPS, xml_path=XML_PATH)
            print(
                f"  sps_sampling={extra['sps_sampling']:.1f}, "
                f"sps_overall={extra['sps_overall']:.1f}, "
                f"sps_e2e={extra['sps_e2e']:.1f}, "
                f"cpu_avg={r.cpu_avg:.1f}%, cpu_max={r.cpu_max:.1f}%, "
                f"peak_rss={r.rss_peak_gb:.3f} GB, wall={r.wall_s:.2f}s"
            )
            results.append(r)
        except Exception as e:
            print(f"[WARN] n_env={n} 运行失败：{e}")

    if results:
        plot_results(results, "results/bench_n_env")
    else:
        print("[ERR] 没有可用结果")


if __name__ == "__main__":
    main()
