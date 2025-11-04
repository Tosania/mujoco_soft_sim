# tools/precompute_workspace.py
import os
import time
import numpy as np
import mujoco
from pathlib import Path
from simulation import SoftRobot
from tqdm import tqdm
import multiprocessing as mp
import threading
from collections import deque


# --- 与项目保持一致的电机名与范围 ---
ACT_NAMES = [
    "mot_north_1",
    "mot_south_1",
    "mot_east_1",
    "mot_west_1",
    "mot_north_2",
    "mot_south_2",
    "mot_east_2",
    "mot_west_2",
]
# 与 env_softrod 保持一致（-30~30）
CTRL_LOW, CTRL_HIGH = -30.0, 30.0

# 成对索引（相对的两根绳子：一拉一放）
PAIR_INDEXES = [
    (0, 1),  # (N1, S1)
    (2, 3),  # (E1, W1)
    (4, 5),  # (N2, S2)
    (6, 7),  # (E2, W2)
]

# ---- 进程内的全局句柄（由 initializer 填充）----
_GLOBALS = dict()


def _init_worker(xml_path: str, dt: float, tick_q):
    """
    每个子进程初始化一次 MuJoCo 模型与常用 id，避免反复构造开销。
    注意：MuJoCo 线程不安全，这里用“进程”是安全的。
    """
    robot = SoftRobot(xml_path)  # 会自动解析相对路径
    m, d = robot.model, robot.data
    m.opt.timestep = float(dt)

    # 预取 actuator 与 site id
    act_ids = np.array(
        [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACT_NAMES],
        dtype=np.int32,
    )
    tip_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "rod_tip")
    if tip_sid == -1:
        raise RuntimeError("找不到末端 site: rod_tip（请检查 XML）")

    _GLOBALS["robot"] = robot
    _GLOBALS["m"] = m
    _GLOBALS["d"] = d
    _GLOBALS["act_ids"] = act_ids
    _GLOBALS["tip_sid"] = tip_sid
    _GLOBALS["tick_q"] = tick_q  # 逐样本 tick 给主进程


def _sample_chunk(args):
    """
    在子进程里执行：采样一批（chunk）动作，返回 tip 坐标 (k,3)。
    参数：
      chunk_size: 本批数量
      settle_steps: 每个样本的稳定步数
      seed: 本批随机种子
    """
    chunk_size, settle_steps, seed = args
    m = _GLOBALS["m"]
    d = _GLOBALS["d"]
    act_ids = _GLOBALS["act_ids"]
    tip_sid = _GLOBALS["tip_sid"]
    tick_q = _GLOBALS["tick_q"]

    rng = np.random.default_rng(int(seed))
    pts = np.empty((chunk_size, 3), dtype=np.float32)

    u = np.zeros(len(ACT_NAMES), dtype=np.float32)
    for i in range(chunk_size):
        # --- 生成一组满足成对约束的控制 ---
        for a, b in PAIR_INDEXES:
            s = rng.uniform(CTRL_LOW, CTRL_HIGH)
            u[a] = s
            u[b] = -s

        # --- 重置并施加控制 ---
        mujoco.mj_resetData(m, d)
        d.ctrl[act_ids] = u
        mujoco.mj_forward(m, d)

        # --- 漫步到稳定 ---
        for _ in range(settle_steps):
            mujoco.mj_step(m, d)

        # --- 记录 tip 世界坐标 ---
        pts[i] = d.site_xpos[tip_sid]

        # ★ 每个样本完成发一个 tick（非阻塞）
        try:
            tick_q.put_nowait(1)
        except Exception:
            pass

    return pts


def _build_jobs(num_samples: int, chunk_size: int, settle_steps: int, base_seed: int):
    n_full = num_samples // chunk_size
    n_rem = num_samples % chunk_size
    jobs = []
    for i in range(n_full):
        jobs.append((chunk_size, settle_steps, base_seed + i))
    if n_rem > 0:
        jobs.append((n_rem, settle_steps, base_seed + len(jobs)))
    return jobs


def main(
    xml_path: str = "./source/two_disks_uj.xml",
    num_samples: int = 3000,
    settle_steps: int = 100,
    dt: float = 0.002,
    out_path: str = "./source/workspace_points.npy",
    n_workers: int = None,
    chunk_size: int = 64,
    base_seed: int = 0,
):
    """
    多进程采样末端可达工作空间（逐样本实时进度与吞吐率）：
      - xml_path: 模型路径
      - num_samples: 总采样数
      - settle_steps: 每个样本持有步数（让构型稳定）
      - dt: 仿真步长
      - out_path: 输出 .npy 文件路径
      - n_workers: 进程数（默认：min(物理核心-1, 8)）
      - chunk_size: 每个任务块包含的样本数（64~256较稳）
      - base_seed: 随机种子基准
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # --- 计算并行参数 ---
    if n_workers is None:
        cpu = max(1, os.cpu_count() or 1)
        n_workers = max(1, min(cpu - 1, 8))  # 留一个核心给系统
    n_workers = 16
    jobs = _build_jobs(num_samples, chunk_size, settle_steps, base_seed)

    all_pts = []

    # --- 速度统计参数 ---
    ema_alpha = 0.2  # 指数滑动平均权重
    ema_rate = None
    window = deque(maxlen=2000)  # 最近窗口（样本时间戳）

    with mp.Manager() as mgr:
        tick_q = mgr.Queue(maxsize=10000)

        stop_flag = False

        def _consume_ticks(pbar, total):
            """主进程里消费逐样本 tick：更新进度 + 估计吞吐率"""
            nonlocal ema_rate
            last_postfix_t = time.time()
            while not stop_flag or not tick_q.empty():
                try:
                    tick_q.get(timeout=0.1)
                except Exception:
                    continue
                pbar.update(1)

                now = time.time()
                window.append(now)
                if len(window) >= 2:
                    dtw = window[-1] - window[0]
                    if dtw > 1e-6:
                        inst_rate = len(window) / dtw
                        ema_rate = (
                            inst_rate
                            if ema_rate is None
                            else (ema_alpha * inst_rate + (1.0 - ema_alpha) * ema_rate)
                        )

                # 节流更新后缀，降低开销
                if now - last_postfix_t >= 0.25:
                    done = pbar.n
                    pct = 100.0 * done / max(1, total)
                    pbar.set_postfix(
                        {
                            "rate": f"{(ema_rate or 0):.1f} samp/s",
                            "workers": n_workers,
                            "chunk": chunk_size,
                            "done%": f"{pct:.1f}",
                        }
                    )
                    last_postfix_t = now

        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(xml_path, dt, tick_q),  # 传入 tick 队列
            maxtasksperchild=64,  # 防止长时间运行潜在内存膨胀
        ) as pool, tqdm(
            total=num_samples,
            desc="Sampling workspace",
            dynamic_ncols=True,
            mininterval=0.05,
            smoothing=0.05,
            leave=True,
        ) as pbar:
            # 启动消费者线程（主进程里安全更新 tqdm）
            t_consumer = threading.Thread(
                target=_consume_ticks, args=(pbar, num_samples), daemon=True
            )
            t_consumer.start()

            # 收集分块结果
            def _on_done(pts):
                all_pts.append(pts)

            # 提交所有任务
            for job in jobs:
                pool.apply_async(_sample_chunk, (job,), callback=_on_done)

            pool.close()
            pool.join()
            stop_flag = True
            t_consumer.join(timeout=2.0)

    # --- 汇总与保存 ---
    if len(all_pts) == 0:
        raise RuntimeError("没有采样结果，请检查并行配置。")
    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    if len(pts) != num_samples:
        pts = pts[:num_samples]
    np.save(out_path, pts)
    print(f"[OK] workspace points: {pts.shape} -> saved to {out_path}")


if __name__ == "__main__":
    # Windows/macOS-spawn 兼容
    mp.freeze_support()
    main()
