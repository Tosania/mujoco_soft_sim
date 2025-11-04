# vis_policy.py
# 可视化已训练模型（MuJoCo Viewer）
# 依赖：mujoco>=3, gymnasium, stable-baselines3>=2.1, numpy

import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from env_softrod import SoftRobotReachEnv

# === 你可以在这里直接改路径 ===
XML_PATH = "./source/two_disks_uj.xml"
LOG_DIR = "runs/softrod"
VECNORM_PATH = os.path.join(LOG_DIR, "vecnorm.pkl")
# 首选 best_model.zip，没有就用 last_model.zip
MODEL_PATHS = [
    os.path.join(LOG_DIR, "best_model.zip"),
    os.path.join(LOG_DIR, "last_model.zip"),
]

FPS = 60
STEPS = 3000
GOAL_CACHE = os.path.join("./source/workspace_points.npy")


def _unwrap_raw_env(venv):
    """从向量环境中取出最里层 gym.Env，以便访问 .model/.data。"""
    env = venv
    if hasattr(env, "venv"):  # VecNormalize
        env = env.venv
    if hasattr(env, "envs"):  # Dummy/Subproc
        env = env.envs[0]
    if hasattr(env, "env"):  # TimeLimit 等
        env = env.env
    return env


def _resolve_model_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _vec_step_compat(venv, action):
    """
    兼容 SB3 VecEnv(4元组) 与 Gymnasium(5元组) 的 step 返回值。
    返回: obs, reward, done(bool), infos
    """
    result = venv.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, infos = result
        done = bool(terminated[0] or truncated[0])
    else:
        obs, reward, done_arr, infos = result
        done = bool(done_arr[0])  # SB3 的 VecEnv：done_arr 是 shape (n_envs,)
    return obs, reward, done, infos


def _draw_goal_marker(viewer, goal_xyz):
    """在 viewer 中画一个代表目标位置的小球（每帧覆盖刷新）。"""
    if not isinstance(goal_xyz, np.ndarray):
        goal_xyz = np.asarray(goal_xyz, dtype=float)
    # 清除上一帧的自定义几何体
    try:
        viewer.user_scn.ngeom = 0
    except Exception:
        pass
    # 添加一个黄色小球作为目标点
    try:
        viewer.add_marker(
            pos=goal_xyz.astype(float),
            size=np.array([0.01, 0.01, 0.01], dtype=float),  # 半径约1cm
            rgba=np.array([1.0, 1.0, 0.0, 1.0], dtype=float),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
        )
    except Exception:
        # 不同版本API差异，出错就忽略目标可视化但不中断
        pass


def _extract_distance(infos, raw_env):
    """优先从 info 里取 distance；没有就用 env._goal 与末端位置估算。"""
    dist = None
    # 1) 从 info 取（兼容 VecEnv: list[0] / dict）
    if (
        isinstance(infos, (list, tuple))
        and len(infos) > 0
        and isinstance(infos[0], dict)
    ):
        dist = infos[0].get("distance", None)
    elif isinstance(infos, dict):
        dist = infos.get("distance", None)

    # 2) 取不到就自己算：||goal - ee||
    if dist is None:
        goal = getattr(raw_env, "_goal", None)
        ee = None
        for name in ("_ee_pos", "ee_pos", "end_effector_pos", "_tip_pos", "tip_pos"):
            if hasattr(raw_env, name):
                ee = getattr(raw_env, name)
                break
        if goal is not None and ee is not None:
            goal = np.asarray(goal, dtype=float).reshape(-1)[:3]
            ee = np.asarray(ee, dtype=float).reshape(-1)[:3]
            dist = float(np.linalg.norm(ee - goal))
    return dist


def visualize():
    # === 构造VecEnv（便于对接 SB3 + VecNormalize）===
    def _factory():
        return SoftRobotReachEnv(
            xml_path=XML_PATH,
            render_mode="none",
            seed=0,
            goal_cache=GOAL_CACHE if os.path.exists(GOAL_CACHE) else None,
        )

    venv = make_vec_env(_factory, n_envs=1, seed=0)

    # === 恢复归一化统计（如存在）===
    if os.path.exists(VECNORM_PATH):
        venv = VecNormalize.load(VECNORM_PATH, venv)
        venv.training = False
        venv.norm_reward = False
        print(f"[VIS] 已加载归一化统计: {VECNORM_PATH}")
    else:
        print("[VIS] 未找到 vecnorm.pkl，将以未归一化观测推理。")

    raw_env = _unwrap_raw_env(venv)

    # === 载入策略（或随机）===
    model_path = _resolve_model_path(MODEL_PATHS)
    policy = None
    if model_path is not None:
        policy = PPO.load(model_path, env=venv, device="cpu")
        print(f"[VIS] 已加载模型: {model_path}")
    else:
        print("[VIS] 未找到模型(zip)，使用随机动作仅演示可视化。")

    # === reset 并获取当前目标 ===
    obs = venv.reset()
    current_goal = None
    try:
        # VecEnv 的 reset 返回 obs；从原始 env 读取 _goal 更稳
        current_goal = getattr(raw_env, "_goal", None)
        print("[VIS] 当前目标:", current_goal)
    except Exception:
        pass

    frame_interval = 1.0 / max(1, int(FPS))
    next_t = None

    # Viewer 被动同步
    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as v:
        for step_idx in range(int(STEPS)):
            if hasattr(v, "is_running") and not v.is_running():
                break

            # === 策略动作 ===
            if policy is None:
                action = venv.action_space.sample()
            else:
                act, _ = policy.predict(obs, deterministic=True)
                action = act

            # 统一为一维向量并打印
            obs, reward, done, infos = _vec_step_compat(venv, action)
            dist_step = _extract_distance(infos, raw_env)
            if dist_step is not None:
                print(f"[DIST {step_idx:05d}] {dist_step:.6f} m")
            else:
                print(f"[DIST {step_idx:05d}] (unavailable)")

            # 环境推进

            # 每帧画目标标记（若已知）
            if current_goal is None:
                # 兜底再读一次
                current_goal = getattr(raw_env, "_goal", None)
            if current_goal is not None:
                _draw_goal_marker(v, current_goal)

            # 同步渲染
            v.sync()

            # 以接近真实时间渲染
            if next_t is None:
                next_t = time.time()
            next_t += frame_interval
            sleep_t = next_t - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)

            if done:
                # 你的 env 的 info 通常带 distance，可选打印
                dist = None
                if isinstance(infos, (list, tuple)) and infos:
                    dist = infos[0].get("distance", None)
                elif isinstance(infos, dict):
                    dist = infos.get("distance", None)
                print(f"[VIS] episode end, distance={dist}")

                # 重置并更新目标
                obs = venv.reset()
                try:
                    current_goal = getattr(raw_env, "_goal", None)
                    print("[VIS] 新目标:", current_goal)
                except Exception:
                    pass

    print("[VIS] 可视化结束。")


if __name__ == "__main__":
    visualize()
