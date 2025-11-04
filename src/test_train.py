# train_softrod.py
# 依赖：gymnasium、stable-baselines3>=2.1、numpy、mujoco
# 运行：
#   python train_softrod.py
# 查看 TensorBoard：
#   tensorboard --logdir runs/softrod --port 6006

from trainer import SoftRodRL

# --- 放在 test_train.py 顶部 import 后 ---
import numpy as np
import mujoco

from env_softrod import SoftRobotReachEnv
from stable_baselines3 import PPO


def visualize_rollout(model_path=None, steps=1500, fps=60):
    """
    在线可视化：打开 Mujoco Viewer，使用训练好的策略（或随机）跑一段时间。
    - model_path: PPO 模型 .zip 路径；None 则用随机策略
    - steps: 总步数
    """
    env = SoftRobotReachEnv(
        xml_path="./xml/two_disks_uj.xml", render_mode="none", seed=0
    )
    # 构造 viewer（直接复用 env 的 model/data，确保看到一致状态）
    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        # 如果有模型就加载
        policy = None
        if model_path is not None:
            policy = PPO.load(model_path, device="cpu")

        obs, info = env.reset()
        print("[VIS] goal =", info["goal"])
        dt = env.model.opt.timestep
        frame_interval = 1.0 / max(1, int(fps))
        next_t = None

        for t in range(int(steps)):
            if hasattr(v, "is_running") and not v.is_running():
                break

            if policy is None:
                # 随机动作（在合法范围内）
                action = env.action_space.sample()
            else:
                # 模型动作
                action, _ = policy.predict(obs, deterministic=True)

            obs, rew, term, trunc, info2 = env.step(action)

            # 同步渲染
            v.sync()

            # 简单的限速渲染以接近真实时间
            if next_t is None:
                import time

                next_t = time.time()
            next_t += frame_interval
            sleep_t = next_t - time.time()
            if sleep_t > 0:
                import time

                time.sleep(sleep_t)

            if term or trunc:
                print(f"[VIS] episode end: distance={info2.get('distance', None):.4f}")
                obs, info = env.reset()
                print("[VIS] new goal =", info["goal"])


def main():
    # 极简训练配置（你可以把 total_timesteps 调大）
    agent = SoftRodRL(
        xml_path="./source/two_disks_uj.xml",  # 你的模型XML
        n_envs=12,  # 最小可用配置
        n_steps=128,
        total_timesteps=20000,
        use_callback=False,
        batch_size=256,
        learning_rate=1e-4,
        gamma=0.98,
        clip_range=0.15,
        vf_coef=0.7,
        render_mode="none",  # 训练禁用渲染以提速
        log_dir="runs/softrod",  # TensorBoard 日志与模型保存目录
        goal_cache="source/workspace_points.npy",
        # policy_kwargs / learn_kwargs 都可保持默认
    )

    # 训练：返回最佳模型路径（若无best则返回 last_model.zip）
    best_model_path = agent.train()
    print(f"[INFO] Best (or last) model saved at: {best_model_path}")

    # 评估：给一个快速的量化结论
    # stats = agent.evaluate(model_path=best_model_path, episodes=5)
    # print(
    #     f"[EVAL] mean_reward={stats['mean_reward']:.3f}, std={stats['std_reward']:.3f}"
    # )


if __name__ == "__main__":
    # visualize_rollout(model_path=None, steps=2000)
    main()
