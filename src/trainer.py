# rl_framework.py
# 依赖：stable-baselines3>=2.0, gymnasium, numpy
# 用法示例（单行）：
#   训练：  from rl_framework import SoftRodRL; SoftRodRL.quick_train(total_timesteps=2_00_000)
#   评估：  from rl_framework import SoftRodRL; SoftRodRL.quick_eval("runs/softrod/best_model.zip", episodes=10)

from typing import Optional, Dict, Any
import os
import numpy as np
from pathlib import Path
from stable_baselines3.common.vec_env import SubprocVecEnv

# 在 SoftRodRL.__init__ 之前，加一个项目根推断（src/ 下、或根目录运行都能对）
PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
    if (Path(__file__).parent.name == "src")
    else Path(__file__).resolve().parent
)

XML_DIR = PROJECT_ROOT / "xml"
RUNS_DIR = PROJECT_ROOT / "runs" / "softrod"
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from env_softrod import SoftRobotReachEnv


class SoftRodRL:
    def __init__(
        self,
        *,
        xml_path: str = None,  # 你的默认模型  :contentReference[oaicite:4]{index=4}
        algo: str = "PPO",
        n_envs: int = 1,
        seed: int = 42,
        device: str = "auto",
        log_dir: str = None,
        total_timesteps: int = 200_000,
        eval_every_steps: int = 10_0,
        eval_episodes: int = 5,
        render_mode: str = "none",  # 训练时一般设为 "none"
        policy_kwargs: Optional[Dict[str, Any]] = None,
        learn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.xml_path = str(xml_path or (XML_DIR / "two_disks_uj.xml"))
        self.algo = algo.upper()
        self.n_envs = int(n_envs)
        self.seed = int(seed)
        self.device = device
        self.log_dir = str(log_dir or RUNS_DIR)
        self.total_timesteps = int(total_timesteps)
        self.eval_every_steps = int(eval_every_steps)
        self.eval_episodes = int(eval_episodes)
        self.render_mode = render_mode
        self.policy_kwargs = policy_kwargs or dict(net_arch=[128, 128])
        self.learn_kwargs = learn_kwargs or {}

        os.makedirs(self.log_dir, exist_ok=True)
        self.model = None
        self.env = None
        self.eval_env = None
        self.best_model_path = os.path.join(self.log_dir, "best_model")

    # ---------- 内部：构造环境 ----------
    def _make_env(self, render_mode: str):
        def _factory():
            # 复用你写好的 env（动作/观测空间、reset/step 已封装好）  :contentReference[oaicite:5]{index=5}
            return SoftRobotReachEnv(
                xml_path=self.xml_path, render_mode=render_mode, seed=self.seed
            )

        return _factory

    def _build_envs(self):
        self.env = make_vec_env(
            self._make_env(self.render_mode),
            n_envs=self.n_envs,
            seed=self.seed,
            vec_env_cls=SubprocVecEnv,
        )
        # 评估环境：单环境、固定 seed 偏移，render 关闭
        self.eval_env = make_vec_env(
            self._make_env("none"), n_envs=1, seed=self.seed + 123
        )

    # ---------- 内部：构造算法 ----------
    def _build_algo(self):
        if self.algo == "PPO":
            self.model = PPO(
                policy="MlpPolicy",
                env=self.env,
                verbose=1,
                tensorboard_log=self.log_dir,
                device=self.device,
                seed=self.seed,
                policy_kwargs=self.policy_kwargs,
                n_steps=64,
            )
        else:
            raise NotImplementedError(
                f"暂不支持的算法：{self.algo}（当前默认提供 PPO）"
            )

    # ---------- 训练 ----------
    def train(self) -> str:
        """启动训练，返回最优模型路径（.zip）"""
        if self.env is None:
            self._build_envs()
        if self.model is None:
            self._build_algo()

        # 评估回调：自动保存最好的模型
        eval_cb = EvalCallback(
            self.eval_env,
            best_model_save_path=self.best_model_path,
            log_path=self.log_dir,
            eval_freq=max(1, self.eval_every_steps // max(1, self.n_envs)),
            n_eval_episodes=self.eval_episodes,
            deterministic=True,
            render=False,
        )

        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=eval_cb,
            **self.learn_kwargs,
            progress_bar=True,
        )

        # 如果没有“最佳”，也保存最终
        final_path = os.path.join(self.log_dir, "last_model.zip")
        self.model.save(final_path)

        # 选择返回最佳模型路径；若不存在，则返回 last
        best_zip = os.path.join(self.best_model_path, "best_model.zip")
        return best_zip if os.path.exists(best_zip) else final_path

    # ---------- 评估 ----------
    def evaluate(
        self,
        model_path: Optional[str] = None,
        episodes: Optional[int] = None,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """评估指定模型（或当前已训练模型），返回 {'mean_reward':..., 'std_reward':...}"""
        if self.eval_env is None:
            self._build_envs()

        # 加载模型（若未给出路径且已有 self.model，则用当前模型）
        if model_path:
            model = PPO.load(model_path, env=self.eval_env, device=self.device)
        else:
            if self.model is None:
                raise RuntimeError("没有可评估的模型：请先训练或提供 model_path")
            model = self.model

        n_ep = int(episodes or self.eval_episodes)
        mean_r, std_r = evaluate_policy(
            model,
            self.eval_env,
            n_eval_episodes=n_ep,
            deterministic=deterministic,
            render=False,
        )
        return {"mean_reward": float(mean_r), "std_reward": float(std_r)}
