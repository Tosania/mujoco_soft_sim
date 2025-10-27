import math
import time
from typing import Optional, Tuple, Dict, Any, Literal

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

from simulation import SoftRobot  # 直接复用你的仿真封装

ACT_NAMES = [
    "len_north_1",
    "len_south_1",
    "len_east_1",
    "len_west_1",
    "len_north_2",
    "len_south_2",
    "len_east_2",
    "len_west_2",
]
CTRL_LOW, CTRL_HIGH = 0.0, 0.4  # 对应 XML 中 <position ... ctrlrange="0 0.4"> 的区间
DEFAULT_BASE_LEN = 0.20  # 初始长度（你项目里常用的基准）
TIP_SITE = "rod_tip"  # 末端位姿观测
BASE_SITE = "rod_root"  # 用于计算整体弯曲角的基准


class SoftRobotReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        *,
        xml_path: str = "two_disks_uj.xml",
        render_mode: Literal["human", "none"] = "none",
        settle_steps: int = 200,
        dt: float = 0.002,
        max_steps: int = 1000,
        goal_radius: float = 0.25,
        goal_center: Tuple[float, float, float] = (0.0, 0.0, 0.70),
        terminate_dist: float = 0.01,
        gaussian_bonus: float = 5.0,
        delta_per_step: float = 0.01,
        substeps_per_step: int = 5,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert render_mode in ("human", "none")
        self.render_mode = render_mode
        self.dt = float(dt)
        self.settle_steps = int(settle_steps)
        self.max_steps = int(max_steps)
        self.terminate_dist = float(terminate_dist)
        self.gaussian_bonus = float(gaussian_bonus)
        self.delta_per_step = float(delta_per_step)
        self.substeps_per_step = int(substeps_per_step)
        self.goal_radius = float(goal_radius)
        self.goal_center = np.array(goal_center, dtype=np.float32)

        # 仿真
        self.robot = SoftRobot(xml_path)  # 内部已加载 model/data
        self.robot.model.opt.timestep = self.dt

        # 动作空间
        self.action_space = spaces.Box(
            low=np.full(8, CTRL_LOW, dtype=np.float32),
            high=np.full(8, CTRL_HIGH, dtype=np.float32),
            dtype=np.float32,
        )

        # 观测空间： [bend_deg, tip_xyz(3), len(8), tension(8), goal_xyz(3)] -> 23 维
        low = np.concatenate(
            [
                np.array([0.0], dtype=np.float32),  # 弯曲角度下界
                np.full(3, -np.inf, dtype=np.float32),  # tip xyz
                np.full(8, 0.0, dtype=np.float32),  # tendon 长度 >= 0
                np.full(8, 0.0, dtype=np.float32),  # 张力 >= 0
                np.full(3, -np.inf, dtype=np.float32),  # goal xyz
            ]
        )
        high = np.concatenate(
            [
                np.array([180.0], dtype=np.float32),  # 弯曲角度上界
                np.full(3, np.inf, dtype=np.float32),
                np.full(8, CTRL_HIGH, dtype=np.float32),
                np.full(8, np.inf, dtype=np.float32),
                np.full(3, np.inf, dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 末端 site id
        self.tip_sid = mujoco.mj_name2id(
            self.robot.model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE
        )
        if self.tip_sid == -1:
            raise RuntimeError(f"找不到末端 site: {TIP_SITE}(请检查 XML)")
        self._tendon_names = [f"tendon{n[3:]}" for n in ACT_NAMES]
        self._tendon_ids = [
            mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_TENDON, name)
            for name in self._tendon_names
        ]
        # 随机数
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._step_count = 0
        self._goal = None
        self._last_action = np.zeros(8, dtype=np.float32)

        # 初始化各执行器到基准长度
        for name in ACT_NAMES:
            self.robot.control(name, DEFAULT_BASE_LEN)
        self._mj_forward()
        self._settle(self.settle_steps)

        print("环境初始化成功")

    # ------------------------- 工具函数 -------------------------
    def _mj_step(self):
        mujoco.mj_step(self.robot.model, self.robot.data)

    def _mj_forward(self):
        mujoco.mj_forward(self.robot.model, self.robot.data)

    def _settle(self, steps: int):
        for _ in range(max(0, int(steps))):
            self._mj_step()

    def _get_tip_pos(self) -> np.ndarray:
        return self.robot.data.site_xpos[self.tip_sid].astype(np.float32)

    def _get_lengths_forces(self):
        d = self.robot.data
        lens = d.ten_length[self._tendon_ids].astype(np.float32)
        tens = d.ten_force[self._tendon_ids].astype(np.float32)
        return lens, tens

    def _get_bend_deg(self) -> float:
        return float(
            self.robot.get_bending_angle(base_site="rod_root", tip_site=TIP_SITE)
        )

    def _get_obs(self) -> np.ndarray:
        tip = self._get_tip_pos()
        bend = self._get_bend_deg()
        lens, tens = self._get_lengths_forces()
        goal = self._goal.astype(np.float32)
        obs = np.concatenate([[bend], tip, lens, tens, goal]).astype(np.float32)
        return obs

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    # ------------------------- Gym API -------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # 重置执行器到基准长度
        self.robot.mj_resetData(self.model, self.data)
        for name in ACT_NAMES:
            self.robot.control(name, DEFAULT_BASE_LEN)
        self._mj_forward()
        # self._settle(self.settle_steps)

        # 采样目标
        self._goal = self._sample_goal_in_sphere()
        self._step_count = 0
        self._last_action[:] = 0.0

        obs = self._get_obs()
        info = {"goal": self._goal.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32).reshape(8)

        # 应用动作
        if self.control_mode == "absolute":
            targets = np.clip(action, CTRL_LOW, CTRL_HIGH)
        else:  # delta
            # 先取当前“目标长度”（近似用当前 ctrl 值；MuJoCo 的 position 执行器 ctrl 即目标量）
            # 为稳妥，这里维持一个缓存：第一次以 DEFAULT_BASE_LEN 为基准，后续在缓存上累加。
            if np.allclose(self._last_action, 0.0) and self._step_count == 1:
                curr = np.full(8, DEFAULT_BASE_LEN, dtype=np.float32)
            else:
                curr = np.clip(self._last_action, CTRL_LOW, CTRL_HIGH)
            targets = np.clip(curr + action, CTRL_LOW, CTRL_HIGH)

        # 下发控制
        for i, name in enumerate(ACT_NAMES):
            self.robot.control(name, float(targets[i]))
        self._last_action = targets.copy()

        # 物理子步
        for _ in range(max(1, self.substeps_per_step)):
            self._mj_step()

        # 计算观测与奖励
        obs = self._get_obs()
        tip = obs[1:4]
        goal = obs[-3:]
        dist = self._distance(tip, goal)
        reward = -dist + math.exp(-(dist**2)) * self.gaussian_bonus

        terminated = dist < self.terminate_dist
        truncated = self._step_count >= self.max_steps

        info = {
            "distance": dist,
            "last_action": self._last_action.copy(),
        }

        # 渲染（可选：human 模式这里仅返回，不阻塞）
        if self.render_mode == "human":
            # 轻量：什么都不做，保持外部可自行用 viewer 可视化
            pass

        return obs, reward, terminated, truncated, info

    def render(self):
        # 若需要可视化，可在外部单独启动 self.robot.runview()
        # 这里保持空实现以兼容 Gymnasium API
        return None

    def close(self):
        # 无持久资源，这里无需特殊清理
        pass


if __name__ == "__main__":
    a = SoftRobotReachEnv()
