import math
from typing import Optional, Dict, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from simulation import SoftRobot
from pathlib import Path
import os

# ---- 固定常量 ----
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
CTRL_LOW, CTRL_HIGH = 0.15, 0.25
DEFAULT_BASE_LEN = 0.20
DT = 0.002
SUBSTEPS = 5
MAX_STEPS = 1000
TIP_SITE = "rod_tip"
BASE_SITE = "rod_root"
GOAL_CENTER = np.array([0.0, 0.0, 0.70], dtype=np.float32)
GOAL_RADIUS = 0.25
TERMINATE_DIST = 0.01  # 成功阈值


class SoftRobotReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
        self,
        *,
        xml_path: str = "two_disks_uj.xml",
        render_mode: str = "none",
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert render_mode in ("human", "none")
        self.render_mode = render_mode

        # --- 仿真 ---
        self.robot = SoftRobot(xml_path)
        xml_path = str(Path(xml_path).resolve())
        self.model, self.data = self.robot.model, self.robot.data
        self.model.opt.timestep = DT

        # site 与 id
        self.tip_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
        if self.tip_sid == -1:
            raise RuntimeError(f"找不到末端 site: {TIP_SITE}")
        self._act_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in ACT_NAMES
            ],
            dtype=np.int32,
        )
        self._ten_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_TENDON, "tendon" + n[3:]
                )
                for n in ACT_NAMES
            ],
            dtype=np.int32,
        )

        # 动作 / 观测空间
        self.action_space = spaces.Box(
            low=np.full(8, CTRL_LOW, np.float32),
            high=np.full(8, CTRL_HIGH, np.float32),
            dtype=np.float32,
        )
        low = np.concatenate(
            [
                [0.0],
                np.full(3, -np.inf),
                np.full(8, 0.0),
                np.full(8, 0.0),
                np.full(3, -np.inf),
            ]
        ).astype(np.float32)
        high = np.concatenate(
            [
                [180.0],
                np.full(3, np.inf),
                np.full(8, CTRL_HIGH),
                np.full(8, np.inf),
                np.full(3, np.inf),
            ]
        ).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 其它
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._step_count = 0
        self._goal = np.zeros(3, dtype=np.float32)

        # 初始状态
        self.data.ctrl[self._act_ids] = DEFAULT_BASE_LEN
        mujoco.mj_forward(self.model, self.data)

    # -------- Gym API --------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[self._act_ids] = DEFAULT_BASE_LEN
        mujoco.mj_forward(self.model, self.data)

        # ✅ 改为三个独立随机数（立方体采样）
        rand_offset = self.np_random.uniform(-GOAL_RADIUS, GOAL_RADIUS, size=3).astype(
            np.float32
        )
        self._goal = GOAL_CENTER + rand_offset

        self._step_count = 0
        return self._get_obs(), {"goal": self._goal.copy()}

    def step(self, action: np.ndarray):
        self._step_count += 1
        self.data.ctrl[self._act_ids] = np.clip(action, CTRL_LOW, CTRL_HIGH).astype(
            np.float32
        )
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        tip = obs[1:4]
        dist = float(np.linalg.norm(tip - self._goal))
        reward = -dist
        terminated = dist < TERMINATE_DIST
        truncated = self._step_count >= MAX_STEPS
        return obs, reward, terminated, truncated, {"distance": dist}

    def render(self):
        if self.render_mode == "human":
            self.robot.runview()

    def close(self):
        pass

    def _get_obs(self) -> np.ndarray:
        tip = self.data.site_xpos[self.tip_sid].astype(np.float32)
        bend = float(
            self.robot.get_bending_angle(base_site=BASE_SITE, tip_site=TIP_SITE)
        )
        lens = self.data.ten_length[self._ten_ids].astype(np.float32)
        tens = np.array(
            [
                self.robot.get_tendon_force("tendon" + n[3:])["tension"]
                for n in ACT_NAMES
            ],
            dtype=np.float32,
        )
        return np.concatenate([[bend], tip, lens, tens, self._goal]).astype(np.float32)
