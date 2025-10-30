# coded by scy
# mujoco version 3.3.7
import time
import mujoco
import numpy as np
import mujoco.viewer
import os
from pathlib import Path


class SoftRobot:
    def __init__(self, xml_path: str):
        p = Path(xml_path)
        if not p.is_absolute():
            # 优先从当前文件上一层找 xml 目录
            base = (
                Path(__file__).resolve().parents[1]
                if (Path(__file__).parent.name == "src")
                else Path(__file__).resolve().parent
            )
            cand = base / "xml" / p.name if p.parent == Path(".") else (base / p)
            p = cand.resolve()
        self.xml_path = str(p)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        print(f"模型加载成功: {self.xml_path}")

    def save_model_info(self, filename: str = "model_info.txt"):
        assert hasattr(self, "model") and hasattr(
            self, "data"
        ), "请先加载 model 与 data"
        model, data = self.model, self.data

        # 计算一次前向学（确保长度、site世界坐标等是最新的）
        mujoco.mj_forward(model, data)

        # 小工具：安全的名字获取
        def safe_name(obj_type, idx, default_prefix):
            name = mujoco.mj_id2name(model, obj_type, idx)
            return name if name is not None else f"{default_prefix}_{idx}"

        # 关节类型映射
        JT = mujoco.mjtJoint
        _jt_map = {
            int(JT.mjJNT_FREE): "free",
            int(JT.mjJNT_BALL): "ball",
            int(JT.mjJNT_SLIDE): "slide",
            int(JT.mjJNT_HINGE): "hinge",
        }

        def jtype_str(code: int) -> str:
            return _jt_map.get(int(code), str(int(code)))

        # 写文件
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"模型文件: {getattr(self, 'xml_path', '<unknown>')}\n")
            f.write("=" * 60 + "\n\n")

            # 基本统计
            f.write("[基本统计]\n")
            f.write(f"body数量: {model.nbody}\n")
            f.write(f"joint数量: {model.njnt}\n")
            f.write(f"actuator数量: {model.nu}\n")
            f.write(f"site数量: {model.nsite}\n")
            f.write(f"tendon数量: {model.ntendon}\n")
            f.write("\n" + "=" * 60 + "\n\n")

            # body 信息（世界位姿用 data.xpos）
            f.write("[body信息]\n")
            for i in range(model.nbody):
                name = safe_name(mujoco.mjtObj.mjOBJ_BODY, i, "unnamed_body")
                pos = data.xpos[i]  # 世界坐标
                f.write(
                    f"{i:3d}: {name:25s} xpos=({pos[0]: .4f}, {pos[1]: .4f}, {pos[2]: .4f})\n"
                )
            f.write("\n" + "=" * 60 + "\n\n")

            # joint 信息（类型、锚点位置/轴）
            f.write("[joint信息]\n")
            for j in range(model.njnt):
                name = safe_name(mujoco.mjtObj.mjOBJ_JOINT, j, "unnamed_joint")
                jtype = jtype_str(model.jnt_type[j])
                pos = model.jnt_pos[j]  # 关节在其父body坐标的锚点
                f.write(
                    f"{j:3d}: {name:25s} type={jtype:6s} jnt_pos=({pos[0]: .4f}, {pos[1]: .4f}, {pos[2]: .4f})\n"
                )
            f.write("\n" + "=" * 60 + "\n\n")

            # 执行器信息（控制/力范围，绑定对象）
            f.write("[执行器信息]\n")
            for a in range(model.nu):
                name = safe_name(mujoco.mjtObj.mjOBJ_ACTUATOR, a, "unnamed_act")
                cr = (
                    model.actuator_ctrlrange[a]
                    if model.actuator_ctrlrange.size
                    else np.array([np.nan, np.nan])
                )
                fr = (
                    model.actuator_forcerange[a]
                    if model.actuator_forcerange.size
                    else np.array([np.nan, np.nan])
                )
                trgid = model.actuator_trnid[
                    a, 0
                ]  # 目标 id（如 tendon/body/joint），-1 表示没有
                trntype = model.actuator_trntype[a]  # 目标类型（见 mjtTrn）
                f.write(
                    f"{a:3d}: {name:25s} ctrl_range=[{cr[0]:.3g}, {cr[1]:.3g}] "
                    f"force_range=[{fr[0]:.3g}, {fr[1]:.3g}] trn_type={int(trntype)} trn_id={int(trgid)}\n"
                )
            f.write("\n" + "=" * 60 + "\n\n")

            # tendon 信息（当前世界长度）
            f.write("[tendon信息]\n")
            for t in range(model.ntendon):
                name = safe_name(mujoco.mjtObj.mjOBJ_TENDON, t, "unnamed_tendon")
                length = (
                    float(data.ten_length[t]) if model.ntendon > 0 else float("nan")
                )
                f.write(f"{t:3d}: {name:25s} length={length:.6f}\n")
            f.write("\n" + "=" * 60 + "\n\n")

            # site 信息（同时给出模型局部坐标与世界坐标，便于比对）
            f.write("[site信息]\n")
            for s in range(model.nsite):
                name = safe_name(mujoco.mjtObj.mjOBJ_SITE, s, "unnamed_site")
                lpos = model.site_pos[s]  # 局部
                wpos = data.site_xpos[s]  # 世界
                f.write(
                    f"{s:3d}: {name:25s} local=({lpos[0]: .4f}, {lpos[1]: .4f}, {lpos[2]: .4f}) "
                    f"world=({wpos[0]: .4f}, {wpos[1]: .4f}, {wpos[2]: .4f})\n"
                )

        print(f"模型信息输出于: {filename}")

    def runview(
        self,
        *,
        dt: float = 0.002,
        real_time: bool = True,
        render_hz: int = 60,
        substeps_per_render: int = 1,
    ) -> dict:
        import time, mujoco, numpy as np

        self.model.opt.timestep = float(dt)

        done_steps = 0
        frame_interval = 1.0 / max(1, int(render_hz))
        next_frame_t = time.time()

        with mujoco.viewer.launch_passive(self.model, self.data) as v:
            while True:
                if hasattr(v, "is_running") and not v.is_running():
                    break
                for _ in range(max(1, int(substeps_per_render))):
                    mujoco.mj_step(self.model, self.data)
                    done_steps += 1

                v.sync()

                if real_time:
                    now = time.time()
                    sleep_t = next_frame_t - now
                    if sleep_t > 0:
                        time.sleep(sleep_t)
                    next_frame_t += frame_interval

        return {"steps": int(done_steps), "sim_time": float(done_steps * dt)}

    def control(self, name: str, value: float):
        act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id == -1:
            raise ValueError(f"执行器 {name} 不存在！")
        self.data.ctrl[act_id] = value

    def get_bending_angle(
        self, base_site: str = "rod_root", tip_site: str = "rod_tip"
    ) -> float:
        """
        计算整体弯曲角度（单位：度）
        方法：根据底端与顶端的连线与竖直方向(0,0,1)的夹角
        """
        sid_base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, base_site)
        sid_tip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, tip_site)
        if sid_base == -1 or sid_tip == -1:
            raise ValueError(f"未找到指定的 site: {base_site} 或 {tip_site}")
        p_base = self.data.site_xpos[sid_base]
        p_tip = self.data.site_xpos[sid_tip]
        vec = p_tip - p_base
        vec_norm = vec / np.linalg.norm(vec)
        z_axis = np.array([0, 0, 1])
        cos_theta = np.clip(np.dot(vec_norm, z_axis), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        return np.degrees(theta)

    def tendon_id(self, name: str) -> int:
        tid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, name)
        if tid == -1:
            raise ValueError(f"找不到tendon: {name}")
        return tid

    def get_tendon_length(self, name: str) -> float:
        mujoco.mj_forward(self.model, self.data)  # 确保状态为最新
        tid = self.tendon_id(name)
        return float(self.data.ten_length[tid])

    def get_tendon_force(self, name: str) -> dict:
        mujoco.mj_forward(self.model, self.data)
        m, d = self.model, self.data
        tid = self.tendon_id(name)

        signed = 0.0
        contributors = []
        for a in range(m.nu):
            if (
                int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_TENDON)
                and int(m.actuator_trnid[a, 0]) == tid
            ):
                f = float(d.actuator_force[a])  # 执行器当前输出（已包含gear的影响）
                g = float(m.actuator_gear[a, 0]) if m.actuator_gear.size else 1.0
                signed += f
                contributors.append((a, f, g))

        return {
            "signed": signed,  # 合力（N），可能为负号（取决于gear方向）
            "tension": abs(signed),  # 张力估计（N）
            "contributors": contributors,  # 可选：[(act_id, actuator_force, gear0), ...]
        }

    def run_all_sp(self):
        a = 0
        print(time.time())
        while 1:
            a += 1
            if a == 1000:
                print(time.time())
            mujoco.mj_step(self.model, self.data)


# if __name__ == "__main__":
#     robot = SoftRobot("two_disks_uj.xml")
#     # robot.save_model_info()

#     # robot.save_model_info()
#     robot.run_all_sp()
