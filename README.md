git clone https://github.com/Tosania/mujoco_soft_sim.git
conda create -m "mujoco-soft" python=3.13.2
conda init bash && source /root/.bashrc
conda activate mujoco-soft
pip install -r requirement.txt
