import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import math
import csv
from scipy.spatial.transform import Rotation as R

# === 可參數化設定 ===
URDF_PATH = "../examples/models/urdf/urdf3/dual_piepr_humanity.urdf"    # 修改為你的路徑
EE_LINK_NAME = "arm2_gripper_point"              # 修改為你的末端連桿名
SAVE_FILE = "reachable_table.csv"

# 掃描空間範圍 (meters)
x_range = np.linspace(0.3, 0.9, 20)
y_range = np.linspace(-0.3, 0.3, 20)
z_range = np.linspace(0.7, 1.1, 10)

# 測試的多組四元數（XYZ 軸旋轉組合）

def generate_orientations(base_quat=[0.548, -0.547, 0.441, -0.454]):
    orientations = []
    base_rot = R.from_quat(base_quat)  # 注意順序是 [x, y, z, w]
    for angle in np.linspace(-np.pi, np.pi, 6):
        z_rot = R.from_euler('z', angle)
        combined_rot = z_rot * base_rot
        quat = combined_rot.as_quat()  # [x, y, z, w]
        orientations.append(quat.tolist())
    return orientations

# 初始化 PyBullet
physicsClient = p.connect(p.DIRECT)  # or p.GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 載入 URDF
robot_id = p.loadURDF(URDF_PATH, useFixedBase=True)

# 取得末端 effector link index
num_joints = p.getNumJoints(robot_id)
ee_index = -1
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    if joint_info[12].decode("utf-8") == EE_LINK_NAME:
        ee_index = i
        break
if ee_index == -1:
    raise ValueError("找不到末端連桿: " + EE_LINK_NAME)

print(f"找到 EE 連桿 Index: {ee_index}")

# 開始遍歷
reachable_data = []
orientations = generate_orientations()

total = len(x_range) * len(y_range) * len(z_range) * len(orientations)
progress = 0

print("開始遍歷空間與姿態...")
for x in x_range:
    for y in y_range:
        for z in z_range:
            for quat in orientations:
                joint_angles = p.calculateInverseKinematics(robot_id, ee_index, [x, y, z], quat)
                # 檢查是否成功反解出角度
                if joint_angles is not None:
                    # 可加上進一步檢查角度限制、碰撞等
                    reachable_data.append([x, y, z, *quat])
                progress += 1
                if progress % 500 == 0:
                    print(f"進度: {progress}/{total} ({progress / total * 100:.2f}%)")

print(f"總共找到可達點數: {len(reachable_data)}")

# 儲存 CSV
with open(SAVE_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    writer.writerows(reachable_data)

print(f"✅ 已儲存結果至 {SAVE_FILE}")

p.disconnect()
