import pybullet as p
import pybullet_data
import csv
import time
import os

URDF_PATH = "../examples/models/urdf/urdf3/dual_piepr_humanity.urdf"    # 修改為你的路徑
EE_LINK_NAME = "arm1_gripper_point"              # 修改為你的末端連桿名
CSV_PATH = "reachable_table.csv"

# 初始化 PyBullet (GUI 可視化)
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 載入 URDF
robot_id = p.loadURDF(URDF_PATH, useFixedBase=True)

# 找到 EE 的 joint index
ee_index = -1
for i in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, i)
    if joint_info[12].decode("utf-8") == EE_LINK_NAME:
        ee_index = i
        break
if ee_index == -1:
    raise ValueError("找不到末端連桿: " + EE_LINK_NAME)

# 讀入 CSV 並逐點 IK 驗證
with open(CSV_PATH, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # skip header

    count = 0
    for row in reader:
        x, y, z = float(row[0]), float(row[1]), float(row[2])
        qx, qy, qz, qw = float(row[3]), float(row[4]), float(row[5]), float(row[6])
        pos = [x, y, z]
        orn = [qx, qy, qz, qw]

        controlled_joint_indices = []
        for i in range(p.getNumJoints(robot_id)):
            joint_type = p.getJointInfo(robot_id, i)[2]
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                controlled_joint_indices.append(i)

        # 嘗試做 IK
        joint_angles = p.calculateInverseKinematics(robot_id, ee_index, pos, orn)

        print("=== Robot Joints ===")
        for i in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, i)
            print(f"[{i}] {info[1].decode('utf-8')} - Type: {info[2]}")

        # 設定可控制 joint 的角度
        for j, joint_id in enumerate(controlled_joint_indices):
            p.resetJointState(robot_id, joint_id, joint_angles[j])
            time.sleep(0.1)
        # 檢查是否有自碰撞
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
        has_self_collision = any([c[3] != c[4] for c in contacts])  # 同一個body但不同link碰撞

        # 取得實際末端位置
        actual_pos, actual_orn = p.getLinkState(robot_id, ee_index)[4:6]

        # 計算誤差距離
        dist = ((actual_pos[0] - x) ** 2 + (actual_pos[1] - y) ** 2 + (actual_pos[2] - z) ** 2) ** 0.5

        # 畫球體標記點
        if has_self_collision:
            color = [0, 0, 1, 0.5]  # 藍色：發生自碰撞
        elif dist < 0.02:
            color = [0, 1, 0, 0.5]  # 綠色：成功且無碰撞
        else:
            color = [1, 0, 0, 0.5]  # 紅色：誤差過大

        # 取得實際末端位置
        actual_pos, actual_orn = p.getLinkState(robot_id, ee_index)[4:6]

        # 計算誤差距離
        dist = ((actual_pos[0] - x)**2 + (actual_pos[1] - y)**2 + (actual_pos[2] - z)**2)**0.5

        # 畫球體標記點（綠色:成功 / 紅色:失敗）
        color = [0, 1, 0, 0.5] if dist < 0.02 else [1, 0, 0, 0.5]
        radius = 0.01
        p.loadURDF("sphere_small.urdf", [x, y, z], globalScaling=radius / 0.25)  # sphere_small 是半徑約0.05的球

        count += 1
        if count % 100 == 0:
            print(f"驗證進度: {count}")
            time.sleep(1)  # 放慢可視化速度

print("✅ 所有點完成檢查！按下 [q] 關閉視窗")
while True:
    keys = p.getKeyboardEvents()
    if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
        break
    time.sleep(1)

p.disconnect()
