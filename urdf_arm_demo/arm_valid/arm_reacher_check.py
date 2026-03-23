#!/usr/bin/env python3
"""
機械臂能力完整評估工具
從URDF檔案開始，全面分析機械臂的工作空間和姿態能力
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import argparse
from datetime import datetime
import time
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import traceback

# 嘗試導入不同的機器人學庫
try:
    import robotics_toolbox as rtb

    HAS_RTB = True
except ImportError:
    HAS_RTB = False

try:
    import pinocchio as pin

    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False

try:
    import pybullet as p

    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']  # 中文字體
plt.rcParams['axes.unicode_minus'] = False  # 負號顯示

class RobotCapabilityAnalyzer:
    def __init__(self, urdf_path, base_link="base_link", end_effector_link="tool0"):
        """
        初始化機械臂能力分析器

        Args:
            urdf_path: URDF檔案路徑
            base_link: 基座連桿名稱
            end_effector_link: 末端執行器連桿名稱
        """
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.end_effector_link = end_effector_link

        # 分析結果存儲
        self.joint_info = {}
        self.workspace_points = []
        self.reachable_poses = {}  # {(x,y,z): [quaternions]}
        self.joint_limits = {}
        self.robot_specs = {}

        print(f"🤖 分析機械臂: {urdf_path}")
        self.analyze_urdf()
        self.initialize_robot_model()

    def analyze_urdf(self):
        """分析URDF檔案獲取機械臂基本信息"""
        print("📋 解析URDF檔案...")

        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()

            # 解析關節信息
            joints = root.findall('.//joint[@type="revolute"]') + root.findall('.//joint[@type="prismatic"]')

            self.joint_info = {}
            for joint in joints:
                joint_name = joint.get('name')
                joint_type = joint.get('type')

                # 獲取關節限制
                limit_elem = joint.find('limit')
                if limit_elem is not None:
                    lower = float(limit_elem.get('lower', '-3.14159'))
                    upper = float(limit_elem.get('upper', '3.14159'))
                    velocity = float(limit_elem.get('velocity', '1.0'))
                    effort = float(limit_elem.get('effort', '100.0'))
                else:
                    lower, upper = -np.pi, np.pi
                    velocity, effort = 1.0, 100.0

                self.joint_info[joint_name] = {
                    'type': joint_type,
                    'lower': lower,
                    'upper': upper,
                    'velocity': velocity,
                    'effort': effort,
                    'range': upper - lower
                }

                self.joint_limits[joint_name] = [lower, upper]

            print(f"✅ 找到 {len(self.joint_info)} 個可動關節")
            for name, info in self.joint_info.items():
                print(f"   {name}: {info['type']}, 範圍 [{info['lower']:.3f}, {info['upper']:.3f}]")

        except Exception as e:
            print(f"❌ URDF解析失敗: {e}")

    def initialize_robot_model(self):
        """初始化機器人模型"""
        print("🔧 初始化機器人模型...")

        self.robot_model = None
        self.fk_solver = None

        # 嘗試使用不同的機器人學庫
        if HAS_PINOCCHIO:
            try:
                self.init_pinocchio_model()
                print("✅ 使用 Pinocchio 模型")
                return
            except Exception as e:
                print(f"⚠️ Pinocchio 初始化失敗: {e}")

        if HAS_PYBULLET:
            try:
                self.init_pybullet_model()
                print("✅ 使用 PyBullet 模型")
                return
            except Exception as e:
                print(f"⚠️ PyBullet 初始化失敗: {e}")

        if HAS_RTB:
            try:
                self.init_robotics_toolbox_model()
                print("✅ 使用 Robotics Toolbox 模型")
                return
            except Exception as e:
                print(f"⚠️ Robotics Toolbox 初始化失敗: {e}")

        print("❌ 無法初始化機器人模型，將使用簡化分析")
        self.init_simplified_model()

    def init_pinocchio_model(self):
        """使用Pinocchio初始化模型"""
        if not HAS_PINOCCHIO:
            raise ImportError("未安裝 Pinocchio")

        self.robot_model = pin.buildModelFromUrdf(self.urdf_path)
        self.robot_data = self.robot_model.createData()
        self.fk_solver = "pinocchio"

    def init_pybullet_model(self):
        """使用PyBullet初始化模型"""
        if not HAS_PYBULLET:
            raise ImportError("未安裝 PyBullet")

        # 啟動PyBullet（無圖形界面）
        p.connect(p.DIRECT)
        self.robot_id = p.loadURDF(self.urdf_path, useFixedBase=True)
        self.fk_solver = "pybullet"

        # 獲取關節信息
        self.pybullet_joint_indices = []
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if joint_name in self.joint_info:
                self.pybullet_joint_indices.append(i)

    def init_robotics_toolbox_model(self):
        """使用Robotics Toolbox初始化模型"""
        if not HAS_RTB:
            raise ImportError("未安裝 Robotics Toolbox")

        self.robot_model = rtb.Robot.URDF(self.urdf_path)
        self.fk_solver = "rtb"

    def init_simplified_model(self):
        """簡化模型（基於關節限制估算）"""
        print("🔧 使用簡化分析模型")
        self.fk_solver = "simplified"

        # 基於關節數量和範圍估算工作空間
        num_joints = len(self.joint_info)
        total_range = sum(info['range'] for info in self.joint_info.values())

        # 粗略估算臂長（假設每個關節對應0.3m的連桿）
        estimated_reach = num_joints * 0.3

        self.robot_specs = {
            'estimated_reach': estimated_reach,
            'num_joints': num_joints,
            'total_joint_range': total_range,
            'method': 'simplified_estimation'
        }

    def compute_forward_kinematics(self, joint_values):
        """
        計算正向運動學

        Args:
            joint_values: 關節角度列表

        Returns:
            (position, orientation): 位置[x,y,z] 和 旋轉矩陣
        """
        try:
            if self.fk_solver == "pinocchio":
                return self._fk_pinocchio(joint_values)
            elif self.fk_solver == "pybullet":
                return self._fk_pybullet(joint_values)
            elif self.fk_solver == "rtb":
                return self._fk_robotics_toolbox(joint_values)
            else:
                return self._fk_simplified(joint_values)
        except Exception as e:
            print(f"⚠️ FK計算失敗: {e}")
            return None, None

    def _fk_pinocchio(self, joint_values):
        """Pinocchio正向運動學"""
        q = np.array(joint_values)
        pin.forwardKinematics(self.robot_model, self.robot_data, q)
        pin.updateFramePlacements(self.robot_model, self.robot_data)

        # 獲取末端執行器位姿
        ee_frame_id = self.robot_model.getFrameId(self.end_effector_link)
        ee_pose = self.robot_data.oMf[ee_frame_id]

        position = ee_pose.translation
        rotation = ee_pose.rotation

        return position, rotation

    def _fk_pybullet(self, joint_values):
        """PyBullet正向運動學"""
        # 設置關節角度
        for i, (joint_idx, value) in enumerate(zip(self.pybullet_joint_indices, joint_values)):
            p.resetJointState(self.robot_id, joint_idx, value)

        # 計算正向運動學
        link_state = p.getLinkState(self.robot_id, len(self.pybullet_joint_indices) - 1)
        position = np.array(link_state[0])
        orientation_quat = np.array(link_state[1])  # [x,y,z,w]

        # 轉換為旋轉矩陣
        rotation = R.from_quat(orientation_quat).as_matrix()

        return position, rotation

    def _fk_robotics_toolbox(self, joint_values):
        """Robotics Toolbox正向運動學"""
        T = self.robot_model.fkine(joint_values)
        position = T.t
        rotation = T.R

        return position, rotation

    def _fk_simplified(self, joint_values):
        """簡化正向運動學（僅供參考）"""
        # 非常簡化的估算，不準確但可以給出大概範圍
        reach = self.robot_specs.get('estimated_reach', 1.0)

        # 假設簡單的球坐標系統
        if len(joint_values) >= 2:
            theta = joint_values[0]  # 水平角
            phi = joint_values[1] if len(joint_values) > 1 else 0  # 垂直角
            r = reach * 0.7  # 假設70%的理論臂長可達

            x = r * np.cos(phi) * np.cos(theta)
            y = r * np.cos(phi) * np.sin(theta)
            z = r * np.sin(phi)

            position = np.array([x, y, z])
            rotation = np.eye(3)  # 簡化為單位矩陣

            return position, rotation

        return np.array([0, 0, 0]), np.eye(3)

    def estimate_workspace_bounds(self):
        """估算工作空間邊界"""
        print("📏 估算工作空間邊界...")

        if self.fk_solver == "simplified":
            reach = self.robot_specs.get('estimated_reach', 1.0)
            bounds = {
                'x_min': -reach, 'x_max': reach,
                'y_min': -reach, 'y_max': reach,
                'z_min': 0, 'z_max': reach,
                'method': 'estimated'
            }
            print(f"📏 估算工作空間: ±{reach:.2f}m")
            return bounds

        # 使用關節限制計算邊界
        sample_configs = self.generate_joint_samples(1000)
        positions = []

        for config in sample_configs:
            pos, rot = self.compute_forward_kinematics(config)
            if pos is not None:
                positions.append(pos)

        if positions:
            positions = np.array(positions)
            bounds = {
                'x_min': positions[:, 0].min(),
                'x_max': positions[:, 0].max(),
                'y_min': positions[:, 1].min(),
                'y_max': positions[:, 1].max(),
                'z_min': positions[:, 2].min(),
                'z_max': positions[:, 2].max(),
                'method': 'sampled'
            }

            print(f"📏 實際工作空間:")
            print(f"   X: [{bounds['x_min']:.3f}, {bounds['x_max']:.3f}]")
            print(f"   Y: [{bounds['y_min']:.3f}, {bounds['y_max']:.3f}]")
            print(f"   Z: [{bounds['z_min']:.3f}, {bounds['z_max']:.3f}]")

        else:
            # 回退到估算
            return self.estimate_workspace_bounds()

        return bounds

    def generate_joint_samples(self, num_samples=10000):
        """生成關節角度樣本"""
        joint_names = list(self.joint_info.keys())
        samples = []

        for _ in range(num_samples):
            sample = []
            for joint_name in joint_names:
                limits = self.joint_info[joint_name]
                value = np.random.uniform(limits['lower'], limits['upper'])
                sample.append(value)
            samples.append(sample)

        return samples

    def comprehensive_workspace_analysis(self, resolution=0.1, max_samples=50000):
        """
        全面的工作空間分析

        Args:
            resolution: 空間解析度 (m)
            max_samples: 最大關節配置樣本數
        """
        print(f"🔍 開始全面工作空間分析...")
        print(f"   空間解析度: {resolution:.3f}m")
        print(f"   最大樣本數: {max_samples}")

        # 估算工作空間邊界
        bounds = self.estimate_workspace_bounds()

        # 生成關節配置樣本
        print("🎲 生成關節配置樣本...")
        joint_samples = self.generate_joint_samples(max_samples)

        # 計算所有樣本的末端位姿
        print("🧮 計算正向運動學...")
        valid_poses = {}
        invalid_count = 0

        start_time = time.time()

        for i, joint_config in enumerate(joint_samples):
            if i % 5000 == 0 and i > 0:
                elapsed = time.time() - start_time
                progress = i / len(joint_samples) * 100
                eta = elapsed / i * (len(joint_samples) - i)
                print(f"   進度: {progress:.1f}% ({i}/{len(joint_samples)}), "
                      f"有效姿態: {len(valid_poses)}, ETA: {eta:.1f}s")

            pos, rot = self.compute_forward_kinematics(joint_config)

            if pos is not None and not np.any(np.isnan(pos)):
                # 將位置量化到網格
                grid_pos = tuple(np.round(pos / resolution) * resolution)

                # 轉換旋轉矩陣為四元數
                quat = R.from_matrix(rot).as_quat()
                quat_wxyz = [quat[3], quat[0], quat[1], quat[2]]  # [w,x,y,z]

                if grid_pos not in valid_poses:
                    valid_poses[grid_pos] = []

                valid_poses[grid_pos].append({
                    'quaternion': quat_wxyz,
                    'joint_config': joint_config.copy()
                })
            else:
                invalid_count += 1

        total_time = time.time() - start_time

        print(f"✅ 工作空間分析完成！")
        print(f"   用時: {total_time:.1f}s")
        print(f"   有效位置: {len(valid_poses)}")
        print(f"   無效樣本: {invalid_count}")
        print(f"   總姿態數: {sum(len(poses) for poses in valid_poses.values())}")

        self.reachable_poses = valid_poses
        self.workspace_points = list(valid_poses.keys())

        return valid_poses

    def analyze_pose_diversity(self):
        """分析每個位置的姿態多樣性"""
        if not self.reachable_poses:
            print("⚠️ 請先執行工作空間分析")
            return {}

        print("📊 分析姿態多樣性...")

        diversity_stats = {}
        orientation_counts = [len(poses) for poses in self.reachable_poses.values()]

        diversity_stats = {
            'total_positions': len(self.reachable_poses),
            'total_orientations': sum(orientation_counts),
            'avg_orientations_per_position': np.mean(orientation_counts),
            'max_orientations': np.max(orientation_counts),
            'min_orientations': np.min(orientation_counts),
            'std_orientations': np.std(orientation_counts)
        }

        print(f"📊 姿態多樣性統計:")
        print(f"   總位置數: {diversity_stats['total_positions']}")
        print(f"   總姿態數: {diversity_stats['total_orientations']}")
        print(f"   平均每位置姿態數: {diversity_stats['avg_orientations_per_position']:.2f}")
        print(f"   最多姿態數: {diversity_stats['max_orientations']}")
        print(f"   最少姿態數: {diversity_stats['min_orientations']}")

        return diversity_stats

    def find_optimal_positions(self, criteria='max_orientations', top_n=10):
        """
        找到最佳位置

        Args:
            criteria: 選擇標準 ('max_orientations', 'center_distance', 'joint_efficiency')
            top_n: 返回前N個位置
        """
        if not self.reachable_poses:
            return []

        print(f"🎯 根據 '{criteria}' 尋找最佳位置...")

        position_scores = []

        for pos, poses in self.reachable_poses.items():
            score = 0

            if criteria == 'max_orientations':
                score = len(poses)
            elif criteria == 'center_distance':
                # 距離中心點的距離（越近越好）
                distance = np.linalg.norm(pos)
                score = 1.0 / (1.0 + distance)
            elif criteria == 'joint_efficiency':
                # 關節角度的效率（接近中間位置的關節配置更好）
                joint_configs = [pose['joint_config'] for pose in poses]
                joint_ranges = [self.joint_info[name]['range'] for name in self.joint_info.keys()]

                efficiencies = []
                for config in joint_configs:
                    # 計算每個關節距離中點的程度
                    efficiency = 0
                    for i, (joint_name, joint_value) in enumerate(zip(self.joint_info.keys(), config)):
                        joint_center = (self.joint_info[joint_name]['lower'] + self.joint_info[joint_name]['upper']) / 2
                        normalized_distance = abs(joint_value - joint_center) / (
                                    self.joint_info[joint_name]['range'] / 2)
                        efficiency += (1.0 - normalized_distance)
                    efficiencies.append(efficiency / len(config))

                score = np.mean(efficiencies) if efficiencies else 0

            position_scores.append((pos, score, len(poses)))

        # 排序並返回前N個
        position_scores.sort(key=lambda x: x[1], reverse=True)

        print(f"🏆 前 {top_n} 個最佳位置:")
        for i, (pos, score, num_poses) in enumerate(position_scores[:top_n]):
            print(f"   {i + 1}. 位置 {pos}: 分數 {score:.3f}, 姿態數 {num_poses}")

        return position_scores[:top_n]

    def visualize_workspace(self, save_path=None):
        """可視化工作空間"""
        if not self.workspace_points:
            print("⚠️ 請先執行工作空間分析")
            return

        print("🎨 生成工作空間可視化...")

        points = np.array(self.workspace_points)
        orientation_counts = [len(self.reachable_poses[pos]) for pos in self.workspace_points]

        fig = plt.figure(figsize=(20, 5))

        # 3D散點圖
        ax1 = fig.add_subplot(141, projection='3d')
        scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                              c=orientation_counts, cmap='viridis', s=20, alpha=0.6)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D工作空間（顏色=姿態數量）')
        plt.colorbar(scatter, ax=ax1, shrink=0.5)

        # XY平面投影
        ax2 = fig.add_subplot(142)
        scatter2 = ax2.scatter(points[:, 0], points[:, 1],
                               c=orientation_counts, cmap='viridis', s=30, alpha=0.7)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY平面投影')
        ax2.set_aspect('equal')
        plt.colorbar(scatter2, ax=ax2)

        # XZ平面投影
        ax3 = fig.add_subplot(143)
        scatter3 = ax3.scatter(points[:, 0], points[:, 2],
                               c=orientation_counts, cmap='viridis', s=30, alpha=0.7)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('XZ平面投影')
        ax3.set_aspect('equal')
        plt.colorbar(scatter3, ax=ax3)

        # 姿態多樣性直方圖
        ax4 = fig.add_subplot(144)
        ax4.hist(orientation_counts, bins=50, alpha=0.7, color='skyblue')
        ax4.set_xlabel('每個位置的姿態數量')
        ax4.set_ylabel('位置數量')
        ax4.set_title('姿態多樣性分佈')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 工作空間圖表已保存到: {save_path}")
        else:
            plt.show()

    def export_results(self, output_dir="robot_analysis_results"):
        """導出分析結果"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 導出完整數據
        full_data = {
            'urdf_path': self.urdf_path,
            'analysis_timestamp': timestamp,
            'joint_info': self.joint_info,
            'robot_specs': self.robot_specs,
            'reachable_positions': [list(pos) for pos in self.workspace_points],
            'workspace_analysis': {
                'total_positions': len(self.workspace_points),
                'total_poses': sum(len(poses) for poses in self.reachable_poses.values()),
                'solver_type': self.fk_solver
            }
        }

        # 導出姿態映射（簡化版，只保存第一個姿態作為示例）
        position_orientation_map = {}
        for pos, poses in self.reachable_poses.items():
            position_orientation_map[str(pos)] = {
                'sample_quaternion': poses[0]['quaternion'],
                'sample_joint_config': poses[0]['joint_config'],
                'total_orientations': len(poses)
            }

        full_data['position_orientation_sample'] = position_orientation_map

        # 保存JSON格式
        json_file = os.path.join(output_dir, f"robot_analysis_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(full_data, f, indent=2, default=str)

        # 保存可達位置列表（供其他程序使用）
        reachable_positions_file = os.path.join(output_dir, f"reachable_positions_{timestamp}.json")
        positions_data = {
            'positions': [list(pos) for pos in self.workspace_points],
            'metadata': {
                'total_positions': len(self.workspace_points),
                'urdf_source': self.urdf_path,
                'analysis_date': timestamp
            }
        }

        with open(reachable_positions_file, 'w') as f:
            json.dump(positions_data, f, indent=2)

        print(f"💾 分析結果已保存到:")
        print(f"   完整數據: {json_file}")
        print(f"   位置列表: {reachable_positions_file}")

        return output_dir

    def generate_capability_report(self):
        """生成機械臂能力報告"""
        print("\n" + "=" * 60)
        print("🤖 機械臂能力評估報告")
        print("=" * 60)

        print(f"\n📁 URDF檔案: {os.path.basename(self.urdf_path)}")
        print(f"🔧 分析方法: {self.fk_solver}")

        print(f"\n🦾 關節信息:")
        for name, info in self.joint_info.items():
            range_deg = np.degrees(info['range'])
            print(f"   {name}: {info['type']}, 範圍 {range_deg:.1f}° ({info['range']:.3f} rad)")

        if self.workspace_points:
            points = np.array(self.workspace_points)
            print(f"\n📏 工作空間範圍:")
            print(f"   X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] m")
            print(f"   Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] m")
            print(f"   Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] m")
            print(f"   可達位置總數: {len(self.workspace_points)}")

            total_poses = sum(len(poses) for poses in self.reachable_poses.values())
            print(f"   總姿態數: {total_poses}")
            print(f"   平均每位置姿態數: {total_poses / len(self.workspace_points):.2f}")

            # 計算工作空間體積（粗略估算）
            workspace_volume = (points[:, 0].max() - points[:, 0].min()) * \
                               (points[:, 1].max() - points[:, 1].min()) * \
                               (points[:, 2].max() - points[:, 2].min())
            print(f"   工作空間體積 (近似): {workspace_volume:.3f} m³")

        print(f"\n💡 建議:")
        if self.fk_solver == "simplified":
            print("   ⚠️ 建議安裝 pinocchio、pybullet 或 robotics-toolbox 以獲得更精確的分析")

        if len(self.joint_info) < 6:
            print("   ⚠️ 關節數量較少，可能限制了姿態靈活性")
        elif len(self.joint_info) > 7:
            print("   ✅ 關節數量充足，具有冗餘度，姿態靈活性較好")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="機械臂能力評估工具")
    parser.add_argument("--urdf", type=str,default="examples/models/urdf/urdf3/dual_piepr_humanity.urdf", help="URDF檔案路徑")
    parser.add_argument("--base_link", type=str, default="base_link", help="基座連桿名稱")
    parser.add_argument("--end_effector", type=str, default="arm2_gripper_point", help="末端執行器連桿名稱")
    parser.add_argument("--resolution", type=float, default=0.08, help="空間解析度 (m)")
    parser.add_argument("--max_samples", type=int, default=30000, help="最大關節配置樣本數")
    parser.add_argument("--output_dir", type=str, default="robot_analysis_results", help="結果輸出目錄")
    parser.add_argument("--skip_analysis", action="store_true", help="跳過詳細分析，僅做基本檢查")

    args = parser.parse_args()

    try:
        # 初始化分析器
        analyzer = RobotCapabilityAnalyzer(
            args.urdf,
            args.base_link,
            args.end_effector
        )

        # 生成基本報告
        analyzer.generate_capability_report()

        if not args.skip_analysis:
            # 執行全面分析
            analyzer.comprehensive_workspace_analysis(
                resolution=args.resolution,
                max_samples=args.max_samples
            )

            # 分析姿態多樣性
            analyzer.analyze_pose_diversity()

            # 找到最佳位置
            analyzer.find_optimal_positions(criteria='max_orientations', top_n=5)

            # 導出結果
            analyzer.export_results(args.output_dir)

            # 生成可視化
            viz_path = os.path.join(args.output_dir, "workspace_visualization.png")
            analyzer.visualize_workspace(viz_path)

            print(f"\n🎉 分析完成！結果已保存到 '{args.output_dir}' 目錄")

        else:
            print("\n⏭️ 跳過詳細分析（使用 --skip_analysis 參數）")
            print("如需完整分析，請移除該參數重新運行")

    except Exception as e:
        print(f"❌ 分析過程中出錯: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("🤖 機械臂能力評估工具")
    print("支持的機器人學庫:")
    print(f"   Pinocchio: {'✅' if HAS_PINOCCHIO else '❌'}")
    print(f"   PyBullet: {'✅' if HAS_PYBULLET else '❌'}")
    print(f"   Robotics Toolbox: {'✅' if HAS_RTB else '❌'}")
    print()

    main()
