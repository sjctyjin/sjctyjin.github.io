#!/usr/bin/env python3
"""
雙臂機器人工作空間分析修復
專門處理包含多個機械臂的URDF文件
"""

import numpy as np
import pickle
import pybullet as p
import os
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DualArmWorkspaceAnalyzer:
    """雙臂機器人工作空間分析器"""

    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.robot_id = None
        self.arm_configs = {}

        self.init_robot()
        self.analyze_robot_structure()

    def init_robot(self):
        """初始化PyBullet和機器人"""
        # 連接PyBullet（靜默模式）
        self.physics_client = p.connect(p.DIRECT)

        # 加載URDF
        self.robot_id = p.loadURDF(self.urdf_path, useFixedBase=True)
        print(f"✅ 成功加載URDF: {os.path.basename(self.urdf_path)}")

    def analyze_robot_structure(self):
        """分析機器人結構，識別各個手臂"""
        num_joints = p.getNumJoints(self.robot_id)
        print(f"📊 總關節數: {num_joints}")

        # 分析所有關節
        all_joints = []
        arm1_joints = []
        arm2_joints = []
        other_joints = []

        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]

            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                joint_data = {
                    'index': i,
                    'name': joint_name,
                    'type': joint_type,
                    'lower_limit': joint_info[8],
                    'upper_limit': joint_info[9]
                }

                all_joints.append(joint_data)

                # 根據名稱分類
                if 'arm1' in joint_name.lower():
                    arm1_joints.append(joint_data)
                elif 'arm2' in joint_name.lower():
                    arm2_joints.append(joint_data)
                else:
                    other_joints.append(joint_data)

        # 構建手臂配置
        self.arm_configs = {
            'arm1': {
                'joints': sorted(arm1_joints, key=lambda x: x['name']),
                'end_effector_link': self._find_end_effector_link('arm1')
            },
            'arm2': {
                'joints': sorted(arm2_joints, key=lambda x: x['name']),
                'end_effector_link': self._find_end_effector_link('arm2')
            }
        }

        # 如果沒有找到分離的手臂，可能是單臂系統
        if not arm1_joints and not arm2_joints:
            # 假設所有關節屬於一個手臂
            main_arm_joints = [j for j in all_joints if 'dummy' not in j['name'].lower()]
            if main_arm_joints:
                self.arm_configs['main_arm'] = {
                    'joints': main_arm_joints,
                    'end_effector_link': len(main_arm_joints) - 1
                }

        # 打印結構信息
        for arm_name, config in self.arm_configs.items():
            joints = config['joints']
            if joints:
                print(f"\n🦾 {arm_name.upper()}:")
                print(f"   關節數: {len(joints)}")
                print(f"   末端執行器: link {config['end_effector_link']}")
                for joint in joints:
                    print(f"   - {joint['name']} (索引 {joint['index']})")

    def _find_end_effector_link(self, arm_prefix):
        """找到指定手臂的末端執行器連桿"""
        num_joints = p.getNumJoints(self.robot_id)

        # 查找包含arm_prefix且最後的連桿
        arm_links = []
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')

            if arm_prefix.lower() in joint_name.lower():
                arm_links.append(i)

        if arm_links:
            # 返回該手臂的最後一個關節索引
            return max(arm_links)

        return num_joints - 1  # 默認返回最後一個

    def forward_kinematics(self, arm_name, joint_values):
        """計算指定手臂的正向運動學"""
        if arm_name not in self.arm_configs:
            return None, None

        config = self.arm_configs[arm_name]
        joints = config['joints']

        if len(joint_values) != len(joints):
            print(f"⚠️ 關節值數量不匹配: 期望{len(joints)}, 得到{len(joint_values)}")
            return None, None

        try:
            # 重置所有關節到零位置
            for i in range(p.getNumJoints(self.robot_id)):
                p.resetJointState(self.robot_id, i, 0.0)

            # 設置指定手臂的關節值
            for joint_data, value in zip(joints, joint_values):
                p.resetJointState(self.robot_id, joint_data['index'], value)

            # 獲取末端執行器狀態
            end_effector_link = config['end_effector_link']
            link_state = p.getLinkState(self.robot_id, end_effector_link)

            position = np.array(link_state[0])
            quat_xyzw = np.array(link_state[1])
            quaternion = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # [w,x,y,z]

            return position, quaternion

        except Exception as e:
            print(f"⚠️ FK計算失敗: {e}")
            return None, None

    def generate_joint_samples(self, arm_name, num_samples):
        """為指定手臂生成隨機關節配置"""
        if arm_name not in self.arm_configs:
            return []

        joints = self.arm_configs[arm_name]['joints']
        samples = []

        for _ in range(num_samples):
            sample = []
            for joint_data in joints:
                lower = joint_data['lower_limit']
                upper = joint_data['upper_limit']

                # 處理無限制關節
                if lower < -1000:
                    lower = -np.pi
                if upper > 1000:
                    upper = np.pi

                value = np.random.uniform(lower, upper)
                sample.append(value)

            samples.append(sample)

        return samples

    def analyze_single_arm_workspace(self, arm_name, x_range=(-0.8, 0.8),
                                     y_range=(-0.8, 0.8), z_range=(0.1, 1.0),
                                     resolution=0.05, max_samples=10000):
        """分析單個手臂的工作空間"""

        if arm_name not in self.arm_configs:
            print(f"❌ 找不到手臂: {arm_name}")
            return None

        print(f"🔍 分析 {arm_name} 工作空間...")
        print(f"   範圍: X{x_range}, Y{y_range}, Z{z_range}")
        print(f"   解析度: {resolution}m")
        print(f"   樣本數: {max_samples}")

        # 生成關節配置樣本
        joint_samples = self.generate_joint_samples(arm_name, max_samples)
        print(f"   生成了 {len(joint_samples)} 個關節配置樣本")

        # 計算正向運動學
        reachable_points = []
        position_orientation_map = {}
        valid_count = 0

        for i, joint_config in enumerate(joint_samples):
            if i % 1000 == 0:
                progress = i / len(joint_samples) * 100
                print(f"   進度: {progress:.1f}% (有效點: {valid_count})")

            position, quaternion = self.forward_kinematics(arm_name, joint_config)

            if position is None:
                continue

            valid_count += 1

            # 檢查是否在目標範圍內
            if (x_range[0] <= position[0] <= x_range[1] and
                    y_range[0] <= position[1] <= y_range[1] and
                    z_range[0] <= position[2] <= z_range[1]):

                # 量化到網格
                grid_x = round(position[0] / resolution) * resolution
                grid_y = round(position[1] / resolution) * resolution
                grid_z = round(position[2] / resolution) * resolution

                grid_pos = (grid_x, grid_y, grid_z)

                if grid_pos not in position_orientation_map:
                    position_orientation_map[grid_pos] = []
                    reachable_points.append([grid_x, grid_y, grid_z])

                position_orientation_map[grid_pos].append(quaternion)

        print(f"✅ {arm_name} 分析完成:")
        print(f"   有效FK計算: {valid_count}/{max_samples}")
        print(f"   範圍內位置: {len(reachable_points)}")
        print(f"   總姿態數: {sum(len(orientations) for orientations in position_orientation_map.values())}")

        # 構建數據結構
        workspace_data = {
            'arm_name': arm_name,
            'reachable_points': reachable_points,
            'unreachable_points': [],
            'position_orientation_map': position_orientation_map,
            'exploration_params': {
                'robot': self.urdf_path,
                'arm': arm_name,
                'x_range': x_range,
                'y_range': y_range,
                'z_range': z_range,
                'resolution': resolution,
                'max_samples': max_samples,
                'valid_fk_count': valid_count,
                'timestamp': datetime.now().isoformat()
            }
        }

        return workspace_data

    def analyze_all_arms(self, **kwargs):
        """分析所有手臂的工作空間"""
        results = {}

        for arm_name in self.arm_configs.keys():
            if self.arm_configs[arm_name]['joints']:  # 只分析有關節的手臂
                print(f"\n{'=' * 50}")
                result = self.analyze_single_arm_workspace(arm_name, **kwargs)
                if result:
                    results[arm_name] = result

        return results

    def save_workspace_data(self, workspace_data, output_dir="workspace_data"):
        """保存工作空間數據"""
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(workspace_data, dict) and 'arm_name' in workspace_data:
            # 單個手臂數據
            arm_name = workspace_data['arm_name']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"workspace_{arm_name}_{timestamp}.pkl"
        else:
            # 多個手臂數據
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"workspace_all_arms_{timestamp}.pkl"

        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(workspace_data, f)

        print(f"💾 工作空間數據已保存: {filepath}")
        return filepath

    def visualize_workspace(self, workspace_data, show_plot=False):
        """可視化工作空間"""
        if isinstance(workspace_data, dict) and 'arm_name' in workspace_data:
            # 單個手臂
            self._plot_single_arm(workspace_data, show_plot)
        else:
            # 多個手臂
            self._plot_multiple_arms(workspace_data, show_plot)

    def _plot_single_arm(self, data, show_plot=False):
        """繪製單個手臂工作空間"""
        points = np.array(data['reachable_points'])
        if len(points) == 0:
            print("⚠️ 沒有可達點，跳過繪圖")
            return

        orientation_counts = [len(data['position_orientation_map'][tuple(pos)])
                              for pos in points]

        fig = plt.figure(figsize=(15, 5))

        # 3D散點圖
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                              c=orientation_counts, cmap='viridis', s=20, alpha=0.7)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'{data["arm_name"]} 3D Workspace')
        plt.colorbar(scatter, ax=ax1, shrink=0.5)

        # XY投影
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(points[:, 0], points[:, 1],
                               c=points[:, 2], cmap='plasma', alpha=0.7)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY Projection (Color = Z)')
        plt.colorbar(scatter2, ax=ax2)

        # 姿態分佈
        ax3 = fig.add_subplot(133)
        ax3.hist(orientation_counts, bins=20, alpha=0.7, color='green')
        ax3.set_xlabel('Orientations per Position')
        ax3.set_ylabel('Number of Positions')
        ax3.set_title('Orientation Diversity')

        plt.tight_layout()

        # 保存圖片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workspace_{data['arm_name']}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 可視化圖表已保存: {filename}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def _plot_multiple_arms(self, all_data, show_plot=False):
        """繪製多個手臂工作空間對比"""
        fig = plt.figure(figsize=(15, 10))

        colors = ['red', 'blue', 'green', 'orange', 'purple']

        # 3D對比圖
        ax1 = fig.add_subplot(221, projection='3d')

        for i, (arm_name, data) in enumerate(all_data.items()):
            points = np.array(data['reachable_points'])
            if len(points) > 0:
                color = colors[i % len(colors)]
                ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                            c=color, alpha=0.6, s=10, label=arm_name)

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Multi-Arm Workspace Comparison')
        ax1.legend()

        # 統計對比
        ax2 = fig.add_subplot(222)
        arm_names = list(all_data.keys())
        position_counts = [len(data['reachable_points']) for data in all_data.values()]

        ax2.bar(arm_names, position_counts, color=colors[:len(arm_names)])
        ax2.set_ylabel('Reachable Positions')
        ax2.set_title('Position Count Comparison')

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workspace_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 對比圖表已保存: {filename}")

        if show_plot:
            plt.show()
        else:
            plt.close()


def main():
    import sys

    # if len(sys.argv) < 2:
    #     print("❌ 請提供URDF文件路徑")
    #     print("使用方法: python dual_arm_workspace_analyzer.py your_robot.urdf [arm_name]")
    #     return

    urdf_path = "../examples/models/urdf/urdf3/dual_piepr_humanity.urdf"
    target_arm = "arm2" #sys.argv[2] if len(sys.argv) > 2 else None

    try:
        # 創建分析器
        analyzer = DualArmWorkspaceAnalyzer(urdf_path)

        if not analyzer.arm_configs:
            print("❌ 沒有找到任何可分析的手臂")
            return

        # 顯示可用的手臂
        print(f"\n📋 可用的手臂:")
        for arm_name, config in analyzer.arm_configs.items():
            joint_count = len(config['joints'])
            if joint_count > 0:
                print(f"   - {arm_name}: {joint_count} 個關節")

        # 分析工作空間
        if target_arm and target_arm in analyzer.arm_configs:
            # 分析指定手臂
            print(f"\n🎯 分析指定手臂: {target_arm}")
            workspace_data = analyzer.analyze_single_arm_workspace(
                target_arm,
                x_range=(-1.0, 1.0),
                y_range=(-1.0, 1.0),
                z_range=(0.0, 1.5),
                resolution=0.08,
                max_samples=15000
            )

            if workspace_data and len(workspace_data['reachable_points']) > 0:
                # 保存數據
                pkl_path = analyzer.save_workspace_data(workspace_data)

                # 可視化
                analyzer.visualize_workspace(workspace_data)

                print(f"\n🎉 成功！現在可以使用:")
                print(f"lookup_table = PoseLookupTable('{pkl_path}')")
            else:
                print(f"❌ {target_arm} 沒有找到可達位置，請檢查參數設置")

        else:
            # 分析所有手臂
            print(f"\n🔄 分析所有手臂...")
            all_workspace_data = analyzer.analyze_all_arms(
                x_range=(-1.0, 1.0),
                y_range=(-1.0, 1.0),
                z_range=(0.0, 1.5),
                resolution=0.08,
                max_samples=15000
            )

            if all_workspace_data:
                # 保存每個手臂的數據
                for arm_name, data in all_workspace_data.items():
                    if len(data['reachable_points']) > 0:
                        pkl_path = analyzer.save_workspace_data(data)
                        print(f"✅ {arm_name} 數據已保存: {pkl_path}")

                # 生成對比可視化
                analyzer.visualize_workspace(all_workspace_data)
            else:
                print("❌ 沒有找到任何可達位置")

    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
