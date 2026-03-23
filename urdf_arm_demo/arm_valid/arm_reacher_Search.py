#!/usr/bin/env python3
"""
機械臂姿態檢索表格系統
基於預計算的工作空間數據，實現快速姿態查詢和選擇
專為水果抓取等實際應用場景設計
"""

import numpy as np
import json
import pickle
import time
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import threading
import warnings

warnings.filterwarnings('ignore')


@dataclass
class GraspPose:
    """抓取姿態數據結構"""
    position: np.ndarray
    quaternion: np.ndarray  # [w, x, y, z]
    joint_config: np.ndarray
    quality_score: float
    grasp_type: str  # 'top', 'side', 'angle'
    success_probability: float


class PoseLookupTable:
    """
    姿態檢索表格類
    用於快速查詢給定位置的可達姿態
    """

    def __init__(self, workspace_data_file=None):
        """
        初始化姿態檢索表格

        Args:
            workspace_data_file: 工作空間數據文件路徑
        """
        self.position_tree = None
        self.pose_database = {}  # {(x,y,z): [GraspPose, ...]}
        self.position_tolerance = 0.01  # 位置容差 1cm
        self.grasp_preferences = {
            'top_down': 1.0,  # 垂直向下抓取權重
            'angled': 0.8,  # 斜向抓取權重
            'side': 0.6  # 側向抓取權重
        }

        if workspace_data_file:
            self.load_workspace_data(workspace_data_file)

    def load_workspace_data(self, data_file):
        """加載工作空間數據並構建檢索表"""
        print(f"📂 加載工作空間數據: {data_file}")

        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        self.raw_reachable_points = np.array(data['reachable_points'])
        self.raw_position_orientation_map = data['position_orientation_map']

        print(f"✅ 原始數據加載完成: {len(self.raw_reachable_points)} 個位置")

        # 構建優化的檢索結構
        self.build_lookup_table()

    def build_lookup_table(self):
        """構建優化的姿態檢索表"""
        print("🔧 構建姿態檢索表...")

        # 轉換數據格式並分類姿態
        positions = []
        pose_database = {}

        for i, pos in enumerate(self.raw_reachable_points):
            pos_key = tuple(pos)
            if pos_key not in self.raw_position_orientation_map:
                continue

            orientations = self.raw_position_orientation_map[pos_key]
            grasp_poses = []

            for orientation in orientations:
                # 分析姿態類型
                grasp_type = self.classify_grasp_orientation(orientation)

                # 計算質量分數
                quality_score = self.calculate_pose_quality(pos, orientation, grasp_type)

                # 創建GraspPose對象
                grasp_pose = GraspPose(
                    position=np.array(pos),
                    quaternion=np.array(orientation),
                    joint_config=np.array([0.0] * 7),  # 如果有關節配置數據可以填入
                    quality_score=quality_score,
                    grasp_type=grasp_type,
                    success_probability=quality_score
                )
                grasp_poses.append(grasp_pose)

            # 按質量分數排序
            grasp_poses.sort(key=lambda x: x.quality_score, reverse=True)

            positions.append(pos)
            pose_database[tuple(pos)] = grasp_poses

        # 構建KD樹用於快速空間查詢
        self.positions = np.array(positions)
        self.position_tree = cKDTree(self.positions)
        self.pose_database = pose_database

        print(f"✅ 檢索表構建完成: {len(positions)} 個位置，"
              f"{sum(len(poses) for poses in pose_database.values())} 個姿態")

    def classify_grasp_orientation(self, quaternion):
        """
        分類抓取姿態類型

        Args:
            quaternion: 四元數 [w, x, y, z]

        Returns:
            str: 抓取類型 ('top', 'side', 'angle')
        """
        # 轉換四元數為旋轉矩陣
        quat_xyzw = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        rotation = R.from_quat(quat_xyzw)

        # 獲取Z軸方向（末端執行器指向）
        z_direction = rotation.as_matrix()[:, 2]

        # 分析與垂直方向的角度
        vertical_angle = np.arccos(np.abs(z_direction[2]))

        if vertical_angle < np.pi / 6:  # 30度內
            return 'top'
        elif vertical_angle > 2 * np.pi / 3:  # 120度外
            return 'side'
        else:
            return 'angle'

    def calculate_pose_quality(self, position, quaternion, grasp_type):
        """
        計算姿態質量分數

        Args:
            position: 位置
            quaternion: 四元數
            grasp_type: 抓取類型

        Returns:
            float: 質量分數 (0-1)
        """
        score = 0.0

        # 基於抓取類型的基礎分數
        base_score = self.grasp_preferences.get(grasp_type, 0.5)
        score += base_score * 0.4

        # 基於高度的分數（中等高度更好）
        height = position[2]
        if 0.2 <= height <= 0.8:
            height_score = 1.0 - abs(height - 0.5) / 0.3
        else:
            height_score = 0.3
        score += height_score * 0.2

        # 基於距離機器人基座的分數（不要太遠）
        distance = np.linalg.norm(position[:2])  # XY平面距離
        if distance <= 0.8:
            distance_score = 1.0 - distance / 0.8
        else:
            distance_score = 0.1
        score += distance_score * 0.2

        # 基於姿態穩定性的分數
        quat_xyzw = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        euler = R.from_quat(quat_xyzw).as_euler('xyz')

        # 偏好較小的roll和pitch角度
        stability_score = 1.0 - (abs(euler[0]) + abs(euler[1])) / (2 * np.pi)
        score += max(0, stability_score) * 0.2

        return min(1.0, max(0.0, score))

    def query_poses(self, target_position, num_candidates=5,
                    preferred_grasp_type=None, min_quality=0.3):
        """
        查詢指定位置的最佳抓取姿態

        Args:
            target_position: 目標位置 [x, y, z]
            num_candidates: 返回的候選數量
            preferred_grasp_type: 偏好的抓取類型
            min_quality: 最小質量分數閾值

        Returns:
            List[GraspPose]: 排序後的抓取姿態列表
        """
        target_pos = np.array(target_position)

        if self.position_tree is None:
            return []

        # 找到最近的位置點
        distances, indices = self.position_tree.query(
            target_pos, k=min(10, len(self.positions))
        )

        candidate_poses = []

        for distance, idx in zip(distances, indices):
            if distance > self.position_tolerance:
                continue

            pos = self.positions[idx]
            poses = self.pose_database.get(tuple(pos), [])

            for pose in poses:
                if pose.quality_score < min_quality:
                    continue

                # 如果有偏好類型，給予額外權重
                adjusted_score = pose.quality_score
                if preferred_grasp_type and pose.grasp_type == preferred_grasp_type:
                    adjusted_score *= 1.2

                # 考慮位置距離
                position_penalty = distance / self.position_tolerance
                final_score = adjusted_score * (1.0 - position_penalty * 0.1)

                # 創建調整後的姿態副本
                adjusted_pose = GraspPose(
                    position=target_pos,  # 使用目標位置
                    quaternion=pose.quaternion.copy(),
                    joint_config=pose.joint_config.copy(),
                    quality_score=final_score,
                    grasp_type=pose.grasp_type,
                    success_probability=final_score
                )

                candidate_poses.append(adjusted_pose)

        # 按調整後的分數排序
        candidate_poses.sort(key=lambda x: x.quality_score, reverse=True)

        return candidate_poses[:num_candidates]

    def get_best_pose(self, target_position, strategy='quality'):
        """
        獲取單個最佳姿態

        Args:
            target_position: 目標位置
            strategy: 選擇策略 ('quality', 'top_down', 'conservative')

        Returns:
            GraspPose: 最佳姿態，如果沒有找到則返回None
        """
        if strategy == 'top_down':
            poses = self.query_poses(target_position, num_candidates=10,
                                     preferred_grasp_type='top')
        elif strategy == 'conservative':
            poses = self.query_poses(target_position, num_candidates=5,
                                     min_quality=0.6)
        else:
            poses = self.query_poses(target_position, num_candidates=1)

        return poses[0] if poses else None

    def batch_query(self, target_positions, strategy='quality'):
        """
        批量查詢多個位置的最佳姿態

        Args:
            target_positions: 目標位置列表
            strategy: 選擇策略

        Returns:
            List[Optional[GraspPose]]: 每個位置的最佳姿態
        """
        results = []

        print(f"🔍 批量查詢 {len(target_positions)} 個位置...")
        start_time = time.time()

        for i, pos in enumerate(target_positions):
            best_pose = self.get_best_pose(pos, strategy)
            results.append(best_pose)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                print(f"   進度: {i + 1}/{len(target_positions)}, "
                      f"平均查詢時間: {avg_time * 1000:.2f}ms")

        total_time = time.time() - start_time
        success_rate = sum(1 for r in results if r is not None) / len(results)

        print(f"✅ 批量查詢完成: {total_time:.2f}s, 成功率: {success_rate * 100:.1f}%")

        return results

    def export_curobo_format(self, poses, output_file):
        """
        將姿態數據導出為CuRobo可用格式

        Args:
            poses: GraspPose列表
            output_file: 輸出文件路徑
        """
        curobo_poses = []

        for pose in poses:
            if pose is not None:
                curobo_pose = {
                    'position': pose.position.tolist(),
                    'quaternion': pose.quaternion.tolist(),  # [w,x,y,z]
                    'quality_score': pose.quality_score,
                    'grasp_type': pose.grasp_type
                }
                curobo_poses.append(curobo_pose)

        with open(output_file, 'w') as f:
            json.dump(curobo_poses, f, indent=2)

        print(f"💾 已導出 {len(curobo_poses)} 個姿態到: {output_file}")

    def visualize_query_result(self, target_position, candidate_poses):
        """可視化查詢結果"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 5))

        # 3D視圖
        ax1 = fig.add_subplot(121, projection='3d')

        # 繪製所有可達位置（淺色）
        if hasattr(self, 'positions'):
            ax1.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2],
                        c='lightgray', alpha=0.1, s=1)

        # 繪製目標位置
        ax1.scatter(*target_position, c='red', s=100, label='Target')

        # 繪製候選姿態位置
        if candidate_poses:
            candidate_positions = np.array([pose.position for pose in candidate_poses])
            colors = [pose.quality_score for pose in candidate_poses]

            scatter = ax1.scatter(candidate_positions[:, 0], candidate_positions[:, 1],
                                  candidate_positions[:, 2], c=colors, cmap='viridis',
                                  s=50, alpha=0.8)
            plt.colorbar(scatter, ax=ax1, label='Quality Score')

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Pose Query Result')
        ax1.legend()

        # 質量分數分佈
        ax2 = fig.add_subplot(122)
        if candidate_poses:
            qualities = [pose.quality_score for pose in candidate_poses]
            grasp_types = [pose.grasp_type for pose in candidate_poses]

            type_colors = {'top': 'green', 'angle': 'orange', 'side': 'blue'}
            colors = [type_colors.get(gt, 'gray') for gt in grasp_types]

            bars = ax2.bar(range(len(qualities)), qualities, color=colors)
            ax2.set_xlabel('Candidate Index')
            ax2.set_ylabel('Quality Score')
            ax2.set_title('Candidate Quality Scores')

            # 添加圖例
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color, label=gtype)
                               for gtype, color in type_colors.items()]
            ax2.legend(handles=legend_elements)

        plt.tight_layout()
        plt.show()


class RealTimeGraspPlanner:
    """
    實時抓取規劃器
    整合姿態檢索表格和CuRobo運動規劃
    """

    def __init__(self, pose_lookup_table, motion_gen):
        """
        初始化實時抓取規劃器

        Args:
            pose_lookup_table: 姿態檢索表格
            motion_gen: CuRobo運動生成器
        """
        self.lookup_table = pose_lookup_table
        self.motion_gen = motion_gen
        self.planning_cache = {}  # 規劃緩存
        self.fallback_strategies = ['quality', 'top_down', 'conservative']

    def plan_grasp(self, fruit_position, current_joint_state,
                   max_attempts=3, timeout=2.0):
        """
        規劃抓取動作

        Args:
            fruit_position: 水果位置 [x, y, z]
            current_joint_state: 當前關節狀態
            max_attempts: 最大嘗試次數
            timeout: 規劃超時時間

        Returns:
            dict: 規劃結果
        """
        result = {
            'success': False,
            'trajectory': None,
            'selected_pose': None,
            'planning_time': 0.0,
            'attempts': 0
        }

        start_time = time.time()

        # 嘗試不同策略
        for strategy in self.fallback_strategies:
            if time.time() - start_time > timeout:
                break

            result['attempts'] += 1

            # 查詢候選姿態
            candidate_poses = self.lookup_table.query_poses(
                fruit_position, num_candidates=5
            )

            if not candidate_poses:
                continue

            # 嘗試每個候選姿態
            for pose in candidate_poses:
                try:
                    # 創建CuRobo目標姿態
                    from curobo.types.math import Pose
                    target_pose = Pose(
                        position=self.motion_gen.tensor_args.to_device(pose.position),
                        quaternion=self.motion_gen.tensor_args.to_device(pose.quaternion)
                    )

                    # 執行運動規劃
                    plan_result = self.motion_gen.plan_single(
                        current_joint_state.unsqueeze(0),
                        target_pose,
                        self.motion_gen.get_plan_config(max_attempts=max_attempts)
                    )

                    if plan_result.success.item():
                        result.update({
                            'success': True,
                            'trajectory': plan_result.get_interpolated_plan(),
                            'selected_pose': pose,
                            'planning_time': time.time() - start_time
                        })

                        print(f"✅ 抓取規劃成功！策略: {strategy}, "
                              f"姿態類型: {pose.grasp_type}, "
                              f"質量分數: {pose.quality_score:.3f}")
                        return result

                except Exception as e:
                    print(f"⚠️ 姿態規劃失敗: {e}")
                    continue

        result['planning_time'] = time.time() - start_time
        print(f"❌ 所有抓取策略均失敗，用時: {result['planning_time']:.2f}s")

        return result


# 使用示例和測試代碼
def example_usage():
    """展示如何使用姿態檢索表格系統"""
    print("🤖 姿態檢索表格系統示例")
    print("=" * 50)

    # 1. 初始化檢索表格
    lookup_table = PoseLookupTable('workspace_data/workspace_fanuc_20250826_091843.pkl')

    # 2. 查詢特定位置的姿態
    target_position = [0.4, 0.00, 0.8]  # 您的示例位置

    print(f"\n🎯 查詢位置: {target_position}")

    # 獲取候選姿態
    candidate_poses = lookup_table.query_poses(target_position, num_candidates=5)

    print(f"找到 {len(candidate_poses)} 個候選姿態:")
    for i, pose in enumerate(candidate_poses):
        print(f"  {i + 1}. 四元數: {pose.quaternion}")
        print(f"      類型: {pose.grasp_type}, 分數: {pose.quality_score:.3f}")

    # 3. 獲取最佳姿態
    best_pose = lookup_table.get_best_pose(target_position, strategy='top_down')
    if best_pose:
        print(f"\n🏆 最佳姿態:")
        print(f"   四元數: {best_pose.quaternion}")
        print(f"   類型: {best_pose.grasp_type}")
        print(f"   分數: {best_pose.quality_score:.3f}")

    # 4. 批量查詢多個水果位置
    fruit_positions = [
        [0.7, 0.1, 0.112],
        [0.4, 0.00, 0.7],
        [0.39, 0.00, 0.8]
    ]

    batch_results = lookup_table.batch_query(fruit_positions, strategy='quality')

    print(f"\n📦 批量查詢結果:")
    for i, result in enumerate(batch_results):
        if result:
            print(f"  位置 {i + 1}: 成功，分數 {result.quality_score:.3f}")
        else:
            print(f"  位置 {i + 1}: 無可達姿態")

    # 5. 導出為CuRobo格式
    valid_poses = [p for p in batch_results if p is not None]
    lookup_table.export_curobo_format(valid_poses, 'grasp_poses_for_curobo.json')

    return lookup_table, candidate_poses


if __name__ == "__main__":
    # 運行示例
    try:
        lookup_table, poses = example_usage()

        # 可視化查詢結果
        if poses:
            lookup_table.visualize_query_result([0.22, 0.75, 0.112], poses)

    except FileNotFoundError:
        print("❌ 請先運行工作空間探索工具生成數據文件")
        print("💡 示例數據文件路徑: workspace_results/workspace_data_YYYYMMDD_HHMMSS.pkl")
