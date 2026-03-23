#!/usr/bin/env python3
"""
機械臂位姿實時驗證工具
快速驗證指定位置和姿態的可達性
"""

import numpy as np
import json
import argparse
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import warnings

warnings.filterwarnings('ignore')

# 嘗試導入機器人學庫
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


class PoseValidator:
    def __init__(self, urdf_path, joint_limits_dict=None):
        """
        初始化位姿驗證器

        Args:
            urdf_path: URDF檔案路徑
            joint_limits_dict: 關節限制字典 {joint_name: [lower, upper]}
        """
        self.urdf_path = urdf_path
        self.joint_limits = joint_limits_dict or {}

        print(f"🔧 初始化位姿驗證器: {urdf_path}")
        self.init_robot_model()

    def init_robot_model(self):
        """初始化機器人模型"""
        self.robot_model = None
        self.solver_type = None

        # 優先使用Pinocchio
        if HAS_PINOCCHIO:
            try:
                self.robot_model = pin.buildModelFromUrdf(self.urdf_path)
                self.robot_data = self.robot_model.createData()
                self.solver_type = "pinocchio"

                # 提取關節限制
                if not self.joint_limits:
                    for i in range(self.robot_model.nq):
                        joint_name = self.robot_model.names[i + 1]  # 跳過universe
                        lower = self.robot_model.lowerPositionLimit[i]
                        upper = self.robot_model.upperPositionLimit[i]
                        self.joint_limits[joint_name] = [float(lower), float(upper)]

                print(f"✅ 使用 Pinocchio，關節數: {self.robot_model.nq}")
                return

            except Exception as e:
                print(f"⚠️ Pinocchio 初始化失敗: {e}")

        # 備選：使用PyBullet
        if HAS_PYBULLET:
            try:
                p.connect(p.DIRECT)
                self.robot_id = p.loadURDF(self.urdf_path, useFixedBase=True)
                self.solver_type = "pybullet"

                # 獲取關節信息
                self.joint_indices = []
                num_joints = p.getNumJoints(self.robot_id)

                for i in range(num_joints):
                    joint_info = p.getJointInfo(self.robot_id, i)
                    joint_name = joint_info[1].decode('utf-8')
                    joint_type = joint_info[2]

                    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                        self.joint_indices.append(i)
                        if joint_name not in self.joint_limits:
                            lower_limit = joint_info[8]
                            upper_limit = joint_info[9]
                            self.joint_limits[joint_name] = [float(lower_limit), float(upper_limit)]

                print(f"✅ 使用 PyBullet，可動關節數: {len(self.joint_indices)}")
                return

            except Exception as e:
                print(f"⚠️ PyBullet 初始化失敗: {e}")

        raise RuntimeError("無法初始化任何機器人學庫，請安裝 pinocchio 或 pybullet")

    def forward_kinematics(self, joint_values):
        """
        正向運動學計算

        Args:
            joint_values: 關節角度數組

        Returns:
            (position, orientation_matrix): 位置和旋轉矩陣
        """
        try:
            if self.solver_type == "pinocchio":
                q = np.array(joint_values)
                pin.forwardKinematics(self.robot_model, self.robot_data, q)
                pin.updateFramePlacements(self.robot_model, self.robot_data)

                # 獲取末端執行器位姿
                ee_pose = self.robot_data.oMi[-1]  # 最後一個連桿
                return ee_pose.translation, ee_pose.rotation

            elif self.solver_type == "pybullet":
                # 設置關節狀態
                for i, (joint_idx, value) in enumerate(zip(self.joint_indices, joint_values)):
                    p.resetJointState(self.robot_id, joint_idx, value)

                # 獲取末端執行器狀態
                link_state = p.getLinkState(self.robot_id, self.joint_indices[-1])
                position = np.array(link_state[0])
                orientation_quat = np.array(link_state[1])  # [x,y,z,w]

                rotation_matrix = R.from_quat(orientation_quat).as_matrix()
                return position, rotation_matrix

        except Exception as e:
            print(f"FK計算錯誤: {e}")
            return None, None

    def inverse_kinematics_numerical(self, target_position, target_orientation=None,
                                     initial_guess=None, position_weight=1.0, orientation_weight=0.1):
        """
        數值逆運動學求解

        Args:
            target_position: 目標位置 [x, y, z]
            target_orientation: 目標方向矩陣 (可選)
            initial_guess: 初始猜測關節角度
            position_weight: 位置權重
            orientation_weight: 姿態權重

        Returns:
            (success, joint_solution, final_error)
        """
        target_pos = np.array(target_position)
        num_joints = len(self.joint_limits)

        # 設置初始猜測
        if initial_guess is None:
            # 使用關節中點作為初始猜測
            initial_guess = []
            for joint_name in self.joint_limits.keys():
                lower, upper = self.joint_limits[joint_name]
                mid_point = (lower + upper) / 2.0
                initial_guess.append(mid_point)

        initial_guess = np.array(initial_guess[:num_joints])

        # 設置關節邊界
        bounds = []
        for joint_name in self.joint_limits.keys():
            bounds.append(self.joint_limits[joint_name])

        def objective_function(joint_values):
            """目標函數：最小化位姿誤差"""
            pos, rot = self.forward_kinematics(joint_values)

            if pos is None:
                return 1e6  # 計算失敗的懲罰

            # 位置誤差
            pos_error = np.linalg.norm(pos - target_pos)
            total_error = position_weight * pos_error

            # 姿態誤差（如果提供了目標姿態）
            if target_orientation is not None and rot is not None:
                # 使用Frobenius範數計算旋轉矩陣差異
                rot_error = np.linalg.norm(rot - target_orientation, 'fro')
                total_error += orientation_weight * rot_error

            return total_error

        # 嘗試多種優化方法
        best_solution = None
        best_error = float('inf')

        # 方法1: scipy.optimize.minimize (L-BFGS-B)
        try:
            result = minimize(
                objective_function,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )

            if result.success and result.fun < best_error:
                best_solution = result.x
                best_error = result.fun

        except Exception as e:
            print(f"L-BFGS-B 優化失敗: {e}")

        # 方法2: 差分進化算法 (全局優化)
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                seed=42,
                maxiter=300,
                popsize=15,
                atol=1e-8
            )

            if result.success and result.fun < best_error:
                best_solution = result.x
                best_error = result.fun

        except Exception as e:
            print(f"差分進化優化失敗: {e}")

        # 判斷是否成功
        success = (best_solution is not None and
                   best_error < 0.01)  # 1cm的位置誤差容差

        return success, best_solution, best_error

    def validate_pose(self, position, orientation_quat=None, orientation_euler=None,
                      tolerance_pos=0.01, tolerance_rot=0.1):
        """
        驗證指定位姿的可達性

        Args:
            position: 目標位置 [x, y, z]
            orientation_quat: 目標四元數 [w, x, y, z] (可選)
            orientation_euler: 目標歐拉角 [roll, pitch, yaw] (可選)
            tolerance_pos: 位置容差 (m)
            tolerance_rot: 姿態容差 (rad)

        Returns:
            dict: 驗證結果
        """
        result = {
            'target_position': position,
            'target_orientation': None,
            'is_reachable': False,
            'joint_solution': None,
            'achieved_position': None,
            'achieved_orientation': None,
            'position_error': None,
            'orientation_error': None,
            'solve_time': None
        }

        # 處理姿態輸入
        target_rotation_matrix = None
        if orientation_quat is not None:
            # 四元數格式轉換
            if len(orientation_quat) == 4:
                if abs(orientation_quat[0]) > abs(orientation_quat[3]):
                    # 假設輸入是 [w,x,y,z]
                    quat_xyzw = [orientation_quat[1], orientation_quat[2], orientation_quat[3], orientation_quat[0]]
                else:
                    # 假設輸入是 [x,y,z,w]
                    quat_xyzw = orientation_quat

                target_rotation_matrix = R.from_quat(quat_xyzw).as_matrix()
                result['target_orientation'] = orientation_quat

        elif orientation_euler is not None:
            target_rotation_matrix = R.from_euler('xyz', orientation_euler).as_matrix()
            result['target_orientation'] = orientation_euler

        # 執行逆運動學求解
        start_time = time.time()

        success, joint_solution, final_error = self.inverse_kinematics_numerical(
            position, target_rotation_matrix
        )

        solve_time = time.time() - start_time
        result['solve_time'] = solve_time

        if success and joint_solution is not None:
            # 驗證解的準確性
            achieved_pos, achieved_rot = self.forward_kinematics(joint_solution)

            if achieved_pos is not None:
                position_error = np.linalg.norm(achieved_pos - np.array(position))

                result.update({
                    'is_reachable': position_error <= tolerance_pos,
                    'joint_solution': joint_solution.tolist(),
                    'achieved_position': achieved_pos.tolist(),
                    'position_error': float(position_error)
                })

                if achieved_rot is not None and target_rotation_matrix is not None:
                    orientation_error = np.linalg.norm(achieved_rot - target_rotation_matrix, 'fro')
                    result.update({
                        'achieved_orientation': R.from_matrix(achieved_rot).as_euler('xyz').tolist(),
                        'orientation_error': float(orientation_error)
                    })

                    # 更新可達性判斷（考慮姿態誤差）
                    if orientation_error <= tolerance_rot:
                        result['is_reachable'] = result['is_reachable'] and True
                    else:
                        result['is_reachable'] = False

        return result

    def batch_validate(self, poses_list, show_progress=True):
        """
        批量驗證多個位姿

        Args:
            poses_list: 位姿列表，每個元素是 {'position': [x,y,z], 'orientation': ...}
            show_progress: 是否顯示進度

        Returns:
            list: 驗證結果列表
        """
        print(f"🔍 開始批量驗證 {len(poses_list)} 個位姿...")

        results = []
        start_time = time.time()

        for i, pose in enumerate(poses_list):
            if show_progress and (i % 10 == 0 or i == len(poses_list) - 1):
                progress = (i + 1) / len(poses_list) * 100
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(poses_list) - i - 1)
                print(f"   進度: {progress:.1f}% ({i + 1}/{len(poses_list)}), ETA: {eta:.1f}s")

            position = pose['position']
            orientation_quat = pose.get('orientation_quat', None)
            orientation_euler = pose.get('orientation_euler', None)

            result = self.validate_pose(position, orientation_quat, orientation_euler)
            result['pose_index'] = i
            results.append(result)

        total_time = time.time() - start_time
        reachable_count = sum(1 for r in results if r['is_reachable'])

        print(f"✅ 批量驗證完成！用時: {total_time:.1f}s")
        print(f"   可達位姿: {reachable_count}/{len(poses_list)} ({reachable_count / len(poses_list) * 100:.1f}%)")

        return results

    def suggest_reachable_orientations(self, position, num_samples=50):
        """
        為指定位置建議可達的姿態

        Args:
            position: 目標位置 [x, y, z]
            num_samples: 姿態樣本數量

        Returns:
            list: 可達的姿態列表
        """
        print(f"🎯 為位置 {position} 尋找可達姿態...")

        # 生成隨機姿態樣本
        orientations = []
        for _ in range(num_samples):
            # 生成隨機歐拉角
            roll = np.random.uniform(-np.pi, np.pi)
            pitch = np.random.uniform(-np.pi / 2, np.pi / 2)
            yaw = np.random.uniform(-np.pi, np.pi)
            orientations.append([roll, pitch, yaw])

        # 批量測試
        poses_list = []
        for euler in orientations:
            poses_list.append({
                'position': position,
                'orientation_euler': euler
            })

        results = self.batch_validate(poses_list, show_progress=False)

        # 提取可達的姿態
        reachable_orientations = []
        for i, result in enumerate(results):
            if result['is_reachable']:
                reachable_orientations.append({
                    'euler_angles': orientations[i],
                    'quaternion': R.from_euler('xyz', orientations[i]).as_quat(),
                    'joint_solution': result['joint_solution'],
                    'position_error': result['position_error']
                })

        print(f"✅ 找到 {len(reachable_orientations)}/{num_samples} 個可達姿態")

        return reachable_orientations

    def visualize_validation_results(self, results, save_path=None):
        """可視化驗證結果"""
        if not results:
            print("⚠️ 沒有驗證結果可視化")
            return

        print("🎨 生成驗證結果可視化...")

        # 分離可達和不可達的點
        reachable_positions = []
        unreachable_positions = []
        position_errors = []

        for result in results:
            pos = result['target_position']
            if result['is_reachable']:
                reachable_positions.append(pos)
                position_errors.append(result['position_error'])
            else:
                unreachable_positions.append(pos)

        fig = plt.figure(figsize=(15, 5))

        # 3D散點圖
        ax1 = fig.add_subplot(131, projection='3d')

        if reachable_positions:
            reach_array = np.array(reachable_positions)
            ax1.scatter(reach_array[:, 0], reach_array[:, 1], reach_array[:, 2],
                        c='green', s=50, alpha=0.7, label='可達')

        if unreachable_positions:
            unreach_array = np.array(unreachable_positions)
            ax1.scatter(unreach_array[:, 0], unreach_array[:, 1], unreach_array[:, 2],
                        c='red', s=50, alpha=0.7, label='不可達')

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('位姿驗證結果')
        ax1.legend()

        # 誤差分佈
        ax2 = fig.add_subplot(132)
        if position_errors:
            ax2.hist(position_errors, bins=20, alpha=0.7, color='blue')
            ax2.set_xlabel('位置誤差 (m)')
            ax2.set_ylabel('數量')
            ax2.set_title('位置誤差分佈')

        # 統計圓餅圖
        ax3 = fig.add_subplot(133)
        reachable_count = len(reachable_positions)
        unreachable_count = len(unreachable_positions)

        if reachable_count > 0 or unreachable_count > 0:
            ax3.pie([reachable_count, unreachable_count],
                    labels=['可達', '不可達'],
                    colors=['green', 'red'],
                    autopct='%1.1f%%')
            ax3.set_title('可達性統計')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 可視化結果已保存到: {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="機械臂位姿實時驗證工具")
    parser.add_argument("--urdf", type=str,default="examples/models/urdf/urdf3/dual_piepr_humanity.urdf", help="URDF檔案路徑")
    parser.add_argument("--position", nargs=3, type=float, help="測試位置 x y z")
    parser.add_argument("--orientation_euler", nargs=3, type=float,
                        help="測試姿態歐拉角 roll pitch yaw (弧度)")
    parser.add_argument("--orientation_quat", nargs=4, type=float,
                        help="測試姿態四元數 w x y z")
    parser.add_argument("--batch_file", type=str,default="robot_analysis_results/reachable_positions_20250825_162317.json",
                        help="批量測試JSON檔案")
    parser.add_argument("--suggest_orientations", action="store_true",
                        help="為指定位置建議可達姿態")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="姿態樣本數量")
    parser.add_argument("--output_dir", type=str, default="pose_validation_results",
                        help="結果輸出目錄")

    args = parser.parse_args()

    try:
        # 初始化驗證器
        validator = PoseValidator(args.urdf)

        # 創建輸出目錄
        import os
        os.makedirs(args.output_dir, exist_ok=True)

        # 單個位姿驗證
        if args.position:
            print(f"\n🎯 驗證位置: {args.position}")

            result = validator.validate_pose(
                args.position,
                args.orientation_quat,
                args.orientation_euler
            )

            print(f"\n驗證結果:")
            print(f"  可達性: {'✅ 可達' if result['is_reachable'] else '❌ 不可達'}")
            print(f"  求解時間: {result['solve_time']:.3f}s")
            if result['position_error'] is not None:
                print(f"  位置誤差: {result['position_error']:.6f}m")
            if result['joint_solution'] is not None:
                print(f"  關節解: {[f'{x:.4f}' for x in result['joint_solution']]}")

            # 建議可達姿態
            if args.suggest_orientations:
                orientations = validator.suggest_reachable_orientations(
                    args.position, args.num_samples
                )

                print(f"\n🎯 建議的可達姿態:")
                for i, orient in enumerate(orientations[:5]):  # 顯示前5個
                    euler_deg = np.degrees(orient['euler_angles'])
                    print(f"  {i + 1}. 歐拉角(度): [{euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°]")
                    print(
                        f"      四元數: [{orient['quaternion'][3]:.3f}, {orient['quaternion'][0]:.3f}, {orient['quaternion'][1]:.3f}, {orient['quaternion'][2]:.3f}]")

        # 批量驗證
        if args.batch_file:
            print(f"\n📁 讀取批量測試檔案: {args.batch_file}")

            with open(args.batch_file, 'r') as f:
                poses_data = json.load(f)

            results = validator.batch_validate(poses_data)

            # 保存結果
            results_file = os.path.join(args.output_dir, "batch_validation_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # 生成可視化
            viz_file = os.path.join(args.output_dir, "validation_visualization.png")
            validator.visualize_validation_results(results, viz_file)

            print(f"💾 結果已保存到: {results_file}")

    except Exception as e:
        import traceback
        print(f"❌ 驗證過程中出錯: {e}")
        traceback.print_exc()


# 示例用法函數
def create_example_batch_file(filename="example_poses.json"):
    """創建示例批量測試檔案"""
    example_poses = [
        {
            "position": [0.3, 0.0, 0.4],
            "orientation_euler": [0, -1.57, 0]
        },
        {
            "position": [0.5, 0.2, 0.3],
            "orientation_quat": [1, 0, 0, 0]
        },
        {
            "position": [0.2, -0.3, 0.5],
            "orientation_euler": [0.5, -1.0, 0.5]
        },
        {
            "position": [0.6, 0.1, 0.2],
            "orientation_euler": [0, -0.785, 1.57]
        }
    ]

    with open(filename, 'w') as f:
        json.dump(example_poses, f, indent=2)

    print(f"📝 示例測試檔案已創建: {filename}")


if __name__ == "__main__":
    print("🔍 機械臂位姿實時驗證工具")
    print("支持的機器人學庫:")
    print(f"   Pinocchio: {'✅' if HAS_PINOCCHIO else '❌'}")
    print(f"   PyBullet: {'✅' if HAS_PYBULLET else '❌'}")
    print()

    print("示例用法:")
    print("# 單個位姿驗證")
    print("python pose_validator.py --urdf robot.urdf --position 0.3 0.0 0.4 --orientation_euler 0 -1.57 0")
    print()
    print("# 建議可達姿態")
    print("python pose_validator.py --urdf robot.urdf --position 0.3 0.0 0.4 --suggest_orientations")
    print()
    print("# 批量驗證")
    print("python pose_validator.py --urdf robot.urdf --batch_file example_poses.json")
    print()

    # 如果沒有參數，創建示例檔案
    import sys

    if len(sys.argv) == 1:
        create_example_batch_file()
        print("💡 使用 --help 查看詳細參數說明")
    else:
        main()
