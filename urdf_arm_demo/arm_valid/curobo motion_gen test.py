#!/usr/bin/env python3
"""
基於CuRobo官方IK的工作空間檢索表構建器
修改官方範例，對每個位置測試多個姿態並保存為檢索表
"""

try:
    import isaacsim
except ImportError:
    pass

import torch
import time
import pickle
import json
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# 檢查CUDA
a = torch.zeros(4, device="cuda:0")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless_mode", type=str, default=None)
parser.add_argument("--visualize_spheres", action="store_true", default=False)
parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration")
parser.add_argument("--grid_x", type=int, default=15, help="X方向網格數")
parser.add_argument("--grid_y", type=int, default=15, help="Y方向網格數") 
parser.add_argument("--grid_z", type=int, default=8, help="Z方向網格數")
parser.add_argument("--max_x", type=float, default=0.8, help="X方向最大範圍")
parser.add_argument("--max_y", type=float, default=0.8, help="Y方向最大範圍")
parser.add_argument("--max_z", type=float, default=0.8, help="Z方向最大範圍")
parser.add_argument("--orientation_samples", type=int, default=12, help="每個位置測試的姿態數量")
parser.add_argument("--output_file", type=str, default="workspace_lookup_table.pkl", help="輸出檢索表文件")
args = parser.parse_args()

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless_mode is not None,
    "width": "1920", 
    "height": "1080",
})

import carb
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

# CuRobo imports
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (get_robot_configs_path, get_world_configs_path, 
                             join_path, load_yaml)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

def get_pose_grid(n_x, n_y, n_z, max_x, max_y, max_z, min_z=0.1):
    """生成測試位置網格"""
    x = np.linspace(-max_x, max_x, n_x)
    y = np.linspace(-max_y, max_y, n_y)
    z = np.linspace(min_z, max_z, n_z)
    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    position_arr = np.zeros((n_x * n_y * n_z, 3))
    position_arr[:, 0] = x.flatten()
    position_arr[:, 1] = y.flatten()
    position_arr[:, 2] = z.flatten()
    return position_arr

def generate_test_orientations(num_samples=12):
    """生成多樣化的測試姿態"""
    orientations = []
    
    # 1. 基本姿態（常用抓取姿態）
    basic_orientations_euler = [
        [0, -np.pi/2, 0],      # 垂直向下
        [0, -np.pi/4, 0],      # 45度向下
        [0, -3*np.pi/4, 0],    # 135度向下
        [0, 0, 0],             # 水平
        [np.pi/2, -np.pi/2, 0], # 側向抓取
        [-np.pi/2, -np.pi/2, 0], # 另一側向抓取
    ]
    
    for euler in basic_orientations_euler:
        quat_xyzw = R.from_euler('xyz', euler).as_quat()
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        orientations.append(quat_wxyz)
    
    # 2. 圍繞Z軸的不同方向
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        euler = [0, -np.pi/2, angle]  # 保持向下，旋轉方向
        quat_xyzw = R.from_euler('xyz', euler).as_quat()
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        orientations.append(quat_wxyz)
    
    # 3. 如果需要更多樣本，添加隨機姿態
    while len(orientations) < num_samples:
        # 生成隨機歐拉角，但限制在合理範圍內
        roll = np.random.uniform(-np.pi/2, np.pi/2)
        pitch = np.random.uniform(-np.pi, -np.pi/6)  # 主要向下的姿態
        yaw = np.random.uniform(-np.pi, np.pi)
        
        quat_xyzw = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        orientations.append(quat_wxyz)
    
    return orientations[:num_samples]

def draw_reachability_points(positions, orientations_per_pos, tensor_args):
    """繪製可達性點，顏色表示可達姿態數量"""
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()
    
    point_list = []
    colors = []
    sizes = []
    
    # 計算顏色映射
    max_orientations = max(orientations_per_pos) if orientations_per_pos else 1
    
    for i, (pos, num_orientations) in enumerate(zip(positions, orientations_per_pos)):
        point_list.append(tuple(pos))
        
        if num_orientations == 0:
            # 不可達 - 紅色
            colors.append((1.0, 0.0, 0.0, 0.3))
            sizes.append(20.0)
        else:
            # 可達 - 綠色到藍色漸變（根據姿態數量）
            intensity = num_orientations / max_orientations
            colors.append((0.0, intensity, 1.0 - intensity, 0.7))
            sizes.append(30.0 + 20.0 * intensity)  # 大小也反映姿態豐富度
    
    draw.draw_points(point_list, colors, sizes)

class WorkspaceAnalyzer:
    """工作空間分析器"""
    
    def __init__(self, ik_solver, tensor_args):
        self.ik_solver = ik_solver
        self.tensor_args = tensor_args
        self.workspace_data = {
            'position_orientation_map': {},
            'reachable_positions': [],
            'analysis_params': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_workspace(self, positions, test_orientations, batch_size=500):
        """分析整個工作空間"""
        print("開始工作空間分析...")
        print(f"總位置數: {len(positions)}")
        print(f"每位置測試姿態數: {len(test_orientations)}")
        print(f"總測試次數: {len(positions) * len(test_orientations)}")
        
        total_positions = len(positions)
        start_time = time.time()
        orientations_per_position = []
        
        # 轉換姿態為張量
        test_orientations_tensor = torch.tensor(test_orientations, device=self.tensor_args.device)
        
        # 分批處理位置以避免GPU記憶體不足
        for batch_start in range(0, total_positions, batch_size):
            batch_end = min(batch_start + batch_size, total_positions)
            batch_positions = positions[batch_start:batch_end]
            
            print(f"處理批次 {batch_start//batch_size + 1}/{(total_positions-1)//batch_size + 1}: "
                  f"位置 {batch_start}-{batch_end}")
            
            # 為每個位置測試所有姿態
            for pos_idx, position in enumerate(batch_positions):
                global_pos_idx = batch_start + pos_idx
                
                if global_pos_idx % 200 == 0:
                    elapsed = time.time() - start_time
                    progress = global_pos_idx / total_positions * 100
                    avg_time = elapsed / (global_pos_idx + 1)
                    eta = avg_time * (total_positions - global_pos_idx)
                    print(f"  進度: {progress:.1f}%, 平均時間: {avg_time*1000:.1f}ms/位置, ETA: {eta:.0f}s")
                
                # 創建當前位置的所有姿態組合
                num_orientations = len(test_orientations)
                batch_positions_repeated = torch.tensor([position] * num_orientations, 
                                                      device=self.tensor_args.device)
                
                # 創建Pose批次
                pose_batch = Pose(
                    position=batch_positions_repeated,
                    quaternion=test_orientations_tensor
                )
                
                # 執行批次IK求解
                try:
                    result = self.ik_solver.solve_batch(pose_batch)
                    success_mask = result.success.cpu().numpy()
                    
                    # 記錄成功的姿態
                    successful_orientations = []
                    for i, is_success in enumerate(success_mask):
                        if is_success:
                            orientation_data = {
                                'quaternion': test_orientations[i],
                                'joint_solution': result.solution[i].cpu().numpy().tolist()
                            }
                            successful_orientations.append(orientation_data)
                    
                    # 記錄結果
                    pos_key = tuple(position.astype(float))
                    if successful_orientations:
                        self.workspace_data['position_orientation_map'][pos_key] = successful_orientations
                        self.workspace_data['reachable_positions'].append(position.tolist())
                    
                    orientations_per_position.append(len(successful_orientations))
                    
                except Exception as e:
                    print(f"位置 {position} IK求解失敗: {e}")
                    orientations_per_position.append(0)
        
        total_time = time.time() - start_time
        
        # 統計結果
        total_reachable_positions = len(self.workspace_data['reachable_positions'])
        total_orientations = sum(len(orients) for orients in 
                               self.workspace_data['position_orientation_map'].values())
        
        print(f"\n=== 工作空間分析完成 ===")
        print(f"總用時: {total_time:.1f}秒")
        print(f"可達位置: {total_reachable_positions}/{total_positions} "
              f"({total_reachable_positions/total_positions*100:.1f}%)")
        print(f"總可達姿態: {total_orientations}")
        print(f"平均每位置姿態數: {total_orientations/max(1,total_reachable_positions):.1f}")
        
        # 保存分析參數
        self.workspace_data['analysis_params'] = {
            'total_positions': total_positions,
            'total_reachable_positions': total_reachable_positions,
            'total_orientations': total_orientations,
            'orientations_per_position': len(test_orientations),
            'analysis_time': total_time,
            'grid_params': {
                'n_x': args.grid_x,
                'n_y': args.grid_y, 
                'n_z': args.grid_z,
                'max_x': args.max_x,
                'max_y': args.max_y,
                'max_z': args.max_z
            }
        }
        
        # 更新可視化
        draw_reachability_points(positions, orientations_per_position, self.tensor_args)
        
        return orientations_per_position
    
    def save_workspace_data(self, filename):
        """保存工作空間數據為檢索表"""
        with open(filename, 'wb') as f:
            pickle.dump(self.workspace_data, f)
        
        # 也保存JSON格式的摘要
        summary_file = filename.replace('.pkl', '_summary.json')
        summary_data = {
            'reachable_positions': self.workspace_data['reachable_positions'],
            'analysis_params': self.workspace_data['analysis_params'],
            'timestamp': self.workspace_data['timestamp']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"工作空間檢索表已保存:")
        print(f"  完整數據: {filename}")
        print(f"  摘要數據: {summary_file}")

def main():
    try:
        setup_curobo_logger("warn")
        add_extensions(simulation_app, args.headless_mode)
        
        # 創建世界
        my_world = World(stage_units_in_meters=1.0)
        stage = my_world.stage
        xform = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(xform)
        stage.DefinePrim("/curobo", "Xform")

        # 添加目標立方體（用於參考）
        target = cuboid.VisualCuboid(
            "/World/target",
            position=np.array([0.6, 0, 1.8]),
            orientation=np.array([0.699, 0.071, 0.708,-0.070]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )

        # 初始化USD和張量
        usd_help = UsdHelper()
        usd_help.load_stage(my_world.stage)
        tensor_args = TensorDeviceType()

        # 加載機器人配置
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
        j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
        default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

        # 創建簡化的世界配置
        world_cfg = WorldConfig(cuboid=[], mesh=[], sphere=[])
        usd_help.add_world_to_stage(world_cfg, base_frame="/World")
        my_world.scene.add_default_ground_plane()

        # 添加機器人
        robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
        
        # 初始化物理
        my_world.reset()
        for _ in range(10):
            my_world.step(render=True)

        print("初始化機器人姿態...")
        # 初始化機器人
        if hasattr(robot, "_articulation_view"):
            robot._articulation_view.initialize()
        idx_list = [robot.get_dof_index(x) for x in j_names]
        robot.set_joint_positions(default_config, idx_list)
        if hasattr(robot._articulation_view, "set_max_efforts"):
            robot._articulation_view.set_max_efforts(
                values=np.array([5000] * len(idx_list)), 
                joint_indices=idx_list
            )

        for _ in range(10):
            my_world.step(render=True)

        # 創建IK求解器
        print("初始化IK求解器...")
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        ik_solver = IKSolver(ik_config)

        # 生成位置網格和測試姿態
        print("生成測試網格和姿態...")
        positions = get_pose_grid(
            args.grid_x, args.grid_y, args.grid_z,
            args.max_x, args.max_y, args.max_z
        )
        test_orientations = generate_test_orientations(args.orientation_samples)
        
        print(f"生成的測試姿態數量: {len(test_orientations)}")
        
        # 預熱IK求解器
        print("預熱IK求解器...")
        sample_pose = Pose(
            position=tensor_args.to_device([[0.6, 0, 1.8]]),
            quaternion=tensor_args.to_device([test_orientations[0]])
        )
        ik_solver.solve_batch(sample_pose)
        print("IK求解器預熱完成")

        # 創建工作空間分析器
        analyzer = WorkspaceAnalyzer(ik_solver, tensor_args)
        
        print("\n" + "="*60)
        print("開始工作空間分析 - 這可能需要幾分鐘時間")
        print("="*60)
        
        # 執行分析
        orientations_per_pos = analyzer.analyze_workspace(positions, test_orientations)
        
        # 保存結果
        analyzer.save_workspace_data(args.output_file)
        
        print(f"\n完成！檢索表已保存為: {args.output_file}")
        print("\n使用方法:")
        print(f"lookup_table = PoseLookupTable('{args.output_file}')")
        
        # 保持可視化一段時間
        print("\n按Ctrl+C停止程序...")
        try:
            while simulation_app.is_running():
                my_world.step(render=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("用戶中斷，正在退出...")
            
    except Exception as e:
        print(f"程序執行出錯: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()