import pandas as pd
import argparse
import alphashape
import os
try:
    from fileio import read_points
except:
    from myalphashape.fileio import read_points
from shapely.geometry import Polygon, GeometryCollection
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy import stats
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to plot the points and the alpha shape 可视化函数
def plot_alpha_shape(points_array: np.ndarray, shape: Polygon, save=False, fig = ""):
    """
    Plot the alpha shape along with the original point cloud.

    Args:
        points_array (np.ndarray): The original point cloud as a numpy array of shape (n, 2).
        shape (Polygon): The Shapely polygon representing the alpha shape.
    """
    # Plot the original point cloud
    plt.scatter(points_array[:, 0], points_array[:, 1], c="blue", label="Points")

    # Plot the alpha shape
    if shape and shape.is_valid:
        x, y = shape.exterior.xy  # Extract exterior coordinates
        plt.plot(x, y, c="red", label="Alpha Shape")

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Alpha Shape and Point Cloud")
    if (save):
        plt.savefig(fig, dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
    plt.show()


def filter_points(input_csv, output_csv, THREED_NAME, t):        #热流密度矢量定义函数
    """
    从APDL生成的CSV文件中筛选符合条件的点
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        THREED_NAME: 3D点云文件名
        t: 温度参数
    """
    try:
        # 1. 从 APDL 生成的 CSV 文件中读取数据
        #清洗数据
        df0 = pd.read_csv(input_csv)
        df0['TEMP'] = pd.to_numeric(df0['TEMP'], errors='coerce')
        df = df0.dropna(subset=['TEMP'])
        
        # 2. 根据z坐标分段
        if 'Z' not in df.columns:
            logger.error("CSV文件中缺少Z列")
            return False
            
        grouped = df.groupby('Z')
        
        # 检查是否存在所需的Z值
        if 0 not in grouped.groups or 0.21 not in grouped.groups:
            logger.warning("找不到Z=0或Z=0.21的数据组，尝试使用最接近的值")
            # 找到最接近的Z值
            z_values = sorted(grouped.groups.keys())
            closest_z0 = min(z_values, key=lambda x: abs(x - 0))
            closest_z21 = min(z_values, key=lambda x: abs(x - 0.21))
            logger.info(f"使用Z={closest_z0}代替Z=0，使用Z={closest_z21}代替Z=0.21")
            df_I = grouped.get_group(closest_z0)
            df_E = grouped.get_group(closest_z21)
        else:
            df_I = grouped.get_group(0)
            df_E = grouped.get_group(0.21)
        
        # 3. 计算内外表面温度的众数
        I_temp = df_I['TEMP']
        E_temp = df_E['TEMP']
        
        # 使用更安全的众数计算方法
        I_normal = I_temp.mode().iloc[0] if not I_temp.mode().empty else I_temp.mean()
        E_normal = E_temp.mode().iloc[0] if not E_temp.mode().empty else E_temp.mean()
        
        print(f'内表面众数是：{I_normal}')
        print(f'外表面众数是：{E_normal}')
        
        TI = t
        TE = 0
        
        # 避免除以零的情况
        temp_diff = TI - TE
        if temp_diff == 0:
            logger.warning("TI等于TE，无法计算比率")
            temp_diff = 1.0  # 设置一个默认值
        
        rate_i = (TI - I_normal) / temp_diff
        rate_e = 1 - (TI - E_normal) / temp_diff
        
        # 4. 筛选内外表面影响点
        I = df_I['TEMP'].values
        rate_i0 = (TI - I) / temp_diff
        # 避免除以零的情况
        valid_rate_i0 = rate_i0 != 0
        TB_I = df_I.index[valid_rate_i0 & (rate_i / rate_i0[valid_rate_i0] < 0.95)]
        
        E = df_E['TEMP'].values
        rate_e0 = 1 - (TI - E) / temp_diff
        # 避免除以零的情况
        valid_rate_e0 = rate_e0 != 0
        TB_E = df_E.index[valid_rate_e0 & (rate_e / rate_e0[valid_rate_e0] < 0.95)]
        
        # 5. 合并筛选结果
        # filtered_points_I = df_I[df_I.index.isin(TB_I)]
        filtered_points_E = df_E[df_E.index.isin(TB_E)]
        
        # 修复pd.concat语法，传入列表
        # final_df = pd.concat([filtered_points_E, filtered_points_I], ignore_index=True)
        filtered_points = filtered_points_E.drop_duplicates(subset=['X', 'Y'], keep='last')
        
        if filtered_points.empty:
            logger.warning('没有筛选出符合条件的点！')
        else:
            # 创建适合alphashapes处理的输出格式
            output_df = filtered_points[['NODE_ID', 'X', 'Y']].copy()
            output_df.to_csv(output_csv, header=False, index=False)
            print(f'筛选出的点已保存到{output_csv}')
            print(f"总节点数: {len(df)}")
            print(f"满足公式条件的节点数: {len(output_df)}")
            
            # 3D点云保存
            output_df1 = filtered_points[['NODE_ID', 'X', 'Y', 'Z']].copy()
            # 使用相对路径替代硬编码路径
            pointcloud_dir = r'C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP3\POINTCLOUD'
            save_path = os.path.join(pointcloud_dir, THREED_NAME)
            output_df1.to_csv(save_path, header=False, index=False)
            logger.info(f'3D点云已保存到{save_path}')
            
        return True
    except Exception as e:
        logger.error(f"filter_points函数执行出错: {str(e)}")
        print(f"错误: {str(e)}")
        return False


def alphamain(input, output, twod_name, threed_name, t, fig_boolean=False, alpha=1):
    """
    主函数，处理点云并生成alpha形状
    
    Args:
        input: 输入文件路径
        output: 输出文件路径
        twod_name: 2D点云文件名
        threed_name: 3D点云文件名
        t: 温度参数
        fig_boolean: 是否显示图形
        alpha: alpha参数
    """
    try:
        # 筛点函数中转路径 - 使用相对路径
        temp_path_folder = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP3\2DPOINTCLOUD"
        os.makedirs(temp_path_folder, exist_ok=True)
        temp_path = os.path.join(temp_path_folder, twod_name)
        
        # 调用筛点函数
        if not filter_points(input, temp_path, threed_name, t):
            logger.error("筛点函数执行失败")
            return 0
        
        # 从中转路径读取文件
        points_array, coords_to_id = read_points(temp_path)
        
        # 生成alpha形状
        shape = alphashape.alphashape(points_array, alpha)
        logger.info('Successfully alphashape!')
        
        # 处理不同类型的几何形状结果
        if not isinstance(shape, Polygon):
            raise ValueError(f"Unexpected shape {type(shape)}. Expecting Polygon.")
        
        # 提取边界点
        boundary_points = np.array(list(shape.exterior.coords)[:-1])  # 移除闭合重复点
        boundary_ids = [coords_to_id[tuple(point)] for point in boundary_points]
        boundary_point_count = len(boundary_ids)
        
        # 创建数据框
        data = {'NODE_ID': boundary_ids, 'X': boundary_points[:, 0], 'Y': boundary_points[:, 1]}
        df = pd.DataFrame(data)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        
        # 保存边界ID到输出文件
        with open(output, "w") as file:
            file.write(str(boundary_point_count) + "\n")
        df.to_csv(output, mode='a', index=False, header=False)
        print(f"Boundary IDs and coordinates are saved to {output}")
        
        # 计算面积
        area = shape.area
        
        # 绘制图形（如果需要）
        if fig_boolean:
            plot_alpha_shape(points_array, shape)
        
        return area
    except Exception as e:
        logger.error(f"alphamain函数执行出错: {str(e)}")
        print(f"错误: {str(e)}")
        return 0