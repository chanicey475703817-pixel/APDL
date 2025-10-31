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

def alphashape_fileio(infile, outfile, alpha, fig = ""):    #读取点数据输出边界点ID
    points_array, coords_to_id = read_points(infile)
    # Generate the alpha shape 生成阿尔法图形
    shape = alphashape.alphashape(points_array, alpha) #计算alpha形状，自动选取最优alpha,输入点集、alpha
    if not isinstance(shape, Polygon):                  #判断shaple是否属于polygon
        logging.error(f"Unexpected shape {type(shape)}. Expecting Polygon.")
        plot_alpha_shape(points_array, shape, True, fig)
        raise ValueError(f"Unexpected shape {type(shape)}. Expecting Polygon.")

    # Get the boundary points from the alpha shape
    boundary_points = np.array(list(shape.exterior.coords)[:-1])  # Remove the closing duplicate point
    # boundary_points = np.array(list(shape.exterior.coords))  # Keep the closing duplicate point
    boundary_ids = [coords_to_id[tuple(point)] for point in boundary_points]
    boundary_point_count = len(boundary_ids)
    
    # Save boundary IDs to the output file
    with open(outfile, "w") as file:
        file.write(str(boundary_point_count) + "\n")   #输出边界点数量
        file.write("\n".join(map(str, boundary_ids)))  #输出map包括字符串和id
    print(f"Boundary IDs saved to {outfile}")

def define_effected_points(df,col,threshold):
    df0 = df[col].round(2)
    normal = df0.mode().loc[0] if not df0.mode().empty else df0.mean()
    num = normal*threshold
    diff_data = df.loc[df[col]>num].index
    return diff_data

# def define_effected_points(df,col,threshold=3):
#     #输入的是z相同的dataframe切片，接下来进行筛选处理
#     D = df[col].values
#     # 计算Z分数并筛选异常值
#     z_scores = np.abs(stats.zscore(D))
#     # 返回异常值（大于阈值）的索引
#     diff_data = df[z_scores >= threshold].index
#     return diff_data
    
def filter_points(input_csv,output_csv,THREED_NAME,q_0=2.5):        #热流密度矢量定义函数
    # 1. 从 APDL 生成的 CSV 文件中读取数据
    df0= pd.read_csv(input_csv)
    df0['HF_X'] = pd.to_numeric(df0['HF_X'], errors='coerce')
    df0['HF_Y'] = pd.to_numeric(df0['HF_Y'], errors='coerce')
    df = df0.dropna(subset=['HF_X', 'HF_Y'])
        # 4. 计算热流密度矢量的模
    df.to_csv("C:\\Users\\47570\\Desktop\\a.csv")

    # 假设输入 CSV 列名如下：NODE_ID, X, Y, Z, HF_X, HF_Y, HF_Z
    # 2. 将热流分量提取为 NumPy 数组，方便进行矢量运算
    q_i = df['HF_Z'].values
    q_j = df['HF_Y'].values
    q_k = df['HF_X'].values

    # 4. 计算热流密度矢量的模
    df.insert(loc=7,column='HF_SUM',value=0)
    q = np.sqrt(q_i**2 + q_j**2 + q_k**2)
    df['HF_SUM'] = q
    #此处有警告，但无视即可，函数该段输出的df没有问题
    # 初始化存储筛选结果的列表
    df_list = []
    
    #按Z排序后按Z进行划分，记录划分点，把点按Z坐标分成若干层
    df = df.sort_values(by=['Z'], ascending=True).reset_index(drop=True)
    start = 0
    cut = [0]  # 初始化cut列表
    for index, row in df.iterrows():
        if index > 0 and row['Z'] != df.loc[start, 'Z']:
            cut.append(index)
            start = index
    # 添加最后一个点作为结束点
    cut.append(len(df))
    
    #按得到的cut对df进行分段调取
    for i in range(len(cut) - 1):
        z_part = df.iloc[cut[i]:cut[i+1]].copy()
        # 5. 应用公式中的条件进行逻辑筛选 - 使用HF_SUM列筛选异常值
        outlier_index = define_effected_points(z_part, 'HF_SUM', q_0)
        # 从z_part中提取对应的q_j和q_k值
        z_part_qj = z_part['HF_Y'].values
        z_part_qk = z_part['HF_X'].values
        # 为z_part创建条件 - 筛选HF_SUM的异常值
        condition_1 = z_part.index.isin(outlier_index)
        condition_2 = z_part_qj != 0
        condition_3 = z_part_qk != 0
        combined_condition = condition_1 & condition_2 & condition_3
        # 6. 筛选出满足公式条件的点并加入列表
        filtered_points = z_part[combined_condition]
        if not filtered_points.empty:
            df_list.append(filtered_points)
            
    #7、合并所有dataframe
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        #清理重复的XY坐标数据,拍平坐标
        df_clean = final_df.drop_duplicates(subset=['X', 'Y'], keep='last')
        #创建适合alphashapes处理的输出格式
        output_df = df_clean[['NODE_ID', 'X', 'Y']].copy()
        output_df.to_csv(output_csv, header=False, index=False)        #想改成相对路径但是没成功
        print(f'筛选出的点已保存到{output_csv}')
        print(f"总节点数: {len(df)}")
        print(f"满足公式条件的节点数: {len(output_df)}")
    else:
        # 如果没有筛选出任何点，创建一个空文件
        open(output_csv, 'w').close()
        print(f'未筛选出满足条件的点，已创建空文件{output_csv}')
        print(f"总节点数: {len(df)}")
        print(f"满足公式条件的节点数: 0")
    #3D点云
    output_df1 = final_df[['NODE_ID', 'X', 'Y', 'Z']].copy()
    pointcloud = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP1\POINTCLOUD"
    save_path = os.path.join(pointcloud,THREED_NAME)
    output_df1.to_csv(save_path, header=False, index=False) 
    
    


def alphamain(input,output,twod_name,threed_name,fig_boolean=False,alpha=0.5,q_0=1.3):#Z:2.7


    #筛点函数中转路径
    
    temp_path_folder = "C:\\mine\\apdl\\ThermalSimulation\\TBSimu1.0\\APDL\\SUB\\TEMP1\\2DPOINTCLOUD"
    temp_path = os.path.join(temp_path_folder,twod_name)
    filter_points(input,temp_path,threed_name,q_0) #筛点程序插入
    #筛点程序输入格式（ID\X\Y\Z\HF_X\HF_Y\HF_Z）,输出中转文件格式（ID\X\Y）


    points_array, coords_to_id = read_points(temp_path) #从中转路径读取文件
    # Generate the alpha shape,
    shape = alphashape.alphashape(points_array, alpha)
    logger.info('Successfully alphashape!')

    # 处理不同类型的几何形状结果
    if not isinstance(shape, Polygon):
        raise ValueError(f"Unexpected shhape {type(shape)}. Expecting Polygon.")
        # 如果是单一多边形
    boundary_points = np.array(list(shape.exterior.coords)[:-1])  # 移除闭合重复点
    
    # boundary_points = np.array(list(shape.exterior.coords))  # Keep the closing duplicate point
    boundary_ids = [coords_to_id[tuple(point)] for point in boundary_points]
    #print(boundary_points)
    boundary_point_count = len(boundary_ids)
    
    data = {'NODE_ID':boundary_ids,'X':boundary_points[:,0],'Y':boundary_points[:,1]}
    df = pd.DataFrame(data)  

    # Save boundary IDs to the output file 保存ID到文件
    with open(output, "w") as file:
        file.write(str(boundary_point_count) + "\n")
    df.to_csv(output, mode='a', index=False, header=False)  #每个id转成str然后换行
    print(f"Boundary IDs and coordinates are saved to {output}")
    area = shape.area
    # Plot the alpha shape if requested 绘制图形
    if fig_boolean:
        plot_alpha_shape(points_array, shape)

    return area