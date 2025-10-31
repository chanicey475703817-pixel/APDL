import datetime
import os
import time
import argparse
import logging
import ansys.mapdl.core as pymapdl
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
from myalphashape.main import alphashape_fileio
#from models.ftconvert import replace_params_in_temp_func
from models.png_move import move_png_files
from models.png_move import result_datetime_write
#from models.curve_fit import eq_thermal_flux_fit
from myalphashape.main import alphamain
# from models.log_write import log_write_and_interrupt

project_root = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB"
# 原始APDL批处理文件路径
original_square = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\model_square.mac"
original_cylinder = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\model_cylinder.mac"
original_lshape = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\model_lshape.mac"
file_list = {'Square':original_square , 'Cylinder':original_cylinder , 'Lshape':original_lshape}
test_list = {'Square':original_square}

original_2dvread = r"C:\mine\apdl\ThermalSimulation\original_batch\.mac\2DVREAD.mac"
original_3dvread = r"C:\mine\apdl\ThermalSimulation\original_batch\.mac\3DVREAD.mac"
original_boundaryvread = r'C:\mine\apdl\ThermalSimulation\original_batch\.mac\BOUNDARYVREAD.mac'
original_2dplot = r"C:\mine\apdl\ThermalSimulation\original_batch\.mac\2DPLOT.mac"
original_3dplot = r"C:\mine\apdl\ThermalSimulation\original_batch\.mac\3DPLOT.mac"

# 临时文件保存的文件夹路径
temp_folder_path = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP1"

eqfind_var_path = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TRANS_BATCH.mac"
thermal_flux_data_path = r'C:\mine\apdl\ThermalSimulation\TBSimu1.0\1D\OUTPUT\TIME_TF_TEMP.txt'
#mainloop主循环已跑通，接下来达成以下目标：
#1、计算热桥面积（有问题，需要完善）
#2、绘图（正视图、轴测图）
    #2.1、正视图：轮廓（线视图）、温度云图、热桥边界定义线
    #2.2、轴测图：轮廓（线视图）、点云图、热桥边界定义线（2D，位于表面）、总剖图

    
def single_run(t_value, mapdl, failure_log="failures.log"):  #输入t值后，对三种模型各自处理，进行一次模拟并输出边界点ID文件csv，接下来进行绘图
    work_path = os.getcwd()
    data = {'shape':[],'temp':[],'area':[]}
    dataframe = pd.DataFrame(data)
    n=0

    for shape,path in file_list.items():        #三种形状均遍历一遍,暂时注释掉用于固定输出
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()             #处理文件中t、k、tbh、tbw的值
            for index, line in enumerate(lines): #获取带索引的列表
                
                if "TO=18" in line:
                    # 修改TO参数值
                    lines[index] = f"TO={t_value}\n"    #f表示格式化字符串，\n表示回车但不换行，\r表示换行但不回车        
                    break
                
        # 生成临时文件名
        temp_file_name = f"{shape}_TO_{t_value}.txt"
        temp_file_path = os.path.join(temp_folder_path, temp_file_name)
        #try:
        #os.chmod(temp_folder_path, 0o777)
        with open(temp_file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)  #修改后的数据写入临时文件
        # except PermissionError:
        #     print('PermissionError but successfully writed!')
        # 启动ANSYS MAPDL
        logging.debug(f"Running script {temp_file_name}")   
        mapdl.clear()
        mapdl.input(temp_file_path)                        #输入脚本

        output_folder = r"C:\\mine\\apdl\\ThermalSimulation\\TBSimu1.0\\APDL\\SUB\\TEMP1\\OUTLINE"
        output_name = f"{shape}_TO_{t_value}_data.csv"
        output_path = os.path.join(output_folder,output_name)
        csv_output = r'C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP1\temp_csv\All_Node_Data.csv'#原始输出文件

        area = alphamain(csv_output,output_path,output_name,output_name)     #边界点id\x\y输出至boundary_path
        
        
        print('热桥面积是：'+str(round(area,6)))
        dataframe.loc[n,'shape']= shape
        dataframe.loc[n,'temp'] = t_value
        dataframe.loc[n,'area'] = round(area,6)
        n += 1
        
        # print(dataframe.tail(3))
        # dataframe.to_csv("C:\\Users\\47570\\Desktop\\Final.csv",mode='a',header=False)
    
        #指标获取部分（热桥面积、热桥形状（图片，在alphamain中实现））面积直接通过shapely获取
        
        #修改后绘图部分APDL代码
        

    #建模及轴侧视角出图部分
        twod_folder = "C:\\mine\\apdl\\ThermalSimulation\\TBSimu1.0\\APDL\\SUB\\TEMP1\\2DPOINTCLOUD"
        twod_path = os.path.join(twod_folder,output_name)
        threed_folder = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP1\POINTCLOUD"
        threed_path = os.path.join(threed_folder,output_name)
    #文件修改部分

        #batch脚本保存地址
        #temp_2dvread = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP1\temp_batch\temp_2d.mac"
        temp_3dvread = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP1\temp_batch\temp_3d.mac"
        temp_boundary = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP1\temp_batch\temp_boundary.mac"
        #temp_2dplot = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP1\temp_batch\temp_2Dplot.mac"
        #temp_3dplot = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP1\temp_batch\temp_3Dplot.mac"
        Fig_path = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP1\FIG"
        
        # #修改2d_vread
        # with open(original_2dvread,'r',encoding = 'utf-8') as afile :
        #     alines = afile.readlines()
        #     for aindex,aline in enumerate(alines):
        #         if r"*VREAD,NodeData(1,1),'C:\mine\apdl\ThermalSimulation\test_output.csv',,,JIK,3,20000,1" in aline:
        #             alines[aindex] = "*VREAD,NodeData(1,1),'" + twod_path +"',,,JIK,3,20000,1"+"\n"
        # with open(temp_2dvread,'w',encoding='utf-8') as afile:
        #     afile.writelines(alines)

        #修改3d_vread
        with open(original_3dvread,'r',encoding = 'utf-8') as afile :
            alines = afile.readlines()
            for aindex,aline in enumerate(alines):
                if r"*VREAD,NodeData(1,1),'C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP1\OUTLINE\Square_TO_19_data.csv',,,JIK,4,20000,1" in aline:
                    alines[aindex] = "*VREAD,NodeData(1,1),'" + threed_path +"',,,JIK,4,20000,1"+"\n"
                if r"*VREAD,NodeData(1,1),'C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP1\OUTLINE\Square_TO_19_data.csv',,,JIK,4,RowCount,1" in aline:
                    alines[aindex] = "*VREAD,NodeData(1,1),'" + threed_path +"',,,JIK,4,RowCount,1"+"\n"
        with open(temp_3dvread,'w',encoding='utf-8') as afile:
            afile.writelines(alines)

        #修改boundary_vread
        with open(original_boundaryvread,'r',encoding='utf-8') as afile:#修改boundaryvread，暂存到暂存文件中，再读取
            alines = afile.readlines()
            for aindex,aline in enumerate(alines):
                if r"*VREAD,BDT,'C:\Users\47570\Desktop\a.csv',,,JIK,1,1,1"in aline:
                    alines[aindex] = "*VREAD,BDT,'"+output_path+"',,,JIK,1,1,1"+"\n"
                if r"*VREAD,BD(1,1),'C:\Users\47570\Desktop\a.csv',,,JIK,3,BDN,1,1" in aline:
                    alines[aindex] = "*VREAD,BD(1,1),'"+output_path+"',,,JIK,3,BDN,1,1"  +"\n"               #将读取路径改为相应的2D点云输出路径
        with open(temp_boundary ,'w',encoding='utf-8') as afile:
            afile.writelines(alines)

        
        #正视绘制部分
        mapdl.input(temp_3dvread)
        print("3dvread succesfully!")
        mapdl.input(temp_boundary)
        print("boundary vread succesfully!")
        mapdl.input(original_2dplot)
        print("2dplot succesfully!")

        #轴侧绘制部分
        mapdl.input(original_3dplot)    
        print("3dplot succesfully!")
        move_png_files(Fig_path,shape,t_value)
    print(dataframe.tail(3))
    
    csv_name = 'area.csv'
    csv_path = os.path.join(temp_folder_path,csv_name)
    dataframe.to_csv(csv_path,mode='a',header=False)


# def discretize_thermal_resistance(longitude, latitude, altitude):
#     temp_boundry_name = replace_params_in_temp_func(longitude, latitude, altitude)
#     with open(eqfind_var_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#         for index, line in enumerate(lines):
#             if "R_TOT = 0.000" in line:
#                 lines[index] = f"R_TOT={rtot_value}\n"            
#             if "R_2 = 1.200" in line:
#                 lines[index] = f"R_2={r2_value}\n"
#             if "R_3 = 0.000" in line:
#                 lines[index] = f"R_3={r3_value}\n"
#             if "E_1 = 0.15" in line:
#                 lines[index] = f"TBW={e1_value}\n"
#             if "E_2 = 0.03" in line:
#                 lines[index] = f"TBW={e2_value}\n"
#             if "E_3 = 0.03" in line:
#                 lines[index] = f"TBW={e3_value}\n"
#             if "E_3 = 0.03" in line:
#                 lines[index] = f"TBW={e3_value}\n"
#             if "TO=0.0000" in line:
#                 lines[index] = f"T0={tout_value}\n"
#             if "TE=0.0000" in line:
#                 lines[index] = f"TBW={tin_value}\n"
#             if "/INPUT,50.0_50.0_50.0_TEMP,MAC,'D:\BaiduSyncdisk\ThermalSimulation\TBSimu 1.0\APDL\SUB\TEMP\TEMPFUNC'" in line:
#                 lines[index] = line.replace("50.0_50.0_50.0_TEMP", temp_boundry_name)

#     # 生成临时文件名
#     temp_file_name = os.path.join(temp_boundry_name, "_batch.mac")

#     mapdl.input(temp_file_path)
#     mapdl.input(os.path.join(project_root, "TIME_POST_OPT.mac"))
#     thermal_flux_data_path = r'D:\BaiduSyncdisk\ThermalSimulation\TBSimu 1.0\1D\OUTPUT\TIME_TF_TEMP.txt'
#     q_eq_int, q_eq_ext = eq_thermal_flux_fit(thermal_flux_data_path)

#     q_eq_int_rounded = [round(float(param), 7) for param in q_eq_int]
#     q_eq_ext_rounded = [round(float(param), 7) for param in q_eq_ext]


#     print("室内热流密度拟合参数:", q_eq_int_rounded)
#     print("室外热流密度拟合参数:", q_eq_ext_rounded)
#     mapdl.clear()


def log_write_and_interrupt(t_value, failure_log="failures.log"):            # k_value, tbh_value, tbw_value,
    max_retries = 3  # 最大重试次数，可根据实际情况调整
    retry_delay = 2  # 重试间隔时间（秒），可根据实际情况调整
    retries = 0
    while retries < max_retries:
        try:
            start_time = time.time()
            mapdl=pymapdl.launch_mapdl()
            single_run(t_value, mapdl)
            elapsed_time = time.time() - start_time
            # Log successful completion and time used
            logging.info(f"Completed successfully with t={t_value},  in {elapsed_time:.2f} seconds")    #k={k_value}, tbh={tbh_value}, tbw={tbw_value}
            mapdl.exit()
            break  # 如果执行成功，跳出重试循环

        except KeyboardInterrupt:
            # 捕获用户中断信号，日志记录并立即退出程序
            logging.warning("KeyboardInterrupt detected. Exiting...")
            mapdl.exit()
            raise SystemExit(0) 
            
        except Exception as e:
            print(f"在参数 t={t_value}时出现错误: {e}")                             #, k_value={k_value}, tbh_value={tbh_value}, tbw_value={tbw_value} 
            retries += 1
            if retries < max_retries:
                mapdl.exit()
                print(f"正在进行第 {retries + 1} 次重试，等待 {retry_delay} 秒...")
            else:
                print("达到最大重试次数，继续下一组参数尝试...")
                mapdl.exit()
                # 记录失败的参数和当前时间
                with open(failure_log, "a") as log_file:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{timestamp} - t={t_value} ({e})\n")            #, k={k_value}, tbh={tbh_value}, tbw={tbw_value}
    

def main_loop( t=-1, failure_log="failures.log"):  # k=-1, tbh=-1, tbw=-1
    
    # 设置默认范围
    t_range = [t] if t != -1 else [round(i / 10.0, 2) for i in range(0, 505, 10)] #便于调试时使用单个参数，运行时自动扫描。这里使用了列表推导式

    #注释了无关参数
    # k_range = [k] if k != -1 else range(10, 180, 20)
    # tbh_range = [tbh] if tbh != -1 else [round(i / 1000.0, 2) for i in range(50, 301, 30)]
    # tbw_range = [tbw] if tbw != -1 else [round(i / 1000.0, 2) for i in range(50, 201, 30)]
    data_list=[]
    for t_value in t_range:
        # for k_value in k_range:
        #     for tbh_value in tbh_range:
        #         for tbw_value in tbw_range:

        log_write_and_interrupt(t_value, failure_log="failures.log") #k_value, tbh_value, tbw_value,
    


# def location_loop(lon=-200, lat=-200, alt=10000):
   
#     # 经度范围是±180（步长20度）
#     longitude_range = [lon] if lon != -200 else [i for i in range(-180, 181, 20)]
#     # 纬度范围是±90（步长20度）
#     latitude_range = [lat] if lat != -200 else [i for i in range(-90, 91, 20)]
#     # 海拔范围是0 - 2000（步长200米）
#     altitude_range = [alt] if alt != 10000 else [i for i in range(0, 2001, 200)]

#     for longitude in longitude_range:
#         for latitude in latitude_range:
#             for altitude in altitude_range:
#                 # 在这里可以添加具体针对每个经纬度和海拔组合需要执行的代码逻辑
#                 replace_params_in_temp_func(longitude, latitude, altitude)
#                 print(f"当前经度: {longitude}, 当前纬度: {latitude}, 当前海拔: {altitude}")
#                 mapdl = pymapdl.launch_mapdl()
#                 discretize_thermal_resistance(longitude, latitude, altitude)
#                 mapdl.exit()

def steady_state_simulation(args):
    
    try:
        if args.log.isdigit():  # 如果是整数值
            log_level = int(args.log)
        else:  # 如果是字符串值
            log_level = getattr(logging, args.log.upper(), logging.INFO)
    except ValueError:
        log_level = logging.INFO
        logging.warning(f"Invalid log level '{args.log}', defaulting to INFO.")
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    result_datetime_write()
    
    main_loop(t=args.t)      #, k=args.k, tbh=args.tbh, tbw=args.tbw
    
    

# def location_bondary_simulation(args):
#     location_loop(lon=args.lon, lat=args.lat, alt=args.alt)
#     # longitude, latitude, altitude = 





# def best_eqwall_find(args):

#     # qi,qe = eq_thermal_flux_fit(thermal_flux_data_path)
#     mapdl = pymapdl.launch_mapdl()
#     mapdl.input(os.path.join(project_root, "TRANS_BATCH.mac"))
#     mapdl.input(os.path.join(project_root, "TIME_POST_OPT.mac"))
#     mapdl.exit()




# def location_bondary_simulation(args):
#     location_loop(lon=args.lon, lat=args.lat, alt=args.alt)
#     # longitude, latitude, altitude = 



# def best_eqwall_find(args):

#     # qi,qe = eq_thermal_flux_fit(thermal_flux_data_path)
#     mapdl = launch_mapdl()
#     mapdl.input(os.path.join(project_root, "TRANS_BATCH.mac"))
#     mapdl.input(os.path.join(project_root, "TIME_POST_OPT.mac"))
#     mapdl.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control which modules to run and their options.")
    # 添加一个参数用于指定要运行的模块，可接受多个值，choices限制输入的模块名只能是指定的几个
    parser.add_argument('modules', nargs='+', choices=['A', 'B', 'C', 'D'], help="Specify which modules to run")
    
    # 为模块A添加参数
    steady_parser = parser.add_argument_group('Module A options')
    steady_parser.add_argument("-t", "--t", type=float, default=-1, help="指定 TO 的值（默认 -1 表示循环）")
    # steady_parser.add_argument("-k", "--k", type=int, default=-1, help="指定 k 的值（默认 -1 表示循环）")
    # steady_parser.add_argument("-th", "--tbh", type=float, default=-1, help="指定 tbh 的值（默认 -1 表示循环）,mm")
    # steady_parser.add_argument("-w", "--tbw", type=float, default=-1, help="指定 tbw 的值（默认 -1 表示循环）,mm")
    # steady_parser.add_argument("-s", "--skip", action="store_true", help="是否跳过 result_datetime_write")
    steady_parser.add_argument(
        "-l", "--log", type=str, default="INFO", 
        help="指定日志级别 (字符串: DEBUG, INFO, WARNING, ERROR, CRITICAL 或整数: 10, 20, 30, 40, 50)"
    )

    # 为模块B添加参数
    # location_parser = parser.add_argument_group('Module B options')
    # location_parser.add_argument("-lon", "--lon", type=float, default=-1, help="指定 TO 的值（默认 -1 表示循环）")
    # location_parser.add_argument("-lat", "--lat", type=float, default=-1, help="指定 TO 的值（默认 -1 表示循环）")
    # location_parser.add_argument("-alt", "--alt", type=float, default=-1, help="指定 TO 的值（默认 -1 表示循环）")

    # 解析命令行参数
    args = parser.parse_args()

    for module in args.modules:
        if module == 'A':
            steady_state_simulation(args)
        # elif module == 'B':
        #     location_bondary_simulation(args)
        # elif module == 'C':
        #     eq_calculate_simulation(args)
        # elif module == 'D':
        #     best_eqwall_find(args)


