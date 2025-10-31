import datetime
import os
import shutil

# 临时文件保存的文件夹路径
temp_folder_path = r"C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\SUB\TEMP1"

def move_png_files(target_folder,shape,t):
    source_folder = r"C:\mine\apdl\ThermalSimulation\working"
    
    # 获取当前日期时间并格式化为指定的字符串格式（例如：250106_21_45_FIG）
    #now = datetime.datetime.now()
    #date_time_str = now.strftime('%y%m%d_%H_%M_FIG')
    file_name = f"{shape}_TO_{t}_fig"
    target_path = os.path.join(target_folder, file_name)
    os.makedirs(target_path)

    # dele1,dele2,dele3,dele4='ThermalBridge000.png','ThermalBridge002.png','ThermalBridge003.png','ThermalBridge004.png'
    # delete = [os.path.join(target_folder,dele1),os.path.join(target_folder,dele2),os.path.join(target_folder,dele3),os.path.join(target_folder,dele4)]
    # os.remove(delete)

    for file in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file)
        if os.path.isfile(file_path) and file.lower().endswith('.png'):
            target_file_path = os.path.join(target_path, file)
            shutil.move(file_path, target_file_path)
    
    print("移动完成！")


def result_datetime_write():
    # 如果临时文件夹不存在，则创建它
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)
    # 定义输出文件路径，用于写入日期时间信息
    output_file_path = r'C:\mine\apdl\ThermalSimulation\TBSimu1.0\APDL\OUTPUT\DATA_YB.txt'
    # 获取当前日期和时间并格式化为字符串
    current_datetime = datetime.datetime.now().strftime(' %Y-%m-%d %H:%M:%S')
    # 将日期时间信息写入到指定的输出文件末尾
    with open(output_file_path, 'a') as file:
        file.write('\n' + current_datetime + '\n')