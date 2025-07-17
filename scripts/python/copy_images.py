import os
import shutil

# 设置你的主文件夹路径
parent_dir = "/data/gs/anting/0712_card/cam2_card1/"  # 修改为你的路径
target_dir = "/data/gs/anting/0712_card/seg5_full/"  # 修改为你的路径

# 遍历 parent_dir 下的所有文件夹
for subfolder in os.listdir(parent_dir):
    subfolder_path = os.path.join(parent_dir, subfolder)
    
    # 如果是目录才处理
    if os.path.isdir(subfolder_path):
        print(subfolder_path)
        for filename in os.listdir(subfolder_path):
            src_file = os.path.join(subfolder_path, filename)
            # 确保是文件再复制
            if os.path.isfile(src_file):
                # 为避免重名，给文件加上前缀（可选）
                dst_file = os.path.join(target_dir, f"{filename}")
                shutil.copy2(src_file, dst_file)

print("图片拷贝完成。")
