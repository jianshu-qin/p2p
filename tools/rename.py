import os


def rename_png_files(folder_path):
    # 获取文件夹下的所有文件
    files = os.listdir(folder_path)

    # 筛选出所有的png文件
    png_files = [file for file in files if file.endswith('.png')]

    # 对png文件进行重命名
    for i, png_file in enumerate(png_files):
        # 生成新的文件名
        new_file_name = f"{i:08d}.png"

        # 获取原文件的完整路径
        old_file_path = os.path.join(folder_path, png_file)

        # 获取新文件的完整路径
        new_file_path = os.path.join(folder_path, new_file_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)

# 调用函数，传入文件夹路径
folder_path = "../stylize/warp_fg/control/controlnet_openpose"
rename_png_files(folder_path)
