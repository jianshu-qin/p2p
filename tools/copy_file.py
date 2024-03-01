import os
import shutil

def copy_images_to_subfolders(input_folder, output_parent_folder):
    # 获取输入文件夹中所有的PNG图像
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

    # 遍历输出父文件夹下的所有子文件夹
    for subfolder in os.listdir(output_parent_folder):
        subfolder_path = os.path.join(output_parent_folder, subfolder)

        # 仅处理子文件夹
        if os.path.isdir(subfolder_path):
            # 在每个子文件夹下创建一个输出文件夹
            output_folder = subfolder_path
            os.makedirs(output_folder, exist_ok=True)

            # 将所有PNG图像复制到输出文件夹中
            for image_file in image_files:
                input_path = os.path.join(input_folder, image_file)
                output_path = os.path.join(output_folder, image_file)
                shutil.copy2(input_path, output_path)

if __name__ == "__main__":
    input_folder = "/raid/cvg_data/lurenjie/animatediff-cli-prompt-travel/result/taiji/zoom_images"
    output_parent_folder = "/raid/cvg_data/lurenjie/animatediff-cli-prompt-travel/stylize/n_taiji_zoom/00_controlnet_image"

    copy_images_to_subfolders(input_folder, output_parent_folder)
