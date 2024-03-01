import cv2
import os

# 输入文件夹和输出文件夹的路径
input_folder = "/raid/cvg_data/lurenjie/AnimateDiff_muti/AnimateDiff-I2V-new/project/dance/frames_6"
output_folder = "/raid/cvg_data/lurenjie/AnimateDiff_muti/AnimateDiff-I2V-new/project/dance/canny_6"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中的所有PNG图像文件
input_images = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# 遍历每张图片进行边缘检测并保存
for image_file in input_images:
    # 读取图片
    input_path = os.path.join(input_folder, image_file)
    image = cv2.imread(input_path)

    if image is not None:
        # 执行边缘检测（这里使用Canny算法）
        edges = cv2.Canny(image, 100, 200)  # 调整阈值以获得合适的边缘

        # 构造输出文件路径
        output_path = os.path.join(output_folder, image_file)

        # 保存边缘检测后的图像
        cv2.imwrite(output_path, edges)

print("边缘检测完成，并保存到", output_folder)
