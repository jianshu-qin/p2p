from PIL import Image
import os

def crop_and_resize_images(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中所有的PNG图像
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # 打开图像
        img = Image.open(input_path)

        # 获取原始尺寸
        original_width, original_height = img.size

        # 计算裁剪的区域
        # left = (original_width - original_width // 6) // 5
        # top = (original_height - original_height // 6) // 5
        # right = original_width - left
        # bottom = original_height - top
        # img_cropped = img.crop((left, top, right, bottom))
        # img_resized = img_cropped.resize((original_width, original_height))
        # img_resized.save(output_path)


        left = 0
        top = 260
        right = original_width
        bottom = 260+1024
        # left = 0
        # top = 224
        # right = original_width
        # bottom = 224+1024
        img_cropped = img.crop((left, top, right, bottom))
        # img_resized = img_cropped.resize((1536, 2048))
        img_cropped.save(output_path)


if __name__ == "__main__":
    input_folder = "/raid/cvg_data/lurenjie/animatediff-cli-prompt-travel/stylize/mp4-7/00_img2img"
    output_folder = "/raid/cvg_data/lurenjie/animatediff-cli-prompt-travel/result/youtube/mp4_7_image"

    crop_and_resize_images(input_folder, output_folder)
