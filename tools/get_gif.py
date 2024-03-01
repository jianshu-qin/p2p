import os

from PIL import Image

# 设置文件夹路径和输出GIF文件名
# folder_path = '/home/jianshu/code/prompt_travel/stylize/replace_bg_video/2024-02-04T11-13-38-sample-cyberrealistic_v33/show_all/14'  # 替换为你的文件夹路径
# output_gif = '/home/jianshu/code/prompt_travel/stylize/replace_bg_video/14.gif'  # 替换为你的输出GIF文件名

def make_gif(folder_path, output_gif):
# 获取文件夹中所有的PNG文件，并按名称排序
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    png_files.sort()

    # 创建一个图像列表，按名称顺序加载PNG文件
    images = []
    for png_file in png_files:
        file_path = os.path.join(folder_path, png_file)
        img = Image.open(file_path)
        images.append(img)

    # 设置帧率（FPS）和保存为GIF
    fps = 10  # 替换为你想要的帧率
    images[0].save(output_gif, save_all=True, append_images=images[1:], duration=1000 // fps, loop=0)

root_dir = "/home/jianshu/code/prompt_travel/stylize/replace_bg_video/w01/show_all"
for fn in os.listdir(root_dir):
    num = int(fn)
    folder_path = os.path.join(root_dir, fn)
    output_gif = os.path.join(root_dir, f"{num:02d}.gif")
    make_gif(folder_path, output_gif)
