import cv2
import os

# 视频文件路径
video_path = '/raid/cvg_data/lurenjie/animatediff-cli-prompt-travel/result/youtube/mp4_1_768_dense.mp4'
# 保存帧的文件夹路径
output_folder = '/raid/cvg_data/lurenjie/animatediff-cli-prompt-travel/result/youtube/mp4_1_dense_image'
# 指定的步长
step = 1
# 指定的帧缩放尺寸
target_width = 768
target_height = 1024

# 创建保存帧的文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频文件")
    exit()

frame_count = 0
num = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if frame_count % step == 0:
        # 缩放帧
        resized_frame = cv2.resize(frame, (target_width, target_height))
        # 生成帧文件名
        frame_filename = os.path.join(output_folder, f'{num:08}.png')
        # 保存帧
        cv2.imwrite(frame_filename, resized_frame)
        num += 1

# 释放视频对象
cap.release()
cv2.destroyAllWindows()

print(f"提取并保存了 {frame_count} 帧到文件夹 {output_folder}")
