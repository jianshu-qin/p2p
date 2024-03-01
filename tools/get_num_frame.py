import cv2

# 视频文件路径
video_path = '/raid/cvg_data/ECCV2024/dataset/UBC_fashion_video/train/81FyMPk-WIS.mp4'

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
else:
    # 获取视频的帧率（FPS）
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 获取视频的总帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("视频帧率（FPS）：", fps)
    print("视频总帧数：", frame_count)

    # 关闭视频文件
    cap.release()

# 释放所有窗口
cv2.destroyAllWindows()
