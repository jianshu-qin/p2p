import cv2

def cut_video(input_file, output_file):
    # 读取视频文件
    video = cv2.VideoCapture(input_file)

    # 获取视频的帧率和总帧数
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算前16秒的帧数范围
    start_frame = fps * 20
    end_frame = fps * 30

    # 设置输出视频的编码器和参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 逐帧读取视频并写入输出文件
    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break
        if i >= start_frame and i <= end_frame:
            out.write(frame)

    # 释放资源
    video.release()
    out.release()

# 调用函数，传入输入文件和输出文件路径
cut_video('/raid/cvg_data/lurenjie/animatediff-cli-prompt-travel/projects/speech/speach.mp4', '/raid/cvg_data/lurenjie/animatediff-cli-prompt-travel/projects/speech/speach_10.mp4')
