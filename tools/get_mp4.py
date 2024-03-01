import os

import cv2
import natsort


def concatenate_images_to_video(folder1, folder2, output_video, length=120, fps=10):
    fns1 = natsort.natsorted(os.listdir(folder1))
    fns2 = natsort.natsorted(os.listdir(folder2))

    img = cv2.imread(os.path.join(folder1, fns1[0]))
    height, width, layers = img.shape
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width*2, height))
    cnt = 0
    for f1, f2 in zip(fns1, fns2):
        assert f1 == f2
        img1 = cv2.imread(os.path.join(folder1, f1))
        img2 = cv2.imread(os.path.join(folder2, f2))
        img = cv2.hconcat([img1, img2])
        video.write(img)
        cnt += 1
        if cnt == length:
            break

    cv2.destroyAllWindows()
    video.release()
    print(f"Video saved as {output_video}, total {length} frames at fps {fps}")



def images_to_video(input_folder, output_video, length=120, fps=24):
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

    if not image_files:
        print(f"No PNG images found in {input_folder}")
        return

    img = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, layers = img.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    cnt = 0
    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)
        video.write(img)
        cnt += 1
        if cnt == length:
            break

    cv2.destroyAllWindows()
    video.release()
    print(f"Video saved as {output_video}")

# 示例
input_folder = '/home/jianshu/code/prompt_travel/stylize/course3_0118/2024-01-17T19-49-34-sample-cyberrealistic_v33/00-8888'
output_video = '/home/jianshu/code/prompt_travel/stylize/course3_0118/10-22s_ori.mp4'
fps = 10
folder1 = "/raid/cvg_data/lurenjie/animatediff-cli-prompt-travel/stylize/mp4_kemu3girl/00_tmp_controlnet_image/controlnet_openpose"
folder2 = "/raid/cvg_data/lurenjie/animatediff-cli-prompt-travel/stylize/mp4_kemu3girl/2024-01-17T20-51-48_00/00-7404840025117271001"

images_to_video(folder1, output_video, fps=fps)
# concatenate_images_to_video(folder1, folder2, output_video, fps=10)
