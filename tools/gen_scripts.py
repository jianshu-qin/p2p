import os

folder = "/home/jianshu/code/prompt_travel/stylize/course3_1227/seg/"
dst1 = "/home/jianshu/code/prompt_travel/stylize/course3_1227/00_controlnet_image/controlnet_openpose"
dst2 = "/home/jianshu/code/prompt_travel/stylize/course3_1227/00_controlnet_image/controlnet_densepose"

with open("infer_seg2.sh", "w+") as f:
    for i in range(20):
        out_dir = folder + str(i)
        f.write(f"ln -s {out_dir}/00_controlnet_image/controlnet_openpose {dst1}\n")
        f.write(f"ln -s {out_dir}/00_controlnet_image/controlnet_densepose {dst2}\n")
        f.write(f"CUDA_VISIBLE_DEVICES=3 animatediff generate -c stylize/course3_1227/config.json -W 768 -H 1024 -L 16 -C 16 -o {out_dir}\n")
        f.write(f"unlink {dst1}\n")
        f.write(f"unlink {dst2}\n\n")

