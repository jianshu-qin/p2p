import os
import shutil

dst = "../stylize/course3_1227/seg"
os.makedirs(dst, exist_ok=True)

src = "/home/jianshu/data/course3_1227/fps_10_densepose"
num = 16

fns = sorted(os.listdir(src))
l = len(fns)
i = 0

while 1:
    dst_i = os.path.join(dst, str(i), "00_controlnet_image/controlnet_densepose")
    os.makedirs(dst_i)
    for j in range(i*num, min(i*num+num, l)):
        fn = fns[j]
        shutil.copyfile(os.path.join(src, fn), os.path.join(dst_i, f'{j-i*num:08}.png'))
    i += 1
    if i*num >= l:
        break
