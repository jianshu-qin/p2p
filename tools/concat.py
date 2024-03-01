import os
import shutil

dst = "../stylize/course3_1227/seg/total"
os.makedirs(dst, exist_ok=True)

src = "../stylize/course3_1227/seg/"

cnt = 0
for i in range(20):
    src_i = src + str(i)
    img_dir = ""
    for fn in os.listdir(src_i):
        if fn.startswith("2023"):
            img_dir = os.path.join(src_i, fn)
    for fn in sorted(os.listdir(img_dir)):
        # f1 = os.path.join(img_dir, fn)
        # f2 = os.path.join(dst, f"{cnt:08}.png")
        if fn.endswith(".gif"):
            f1 = os.path.join(img_dir, fn)
            f2 = os.path.join(dst, f"{cnt:02}.gif")
            shutil.copyfile(f1, f2)
            cnt += 1
