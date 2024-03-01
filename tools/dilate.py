import os

import cv2
import numpy as np


def dilate(img, out_img):
# 读取图像
    image = cv2.imread(img)

    # 定义膨胀核（kernel）
    kernel = np.ones((5,5), np.uint8)  # 5x5的全白方块，用于膨胀操作

    # 使用cv2.dilate进行膨胀操作
    dilated_image = cv2.dilate(image, kernel, iterations=2)
    cv2.imwrite(out_img, dilated_image)

in_dir = "../stylize/replace_bg_video/mask"
out_dir = "../stylize/replace_bg_video/mask_dilate"
os.makedirs(out_dir, exist_ok=True)
for fn in os.listdir(in_dir):
    in_fn = os.path.join(in_dir, fn)
    out_fn = os.path.join(out_dir, fn)
    dilate(in_fn, out_fn)
