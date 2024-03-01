import os

import cv2


def d2m(in_img, out_img):
# in_img = "../stylize/warp_test/50_densepose_384.png"
# out_img = "../stylize/warp_test/50_densepose_384_mask.png"
    img = cv2.imread(in_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(out_img, img)

in_dir = "/home/jianshu/code/prompt_travel/stylize/course3_1227/2023-12-26T21-56-18-sample-cyberrealistic_v33/00_detectmap/controlnet_densepose"
out_dir = "/home/jianshu/code/prompt_travel/stylize/replace_bg_video"
os.makedirs(out_dir, exist_ok=True)
for fn in os.listdir(in_dir):
    in_img = os.path.join(in_dir, fn)
    out_img = os.path.join(out_dir, fn)
    d2m(in_img, out_img)

