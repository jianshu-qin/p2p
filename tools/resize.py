import os

import cv2

input = "/home/jianshu/code/prompt_travel/stylize/warp_test/00000000.png"
output = "/home/jianshu/code/prompt_travel/stylize/warp_test/0_384.png"
size = (384, 512)
img = cv2.imread(input)
img = cv2.resize(img, size)
cv2.imwrite(output, img)
