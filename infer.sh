
CUDA_VISIBLE_DEVICES=2 animatediff generate -c config.json \
-W 384 -H 512 -L 16 -C 16 -o stylize/replace_bg_video \
--edit 0 --inverse 0 --motion_step 0 \
# -p1 ""


