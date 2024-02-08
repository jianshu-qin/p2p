The code should only requires diffusers==0.18.2.My enviroment is same as https://github.com/haohantianchen/Animate.

# usage
replace the files in `src` dir in https://github.com/haohantianchen/Animate   
copy the `p2p` dir into the Animate dir  
```
bash infer.sh
```
The default config will generate bg using prompts `a photo of a living room, 4k, HD`,and load fg latents from `video_fg_all_20steps.pt`, and blend them with bg weights `w1=[0.9,0.1]` and fg weights `w2=[0.5,0.1]`.

To modify blend steps:
```
cli.py, line 416
```
To modify bg prompts:
```
infer.sh -p1 "******"
```
To modify fg prompts:
```
config.json, line 17
```
To modify blend weights w1 or w2:
```
p2p.py, line 43
```
Visualize the intermediate noisy images:
```
cli.py, line 479
```