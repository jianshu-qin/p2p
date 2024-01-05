import os
import time

import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       StableDiffusionPipeline)
from PIL import Image

import p2p.nti
import p2p.p2p
import p2p.pipeline
import p2p.ptp_utils
import p2p.seq_aligner

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
MODEL = "runwayml/stable-diffusion-v1-5"


pipe = StableDiffusionPipeline.from_pretrained(MODEL).to(device)
tokenizer = pipe.tokenizer
height, width = 512, 512
sample_steps = 20

g_cpu = torch.Generator().manual_seed(8888)
prompts = ["A painting of a squirrel eating a burger",
           "A painting of a lion eating a burger"]

store_controller = p2p.p2p.AttentionStore(tokenizer, device)

p2p.ptp_utils.register_attention_control(pipe, store_controller)

pipe.safety_checker = None
latent = torch.randn(
    (1, pipe.unet.in_channels, height // 8, width // 8),
    generator=g_cpu,
    # dtype=torch.float16
)
latents = latent.expand(1, pipe.unet.in_channels, height // 8, width // 8).to(device)
images = pipe(prompt=prompts[0], latents=latents, num_inference_steps=sample_steps).images
images[0].save("p2p_vis/squirrel.png")


replace_controller = p2p.p2p.AttentionReplace(store_controller.all_attn_store, tokenizer,prompts, sample_steps, cross_replace_steps=.8, self_replace_steps=0.4, local_blend=None, device=device)
p2p.ptp_utils.register_attention_control(pipe, replace_controller)
images = pipe(prompt=prompts[1], latents=latents, num_inference_steps=sample_steps).images
images[0].save("p2p_vis/lion.png")



# p2p.p2p.show_cross_attention(prompts, store_controller, res=16, from_where=("up", "down"), select=1, save_dir="./1109")
# p2p.p2p.show_cross_attention(prompts, controlnet_controller, res=16, from_where=["down"], select=1, save_dir="./1109")
