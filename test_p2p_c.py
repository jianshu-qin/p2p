import os
import time

import torch
from diffusers import (ControlNetModel, DDIMScheduler,
                       StableDiffusionControlNetPipeline,
                       StableDiffusionPipeline)
from PIL import Image

import p2p.nti
import p2p.p2p
import p2p.pipeline
import p2p.ptp_utils
import p2p.seq_aligner

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
MODEL = "runwayml/stable-diffusion-v1-5"
MODEL2 = "/home/jianshu/code/prompt_travel/data/models/sd/cyberrealistic_v33.safetensors"
dtype = torch.float32
out_dir = "./stylize/warp_test/output2"
os.makedirs(out_dir, exist_ok=True)

controlnet_path = "/home/jianshu/code/sd/controlnet/densepose/checkpoint-17000/controlnet"
controlnet_path = "lllyasviel/control_v11p_sd15_openpose"
img_path = "stylize/warp_test/0_openpose_768.png"
controlnet = ControlNetModel.from_pretrained(controlnet_path, load_files_only=True, torch_dtype=dtype)

# pipe = p2p.pipeline.ReconstructStableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=dtype).to(device)
pipe = p2p.pipeline.ReconstructStableDiffusionPipeline.from_single_file(MODEL2, use_safetensors=True, torch_dtype=dtype)
unet_state_dict = pipe.unet.state_dict()
tenc_state_dict = pipe.text_encoder.state_dict()
vae_state_dict = pipe.vae.state_dict()
# pipe.unet.load_state_dict(unet_state_dict, strict=False)
# pipe.unet.load_state_dict(tenc_state_dict, strict=False)
# pipe.unet.load_state_dict(vae_state_dict, strict=False)
pipe.scheduler = scheduler
tokenizer = pipe.tokenizer
height, width = 1024, 768
sample_steps = 50

g_cpu = torch.Generator().manual_seed(8888)

bg_prompts = " a photo of a living room, 4k, HD"
prompts = [
    "a photo of a girl in white tank-top, shorts",
    "a photo of a girl in white tank-top, shorts"
]
prompts = ["a girl in beige top, black skirt, black boots, stands still with arms hanging straight on her sides",
           "a girl in beige top, black skirt, black boots"
        ]

pipe.disable_xformers_memory_efficient_attention()
pipe.safety_checker = None
pipe = pipe.to(device)
latent = torch.randn(
    (1, pipe.unet.in_channels, height // 8, width // 8),
    generator=g_cpu,
    dtype=dtype
)
latents = latent.expand(1, pipe.unet.in_channels, height // 8, width // 8).to(device)
show_img = []

x_t, uncond_embeddings = torch.load(f"xt_unc/xt_girl_{width}_{height}_{sample_steps}.pt"), torch.load(f"xt_unc/unc_girl_{width}_{height}_{sample_steps}.pt")
x_t = x_t.expand(1, pipe.unet.in_channels, height // 8, width // 8).to(device)
latents = x_t
# images, bg_latents = pipe(
#     prompt=bg_prompts,
#     height=height,
#     width=width,
#     latents=latents,
#     num_inference_steps=sample_steps,
# )

# show_img.append(images[0])

# pipe = p2p.pipeline.ReconstructStableDiffusionControlNetPipeline.from_pretrained(MODEL, controlnet=controlnet,torch_dtype=dtype).to(device)
# pipe.unet.load_state_dict(unet_state_dict, strict=False)
# pipe.unet.load_state_dict(tenc_state_dict, strict=False)
# pipe.unet.load_state_dict(vae_state_dict, strict=False)
store_controller = p2p.p2p.AttentionStore(tokenizer, device)
control_image = Image.open(img_path)

# p2p.ptp_utils.register_attention_control(pipe, store_controller)

images, fg_latents = pipe(
    prompt=prompts[0],
    image=control_image,
    height=height,
    width=width,
    latents=latents,
    num_inference_steps=sample_steps,
    uncond_embeddings=uncond_embeddings,
    num_inverse_steps=sample_steps,
)
images[0].save(os.path.join(out_dir, "0.png"))
torch.save(fg_latents, "rec_girl_fg_latents.pt")
exit()
show_img.append(images[0])

# p2p.p2p.show_cross_attention(prompts, store_controller, res=16, from_where=["up", "down"], select=1)

pipe = p2p.pipeline.ReconstructStableDiffusionControlNetPipeline.from_pretrained(MODEL, controlnet=controlnet,torch_dtype=dtype).to(device)
pipe.unet.load_state_dict(unet_state_dict, strict=False)
pipe.unet.load_state_dict(tenc_state_dict, strict=False)
pipe.unet.load_state_dict(vae_state_dict, strict=False)
pipe.scheduler = scheduler
pipe.safety_checker = None
latents = latent.expand(len(prompts)-1, pipe.unet.in_channels, height // 8, width // 8).to(device)
lb = p2p.p2p.LocalBlend(pipe.tokenizer, ddim_steps=sample_steps, prompts=prompts[1:]*2,
                        words=(("girl", ), ("girl", )))
replace_background_steps = [0, sample_steps//2]
replace_controller = p2p.p2p.AttentionReplace(
    store_controller.all_attn_store,
    tokenizer,prompts[1:]*2, sample_steps,
    cross_replace_steps=0.8,
    self_replace_steps=0.4,
    local_blend=lb,
    device=device,
)

control_image = Image.open("stylize/warp_test/50_openpose_768.png")
# p2p.ptp_utils.register_attention_control(pipe, replace_controller)
images, _ = pipe(
    prompt=prompts[1:],
    image=control_image,
    height=height,
    width=width,
    latents=latents,
    num_inference_steps=sample_steps,
    controller=replace_controller,
    #ref_bg_latents=bg_latents,
    #replace_background_steps=replace_background_steps,
    ref_fg_latents=fg_latents,
    warp_belnd_steps=[0, sample_steps//2],
)
images[0].save(os.path.join(out_dir, "50.png"))
show_img.append(images[0])

images, all_latents = pipe(
    prompt=prompts[1:],
    image=control_image,
    height=height,
    width=width,
    latents=latents,
    num_inference_steps=sample_steps,
    controller=replace_controller,
    #ref_bg_latents=bg_latents,
    #replace_background_steps=replace_background_steps,
    ref_fg_latents=fg_latents,
    use_warp_blend=True,
    warp_belnd_steps=[0, sample_steps//2],
)
images[0].save(os.path.join(out_dir, "50_warp.png"))
show_img.append(images[0])

images, all_latents = pipe(
    prompt=prompts[1:],
    image=control_image,
    height=height,
    width=width,
    latents=latents,
    num_inference_steps=sample_steps,
    controller=replace_controller,
    #ref_bg_latents=bg_latents,
    #replace_background_steps=replace_background_steps,
    ref_fg_latents=fg_latents,
    use_warp_blend=True,
    warp_belnd_steps=[0, sample_steps],
)
images[0].save(os.path.join(out_dir, "50_warp1.png"))
show_img.append(images[0])

# replace_background_steps = [0, sample_steps-30]
# replace_controller = p2p.p2p.AttentionReplace(
#     store_controller.all_attn_store,
#     tokenizer,prompts, sample_steps,
#     cross_replace_steps=0.8,
#     self_replace_steps=0.4,
#     local_blend=lb,
#     device=device,
# )
# control_image = Image.open("stylize/warp_test/50_openpose_384.png")
# # p2p.ptp_utils.register_attention_control(pipe, replace_controller)
# images, all_latents = pipe(
#     prompt=prompts[1:],
#     image=control_image,
#     height=height,
#     width=width,
#     latents=latents,
#     num_inference_steps=sample_steps,
#     controller=replace_controller,
#     ref_bg_latents=bg_latents,
#     replace_background_steps=replace_background_steps
# )
# images[0].save(os.path.join(out_dir, "50.png"))
# show_img.append(images[0])

# replace_background_steps = [0, sample_steps-20]
# replace_controller = p2p.p2p.AttentionReplace(
#     store_controller.all_attn_store,
#     tokenizer,prompts, sample_steps,
#     cross_replace_steps=0.8,
#     self_replace_steps=0.4,
#     local_blend=lb,
#     device=device,
# )
# control_image = Image.open("stylize/warp_test/50_openpose_384.png")
# # p2p.ptp_utils.register_attention_control(pipe, replace_controller)
# images, all_latents = pipe(
#     prompt=prompts[1:],
#     image=control_image,
#     height=height,
#     width=width,
#     latents=latents,
#     num_inference_steps=sample_steps,
#     controller=replace_controller,
#     ref_bg_latents=bg_latents,
#     replace_background_steps=replace_background_steps
# )
# images[0].save(os.path.join(out_dir, "50.png"))
# show_img.append(images[0])

# replace_background_steps = [0, sample_steps-10]
# replace_controller = p2p.p2p.AttentionReplace(
#     store_controller.all_attn_store,
#     tokenizer,prompts, sample_steps,
#     cross_replace_steps=0.8,
#     self_replace_steps=0.4,
#     local_blend=lb,
#     device=device,
# )
# control_image = Image.open("stylize/warp_test/50_openpose_384.png")
# # p2p.ptp_utils.register_attention_control(pipe, replace_controller)
# images, all_latents = pipe(
#     prompt=prompts[1:],
#     image=control_image,
#     height=height,
#     width=width,
#     latents=latents,
#     num_inference_steps=sample_steps,
#     controller=replace_controller,
#     ref_bg_latents=bg_latents,
#     replace_background_steps=replace_background_steps
# )
# images[0].save(os.path.join(out_dir, "50.png"))
# show_img.append(images[0])

# replace_background_steps = [0, sample_steps]
# replace_controller = p2p.p2p.AttentionReplace(
#     store_controller.all_attn_store,
#     tokenizer,prompts, sample_steps,
#     cross_replace_steps=0.8,
#     self_replace_steps=0.4,
#     local_blend=lb,
#     device=device,
# )
# control_image = Image.open("stylize/warp_test/50_openpose_384.png")
# # p2p.ptp_utils.register_attention_control(pipe, replace_controller)
# images, all_latents = pipe(
#     prompt=prompts[1:],
#     image=control_image,
#     height=height,
#     width=width,
#     latents=latents,
#     num_inference_steps=sample_steps,
#     controller=replace_controller,
#     ref_bg_latents=bg_latents,
#     replace_background_steps=replace_background_steps
# )
# images[0].save(os.path.join(out_dir, "50.png"))
# show_img.append(images[0])



img = p2p.ptp_utils.merge_images(show_img)
img.save(os.path.join(out_dir, "show_replace_fg.png"))




