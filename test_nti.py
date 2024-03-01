import argparse
import os

import torch
from diffusers import (ControlNetModel, DDIMScheduler,
                       StableDiffusionControlNetPipeline,
                       StableDiffusionPipeline)
from PIL import Image

import p2p.nti
import p2p.pipeline
import p2p.ptp_utils
import p2p.seq_aligner

parser = argparse.ArgumentParser()
parser.add_argument('--inverse', default=1, type=int ,help='whether perform null-text inversion')
parser.add_argument('--edit', default=0, type=int, help='whether perform p2p edit')
parser.add_argument('--image_path', default="ref_image.png", type=str ,help='image to be reconstructed')
parser.add_argument('--prompts', default="a girl in beige top, black skirt, black boots, stands still with arms hanging straight on her sides", type=str ,help='prompts in reconstrution')
parser.add_argument('--height', default=1024, type=int ,help='')
parser.add_argument('--width', default=768, type=int ,help='')
parser.add_argument('--sample_steps', default=50, type=int ,help='sample steps when reconstrution')
parser.add_argument('--inner_steps', default=50, type=int ,help="iter steps for every sample step's optimization")
parser.add_argument('--save_dir', default="xt_unc", type=str ,help='path to save xt, unc and latents')
args = parser.parse_args()

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

LOW_RESOURCE = False
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
# MODEL = "runwayml/stable-diffusion-v1-5"
MODEL = "/home/jianshu/code/prompt_travel/data/models/sd/cyberrealistic_v33.safetensors"
height = args.height
width = args.width
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
sample_steps = args.sample_steps
os.makedirs(args.save_dir, exist_ok=True)

pipe = p2p.pipeline.ReconstructStableDiffusionPipeline.from_single_file(MODEL,use_safetensors=True).to(device)
pipe.scheduler = scheduler
pipe.disable_xformers_memory_efficient_attention()
pipe.safety_checker = None
null_inversion = p2p.nti.NullInversion(pipe, num_ddim_steps=sample_steps)

# real image path and its prompt
image_path = args.image_path
prompt = args.prompts
if args.inverse:
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,200,0), num_inner_steps=args.inner_steps, verbose=True, resize=False)
    torch.save(x_t, f"{args.save_dir}/xt_girl_{width}_{height}_{sample_steps}.pt")
    torch.save(uncond_embeddings, f"{args.save_dir}/unc_girl_{width}_{height}_{sample_steps}.pt")
x_t, uncond_embeddings = torch.load(f"{args.save_dir}/xt_girl_{width}_{height}_{sample_steps}.pt"), torch.load(f"{args.save_dir}/unc_girl_{width}_{height}_{sample_steps}.pt")

x_t = x_t.expand(1, pipe.unet.in_channels, height // 8, width // 8).to(device)

print("null-text inversion complete!")

prompts = [prompt]
# original prompt and new prompt
prompts = ["a girl in beige top, black skirt, black boots, stands still with arms hanging straight on her sides",
           "a boy in beige top, black skirt, pink boots, stands still with arms hanging straight on her sidesr"
        ]
store_controller = p2p.p2p.AttentionStore(pipe.tokenizer)
if args.edit:
    p2p.ptp_utils.register_attention_control(pipe, store_controller)
images, all_step_latents = pipe(prompt=prompts[0], latents=x_t, height=height, width=width, num_inference_steps=sample_steps, uncond_embeddings=uncond_embeddings, num_inverse_steps=sample_steps)
images[0].save("rec_girl_20steps_inverse20.png")
torch.save(all_step_latents, f"{args.save_dir}/rec_all_steps_latents_{width}_{height}_{sample_steps}.pt")

all_step_imgs = []
with torch.no_grad():
    for x in all_step_latents[::2]:
        all_step_imgs.append(pipe.vae.decode(x / pipe.vae.config.scaling_factor, return_dict=False)[0])
all_step_imgs = torch.cat(all_step_imgs)
do_denormalize = [True] * all_step_imgs.shape[0]
all_step_imgs = pipe.image_processor.postprocess(all_step_imgs, output_type="pil", do_denormalize=do_denormalize)
all_step_imgs = p2p.ptp_utils.merge_images(all_step_imgs)
all_step_imgs.save("rec_girl_all_steps.png")

# replace_controller = p2p.pipeline.AttentionReplace(store_controller.all_attn_store, pipe.tokenizer,prompts, sample_steps, cross_replace_steps=.8, self_replace_steps=0.4, local_blend=None, device=device)
# p2p.ptp_utils.register_attention_control(pipe, replace_controller)
# images = pipe(prompt=prompts[1], latents=x_t, num_inference_steps=sample_steps, uncond_embeddings=uncond_embeddings).images
# images[0].save("p2p_vis/rec_cat2.png")
# p2p.p2p.show_cross_attention(prompts[1:], replace_controller, 16, ["up", "down"], select=0)

if args.edit:
    # blend_word = ((('cat',), ("dog",)))
    # eq_params = {"words": ("tiger",), "values": (2,)}
    # lb = p2p.p2p.LocalBlend(pipe.tokenizer, sample_steps, prompts, blend_word)
    lb = None
    replace_controller = p2p.p2p.AttentionReplace(store_controller.all_attn_store, pipe.tokenizer,prompts, sample_steps, cross_replace_steps=.8, self_replace_steps=0.4, local_blend=lb, device=device)

    p2p.ptp_utils.register_attention_control(pipe, replace_controller)
    images, _ = pipe(prompt=prompts[1], latents=x_t, num_inference_steps=sample_steps, uncond_embeddings=uncond_embeddings, controller=replace_controller, ref_latents=all_step_latents)
    images[0].save("p2p_vis/rec_boy.png")
    p2p.p2p.show_cross_attention(prompts[1:], replace_controller, 16, ["up", "down"], select=0)
    replace_controller.attention_store = replace_controller.ref_attention_store
    p2p.p2p.show_cross_attention(prompts[1:], replace_controller, 16, ["up", "down"], select=0)


