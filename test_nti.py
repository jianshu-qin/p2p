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
args = parser.parse_args()

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

LOW_RESOURCE = False
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
MODEL = "runwayml/stable-diffusion-v1-5"
height = width = 512
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
sample_steps = 20
os.makedirs("p2p_vis", exist_ok=True)

pipe = p2p.pipeline.ReconstructStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=scheduler).to(device)
pipe.disable_xformers_memory_efficient_attention()
null_inversion = p2p.nti.NullInversion(pipe, num_ddim_steps=sample_steps)

# real image path and its prompt
image_path = "./gnochi_mirror.jpeg"
prompt = "a cat sitting next to a mirror"

if args.inverse:
        (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,200,0), verbose=True)
        torch.save(x_t, "xt_cat_20.pt")
        torch.save(uncond_embeddings, "unc_cat_20.pt")
x_t, uncond_embeddings = torch.load("xt_cat_20.pt"), torch.load("unc_cat_20.pt")
x_t = x_t.expand(1, pipe.unet.in_channels, height // 8, width // 8).to(device)

print("null-text inversion complete!")

prompts = [prompt]
# original prompt and new prompt
prompts = ["a cat sitting next to a mirror",
           "a cat sitting next to a mirror"
        ]
store_controller = p2p.p2p.AttentionStore(pipe.tokenizer)
p2p.ptp_utils.register_attention_control(pipe, store_controller)
images = pipe(prompt=prompts[0], latents=x_t, num_inference_steps=sample_steps, uncond_embeddings=uncond_embeddings).images
images[0].save("p2p_vis/rec_cat.png")
p2p.p2p.show_cross_attention(prompts[:1], store_controller, 16, ["up", "down"], select=0)


# replace_controller = p2p.pipeline.AttentionReplace(store_controller.all_attn_store, pipe.tokenizer,prompts, sample_steps, cross_replace_steps=.8, self_replace_steps=0.4, local_blend=None, device=device)
# p2p.ptp_utils.register_attention_control(pipe, replace_controller)
# images = pipe(prompt=prompts[1], latents=x_t, num_inference_steps=sample_steps, uncond_embeddings=uncond_embeddings).images
# images[0].save("p2p_vis/rec_cat2.png")
# p2p.p2p.show_cross_attention(prompts[1:], replace_controller, 16, ["up", "down"], select=0)

if args.edit:
        prompts[1] = "a tiger sitting next to a mirror"
        blend_word = ((('cat',), ("tiger",)))
        eq_params = {"words": ("tiger",), "values": (2,)}
        lb = p2p.p2p.LocalBlend(pipe.tokenizer, sample_steps, prompts, blend_word)
        replace_controller = p2p.p2p.AttentionReplace(store_controller.all_attn_store, pipe.tokenizer,prompts, sample_steps, cross_replace_steps=.8, self_replace_steps=0.4, local_blend=lb, device=device)
        p2p.ptp_utils.register_attention_control(pipe, replace_controller)
        images = pipe(prompt=prompts[1], latents=x_t, num_inference_steps=sample_steps, uncond_embeddings=uncond_embeddings, controller=replace_controller).images
        images[0].save("p2p_vis/rec_tiger.png")
        p2p.p2p.show_cross_attention(prompts[1:], replace_controller, 16, ["up", "down"], select=0)


