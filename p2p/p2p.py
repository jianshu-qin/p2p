import abc
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as nnf
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       StableDiffusionPipeline)
from PIL import Image

from . import ptp_utils, seq_aligner

LOW_RESOURCE = False
MAX_NUM_WORDS = 77
device = torch.device("cuda:0")

class LocalBlend:

    def get_mask(self, x_t, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask

    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:

            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1).to(self.alpha_layers.device)
            mask = self.get_mask(x_t, maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(x_t, maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, tokenizer, ddim_steps, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1

        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers[1:].to(device)
        self.start_blend = int(start_blend * ddim_steps)
        self.counter = 0
        self.th=th

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @staticmethod
    def init_place_layers():
        return {"down_cross": 0, "mid_cross": 0, "up_cross": 0,
                "down_self": 0,  "mid_self": 0,  "up_self": 0}

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):

        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                # only modify condition part
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.place_layers[key] += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
            self.place_layers = self.init_place_layers()
        return attn

    def __init__(self):
        self.cur_step = 0
        # init in register_attention_control()
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.place_layers = self.init_place_layers()

class EmptyControl(AttentionControl):

    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # print(key, ":", attn.shape)
        if attn.shape[1] <= 64 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn.cpu())
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        if self.is_store:
            self.all_attn_store.append(self.step_store.copy())
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def __init__(self, tokenizer, device=0, is_store=[1]):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.tokenizer = tokenizer
        self.device = device
        self.all_attn_store = []
        self.is_store = is_store


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    if self.is_store:
                        self.attention_store[key][i] += self.step_store[key][i]
                    elif 0 <= self.cur_step-1 < len(self.attn_store):
                        self.attention_store[key][i] += self.attn_store[self.cur_step-1][key][i]
                    else:
                        print(f"wrong steps in {self.cur_step}")
        if self.is_store:
            self.all_attn_store.append(self.step_store.copy())
        self.step_store = self.get_empty_store()

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        attn_base = self.attn_store[self.cur_step][key][self.place_layers[key]].to(attn.device)
        # if torch.equal(attn, attn_base):
        #     print("yes", key, self.cur_step, self.place_layers[key])
        # else:
        #     print(f"no!", key, self.cur_step, self.place_layers[key])
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if self.cur_step == 0:
            return attn
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_replace = attn[:]
            # key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            # attn_base = self.attn_store[self.cur_step][key][self.place_layers[key]].to(attn.device)
            #print(attn.shape, attn_base.shape, key)
            # attn_base = attn_base.reshape(self.batch_size, h, *attn_base.shape[1:])
            # print(attn.shape, attn_base.shape, attn_replace.shape)

            if is_cross:
                # if self.cur_step == 51:
                #     self.cur_step = len(self.cross_replace_alpha)-1
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_replace) * alpha_words + (1 - alpha_words) * attn_replace
                attn = attn_repalce_new
                #print("cross")
                # attn = attn_base
            else:
                #print("self")
                # attn = attn_base
                attn = self.replace_self_attention(attn_base, attn_replace)

            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, attn_store, tokenizer, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend], device: int=0, is_store: List[int]=None):
        super(AttentionControlEdit, self).__init__(tokenizer, device, is_store)
        self.device = device
        self.batch_size = len(prompts) - 1
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        self.attn_store = attn_store
        self.is_store = is_store

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        self.mapper = self.mapper.to(attn_base.dtype)
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, attn_store, tokenizer, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, device: int=0, is_store: List[int]=None):
        super(AttentionReplace, self).__init__(attn_store, tokenizer, prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend,device, is_store)
        self.device = device
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)



class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, tokenizer, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, tokenizer, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(tokenizer, text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer

def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):

    attention_maps = attention_store.get_average_attention()
    frames = 16
    for item in attention_maps[f"{from_where[0]}_{'cross' if is_cross else 'self'}"]:
        frames = item.shape[0]//8
        break
    out = [[] for _ in range(frames)]
    num_pixels = res ** 2
    print(prompts)
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            # print(item.shape)
            #
            if item.shape[1] == num_pixels:
                for i, item_frame in enumerate(torch.chunk(item, frames, dim=0)):
                    # print("item_frame shape: ", item_frame.shape)
                    cross_maps = item_frame.reshape(len(prompts), -1, res, res, item_frame.shape[-1])[select]
                    # print(cross_maps.shape)
                    out[i].append(cross_maps)
    for i, out_frame in enumerate(out):
        out_frame = torch.cat(out_frame, dim=0)
        out_frame = out_frame.sum(0) / out_frame.shape[0]
        out_frame = out_frame.cpu()
        out[i] = out_frame
    # out = torch.cat(out, dim=0)
    # out = out.sum(0) / out.shape[0]
    return out


def show_cross_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, save_dir="./p2p_vis"):
    tokens = attention_store.tokenizer.encode(prompts[select])
    decoder = attention_store.tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    total_images = []
    for attention_maps_frame in attention_maps:
        for i in range(len(tokens)+5):
            image = attention_maps_frame[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            if i < len(tokens):
                image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            else:
                continue
                # image = ptp_utils.text_under_image(image, "paddings")
            images.append(image)
        total_images.append(np.stack(images, axis=0))
        images = []
    vis_gif = []
    save_dir = os.path.join(save_dir, f"p2p_vis_{prompts[select][:30]}")
    os.makedirs(save_dir, exist_ok=True)
    for img in total_images:
        vis_gif.append(ptp_utils.view_images(img, save_dir=save_dir, save_name=f"p2p_vis_{prompts[select][:30]}.png"))
    imageio.mimsave(f"{save_dir}/{prompts[select][:30]}.gif", vis_gif, fps=1)



def show_self_attention_comp(prompts, attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))

# def run_and_display(pipeline, prompts, controller, latent=None, run_baseline=False, generator=None):
#     if run_baseline:
#         print("w.o. prompt-to-prompt")
#         images, latent = run_and_display(pipeline, prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
#         print("with prompt-to-prompt")
#     images, x_t = ptp_utils.text2image_ldm_stable(pipeline, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE)
#     ptp_utils.view_images(images)
#     return images, x_t

# def run_and_display_controlnet(pipeline, img_path, prompts, controller, controlnet_controller, latent=None, run_baseline=False, generator=None):
#     if run_baseline:
#         print("w.o. prompt-to-prompt")
#         images, latent = run_and_display_controlnet(pipeline, img_path, prompts, EmptyControl(), EmptyControl(), latent=latent, run_baseline=False, generator=generator)
#         print("with prompt-to-prompt")
#     images, x_t = ptp_utils.controlnet_stablediffusion(pipeline, img_path, prompts, controller, controlnet_controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE)
#     ptp_utils.view_images(images)
#     return images, x_t
