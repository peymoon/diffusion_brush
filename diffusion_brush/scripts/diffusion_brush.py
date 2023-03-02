from __future__ import annotations
from collections import defaultdict
import os
import re
import traceback
import copy
import gradio as gr
import modules.images as images
import modules.scripts as scripts
import torch
import numpy as np
from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
import open_clip.tokenizer
from modules import script_callbacks
from modules import script_callbacks, sd_hijack_clip, sd_hijack_open_clip
from modules.processing import (Processed, StableDiffusionProcessing, fix_seed,
                                process_images)
from modules.shared import cmd_opts, opts, state
import modules.shared as shared
from PIL import Image
from modules.sd_samplers_kdiffusion import KDiffusionSampler
from modules.sd_samplers import sample_to_image
orig_callback_state = KDiffusionSampler.callback_state
import modules.processing
from tqdm.auto import trange, tqdm
import k_diffusion.sampling
from k_diffusion.sampling import to_d
from modules import devices
from modules import prompt_parser
import json
from modules.ui_components import FormRow
import pandas as pd
import random
global mask
global start_noise
before_image_saved_handler = None
images_to_show = []
def slerp_with_mask(val, noise, subnoise):
    global mask
    global start_noise
    mask = mask.bool()
    mask[3,:,:] = mask[0,:,:]
    mask_float = torch.ones_like(noise)
    mask_float = mask_float * mask * 0.05
    start_noise = subnoise
    new_noise = mask_float * subnoise + (1 - mask_float) * noise
    low_norm = noise/torch.norm(noise, dim=1, keepdim=True)
    high_norm = new_noise/torch.norm(new_noise, dim=1, keepdim=True)
    return new_noise


def get_new_prompt(prompts, p):
    with devices.autocast():
        c = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, [prompts], p.steps)
        return c


# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp_no_mask(val, low, high):
    global start_noise
    global mask
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.ndim ==3:
        mask[3,:,:] = mask[0,:,:]
    start_noise = high
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

@torch.no_grad()
def sample_euler_with_mask(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    global mask
    global seednew
    global run_denosing_different_seed
    global intermediate_step_different_seed
    global apply_intermediate_denoising
    global intermediate_step
    global new_cond
    global new_pr
    global alpha
    global new_seed_list


    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

        if run_denosing_different_seed:
            # running denoising with a different seed
            extra_args2 = copy.deepcopy(extra_args)
            # if new_pr != '':
            #     extra_args2['cond'] = new_cond
            denoised2 = model(start_noise, sigma_hat * s_in, **extra_args2)
            d2 = to_d(start_noise, sigma_hat, denoised2)
            dt2 = sigmas[i + 1] - sigma_hat
            start_noise = start_noise + d2 * dt2

            # combine them after certain number of iterations
            if i >  intermediate_step_different_seed:
                x = mask * start_noise + mask.bitwise_not() * x
                start_noise = x
                if new_pr != '':
                    extra_args['cond'] = new_cond   
        
             
        denoised = model(x, sigma_hat * s_in, **extra_args)

        if apply_intermediate_denoising:
            previous_denoised = denoised
            for j in range(len(step_list)):
                step = int(step_list[j])
                if i == step:
                    masks = prepare_mask(masks_list)
                    denoinsed_masked = torch.zeros_like(denoised)
                    if mask_active_list[j]:
                        mask = masks_list[j]
                        mask = mask.bool()
                        mask[3,:,:] = mask[0,:,:]
                        mask = mask.unsqueeze(0)
                        masks = masks + mask
                        if new_seed_list[j] == -1:
                            new_seed_list[j] = int(random.randrange(4294967294))
                        start_noise = devices.randn(new_seed_list[j], [4,64,64]).unsqueeze(0)
                        start_noise = start_noise * sigmas[int(sigma_step_list[j])]
                        denoinsed_masked += (previous_denoised + 5* alpa_list[j] * start_noise) * mask
                
                        denoised = denoinsed_masked + masks.bitwise_not() * denoised

        
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x

@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

def prepare_masks(mask_list, active_mask_flag_list):
    masks = []
    for i in range(len(mask_list)):
        if mask_list[i] is None:
            masks.append(torch.zeros(4,64,64).cuda())
        else:
            resized_mask = mask_list[i]['mask'].resize((64,64), Image.NEAREST)
            masks.append(torch.from_numpy(np.array(resized_mask)).permute(2,0,1).cuda())
    return masks
def create_active_masks(mask_list, active_mask_flag_list):
    masks = []
    for i in range(len(mask_list)):
        if mask_list[i] is None:
            masks.append(torch.zeros(4,64,64).cuda())
        else:
            resized_mask = mask_list[i]['mask'].resize((64,64), Image.NEAREST)
            masks.append(torch.from_numpy(np.array(resized_mask)).permute(2,0,1).cuda())
   
    active_mask = torch.zeros_like(masks[0]).cuda()
    for i in range(len(masks)):
        active_mask = active_mask + active_mask_flag_list[i] * masks[i]
    
    return active_mask

def variable_outputs(k):
        max_textboxes = 10
        k = int(k)
        return [gr.Textbox.update(visible=True)]*k + [gr.Textbox.update(visible=False)]*(max_textboxes-k)
def prepare_mask(masks_list):
    masks = torch.zeros_like(masks_list[0])
    masks = masks.bool()
    masks[3,:,:] = masks[0,:,:]
    masks = masks.unsqueeze(0)
    return masks
def variable_outputs_tab(k):
        max_textboxes = 10
        k = int(k)
        return [gr.Textbox.update(visible=True)]*k + [gr.Textbox.update(visible=False)]*(max_textboxes-k)



class Script(scripts.Script):
    
    GRID_LAYOUT_AUTO = "Auto"
    GRID_LAYOUT_PREVENT_EMPTY = "Prevent Empty Spot"
    GRID_LAYOUT_BATCH_LENGTH_AS_ROW = "Batch Length As Row"
    
    def title(self):
        return "seed_mask"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        
        with gr.Row():
            is_active = gr.Checkbox(
                        label="seed_mask",
                        value=False)
            mask_intermediate = gr.Checkbox(
                        label="mask_intermediate",
                        value=False)
            

        with FormRow().style(equal_height=False):
            with gr.Column(variant='compact', elem_id="diffusion_brush_settings"):
                copy_image_buttons = []
                copy_image_destinations = {}

            def add_copy_image_controls(tab_name, elem):
                with gr.Row(variant="compact", elem_id=f"magic_brush_copy_to_{tab_name}"):
                    gr.HTML("Copy image to: ", elem_id=f"magic_brush_label_copy_to_{tab_name}")

                    for title, name in zip(['mask1', 'mask2', 'mask3'], ['mask1', 'mask2', 'mask3']):
                        if name == tab_name:
                            gr.Button(title, interactive=False)
                            copy_image_destinations[name] = elem
                            continue

                        button = gr.Button(title)
                        copy_image_buttons.append((button, name, elem))
        mask_list = []
        mask_active_list = []
        alpa_list = []
        step_list = []
        new_seed_list = []
        sigma_step_list = []
        with gr.Tabs(elem_id="magic_brush"):
            with gr.TabItem('mask1', id='mask1', elem_id="mask1_tab") as tab_mask1:
                with FormRow().style(equal_height=False):
                    enable_mask_1 = gr.Checkbox(label="enable mask 1",value=False)
                    mask_active_list.append(enable_mask_1.value)
                    alpha_1 = gr.Slider(minimum=0.0, maximum=5.0, step=0.01, value=0.52 ,label='alpha1', elem_id="alpha_1")
                    alpa_list.append(alpha_1)
                with FormRow().style(equal_height=False):
                    step1 = gr.Number(label='step1', placeholder="20")
                    step_list.append(step1)
                    new_seed_1 = gr.Number(label='new_seed_1', value=-1)
                    new_seed_list.append(new_seed_1)
                    sigma_step_input1 = gr.Number(label='sigma_step1', value=0)
                    sigma_step_list.append(sigma_step_input1)

                
                init_img_with_mask_1 = gr.Image(label="Image for brushing with mask1", show_label=False, elem_id="img2maskimg1", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA").style(height=480)
                add_copy_image_controls('mask1', init_img_with_mask_1)
                mask_list.append(init_img_with_mask_1)

            with gr.TabItem('mask2', id='mask2', elem_id="mask2_tab") as tab_mask2:
                with gr.Row():
                    enable_mask_2 = gr.Checkbox(label="enable mask 2",value=False)
                    mask_active_list.append(enable_mask_2.value)
                    alpha_2 = gr.Slider(label='alpha2', minimum=0.0, maximum=5.0, step=0.01, value=0.5, elem_id="alpha_2")
                    alpa_list.append(alpha_2)
                with gr.Row():
                    step2 = gr.Number(label='step2', placeholder="20")
                    step_list.append(step2)
                    new_seed_2 = gr.Number(label='new_seed 2', value=-1)
                    new_seed_list.append(new_seed_2)
                    sigma_step_input2 = gr.Number(label='sigma_step2', value=0)
                    sigma_step_list.append(sigma_step_input2)
                with gr.Row():
                    init_img_with_mask_2 = gr.Image(label="Image for brushing with mask2", show_label=False, elem_id="img2maskimg2", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA").style(height=480)
                add_copy_image_controls('mask2', init_img_with_mask_2)
                mask_list.append(init_img_with_mask_2)

            with gr.TabItem('mask3', id='mask3', elem_id="mask3_tab") as tab_mask3:
                with gr.Row():
                    enable_mask_3 = gr.Checkbox(label="enable mask 3",value=False)
                    mask_active_list.append(enable_mask_3.value)
                    alpha_3 = gr.Slider(label='alpha3', minimum=0.0, maximum=5.0, step=0.01, value=0.51, elem_id="alpha_3")
                    alpa_list.append(alpha_3)
                with gr.Row():
                    step3 = gr.Number(label='step3', placeholder="20")
                    step_list.append(step3)
                    new_seed_3 = gr.Number(label='new_seed 3', value=-1)
                    new_seed_list.append(new_seed_3)
                    sigma_step_input3 = gr.Number(label='sigma_step3', value=0)
                    sigma_step_list.append(sigma_step_input3)

                with gr.Row():
                    init_img_with_mask_3 = gr.Image(label="Image for brushing with mask3", show_label=False, elem_id="img2maskimg3", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA").style(height=480)
                add_copy_image_controls('mask3', init_img_with_mask_3)
                mask_list.append(init_img_with_mask_3)


            def copy_image(img):
                if isinstance(img, dict) and 'image' in img:
                    return img['image']

                return img

            for button, name, elem in copy_image_buttons:
                button.click(
                    fn=copy_image,
                    inputs=[elem],
                    outputs=[copy_image_destinations[name]],
                )
                button.click(
                    fn=lambda: None,
                    _js="switch_to_"+name.replace(" ", "_"),
                    inputs=[],
                    outputs=[],
                )

            def select_magic_brush_tab(tab):
                return gr.update(visible=tab in [2, 3, 4]), gr.update(visible=tab == 3),

            for i, elem in enumerate([tab_mask1, tab_mask2]):
                elem.select(
                    fn=lambda tab=i: select_magic_brush_tab(tab),
                    inputs=[],
                    outputs=[],
                )
        mask_dict = {}
        mask_dict['mask'] = mask_list
        mask_dict['is_active'] = mask_active_list
        mask_dict['alpha'] = alpa_list
        mask_dict['step'] = step_list

        with gr.Row():
            run_denosing_different_seed_input = gr.Checkbox(label="run denosing_different seed",value=False, visible=False)
            intermediate_step_different_seed_input = gr.Number(label='intermediate step different seed', placeholder="10", visible=False)
            new_seed_intermediate = gr.Number(label='new_seed', value=-1, visible=False)
            new_prompt = gr.Textbox(label='new prompt', placeholder="new prompt to use", visible=False)
        with gr.Row():
            apply_intermediate_denoising_input = gr.Checkbox(label="apply intermediate denoising",value=False)
            intermediate_step_input = gr.Number(label='intermediate step', placeholder="20", visible=False)
            alpha_input = gr.Slider(label='alpha', min=0, max=1, step=0.1, default=0.0, visible=False)
    

        return [is_active, mask_intermediate, new_seed_intermediate, new_prompt, run_denosing_different_seed_input, intermediate_step_different_seed_input, apply_intermediate_denoising_input, intermediate_step_input, alpha_input,
                init_img_with_mask_1, init_img_with_mask_2, init_img_with_mask_3,
                enable_mask_1, enable_mask_2, enable_mask_3,
                step1, step2, step3,
                alpha_1, alpha_2, alpha_3,
                new_seed_1, new_seed_2, new_seed_3,
                sigma_step_input1, sigma_step_input2, sigma_step_input3]
    
    def process(self,p, is_active, mask_intermediate, new_seed, new_prompt, run_denosing_different_seed_input, intermediate_step_different_seed_input, apply_intermediate_denoising_input, intermediate_step_input, alpha_input,
                init_img_with_mask_1, init_img_with_mask_2, init_img_with_mask_3,
                enable_mask_1, enable_mask_2, enable_mask_3,
                step1, step2, step3,
                alpha_1, alpha_2, alpha_3,
                new_seed_1, new_seed_2, new_seed_3, 
                sigma_step_input1, sigma_step_input2, sigma_step_input3):
        
        global mask
        global start_noise
        global seednew 
        global new_cond
        global run_denosing_different_seed
        global intermediate_step_different_seed
        global apply_intermediate_denoising
        global intermediate_step
        global new_pr 
        global alpha
        
        global masks_list
        global mask_active_list
        global step_list
        global alpa_list
        global new_seed_list
        global sigma_step_list
        sigma_step = sigma_step_input1
        new_pr = new_prompt
        run_denosing_different_seed = run_denosing_different_seed_input
        intermediate_step_different_seed = intermediate_step_different_seed_input
        apply_intermediate_denoising = mask_intermediate
        intermediate_step = intermediate_step_input
        seednew = new_seed
        alpha = alpha_input
          
        mask = torch.zeros(4,64,64).cuda()
        new_cond = get_new_prompt(new_prompt, p)

        masks_list = [init_img_with_mask_1, init_img_with_mask_2, init_img_with_mask_3]
        mask_active_list = [enable_mask_1, enable_mask_2, enable_mask_3]
        step_list = [step1, step2, step3]
        alpa_list = [alpha_1, alpha_2, alpha_3]
        new_seed_list = [new_seed_1, new_seed_2, new_seed_3]
        sigma_step_list = [sigma_step_input1, sigma_step_input2, sigma_step_input3]
        masks_list = prepare_masks(masks_list, mask_active_list)

        if is_active:
            resized_mask = mask_dict['mask'].resize((64,64), Image.NEAREST)
            mask = torch.from_numpy(np.array(resized_mask)).permute(2,0,1).cuda()
            modules.processing.slerp = slerp_with_mask
            if mask_intermediate:
                k_diffusion.sampling.sample_euler = sample_euler_with_mask
            else:
                k_diffusion.sampling.sample_euler = sample_euler
        else:
            modules.processing.slerp = slerp_no_mask
            
            if mask_intermediate:
                k_diffusion.sampling.sample_euler = sample_euler_with_mask
            else:
                k_diffusion.sampling.sample_euler = sample_euler

# to save:
# c = mask.squeeze(0).permute(2,1,0).cpu().numpy().astype('uint8')
# c = Image.fromarray(c)
# c.save('test.png')
