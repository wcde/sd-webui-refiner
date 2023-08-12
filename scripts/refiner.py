from math import ceil
import torch
from modules import scripts, script_callbacks, devices, sd_models, sd_models_config, shared
import gradio as gr
import sgm.modules.diffusionmodules.denoiser_scaling
import sgm.modules.diffusionmodules.discretizer
from sgm.modules.encoders.modules import ConcatTimestepEmbedderND
from safetensors.torch import load_file, load
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from omegaconf import OmegaConf
from sgm.util import (
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
)

class Refiner(scripts.Script):
    def __init__(self):
        super().__init__()
        self.callback_set = False
        self.model = None
        self.conditioner = None
        self.base = None
        self.swapped = False
        self.model_name = ''
        self.embedder = ConcatTimestepEmbedderND(256)
        self.c_ae = None
        self.uc_ae = None
        
    def title(self):
        return "Refiner"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def build_model(self):
        refiner_config = OmegaConf.load(sd_models_config.config_sdxl_refiner).model.params.network_config
        self.model = instantiate_from_config(refiner_config)
        self.model = get_obj_from_str(OPENAIUNETWRAPPER)(
            self.model, compile_model=False
        ).eval()
        self.model.to('cpu', devices.dtype_unet)
        self.model.train = disabled_train
        self.model.diffusion_model.dtype = devices.dtype_unet
        self.model.conditioning_key = 'crossattn'
        self.model.cond_stage_key = 'txt'
        self.model.parameterization = 'v'
        discretization = sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization()
        self.model.alphas_cumprod = torch.asarray(discretization.alphas_cumprod, device=devices.device, dtype=devices.dtype_unet)
        for param in self.model.parameters():
            param.requires_grad = False
    
    def load_model(self, model_name):
        if not shared.opts.disable_mmap_load_safetensors:
            ckpt = load_file(sd_models.checkpoints_list[model_name].filename)
        else:
            ckpt = load(open(sd_models.checkpoints_list[model_name].filename, 'rb').read())
        model_type = ''
        for key in ckpt.keys():
            if 'conditioner' in key: 
                model_type = 'Refiner'
            if 'input_blocks.7.1.transformer_blocks.4.attn1.to_k.weight' in key:
                model_type = 'Base'
                break
        if model_type != 'Refiner': 
            self.enable = False
            script_callbacks.remove_current_script_callbacks()
            if model_type == 'Base':
                print('\nIt\'s Base model, use Refiner, extension disabled!\n')
            else:
                print('\nNot refiner, extension disabled!\n')
            return False
        
        print('\nLoading refiner...\n')
        self.build_model()
            
        state_dict = dict()
        for key in ckpt.keys():
            if 'model.diffusion_model' in key:
                state_dict[key.replace('model.d', 'd')] = ckpt[key].half()
        self.model.load_state_dict(state_dict)
        self.model_name = model_name
        return True
        
    def ui(self, is_img2img):
        with gr.Accordion(label='Refiner', open=False):
            enable = gr.Checkbox(label='Enable Refiner', value=False)
            with gr.Row():
                checkpoint = gr.Dropdown(choices=['None', *sd_models.checkpoints_list.keys()], label='Model', value='None')
                steps = gr.Slider(minimum=0, maximum=50, step=1, label='Percent of refiner steps from total sampling steps', value=20)

            gr.HTML('<p style="margin-bottom:0.8em"> It\'s recommended to keep the percentage at 20% (80% base steps, 20% refiner steps). Higher values may result in distortions. </p>')
            
        ui = [enable, checkpoint, steps]
        return ui
    
    def process(self, p, enable, checkpoint, steps):
        if self.base != None or self.swapped == True or self.callback_set == True:
            self.reset(p)
        if not enable or checkpoint == 'None':
            script_callbacks.remove_current_script_callbacks()
            self.model = None
            return
        if self.model == None or self.model_name != checkpoint:
            if not self.load_model(checkpoint): return
        self.c_ae = self.embedder(torch.tensor(shared.opts.sdxl_refiner_high_aesthetic_score).unsqueeze(0).to(devices.device).repeat(p.batch_size, 1))
        self.uc_ae = self.embedder(torch.tensor(shared.opts.sdxl_refiner_low_aesthetic_score).unsqueeze(0).to(devices.device).repeat(p.batch_size, 1))
        p.extra_generation_params['Refiner model'] = checkpoint.rsplit('.', 1)[0]
        p.extra_generation_params['Refiner steps'] = ceil((p.steps * (steps / 100)))
        
        def denoiser_callback(params: script_callbacks.CFGDenoiserParams):
            if params.sampling_step > params.total_sampling_steps * (1 - steps / 100) - 2:
                params.text_cond['vector'] = torch.cat((params.text_cond['vector'][:, :2304], self.c_ae), 1)
                params.text_uncond['vector'] = torch.cat((params.text_uncond['vector'][:, :2304], self.uc_ae), 1)
                params.text_cond['crossattn'] = params.text_cond['crossattn'][:, :, -1280:]
                params.text_uncond['crossattn'] = params.text_uncond['crossattn'][:, :, -1280:]
                if not self.swapped:
                    self.base = p.sd_model.model.to('cpu', devices.dtype_unet)
                    devices.torch_gc()
                    p.sd_model.model = self.model.to(devices.device, devices.dtype_unet)
                    self.swapped = True
        
        def denoised_callback(params: script_callbacks.CFGDenoiserParams):
            if params.sampling_step == params.total_sampling_steps - 2:
                self.reset(p, keep_hook=True)
        
        if not self.callback_set:
            script_callbacks.on_cfg_denoiser(denoiser_callback)
            script_callbacks.on_cfg_denoised(denoised_callback)
            self.callback_set = True
    
    def reset(self, p, keep_hook=False):
        if self.model is not None:
            self.model.to('cpu', devices.dtype_unet)
        p.sd_model.model = (self.base or p.sd_model.model).to(devices.device, devices.dtype_unet)
        devices.torch_gc()
        self.base = None
        self.swapped = False
        if not keep_hook:
            script_callbacks.remove_current_script_callbacks()
            self.callback_set = False
        
    def postprocess(self, p, processed, enable, checkpoint, steps):
        if enable and checkpoint != 'None':
            self.reset(p)
        
