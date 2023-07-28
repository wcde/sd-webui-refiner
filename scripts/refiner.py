from pathlib import Path
import torch
from modules import scripts, shared, processing, sd_samplers, script_callbacks
from modules import devices, prompt_parser, sd_models, sd_models_config, sd_models_xl
import gradio as gr

import sgm.models.diffusion as diffusion
from sgm.modules import UNCONDITIONAL_CONFIG
import sgm.modules.diffusionmodules.denoiser_scaling
import sgm.modules.diffusionmodules.discretizer
from safetensors.torch import load_file
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)

def safe_import(import_name: str, pkg_name: str | None = None):
    try:
        __import__(import_name)
    except Exception:
        pkg_name = pkg_name or import_name
        import pip
        if hasattr(pip, 'main'):
            pip.main(['install', pkg_name])
        else:
            pip._internal.main(['install', pkg_name])
        __import__(import_name)
        

safe_import('omegaconf')
from omegaconf import DictConfig, OmegaConf
config_path = Path(__file__).parent.resolve() / '../config.yaml'

class Refiner(scripts.Script):
    def __init__(self):
        super().__init__()
        if not config_path.exists():
            open(config_path, 'w').close()
        self.config: DictConfig = OmegaConf.load(config_path)
        self.callback_set = False
        self.model = None
        self.conditioner = None
        self.base = None
        self.swapped = False
        self.model_name = ''
        
    def title(self):
        return "Refiner"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def load_model(self, model_name):
        ckpt = load_file(sd_models.checkpoints_list[model_name].filename)
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
        refiner_config = OmegaConf.load(sd_models_config.config_sdxl_refiner).model.params.network_config
        self.model = instantiate_from_config(refiner_config)
        self.model = get_obj_from_str(OPENAIUNETWRAPPER)(
            self.model, compile_model=False
        ).eval()
        self.model.train = disabled_train
        dtype = next(self.model.diffusion_model.parameters()).dtype
        self.model.diffusion_model.dtype = dtype
        self.model.conditioning_key = 'crossattn'
        self.model.cond_stage_key = 'txt'
        self.model.parameterization = 'v'
        discretization = sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization()
        self.model.alphas_cumprod = torch.asarray(discretization.alphas_cumprod, device=devices.device, dtype=dtype)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        state_dict = dict()
        for key in ckpt.keys():
            if 'model.diffusion_model' in key:
                state_dict[key.replace('model.d', 'd')] = ckpt[key]
        self.model.load_state_dict(state_dict)
        self.model_name = model_name
        return True
        

    def ui(self, is_img2img):
        with gr.Accordion(label='Refiner', open=False):
            enable = gr.Checkbox(label='Enable Refiner', value=self.config.get('enable', False))
            with gr.Row():
                checkpoint = gr.Dropdown(choices=['None', *sd_models.checkpoints_list.keys()], label='Model', value=self.config.get('checkpoint', 'None'))
                steps = gr.Slider(minimum=0, maximum=35, step=1, label='Steps', value=self.config.get('steps', 10))
            keep_in_gpu = gr.Checkbox(label='Keep Refiner in GPU (not recommended, requires 24 Gb VRAM)', value=self.config.get('keep_in_gpu', False))
            
        ui = [enable, checkpoint, steps, keep_in_gpu]
        for elem in ui:
            setattr(elem, "do_not_save_to_config", True)
        return ui
    
    
    def process(self, p, enable, checkpoint, steps, keep_in_gpu):
        if not enable or checkpoint == 'None':
            script_callbacks.remove_current_script_callbacks()
            self.model = None
            return
        if self.model == None or self.model_name != checkpoint:
            if not self.load_model(checkpoint): return
        self.config.enable = enable
        self.config.checkpoint = checkpoint
        self.config.steps = steps
        self.config.keep_in_gpu = keep_in_gpu
        
        def denoiser_callback(params: script_callbacks.CFGDenoiserParams):
            if params.sampling_step > params.total_sampling_steps - (steps + 2):
                params.text_cond['vector'] = params.text_cond['vector'][:, :2560]
                params.text_uncond['vector'] = params.text_uncond['vector'][:, :2560]
                params.text_cond['crossattn'] = params.text_cond['crossattn'][:, :, -1280:]
                params.text_uncond['crossattn'] = params.text_uncond['crossattn'][:, :, -1280:]
                if not self.swapped:
                    self.base = p.sd_model.model
                    if not self.config.keep_in_gpu:
                        self.base.cpu()
                    p.sd_model.model = self.model.cuda()
                    self.swapped = True
        
        if not self.callback_set:
            script_callbacks.on_cfg_denoiser(denoiser_callback)
            self.callback_set = True
        
    
    def postprocess(self, p, processed, *args):
        if self.swapped:
            if not self.config.keep_in_gpu:
                self.model.cpu()
            p.sd_model.model = self.base.cuda()
            self.swapped = False
            self.callback_set = False
            script_callbacks.remove_current_script_callbacks()
            OmegaConf.save(self.config, config_path)
        
        
        
        
            
    