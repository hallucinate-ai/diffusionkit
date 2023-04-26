import torch
import gc
from safetensors import safe_open
from .modules.diffusion.ddpm import LatentDiffusion
from .modules.diffusion.openaimodel import UNetModel
from .modules.autoencoder import AutoencoderKL
from .modules.encoders import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
from .modules.controlnet import ControlLDM, ControlNet, ControlledUNetModel

models = dict()


def load_stable_diffusion(config, controlnet=False):
	if config.checkpoint_sd in models:
		return models[config.checkpoint_sd]

	if config.checkpoint_sd.endswith('.safetensor'):
		state_dict = {}
		with safe_open(config.checkpoint_sd, framework='pt', device='cpu') as f:
			for key in f.keys():
				state_dict[key] = f.get_tensor(key)
	else:
		checkpoint = torch.load(config.checkpoint_sd, map_location='cpu')
		state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

	is_v2_model, is_inpainting_model = guess_stable_diffusion_version(state_dict)
	is_v_model = is_v2_model and not is_inpainting_model # ugly hack

	latent_diffusion_config = {
		'linear_start': 0.00085,
		'linear_end': 0.0120,
		'first_stage_key': 'jpg',
		'image_size': 64,
		'channels': 4,
		'scale_factor': 0.18215,
		**(
			{
				'conditioning_key': 'hybrid'
			} if is_inpainting_model else {
				'conditioning_key': 'crossattn'
			}
		),
		**(
			{
				'parameterization': 'v'
			} if is_v_model else {
				'parameterization': 'eps'
			}
		),
		**(
			{
				'control_key': 'hint',
				'only_mid_control': False
			} if controlnet else {
			}
		)
	}

	unet_config = {
		'out_channels': 4,
		'model_channels': 320,
		'attention_resolutions': [ 4, 2, 1 ],
		'num_res_blocks': 2,
		'channel_mult': [ 1, 2, 4, 4 ],
		'use_spatial_transformer': True,
		'transformer_depth': 1,
		**(
			{
				'num_head_channels': 64,
				'use_linear_in_transformer': True,
				'context_dim': 1024,
				'use_fp16': True,
				'num_heads': -1
			} if is_v2_model else {
				'context_dim': 768,
				'num_heads': 8,
			}
		),
		**(
			{
				'in_channels': 9
			} if is_inpainting_model else {
				'in_channels': 4
			}
		)
	}

	first_stage_config = {
		'embed_dim': 4,
		'ddconfig': {
			'double_z': True,
			'z_channels': 4,
			'resolution': 256,
			'in_channels': 3,
			'out_ch': 3,
			'ch': 128,
			'ch_mult': [ 1, 2, 4, 4 ],
			'num_res_blocks': 2,
			'attn_resolutions': [],
			'dropout': 0.0
		}
	}

	if is_v2_model:
		cond_stage = FrozenOpenCLIPEmbedder(
			pretrained=config.checkpoint_clip,
			arch='ViT-H-14',
			layer='penultimate'
		)
	else:
		cond_stage = FrozenCLIPEmbedder(
			pretrained=config.checkpoint_clip
		)

	first_stage = AutoencoderKL(
		**first_stage_config
	)

	if controlnet:
		control_stage_config = {
			'in_channels': 4,
			'hint_channels': 3,
			'model_channels': 320,
			'attention_resolutions': [ 4, 2, 1 ],
			'num_res_blocks': 2,
			'channel_mult': [ 1, 2, 4, 4 ],
			'num_heads': 8,
			'use_spatial_transformer': True,
			'transformer_depth': 1,
			'context_dim': 768,
			'use_checkpoint': True,
			'legacy': False
		}

		control_stage = ControlNet(
			**control_stage_config
		)

		unet = ControlledUNetModel(
			**unet_config
		)

		models[config.checkpoint_ctrl] = model = ControlLDM(
			unet=unet,
			control_stage=control_stage,
			first_stage=first_stage,
			cond_stage=cond_stage,
			**latent_diffusion_config,
		)
	else:
		unet = UNetModel(
			**unet_config
		)

		models[config.checkpoint_sd] = model = LatentDiffusion(
			unet=unet,
			first_stage=first_stage,
			cond_stage=cond_stage,
			**latent_diffusion_config,
		)

	model.load_state_dict(state_dict, strict=False)
	model.is_inpainting_model = is_inpainting_model
	model.is_v_model = is_v_model

	model.half()
	model.cuda()
	model.eval()

	return model


def guess_stable_diffusion_version(state_dict):
	is_v2 = state_dict['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight'].shape[1] == 1024
	is_inpainting = state_dict['model.diffusion_model.input_blocks.0.0.weight'].shape[1] == 9

	return is_v2, is_inpainting


def unload(name):
	if name not in models:
		return

	del models[name]
	gc.collect()
	torch.cuda.empty_cache()


def unload_all_models():
	for name in list(models.keys()):
		unload(name)