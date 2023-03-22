import torch
import gc
from .modules.diffusion.ddpm import LatentDiffusion
from .modules.diffusion.openaimodel import UNetModel
from .modules.autoencoder import AutoencoderKL
from .modules.encoders import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder

models = dict()

latent_diffusion_config = {
	'linear_start': 0.00085,
	'linear_end': 0.0120,
	'first_stage_key': 'jpg',
	'image_size': 64,
	'channels': 4,
	'scale_factor': 0.18215,
	'conditioning_key': 'crossattn'
}

unet_config = {
	'in_channels': 4,
	'out_channels': 4,
	'model_channels': 320,
	'attention_resolutions': [ 4, 2, 1 ],
	'num_res_blocks': 2,
	'channel_mult': [ 1, 2, 4, 4 ],
	'num_heads': 8,
	'use_spatial_transformer': True,
	'transformer_depth': 1,
	'context_dim': 768
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


def load_stable_diffusion(config):
	if config.checkpoint_sd in models:
		return models[config.checkpoint_sd]


	checkpoint = torch.load(config.checkpoint_sd, map_location='cpu')
	state_dict = checkpoint['state_dict']

	is_v2_model, is_inpainting_model = guess_stable_diffusion_version(state_dict)
	is_v_model = is_v2_model and not is_inpainting_model # ugly hack


	if is_v2_model:
		unet_config['num_head_channels'] = 64
		unet_config['use_linear_in_transformer'] = True
		unet_config['context_dim'] = 1024
		unet_config['use_fp16'] = True
		unet_config['num_heads'] = -1
		cond_stage = FrozenOpenCLIPEmbedder(
			pretrained=config.checkpoint_clip,
			arch='ViT-H-14',
			layer='penultimate'
		)
	else:
		cond_stage = FrozenCLIPEmbedder(
			pretrained=config.checkpoint_clip
		)

	if is_v_model:
		latent_diffusion_config['parameterization'] = 'v'

	if is_inpainting_model:
		latent_diffusion_config['conditioning_key'] = 'hybrid'
		unet_config['in_channels'] = 9


	unet = UNetModel(
		**unet_config
	)

	first_stage = AutoencoderKL(
		**first_stage_config
	)

	models[config.checkpoint_sd] = model = LatentDiffusion(
		unet=unet,
		first_stage=first_stage,
		cond_stage=cond_stage,
		**latent_diffusion_config,
	)

	model.load_state_dict(checkpoint['state_dict'], strict=False)
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
	for name in models.keys():
		unload(name)