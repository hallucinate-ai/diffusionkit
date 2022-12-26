import torch
import numpy as np
from PIL import Image
from math import ceil

from . import config
from .loader import load_stable_diffusion
from .interfaces import DiffuseParams, SamplerInterface
from .image import resize_image, image_to_tensor, mask_to_tensor
from .modules.utils import create_random_tensors, latent_to_images
from .context import DiffusionContext



def diffuse(params: DiffuseParams, sampler: SamplerInterface, image: Image = None, mask: Image = None):
	assert 0. <= params.denoising_strength <= 1, 'denoising_strength must be between [0.0, 1.0]'
	assert image is not None if mask is not None else True, 'image must be set if mask is set'

	if not params.width or not params.height:
		assert image is not None, 'either set width and height or supply an image'
		params.width = image.width
		params.height = image.height


	ctx = DiffusionContext(params=params, image=image)
	ctx.report_stage('init')
	
	result_images = []
	batch_size = 1
	prompt = params.prompt
	prompt_negative = params.prompt_negative
	seeds = [params.seed + x for x in range(params.count)]

	width = ceil(params.width / 64) * 64
	height = ceil(params.height / 64) * 64
	width_latent = width // 8
	height_latent = height // 8

	if image:
		image = resize_image(image, width, height)
		image = image_to_tensor(image)

	if mask:
		mask = resize_image(mask, width, height)
		mask = mask_to_tensor(mask)


	if not config.weights.stable_diffusion:
		raise Exception('no weights file path specified for stable diffusion')
	

	model = load_stable_diffusion(config.weights.stable_diffusion)
	cond = model.get_learned_conditioning([prompt] * params.count)
	uncond = model.get_learned_conditioning([prompt_negative] * params.count)

	sampler.use_model(model)

	assert mask is not None if model.is_inpainting_model else True, 'the loaded model is an inpainting model and thus needs an image and mask'


	with torch.no_grad(), torch.autocast('cuda'):
		if image is not None:
			denoising_steps = int(
				min(params.denoising_strength, 0.999) 
				* params.steps
			)

			ctx.report_sampling_steps(denoising_steps)
			ctx.report_stage('encode')

			first_encoding = model.encode_first_stage(image)
			init_latent = model.get_first_stage_encoding(first_encoding)
		else:
			denoising_steps = params.steps
			ctx.report_sampling_steps(denoising_steps)

		
		if model.is_inpainting_model:
			conditioning_mask = torch.round(mask)
			conditioning_image = torch.lerp(
				image,
				image * (1.0 - conditioning_mask),
				1 # todo: make parameter inpainting_mask_weight
			)

			conditioning_encoding = model.encode_first_stage(conditioning_image)
			conditioning_image = model.get_first_stage_encoding(conditioning_encoding)
			conditioning_mask = torch.nn.functional.interpolate(conditioning_mask, size=init_latent.shape[-2:])
			conditioning_mask = conditioning_mask.expand(conditioning_image.shape[0], -1, -1, -1)

			image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)
			image_conditioning = image_conditioning.cuda().half()
		else:
			image_conditioning = None


		for i in range(0, params.count, batch_size):
			batch_seeds = seeds[i:i+batch_size]

			noise = create_random_tensors([4, height_latent, width_latent], seeds=batch_seeds)
			noise = noise.cuda()

			if image is None:
				samples = sampler.sample(
					ctx=ctx,
					noise=noise, 
					cond=cond, 
					uncond=uncond, 
					steps=params.steps
				)
			else:
				samples = sampler.sample(
					ctx=ctx,
					noise=noise, 
					cond=cond, 
					uncond=uncond, 
					steps=denoising_steps, 
					init_latent=init_latent, 
					mask=mask,
					image_conditioning=image_conditioning
				)

			ctx.report_stage('decode')

			#latent_mask = torch.nn.functional.interpolate(mask, size=(height_latent, width_latent))
			#samples = samples * (1.0 - latent_mask) + init_latent * latent_mask

			images = latent_to_images(samples, model)

			for image in images:
				image = resize_image(image, params.width, params.height)
				result_images.append(image)


	ctx.finish()

	return result_images







