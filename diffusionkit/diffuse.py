import torch
import numpy as np
from PIL import Image
from math import ceil

from .loader import load
from .interfaces import DiffuseParams, SamplerInterface
from .modules.utils import create_random_tensors, latent_to_images, resize_image
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

	model = load('stable_diffusion_v1')
	cond = model.get_learned_conditioning([prompt] * params.count)
	uncond = model.get_learned_conditioning([prompt_negative] * params.count)

	sampler.use_model(model)

	if image:
		image = resize_image(image, width, height)
		image = image.convert('RGB')
		image = np.array(image, dtype=np.float32)
		image = 2. * (image / 255.0) - 1.
		image = np.transpose(image, (2, 0, 1))
		image = np.tile(image, (batch_size, 1, 1, 1))
		image = torch.from_numpy(image)
		image = image.half()
		image = image.cuda()

	if mask:
		mask = mask.convert('RGBA')
		mask = resize_image(mask, width=width_latent, height=height_latent)
		mask = mask.split()[1]
		mask = np.array(mask).astype(np.float32) / 255.0
		mask = np.tile(mask, (4, 1, 1))
		mask = mask[None].transpose(0, 1, 2, 3)
		mask = torch.from_numpy(mask)
		mask = mask.half()
		mask = mask.cuda()


	with torch.no_grad(), torch.autocast('cuda'):
		if image is not None:
			denoising_steps = int(
				min(params.denoising_strength, 0.999) 
				* params.steps
			)

			ctx.report_sampling_steps(denoising_steps)
			ctx.report_stage('encode')

			init_latent = model.get_first_stage_encoding(
				model.encode_first_stage(image)
			)
		else:
			denoising_steps = params.steps
			ctx.report_sampling_steps(denoising_steps)


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
					mask=mask
				)

			ctx.report_stage('decode')

			images = latent_to_images(samples, model)

			for image in images:
				image = resize_image(image, params.width, params.height)
				result_images.append(image)


	ctx.finish()

	return result_images







