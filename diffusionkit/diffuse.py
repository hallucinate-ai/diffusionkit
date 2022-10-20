import torch
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import ceil

from .loader import load
from .modules.samplers.common import SamplerInterface
from .modules.utils import create_random_tensors, resize_image
from .context import DiffusionContext


@dataclass
class DiffuseParams:
	prompt: str
	prompt_negative: str = ''
	width: int = 512
	height: int = 512
	steps: int = 50
	cfg_scale: float = 5.0
	denoising_strength: float = 0.75
	seed: int = 0
	count: int = 1



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
		alpha = mask.convert('RGBA')
		alpha = resize_image(alpha, width=width_latent, height=height_latent)
		mask = alpha.split()[1]
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
	
				'''
				if params.sampler_name == 'ddim':
					sampler.make_schedule(
						ddim_num_steps=params.ddim_steps, 
						ddim_eta=0.0, 
						verbose=False
					)

					z_enc = sampler.stochastic_encode(
						init_latent, 
						torch.tensor([t_enc_steps] * batch_size).cuda()
					)

					samples_ddim = sampler.decode(
						z_enc,
						conditioning,
						t_enc_steps,
						unconditional_guidance_scale=params.cfg_scale,
						unconditional_conditioning=conditioning_negative,
						z_mask=mask, 
						x0=init_latent,
						progress_callback=progress_callback
					)
				'''
				

			ctx.report_stage('decode')

			for i in range(len(samples)):
				x_sample = model.decode_first_stage(samples[i].unsqueeze(0))
				x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
				x_sample = x_sample[0].cpu().numpy()
				x_sample = 255. * np.transpose(x_sample, (1, 2, 0))
				x_sample = x_sample.astype(np.uint8)
				
				image = Image.fromarray(x_sample)
				image = resize_image(image, params.width, params.height)
				result_images.append(image)

	ctx.finish()

	return result_images







