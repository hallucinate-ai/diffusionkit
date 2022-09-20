import torch
import numpy as np
import k_diffusion
from dataclasses import dataclass
from loader import load
from PIL import Image
from ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange, repeat
from utils import create_random_tensors


opt_C = 4
opt_f = 8


@dataclass
class DiffuseParams:
	prompt: str
	width: int = 512
	height: int = 512
	ddim_steps: int = 50
	sampler_name: str = 'ddim'
	cfg_scale: float = 5.0
	denoising_strength: float = 0.75
	seed: int = 0
	count: int = 1



def diffuse(params: DiffuseParams, image: Image = None, mask: Image = None):
	assert 0. <= params.denoising_strength <= 1, 'denoising_strength must be between [0.0, 1.0]'

	result_images = []
	batch_size = 1
	model = load('stable_diffusion_v1')

	if params.sampler_name == 'ddim':
		sampler = DDIMSampler(model)
	else:
		sampler = KDiffusionSampler(model, params.sampler_name)
	

	prompt = params.prompt
	prompt_negative = ''

	if '###' in prompt:
		prompt, prompt_negative = prompt.split('###', 1)
		prompt = prompt.strip()
		prompt_negative = prompt_negative.strip()

	conditioning = model.get_learned_conditioning([prompt] * params.count)
	conditioning_negative = model.get_learned_conditioning([prompt_negative] * params.count)

	seeds = [params.seed + x for x in range(params.count)]


	if image:
		width = image.size[0]
		height = image.size[1]

		image = image.convert('RGB')
		image = np.array(image).astype(np.float32) / 255.0
		image = image[None].transpose(0, 3, 1, 2)
		image = torch.from_numpy(image)

		#mask_channel = None
		#if image_editor_mode == "Mask":
		#	alpha = init_mask.convert("RGBA")
		#	alpha = resize_image(resize_mode, alpha, width // 8, height // 8)
		#	mask_channel = alpha.split()[1]

		#mask = None
		#if mask_channel is not None:
		#	mask = np.array(mask_channel).astype(np.float32) / 255.0
		#	mask = (1 - mask)
		#	mask = np.tile(mask, (4, 1, 1))
		#	mask = mask[None].transpose(0, 1, 2, 3)
		#	mask = torch.from_numpy(mask).to(device)


		image = 2. * image - 1.
		images = repeat(image, '1 ... -> b ...', b=batch_size)
		images = images.half()
		images = images.cuda()

		init_latent = model.get_first_stage_encoding(
			model.encode_first_stage(images)
		)
	else:
		width = params.width
		height = params.height

	mask = None
	masks = None

	shape = [opt_C, height // opt_f, width // opt_f]

	
	with torch.no_grad(), torch.autocast('cuda'):
		for i in range(0, params.count, batch_size):
			batch_seeds = seeds[i:i+batch_size]
			x = create_random_tensors(shape, seeds=batch_seeds)

			t_enc_steps = int(params.denoising_strength * params.ddim_steps)
			obliterate = False

			if params.ddim_steps == t_enc_steps:
				t_enc_steps = t_enc_steps - 1
				obliterate = True

			if params.sampler_name != 'ddim':
				sigmas = sampler.model_wrap.get_sigmas(params.ddim_steps)
				noise = x * sigmas[params.ddim_steps - t_enc_steps - 1]
				xi = init_latent + noise

				if obliterate and masks is not None:
					random = torch.randn(masks.shape, device=xi.device)
					xi = (masks * noise) + ((1 - masks) * xi)

				sigma_sched = sigmas[ddim_steps - t_enc_steps - 1:]
				model_wrap_cfg = CFGMaskedDenoiser(sampler.model_wrap)
				sampling_method = k_diffusion.sampling.__dict__[f'sample_{sampler.get_sampler_name()}']
				samples_ddim = sampling_method(
					model_wrap_cfg, 
					xi, 
					sigma_sched, 
					extra_args={
						'cond': conditioning, 
						'uncond': conditioning_negative, 
						'cond_scale': cfg_scale, 
						'mask': z_mask, 
						'x0': init_latent, 
						'xi': xi
					}, 
					disable=False
				)
			else:
				sampler.make_schedule(
					ddim_num_steps=params.ddim_steps, 
					ddim_eta=0.0, 
					verbose=False
				)

				z_enc = sampler.stochastic_encode(
					init_latent, 
					torch.tensor([t_enc_steps] * batch_size).cuda()
				)

				if obliterate and masks is not None:
					random = torch.randn(masks.shape, device=z_enc.device)
					z_enc = (masks * random) + ((1 - masks) * z_enc)

				samples_ddim = sampler.decode(
					z_enc, 
					conditioning, 
					t_enc_steps,
					unconditional_guidance_scale=params.cfg_scale,
					unconditional_conditioning=conditioning_negative,
					z_mask=masks, 
					x0=init_latent
				)

			for i in range(len(samples_ddim)):
				x_samples_ddim = model.decode_first_stage(samples_ddim[i].unsqueeze(0))
				x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

				x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
				x_sample = x_sample.astype(np.uint8)
				
				image = Image.fromarray(x_sample)
				result_images.append(image)

	return result_images
				



			
			

class CFGMaskedDenoiser(torch.nn.Module):
	def __init__(self, model):
		super().__init__()
		self.inner_model = model

	def forward(self, x, sigma, uncond, cond, cond_scale, mask, x0, xi):
		x_in = x
		x_in = torch.cat([x_in] * 2)
		sigma_in = torch.cat([sigma] * 2)
		cond_in = torch.cat([uncond, cond])
		uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
		denoised = uncond + (cond - uncond) * cond_scale

		if mask is not None:
			assert x0 is not None
			img_orig = x0
			mask_inv = 1. - mask
			denoised = (img_orig * mask_inv) + (mask * denoised)

		return denoised


class KDiffusionSampler:
	def __init__(self, m, sampler):
		self.model = m
		self.model_wrap = k_diffusion.external.CompVisDenoiser(m)
		self.schedule = sampler

	def get_sampler_name(self):
		return self.schedule

	def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T, img_callback = None ):
		sigmas = self.model_wrap.get_sigmas(S)
		x = x_T * sigmas[0]
		model_wrap_cfg = CFGDenoiser(self.model_wrap)

		samples_ddim = k_diffusion.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale}, disable=False)

		return samples_ddim, None