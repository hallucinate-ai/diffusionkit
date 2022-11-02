import torch
import numpy as np

from ..interfaces import SamplerInterface
from .denoisers import MaskedCompVisDenoiser
from .diffusion.util import make_ddim_timesteps, make_ddim_sampling_parameters, extract_into_tensor


class KSampler(SamplerInterface):
	def __init__(self, schedule='linear', **schedule_args):
		self.schedule = schedule
		self.schedule_args = schedule_args


	def use_model(self, model):
		self.model = model
		self.ddpm_num_timesteps = model.num_timesteps
		self.denoiser = MaskedCompVisDenoiser(model)


	def make_schedule(self, ddim_num_steps, ddim_discretize='uniform', ddim_eta=0.):
		self.ddim_timesteps = make_ddim_timesteps(
			ddim_discr_method=ddim_discretize, 
			num_ddim_timesteps=ddim_num_steps,
			num_ddpm_timesteps=self.ddpm_num_timesteps
		)

		alphas_cumprod = self.model.alphas_cumprod

		assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'

		to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

		self.betas = to_torch(self.model.betas)
		self.alphas_cumprod = to_torch(alphas_cumprod)
		self.alphas_cumprod_prev = to_torch(self.model.alphas_cumprod_prev)
		self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod.cpu()))
		self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod.cpu()))
		self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod.cpu()))
		self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod.cpu()))
		self.sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1))

		ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
			alphacums=alphas_cumprod.cpu(),
			ddim_timesteps=self.ddim_timesteps,
			eta=ddim_eta
		)

		sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
			(1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * 
			(1 - self.alphas_cumprod / self.alphas_cumprod_prev)
		)

		self.ddim_sigmas = ddim_sigmas.cuda()
		self.ddim_alphas = ddim_alphas.cuda()
		self.ddim_alphas_prev = ddim_alphas_prev.cuda()
		self.ddim_sqrt_one_minus_alphas = np.sqrt(1. - ddim_alphas).cuda()
		self.ddim_sigmas_for_original_num_steps = sigmas_for_original_sampling_steps.cuda()


	def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
		if use_original_steps:
			sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
			sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
		else:
			sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
			sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

		if noise is None:
			noise = torch.randn_like(x0)

		a = extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
		b = extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise

		return a + b


	def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
			   use_original_steps=False, z_mask = None, x0=None, progress_callback=None,):

		timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
		timesteps = timesteps[:t_start]

		time_range = np.flip(timesteps)
		total_steps = timesteps.shape[0]
		#print(f'Running DDIM Sampling with {total_steps} timesteps')

		#iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
		iterator = time_range
		x_dec = x_latent

		if progress_callback:
			iterator = make_sampling_progress_iterator(time_range, progress_callback)

		for i, step in enumerate(iterator):
			index = total_steps - i - 1
			ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)

			if z_mask is not None and i < total_steps - 2:
				assert x0 is not None
				img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
				mask_inv = 1. - z_mask
				x_dec = (img_orig * mask_inv) + (z_mask * x_dec)

			x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
										  unconditional_guidance_scale=unconditional_guidance_scale,
										  unconditional_conditioning=unconditional_conditioning,
										  progress_callback=progress_callback)
		return x_dec

	def sample(self, ctx, noise, cond, uncond, steps, init_latent=None, mask=None):
		self.make_schedule(ddim_num_steps=ctx.params.steps, ddim_eta=0.0)

		z_enc = self.stochastic_encode(
			init_latent, 
			torch.tensor([steps]).cuda()
		)

		return self.decode(
			z_enc,
			cond,
			steps,
			unconditional_guidance_scale=ctx.params.cfg_scale,
			unconditional_conditioning=uncond,
			z_mask=mask, 
			x0=init_latent
		)

'''
sampler.make_schedule(
						ddim_num_steps=params.ddim_steps, 
						ddim_eta=0.0, 
						verbose=False
					)

					z_enc = sampler.stochastic_encode(
						init_latent, 
						torch.tensor([t_enc_steps] * batch_size).cuda()
					)

					if obliterate and mask is not None:
						random = torch.randn(mask.shape, device=z_enc.device)
						z_enc = (mask * random) + ((1 - mask) * z_enc)

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