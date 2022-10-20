import torch
import numpy as np
from .. import prompt_parser
from ..diffusion.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
from ..utils import make_sampling_progress_iterator


class VanillaDDIMSampler():
	def __init__(self, model, schedule='linear'):
		self.model = model
		self.ddpm_num_timesteps = model.num_timesteps
		self.schedule = schedule

	def make_schedule(self, ddim_num_steps, ddim_discretize='uniform', ddim_eta=0., verbose=False):
		self.ddim_timesteps = make_ddim_timesteps(
			ddim_discr_method=ddim_discretize, 
			num_ddim_timesteps=ddim_num_steps,
			num_ddpm_timesteps=self.ddpm_num_timesteps,
			verbose=verbose
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

		# ddim sampling parameters
		ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
			alphacums=alphas_cumprod.cpu(),
			ddim_timesteps=self.ddim_timesteps,
			eta=ddim_eta,verbose=verbose
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


	@torch.no_grad()
	def sample(self,
			   S,
			   batch_size,
			   shape,
			   conditioning=None,
			   quantize_x0=False,
			   eta=0.,
			   mask=None,
			   x0=None,
			   temperature=1.,
			   noise_dropout=0.,
			   score_corrector=None,
			   corrector_kwargs=None,
			   verbose=False,
			   x_T=None,
			   log_every_t=100,
			   unconditional_guidance_scale=1.,
			   unconditional_conditioning=None,
			   # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
			   **kwargs
			   ):

		if conditioning is not None:
			if isinstance(conditioning, dict):
				cbs = conditioning[list(conditioning.keys())[0]].shape[0]
				if cbs != batch_size:
					print(f'Warning: Got {cbs} conditionings but batch-size is {batch_size}')
			else:
				if conditioning.shape[0] != batch_size:
					print(f'Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}')

		self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
		# sampling
		C, H, W = shape
		size = (batch_size, C, H, W)
		print(f'Data shape for DDIM sampling is {size}, eta {eta}')

		samples, intermediates = self.ddim_sampling(conditioning, size,
													callback=callback,
													img_callback=img_callback,
													quantize_denoised=quantize_x0,
													mask=mask, x0=x0,
													ddim_use_original_steps=False,
													noise_dropout=noise_dropout,
													temperature=temperature,
													score_corrector=score_corrector,
													corrector_kwargs=corrector_kwargs,
													x_T=x_T,
													log_every_t=log_every_t,
													unconditional_guidance_scale=unconditional_guidance_scale,
													unconditional_conditioning=unconditional_conditioning
													)
		return samples, intermediates

	@torch.no_grad()
	def ddim_sampling(self, cond, shape,
					  x_T=None, ddim_use_original_steps=False,
					  callback=None, timesteps=None, quantize_denoised=False,
					  mask=None, x0=None, img_callback=None, log_every_t=100,
					  temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
					  unconditional_guidance_scale=1., unconditional_conditioning=None):
		device = self.model.betas.device
		b = shape[0]
		if x_T is None:
			img = torch.randn(shape, device=device)
		else:
			img = x_T

		if timesteps is None:
			timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
		elif timesteps is not None and not ddim_use_original_steps:
			subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
			timesteps = self.ddim_timesteps[:subset_end]

		intermediates = {'x_inter': [img], 'pred_x0': [img]}
		time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
		total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
		#print(f'Running DDIM Sampling with {total_steps} timesteps')

		#iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
		iterator = time_range

		for i, step in enumerate(iterator):
			index = total_steps - i - 1
			ts = torch.full((b,), step, device=device, dtype=torch.long)

			if mask is not None:
				assert x0 is not None
				img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
				img = img_orig * mask + (1. - mask) * img

			outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
									  quantize_denoised=quantize_denoised, temperature=temperature,
									  noise_dropout=noise_dropout, score_corrector=score_corrector,
									  corrector_kwargs=corrector_kwargs,
									  unconditional_guidance_scale=unconditional_guidance_scale,
									  unconditional_conditioning=unconditional_conditioning)
			img, pred_x0 = outs
			if callback: callback(i)
			if img_callback: img_callback(pred_x0, i)

			if index % log_every_t == 0 or index == total_steps - 1:
				intermediates['x_inter'].append(img)
				intermediates['pred_x0'].append(pred_x0)

		return img, intermediates

	@torch.no_grad()
	def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
					  temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
					  unconditional_guidance_scale=1., unconditional_conditioning=None, progress_callback=None):
		b, *_, device = *x.shape, x.device

		if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
			e_t = self.model.apply_model(x, t, c)
		else:
			x_in = torch.cat([x] * 2)
			t_in = torch.cat([t] * 2)
			c_in = torch.cat([unconditional_conditioning, c])
			e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
			e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

		if score_corrector is not None:
			assert self.model.parameterization == 'eps'
			e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

		alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
		alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
		sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
		sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
		# select parameters corresponding to the currently considered timestep
		a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
		a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
		sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
		sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

		# current prediction for x_0
		pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
		if quantize_denoised:
			pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
		# direction pointing to x_t
		dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
		noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
		if noise_dropout > 0.:
			noise = torch.nn.functional.dropout(noise, p=noise_dropout)
		x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
		return x_prev, pred_x0

	@torch.no_grad()
	def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
		# fast, but does not allow for exact reconstruction
		# t serves as an index to gather the correct alphas
		if use_original_steps:
			sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
			sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
		else:
			sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
			sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

		if noise is None:
			noise = torch.randn_like(x0)
		return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
				extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

	@torch.no_grad()
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




class DDIMSampler:
	def __init__(self, model):
		self.sampler = VanillaDDIMSampler(model)
		self.orig_p_sample_ddim = self.sampler.p_sample_ddim
		self.sampler.p_sample_ddim = self.p_sample_ddim_hook

	def p_sample_ddim_hook(self, x_dec, cond, ts, unconditional_conditioning, *args, **kwargs):
		conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
		unconditional_conditioning = prompt_parser.reconstruct_cond_batch(unconditional_conditioning, self.step)

		assert all([len(conds) == 1 for conds in conds_list]), 'composition via AND is not supported for DDIM/PLMS samplers'
		cond = tensor

		# for DDIM, shapes must match, we can't just process cond and uncond independently;
		# filling unconditional_conditioning with repeats of the last vector to match length is
		# not 100% correct but should work well enough
		if unconditional_conditioning.shape[1] < cond.shape[1]:
			last_vector = unconditional_conditioning[:, -1:]
			last_vector_repeated = last_vector.repeat([1, cond.shape[1] - unconditional_conditioning.shape[1], 1])
			unconditional_conditioning = torch.hstack([unconditional_conditioning, last_vector_repeated])
		elif unconditional_conditioning.shape[1] > cond.shape[1]:
			unconditional_conditioning = unconditional_conditioning[:, :cond.shape[1]]

		if self.mask is not None:
			img_orig = self.sampler.model.q_sample(self.init_latent, ts)
			x_dec = img_orig * self.mask + self.nmask * x_dec

		res = self.orig_p_sample_ddim(x_dec, cond, ts, unconditional_conditioning=unconditional_conditioning, *args, **kwargs)

		if self.mask is not None:
			self.last_latent = self.init_latent * self.mask + self.nmask * res[1]
		else:
			self.last_latent = res[1]

		store_latent(self.last_latent)

		self.step += 1
		state.sampling_step = self.step
		shared.total_tqdm.update()

		return res

	def sample(self, ctx, init_latent, noise, conditioning, conditioning_negative):
		steps = ctx.params.steps
		t_enc = int(min(ctx.params.denoising_strength, 0.999) * steps)

		self.sampler.make_schedule(
			ddim_num_steps=steps,
			ddim_eta=self.eta, 
			#ddim_discretize=p.ddim_discretize, 
			verbose=False
		)

		


	def initialize(self, p):
		self.eta = p.eta if p.eta is not None else opts.eta_ddim

		for fieldname in ['p_sample_ddim', 'p_sample_plms']:
			if hasattr(self.sampler, fieldname):
				setattr(self.sampler, fieldname, self.p_sample_ddim_hook)

		self.mask = p.mask if hasattr(p, 'mask') else None
		self.nmask = p.nmask if hasattr(p, 'nmask') else None

	def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None):
		steps, t_enc = setup_img2img_steps(p, steps)

		self.initialize(p)

		# existing code fails with certain step counts, like 9
		try:
			self.sampler.make_schedule(ddim_num_steps=steps,  ddim_eta=self.eta, ddim_discretize=p.ddim_discretize, verbose=False)
		except Exception:
			self.sampler.make_schedule(ddim_num_steps=steps+1, ddim_eta=self.eta, ddim_discretize=p.ddim_discretize, verbose=False)

		x1 = self.sampler.stochastic_encode(x, torch.tensor([t_enc] * int(x.shape[0])).to(shared.device), noise=noise)

		self.init_latent = x
		self.step = 0

		samples = self.launch_sampling(steps, lambda: self.sampler.decode(x1, conditioning, t_enc, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning))

		return samples

	def sample(self, p, x, conditioning, unconditional_conditioning, steps=None):
		self.initialize(p)

		self.init_latent = None
		self.step = 0

		steps = steps or p.steps

		# existing code fails with certain step counts, like 9
		try:
			samples_ddim = self.launch_sampling(steps, lambda: self.sampler.sample(S=steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning, x_T=x, eta=self.eta)[0])
		except Exception:
			samples_ddim = self.launch_sampling(steps, lambda: self.sampler.sample(S=steps+1, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning, x_T=x, eta=self.eta)[0])

		return samples_ddim


