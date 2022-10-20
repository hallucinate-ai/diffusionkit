import torch
import math

from ..utils import latent_to_images, to_d, linear_multistep_coeff, get_ancestral_step


def schedule_euler(ctx, denoiser, x, sigmas, cond, uncond, init_latent, mask, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
	steps = len(sigmas) - 1
	s_in = x.new_ones([x.shape[0]])

	for i in ctx.make_sampling_iter(range(steps)):
		gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
		eps = torch.randn_like(x) * s_noise
		sigma_hat = sigmas[i] * (gamma + 1)

		if gamma > 0:
			x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

		denoised = denoiser(
			x=x, 
			sigma=sigma_hat * s_in, 
			cond=cond, 
			uncond=uncond, 
			cond_scale=ctx.params.cfg_scale, 
			init_latent=init_latent, 
			mask=mask
		)

		d = to_d(x, sigma_hat, denoised)

		#if callback is not None:
		#	callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

		dt = sigmas[i + 1] - sigma_hat
		x = x + d * dt
		
	return x


def schedule_euler_ancestral(ctx, denoiser, x, sigmas, cond, uncond, init_latent, mask, eta=1.):
	steps = len(sigmas) - 1
	s_in = x.new_ones([x.shape[0]])
	
	for i in ctx.make_sampling_iter(range(steps)):
		denoised = denoiser(
			x=x, 
			sigma=sigmas[i] * s_in, 
			cond=cond, 
			uncond=uncond, 
			cond_scale=ctx.params.cfg_scale, 
			init_latent=init_latent, 
			mask=mask
		)

		sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

		#if callback is not None:
		#	callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

		d = to_d(x, sigmas[i], denoised)
		dt = sigma_down - sigmas[i]

		x = x + d * dt
		x = x + torch.randn_like(x) * sigma_up

	return x


def sample_heun(ctx, denoiser, x, sigmas, cond, uncond, init_latent, mask, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
	steps = len(sigmas) - 1
	s_in = x.new_ones([x.shape[0]])
	
	for i in ctx.make_sampling_iter(range(steps)):
		gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
		eps = torch.randn_like(x) * s_noise
		sigma_hat = sigmas[i] * (gamma + 1)

		if gamma > 0:
			x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

		denoised = denoiser(
			x=x, 
			sigma=sigma_hat * s_in, 
			cond=cond, 
			uncond=uncond, 
			cond_scale=ctx.params.cfg_scale, 
			init_latent=init_latent, 
			mask=mask
		)

		d = to_d(x, sigma_hat, denoised)

		#if callback is not None:
		#	callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
		
		dt = sigmas[i + 1] - sigma_hat

		if sigmas[i + 1] == 0:
			x = x + d * dt
		else:
			x_2 = x + d * dt
			denoised_2 = denoiser(
				x=x_2, 
				sigma=sigmas[i + 1] * s_in, 
				cond=cond, 
				uncond=uncond, 
				cond_scale=ctx.params.cfg_scale, 
				init_latent=init_latent, 
				mask=mask
			)

			d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
			d_prime = (d + d_2) / 2
			x = x + d_prime * dt
			
	return x



def sample_dpm_2(ctx, denoiser, x, sigmas, cond, uncond, init_latent, mask, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
	steps = len(sigmas) - 1
	s_in = x.new_ones([x.shape[0]])
	
	for i in ctx.make_sampling_iter(range(steps)):
		gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
		eps = torch.randn_like(x) * s_noise
		sigma_hat = sigmas[i] * (gamma + 1)

		if gamma > 0:
			x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5

		denoised = denoiser(
			x=x, 
			sigma=sigma_hat * s_in, 
			cond=cond, 
			uncond=uncond, 
			cond_scale=ctx.params.cfg_scale, 
			init_latent=init_latent, 
			mask=mask
		)

		d = to_d(x, sigma_hat, denoised)

		#if callback is not None:
		#	callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

		if sigmas[i + 1] == 0:
			dt = sigmas[i + 1] - sigma_hat
			x = x + d * dt
		else:
			sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
			dt_1 = sigma_mid - sigma_hat
			dt_2 = sigmas[i + 1] - sigma_hat
			x_2 = x + d * dt_1

			denoised_2 = denoiser(
				x=x_2, 
				sigma=sigma_mid * s_in, 
				cond=cond, 
				uncond=uncond, 
				cond_scale=ctx.params.cfg_scale, 
				init_latent=init_latent, 
				mask=mask
			)

			d_2 = to_d(x_2, sigma_mid, denoised_2)
			x = x + d_2 * dt_2

	return x


def sample_dpm_2_ancestral(ctx, denoiser, x, sigmas, cond, uncond, init_latent, mask, eta=1.):
	steps = len(sigmas) - 1
	s_in = x.new_ones([x.shape[0]])
	
	for i in ctx.make_sampling_iter(range(steps)):
		denoised = denoiser(
			x=x, 
			sigma=sigmas[i] * s_in, 
			cond=cond, 
			uncond=uncond, 
			cond_scale=ctx.params.cfg_scale, 
			init_latent=init_latent, 
			mask=mask
		)

		sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

		#if callback is not None:
		#	callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

		d = to_d(x, sigmas[i], denoised)
		
		if sigma_down == 0:
			dt = sigma_down - sigmas[i]
			x = x + d * dt
		else:
			sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
			dt_1 = sigma_mid - sigmas[i]
			dt_2 = sigma_down - sigmas[i]
			x_2 = x + d * dt_1

			denoised_2 = denoiser(
				x=x_2, 
				sigma=sigma_mid * s_in, 
				cond=cond, 
				uncond=uncond, 
				cond_scale=ctx.params.cfg_scale, 
				init_latent=init_latent, 
				mask=mask
			)

			d_2 = to_d(x_2, sigma_mid, denoised_2)
			x = x + d_2 * dt_2
			x = x + torch.randn_like(x) * sigma_up

	return x


def schedule_lms(ctx, denoiser, x, sigmas, cond, uncond, init_latent, mask, order=4):
	steps = len(sigmas) - 1
	s_in = x.new_ones([x.shape[0]])
	ds = []

	for i in ctx.make_sampling_iter(range(steps)):
		cur_order = min(i + 1, order)
		denoised = denoiser(
			x=x, 
			sigma=sigmas[i] * s_in, 
			cond=cond, 
			uncond=uncond, 
			cond_scale=ctx.params.cfg_scale, 
			init_latent=init_latent, 
			mask=mask
		)

		d = to_d(x, sigmas[i], denoised)
		ds.append(d)

		if len(ds) > order:
			ds.pop(0)

		if ctx.wants_intermediate():
			ctx.put_intermediate(
				latent_to_images(denoised, model=denoiser.inner_model)
			)

		coeffs = [
			linear_multistep_coeff(cur_order, sigmas.cpu(), i, j) 
			for j in range(cur_order)
		]

		x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

	return x