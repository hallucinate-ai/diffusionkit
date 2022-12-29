import math
import torch
from torch import nn

from . import utils



class DiscreteSchedule(nn.Module):
	'''A mapping between continuous noise levels (sigmas) and a list of discrete noise levels.'''

	def __init__(self, sigmas, quantize):
		super().__init__()
		self.register_buffer('sigmas', sigmas)
		self.quantize = quantize


	def get_sigmas(self, n=None):
		if n is None:
			return utils.append_zero(self.sigmas.flip(0))

		t_max = len(self.sigmas) - 1
		t = torch.linspace(t_max, 0, n, device=self.sigmas.device)

		return utils.append_zero(self.t_to_sigma(t))


	def sigma_to_t(self, sigma, quantize=None):
		quantize = self.quantize if quantize is None else quantize
		dists = torch.abs(sigma - self.sigmas[:, None])

		if quantize:
			return torch.argmin(dists, dim=0).view(sigma.shape)

		low_idx, high_idx = torch.sort(torch.topk(dists, dim=0, k=2, largest=False).indices, dim=0)[0]
		low, high = self.sigmas[low_idx], self.sigmas[high_idx]

		w = (low - sigma) / (low - high)
		w = w.clamp(0, 1)
		t = (1 - w) * low_idx + w * high_idx

		return t.view(sigma.shape)


	def t_to_sigma(self, t):
		t = t.float()
		low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
		return (1 - w) * self.sigmas[low_idx] + w * self.sigmas[high_idx]



class DiscreteEpsDDPMDenoiser(DiscreteSchedule):
	'''A wrapper for discrete schedule DDPM models that output eps (the predicted noise).'''

	def __init__(self, model, alphas_cumprod, quantize):
		super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)
		self.inner_model = model
		self.sigma_data = 1.


	def get_scalings(self, sigma):
		c_out = -sigma
		c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

		return c_out, c_in


	def get_eps(self, *args, **kwargs):
		return self.inner_model(*args, **kwargs)


	def loss(self, input, noise, sigma, **kwargs):
		_, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
		noised_input = input + noise * utils.append_dims(sigma, input.ndim)
		eps = self.get_eps(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)

		return (eps - noise).pow(2).flatten(1).mean(1)


	def forward(self, input, sigma, **kwargs):
		c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
		eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs)

		return input + eps * c_out


class DiscreteVDDPMDenoiser(DiscreteSchedule):
	def __init__(self, model, alphas_cumprod, quantize):
		super().__init__(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, quantize)
		self.inner_model = model
		self.sigma_data = 1.

	def get_scalings(self, sigma):
		c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
		c_out = -sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
		c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
		return c_skip, c_out, c_in

	def get_v(self, *args, **kwargs):
		return self.inner_model(*args, **kwargs)

	def loss(self, input, noise, sigma, **kwargs):
		c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
		noised_input = input + noise * utils.append_dims(sigma, input.ndim)
		model_output = self.get_v(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
		target = (input - c_skip * noised_input) / c_out
		return (model_output - target).pow(2).flatten(1).mean(1)

	def forward(self, input, sigma, **kwargs):
		c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
		return self.get_v(input * c_in, self.sigma_to_t(sigma), **kwargs) * c_out + input * c_skip



class CompVisDenoiser(DiscreteEpsDDPMDenoiser):
	def __init__(self, model, quantize=False, device='cpu'):
		super().__init__(model, model.alphas_cumprod, quantize=quantize)

	def get_eps(self, *args, **kwargs):
		return self.inner_model.apply_model(*args, **kwargs)


class CompVisVDenoiser(DiscreteVDDPMDenoiser):
    def __init__(self, model, quantize=False, device='cpu'):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)

    def get_v(self, x, t, cond, **kwargs):
        return self.inner_model.apply_model(x, t, cond, **kwargs)



class MaskedCompVisDenoiser(CompVisDenoiser):
	def forward(self, x, sigma, cond, uncond, cond_scale, init_latent=None, mask=None,  image_conditioning=None):
		x_in = torch.cat([x] * 2)
		sigma_in = torch.cat([sigma] * 2)
		cond_and_uncond = torch.cat([cond, uncond])

		if image_conditioning is not None:
			image_conditioning = torch.cat([image_conditioning] * 2)

		cond_in = (
			cond_and_uncond if image_conditioning is None 
			else {'c_crossattn': [cond_and_uncond], 'c_concat': [image_conditioning]}
		)

		cond, uncond = super().forward(x_in, sigma_in, cond=cond_in).chunk(2)
		
		denoised = uncond + (cond - uncond) * cond_scale

		if mask is not None:
			mask_inverse = 1.0 - mask
			denoised = (init_latent * mask_inverse) + (mask * denoised)

		return denoised


class MaskedCompVisVDenoiser(CompVisVDenoiser):
	def forward(self, x, sigma, cond, uncond, cond_scale, init_latent=None, mask=None,  image_conditioning=None):
		x_in = torch.cat([x] * 2)
		sigma_in = torch.cat([sigma] * 2)
		cond_and_uncond = torch.cat([cond, uncond])

		if image_conditioning is not None:
			image_conditioning = torch.cat([image_conditioning] * 2)

		cond_in = (
			cond_and_uncond if image_conditioning is None 
			else {'c_crossattn': [cond_and_uncond], 'c_concat': [image_conditioning]}
		)

		cond, uncond = super().forward(x_in, sigma_in, cond=cond_in).chunk(2)
		
		denoised = uncond + (cond - uncond) * cond_scale

		if mask is not None:
			mask_inverse = 1.0 - mask
			denoised = (init_latent * mask_inverse) + (mask * denoised)

		return denoised