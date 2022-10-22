import torch

from ..utils import latent_to_images, to_d, linear_multistep_coeff, get_ancestral_step


class common_gamma:
	def __init__(self, sigmas, denoise, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
		self.denoise = denoise
		self.sigmas = sigmas
		self.s_churn = s_churn
		self.s_tmin = s_tmin
		self.s_tmax = s_tmax
		self.s_noise = s_noise

	def apply_gamma(self, x, i):
		gamma = min(self.s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1) if self.s_tmin <= self.sigmas[i] <= self.s_tmax else 0.
		eps = torch.randn_like(x) * self.s_noise
		sigma_hat = self.sigmas[i] * (gamma + 1)

		if gamma > 0:
			x = x + eps * (sigma_hat ** 2 - self.sigmas[i] ** 2) ** 0.5

		return x, sigma_hat



class schedule_euler(common_gamma):
	def step(self, x, i):
		x, sigma_hat = self.apply_gamma(x, i)
		denoised = self.denoise(x, sigma=sigma_hat * x.new_ones([x.shape[0]]))

		d = to_d(x, sigma_hat, denoised)
		dt = self.sigmas[i + 1] - sigma_hat
		x = x + d * dt

		return x



class schedule_heun(common_gamma):
	def step(self, x, i):
		x, sigma_hat = self.apply_gamma(x, i)
		denoised = self.denoise(x, sigma=sigma_hat * x.new_ones([x.shape[0]]))

		d = to_d(x, sigma_hat, denoised)
		dt = self.sigmas[i + 1] - sigma_hat
		
		if self.sigmas[i + 1] == 0:
			x = x + d * dt
		else:
			x_2 = x + d * dt
			denoised_2 = self.denoise(x_2, sigma=self.sigmas[i + 1] * x.new_ones([x.shape[0]]))
			d_2 = to_d(x_2, self.sigmas[i + 1], denoised_2)
			d_prime = (d + d_2) / 2
			x = x + d_prime * dt

		return x



class schedule_dpm_2(common_gamma):
	def step(self, x, i):
		x, sigma_hat = self.apply_gamma(x, i)
		denoised = self.denoise(x, sigma=sigma_hat * x.new_ones([x.shape[0]]))

		d = to_d(x, sigma_hat, denoised)
		dt = self.sigmas[i + 1] - sigma_hat
		
		if self.sigmas[i + 1] == 0:
			dt = self.sigmas[i + 1] - sigma_hat
			x = x + d * dt
		else:
			sigma_mid = sigma_hat.log().lerp(self.sigmas[i + 1].log(), 0.5).exp()
			dt_1 = sigma_mid - sigma_hat
			dt_2 = self.sigmas[i + 1] - sigma_hat
			x_2 = x + d * dt_1
			denoised_2 = self.denoise(x_2, sigma=sigma_mid * x.new_ones([x.shape[0]]))
			d_2 = to_d(x_2, sigma_mid, denoised_2)
			x = x + d_2 * dt_2

		return x



class schedule_euler_ancestral:
	def __init__(self, sigmas, denoise, eta=1.):
		self.denoise = denoise
		self.sigmas = sigmas
		self.eta = eta

	def step(self, x, i):
		denoised = self.denoise(x, sigma=self.sigmas[i] * x.new_ones([x.shape[0]]))
		sigma_down, sigma_up = get_ancestral_step(self.sigmas[i], self.sigmas[i + 1], eta=self.eta)

		d = to_d(x, self.sigmas[i], denoised)
		dt = sigma_down - self.sigmas[i]

		x = x + d * dt
		x = x + torch.randn_like(x) * sigma_up

		return x



class schedule_dpm_2_ancestral:
	def __init__(self, sigmas, denoise, eta=1.):
		self.denoise = denoise
		self.sigmas = sigmas
		self.eta = eta

	def step(self, x, i):
		denoised = self.denoise(x, sigma=self.sigmas[i] * x.new_ones([x.shape[0]]))
		sigma_down, sigma_up = get_ancestral_step(self.sigmas[i], self.sigmas[i + 1], eta=self.eta)

		d = to_d(x, self.sigmas[i], denoised)

		if sigma_down == 0:
			dt = sigma_down - self.sigmas[i]
			x = x + d * dt
		else:
			sigma_mid = self.sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
			dt_1 = sigma_mid - self.sigmas[i]
			dt_2 = sigma_down - self.sigmas[i]
			x_2 = x + d * dt_1
			denoised_2 = self.denoise(x_2, sigma=sigma_mid * x.new_ones([x.shape[0]]))
			d_2 = to_d(x_2, sigma_mid, denoised_2)
			x = x + d_2 * dt_2
			x = x + torch.randn_like(x) * sigma_up

		return x



class schedule_lms:
	def __init__(self, sigmas, denoise, order=4):
		self.denoise = denoise
		self.sigmas = sigmas
		self.order = order
		self.ds = []

	def step(self, x, i):
		cur_order = min(i + 1, self.order)
		denoised = self.denoise(x, sigma=self.sigmas[i] * x.new_ones([x.shape[0]]))

		d = to_d(x, self.sigmas[i], denoised)
		self.ds.append(d)

		if len(self.ds) > self.order:
			self.ds.pop(0)

		coeffs = [
			linear_multistep_coeff(cur_order, self.sigmas.cpu(), i, j) 
			for j in range(cur_order)
		]

		return x + sum(coeff * d for coeff, d in zip(coeffs, reversed(self.ds)))