from diffusionkit.modules.utils import latent_to_images
from .common import SamplerInterface
from .schedules_k import __dict__ as schedules
from ..denoisers import MaskedCompVisDenoiser


class KSampler(SamplerInterface):
	def __init__(self, schedule, **schedule_args):
		self.schedule = schedule
		self.schedule_args = schedule_args


	def use_model(self, model):
		self.model = model
		self.denoiser = MaskedCompVisDenoiser(model)


	def sample(self, ctx, noise, cond, uncond, steps, init_latent=None, mask=None):
		sigmas = self.denoiser.get_sigmas(ctx.params.steps)

		if init_latent is not None:
			offset = ctx.params.steps - steps - 1
			x = noise * sigmas[offset]
			x = x + init_latent
			sigmas = sigmas[offset:]
		else:
			x = noise * sigmas[0]
			

		def denoise(x, sigma):
			denoised = self.denoiser(
				x, 
				sigma=sigma, 
				cond=cond, 
				uncond=uncond, 
				cond_scale=ctx.params.cfg_scale, 
				init_latent=init_latent, 
				mask=mask
			)

			if ctx.wants_intermediate():
				ctx.put_intermediate(latent_to_images(denoised, model=self.model))

			return denoised


		schedule = schedules['schedule_%s' % self.schedule]
		scheduler = schedule(sigmas, denoise=denoise, **self.schedule_args)

		steps = len(sigmas) - 1

		for i in ctx.make_sampling_iter(range(steps)):
			x = scheduler.step(x, i)

		return x