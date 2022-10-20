from .common import SamplerInterface
from .schedules_k import __dict__ as schedules
from ..denoisers import MaskedCompVisDenoiser


class KSampler(SamplerInterface):
	def __init__(self, schedule, **opts):
		self.schedule = schedule
		self.scheduler = schedules['schedule_%s' % schedule]
		self.opts = opts


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

		
		return self.scheduler(
			ctx=ctx,
			denoiser=self.denoiser,
			x=x,
			sigmas=sigmas,
			cond=cond,
			uncond=uncond,
			init_latent=init_latent,
			mask=mask,
			**self.opts
		)


