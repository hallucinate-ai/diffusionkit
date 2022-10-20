class SamplerInterface:
	def use_model(self, model):
		pass

	def sample(self, ctx, noise, cond, uncond, steps, init_latent=None, mask=None):
		raise NotImplemented