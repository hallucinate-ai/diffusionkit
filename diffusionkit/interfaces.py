from dataclasses import dataclass


@dataclass
class DiffusionModelConfig:
	checkpoint_sd: str
	checkpoint_clip: str

@dataclass
class ControlDiffusionModelConfig:
	checkpoint_sd: str
	checkpoint_ctrl: str
	checkpoint_clip: str


@dataclass
class Txt2ImgParams:
	prompt: str
	prompt_negative: str = ''
	width: int = None
	height: int = None
	sampler: str = 'k_lms'
	steps: int = 50
	cfg_scale: float = 5.0
	seed: int = 0
	count: int = 1

@dataclass
class Img2ImgParams:
	prompt: str
	prompt_negative: str = ''
	width: int = None
	height: int = None
	sampler: str = 'k_lms'
	steps: int = 50
	cfg_scale: float = 5.0
	denoising_strength: float = 0.75
	seed: int = 0
	count: int = 1

@dataclass
class Ctrl2ImgParams:
	prompt: str
	prompt_negative: str = ''
	width: int = None
	height: int = None
	sampler: str = 'k_lms'
	steps: int = 50
	cfg_scale: float = 5.0
	ctrl_strength: float = 2
	seed: int = 0
	count: int = 1


class SamplerInterface:
	def use_model(self, model):
		pass

	def sample(self, ctx, noise, cond, uncond, steps, init_latent=None, mask=None):
		raise NotImplemented