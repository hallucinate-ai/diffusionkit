from setuptools import setup

setup(
	name='diffusionkit',
	version='0.9',
	packages=[
		'diffusionkit',
		'diffusionkit.configs',
		'diffusionkit.models',
		'diffusionkit.models.diffusion',
		'diffusionkit.modules',
		'diffusionkit.modules.diffusion',
		'diffusionkit.modules.image_degradation',
		'diffusionkit.modules.losses'
	],
	install_requires=[
		'diffusers',
		'einops',
		'kornia',
		'omegaconf',
		'transformers',
		'pytorch-lightning',
		'clip @ git+https://github.com/openai/CLIP#egg=clip',
		'taming-transformers @ git+https://github.com/illeatmyhat/taming-transformers#egg=taming-transformers',
		'k_diffusion @ git+https://github.com/hlky/k-diffusion-sd#egg=k_diffusion'
	],
	package_data={
		'diffusionkit.configs': ['*.yaml']
	}
)