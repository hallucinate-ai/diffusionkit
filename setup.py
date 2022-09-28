from setuptools import setup

setup(
	name='diffusionkit',
	version='0.11',
	packages=[
		'diffusionkit',
		'diffusionkit.configs',
		'diffusionkit.models',
		'diffusionkit.models.diffusion',
		'diffusionkit.modules',
		'diffusionkit.modules.diffusion',
	],
	install_requires=[
		'einops',
		'omegaconf',
		'transformers',
		'pytorch-lightning',
		'k_diffusion @ git+https://github.com/hlky/k-diffusion-sd#egg=k_diffusion'
	],
	package_data={
		'diffusionkit.configs': ['*.yaml']
	}
)