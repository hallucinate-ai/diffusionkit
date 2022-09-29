from setuptools import setup

setup(
	name='diffusionkit',
	version='0.12',
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
		'pytorch-lightning'
	],
	package_data={
		'diffusionkit.configs': ['*.yaml']
	}
)