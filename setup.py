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
		'Pillow',
		'scipy',
		'einops',
		'omegaconf',
		'transformers'
	],
	package_data={
		'diffusionkit.configs': ['*.yaml']
	}
)