from setuptools import setup

setup(
	name='diffusionkit',
	version='0.16',
	packages=[
		'diffusionkit',
		'diffusionkit.models',
		'diffusionkit.models.diffusion',
		'diffusionkit.modules',
		'diffusionkit.modules.diffusion',
	],
	install_requires=[
		'Pillow',
		'scipy',
		'einops',
		'transformers>=4.25.1',
		'open-clip-torch==2.7.0'
	]
)