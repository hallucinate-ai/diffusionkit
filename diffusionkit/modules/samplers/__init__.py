from .sampler_k import *

def pick_sampler(name):
	if name == 'k_lms':
		return KSampler(schedule=schedule_lms)
	elif name == 'k_euler':
		return KSampler(schedule=schedule_euler)
	elif name == 'k_euler_a':
		return KSampler(schedule=schedule_euler_ancestral)
	elif name == 'k_heun':
		return KSampler(schedule=schedule_heun)
	elif name == 'k_dpm_2':
		return KSampler(schedule=schedule_dpm_2)
	elif name == 'k_dpm_2_a':
		return KSampler(schedule=schedule_dpm_2_ancestral)
	elif name == 'k_dpmpp_2m':
		return KSampler(schedule=schedule_dpmpp_2m)
	else:
		raise Exception('sampler "%s" does not exist' % name)