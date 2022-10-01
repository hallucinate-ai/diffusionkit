import os
import importlib
import torch
import gc
from omegaconf import OmegaConf

from . import config


models = dict()

def load(name):
	if name in models:
		return models[name]

	if name not in config.weights:
		raise Exception('no weights file path specified for model "%s"' % name)

	weightfile = config.weights[name]
	conf = OmegaConf.load(
		os.path.join(os.path.dirname(__file__), 'configs', '%s.yaml' % name)
	)

	modulename, classname = conf.model.target.rsplit('.', 1)
	module = importlib.import_module(modulename)
	cls = getattr(module, classname)
	model = cls(**conf.model.get("params", dict()))
	checkpoint = torch.load(weightfile, map_location='cpu')

	model.load_state_dict(checkpoint['state_dict'], strict=False)
	model.half()
	model.cuda()
	model.eval()
	models[name] = model

	return model

def unload(name):
	if name not in models:
		return

	print(torch.cuda.memory_allocated())

	del models[name]
	gc.collect()
	torch.cuda.empty_cache()

	print(torch.cuda.memory_allocated())