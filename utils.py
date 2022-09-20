import torch

def create_random_tensors(shape, seeds):
	xs = []

	for seed in seeds:
		torch.manual_seed(seed)
		xs.append(torch.randn(shape, device='cpu'))

	return torch.stack(xs)