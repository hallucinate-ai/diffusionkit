import torch
import torch.nn as nn
from functools import partial
from transformers import CLIPTokenizer, CLIPTextModel, logging
from .. import config


logging.set_verbosity_error()


class AbstractEncoder(nn.Module):
	def __init__(self):
		super().__init__()

	def encode(self, *args, **kwargs):
		raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
	def __init__(self, weights_file, device='cuda', max_length=77):
		super().__init__()
		
		self.tokenizer = CLIPTokenizer.from_pretrained(weights_file, local_files_only=True)
		self.transformer = CLIPTextModel.from_pretrained(weights_file, local_files_only=True)
		self.device = device
		self.max_length = max_length
		self.freeze()

	def freeze(self):
		self.transformer = self.transformer.eval()
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, text):
		batch_encoding = self.tokenizer(
			text, 
			truncation=True, 
			max_length=self.max_length, 
			return_length=True, 
			return_overflowing_tokens=False, 
			padding='max_length', 
			return_tensors='pt'
		)

		tokens = batch_encoding['input_ids'].to(self.device)
		outputs = self.transformer(input_ids=tokens)

		return outputs.last_hidden_state

	def encode(self, text):
		return self(text)