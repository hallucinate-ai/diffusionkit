import torch
import torch.nn as nn
import open_clip
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
	def __init__(self, pretrained, device='cuda', max_length=77):
		super().__init__()
		
		self.tokenizer = CLIPTokenizer.from_pretrained(pretrained, local_files_only=True)
		self.transformer = CLIPTextModel.from_pretrained(pretrained, local_files_only=True)
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


class FrozenOpenCLIPEmbedder(AbstractEncoder):
	def __init__(
		self, 
		arch, 
		pretrained, 
		device='cuda', 
		max_length=77,
		layer='last'
	):
		super().__init__()

		self.device = device
		self.max_length = max_length
		self.model = open_clip.create_model(
			arch, 
			device=torch.device('cpu'), 
			pretrained=pretrained
		)

		del self.model.visual

		self.freeze()

		if layer == 'last':
			self.layer_idx = 0
		elif layer == 'penultimate':
			self.layer_idx = 1
		else:
			raise NotImplementedError()


	def freeze(self):
		self.model = self.model.eval()

		for param in self.parameters():
			param.requires_grad = False


	def forward(self, text):
		tokens = open_clip.tokenize(text)
		z = self.encode_with_transformer(tokens.to(self.device))

		return z


	def encode_with_transformer(self, text):
		x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
		x = x + self.model.positional_embedding
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
		x = x.permute(1, 0, 2)  # LND -> NLD
		x = self.model.ln_final(x)

		return x


	def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
		for i, r in enumerate(self.model.transformer.resblocks):
			if i == len(self.model.transformer.resblocks) - self.layer_idx:
				break

			x = r(x, attn_mask=attn_mask)

		return x


	def encode(self, text):
		return self(text)