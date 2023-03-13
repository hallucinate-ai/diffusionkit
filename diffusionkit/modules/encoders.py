import torch
import torch.nn as nn
import open_clip
from transformers import CLIPTokenizer, CLIPTextModel, logging
from .prompt import parse_prompt_weights


logging.set_verbosity_error()


class PromptChunk:
	def __init__(self):
		self.tokens = []
		self.multipliers = []
		self.fixes = []


class AbstractEncoder(nn.Module):
	def __init__(self):
		super().__init__()

	def encode(self, *args, **kwargs):
		raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
	def __init__(self, pretrained, device='cuda', max_length=75):
		super().__init__()
		
		self.device = device
		self.tokenizer = CLIPTokenizer.from_pretrained(pretrained, local_files_only=True)
		self.transformer = CLIPTextModel.from_pretrained(pretrained, local_files_only=True)
		self.vocab = self.tokenizer.get_vocab()
		self.token_start = self.tokenizer.bos_token_id
		self.token_end = self.tokenizer.eos_token_id
		self.max_length = max_length
		self.freeze()

	def freeze(self):
		self.transformer = self.transformer.eval()
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, text):
		batch = []

		for line in text:
			parsed = parse_prompt_weights(line)

			batch_encoding = self.tokenizer(
				[text for text, _ in parsed], 
				truncation=True, 
				max_length=self.max_length, 
				return_length=True, 
				return_overflowing_tokens=False, 
				padding='max_length', 
				return_tensors='pt'
			)

			batch.append(
				self.make_weighted_chunks(
					[text for text, _ in parsed],
					batch_encoding['input_ids'],
					[weight for _, weight in parsed],
				)
			)
			
		zs = []
		chunks_len = max([len(x) for x in batch])

		for i in range(chunks_len):
			chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch]
			
			tokens = torch.asarray([x.tokens for x in chunk]).to(self.device)
			multipliers = torch.asarray([x.multipliers for x in chunk]).to(self.device)
			#self.hijack.fixes = [x.fixes for x in chunk]

			z = self.transformer(input_ids=tokens).last_hidden_state

			original_mean = z.mean()
			z = z * multipliers.reshape(multipliers.shape + (1,)).expand(z.shape)
			new_mean = z.mean()
			z = z * (original_mean / new_mean)

			zs.append(z)

		return torch.hstack(zs)


	def encode(self, text):
		return self(text)

	def make_weighted_chunks(self, texts, tokens, weights):
		chunks = []
		chunk = PromptChunk()
		token_count = 0
		last_comma = -1

		def next_chunk(is_last=False):
			nonlocal token_count
			nonlocal last_comma
			nonlocal chunk

			if is_last:
				token_count += len(chunk.tokens)
			else:
				token_count += self.max_length

			to_add = self.max_length - len(chunk.tokens)

			if to_add > 0:
				chunk.tokens += [self.token_end] * to_add
				chunk.multipliers += [1.0] * to_add

			chunk.tokens = [self.token_start] + chunk.tokens + [self.token_end]
			chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

			last_comma = -1
			chunks.append(chunk)
			chunk = PromptChunk()

		for text, toks, weight in zip(texts, tokens, weights):
			if text == 'BREAK' and weight == -1:
				next_chunk()
				continue

			position = 0

			while position < len(toks):
				token = toks[position]

				if len(chunk.tokens) == self.max_length:
					next_chunk()
				
				chunk.tokens.append(token)
				chunk.multipliers.append(weight)
				position += 1

		if len(chunk.tokens) > 0 or len(chunks) == 0:
			next_chunk(is_last=True)

		return chunks




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