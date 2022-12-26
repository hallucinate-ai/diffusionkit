import torch
import numpy as np
from PIL import Image


def image_to_tensor(image):
	if image.mode == 'RGBA':
		image = image.to('RGB')

	if image.mode == 'RGB':
		image = np.array(image, dtype=np.float32)
		image = 2. * (image / 255.0) - 1.
		image = torch.from_numpy(image)
		image = torch.permute(image, (2, 0, 1))
		image = image.unsqueeze(0)

	elif image.mode == 'L':
		image = np.array(image, dtype=np.float32)
		image = 2. * (image / 255.0) - 1.
		image = torch.from_numpy(image)
		image = image.unsqueeze(0)
		image = image.unsqueeze(0)
	
	image = image.half()
	image = image.cuda()

	return image


def resize_image(im, width, height, mode='stretch'):
	if im.width == width and im.height == height:
		return im

	if mode == 'stretch':
		res = im.resize((width, height), resample=Image.LANCZOS)
	elif mode == 'pad':
		ratio = width / height
		src_ratio = im.width / im.height

		src_w = width if ratio > src_ratio else im.width * height // im.height
		src_h = height if ratio <= src_ratio else im.height * width // im.width

		resized = im.resize((src_w, src_h), resample=Image.LANCZOS)
		res = Image.new("RGBA", (width, height))
		res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
	elif mode == 'repeat':
		ratio = width / height
		src_ratio = im.width / im.height

		src_w = width if ratio < src_ratio else im.width * height // im.height
		src_h = height if ratio >= src_ratio else im.height * width // im.width

		resized = im.resize((src_w, src_h), resample=Image.LANCZOS)
		res = Image.new("RGBA", (width, height))
		res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

		if ratio < src_ratio:
			fill_height = height // 2 - src_h // 2
			res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
			res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
		elif ratio > src_ratio:
			fill_width = width // 2 - src_w // 2
			res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
			res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

	return res