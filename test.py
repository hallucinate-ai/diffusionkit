from diffuse import diffuse, DiffuseParams
from config import checkpoint_files
from PIL import Image

checkpoint_files['stable_diffusion_v1'] = '/home/diffusion/stablediffusion/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt'

image = Image.open('test.jpg')
params = DiffuseParams(
	prompt='A purple car'
)

results = diffuse(params, image)
results[0].save('test.result.jpg')