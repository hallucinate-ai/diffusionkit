import torch
import numpy as np
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.import_path import import_parent
from diffusers import EulerDiscreteScheduler

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.pipeline_stable_diffusion_ait import StableDiffusionAITPipeline
from src.pipeline_stable_diffusion_img2img_ait import StableDiffusionImg2ImgAITPipeline
from src.pipeline_stable_diffusion_controlnet_ait import StableDiffusionAITPipeline

from PIL import Image
from math import ceil

from .loader import load_stable_diffusion
from .interfaces import DiffusionModelConfig, ControlDiffusionModelConfig, Txt2ImgParams, Img2ImgParams, Ctrl2ImgParams
from .image import resize_image, image_to_tensor, mask_to_tensor
from .modules.utils import create_random_tensors, latent_to_images
from .modules.samplers import pick_sampler
from .context import DiffusionContext

def prepare_image(
    image,
    width,
    height,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
    do_classifier_free_guidance=False,
    guess_mode=False,
):
    if not isinstance(image, torch.Tensor):
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image[0], Image.Image):
            images = []

            for image_ in image:
                image_ = image_.convert("RGB")
                image_ = image_.resize((width, height), resample=Image.LANCZOS)
                image_ = np.array(image_)
                image_ = image_[None, :]
                images.append(image_)

            image = images

            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)

    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    if do_classifier_free_guidance and not guess_mode:
        image = torch.cat([image] * 2)

    return image




def txt2img_AIT(config: DiffusionModelConfig, params: Txt2ImgParams):
    result_images = []
    checkpoint = config.checkpoint_sd
    local_dir = "./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2"
    pipe = StableDiffusionAITPipeline.from_pretrained(
        local_dir,
        scheduler=EulerDiscreteScheduler.from_pretrained(
            local_dir, subfolder="scheduler"
        ),
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")
    batch = 1
    benchmark = params.benchmark
    prompt = params.prompt
    width = params.width
    height = params.height
    negative_prompt = params.prompt_negative
    batch = params.count
    cfg_scale = params.cfg_scale
    steps = params.steps
    
    with torch.autocast("cuda"):
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
        ).images[0]
        if benchmark:
            t = benchmark_torch_function(10, pipe, prompt, height=height, width=width)
            print(
                f"sd e2e: width={width}, height={height}, batchsize={batch}, latency={t} ms"
            )
    #image.save("example_ait.png")
    image = resize_image(image, params.width, params.height)
    result_images.append(image)
    return result_images


def img2img_AIT(config: DiffusionModelConfig, params: Img2ImgParams, image: Image):
    batch = 1
    benchmark = params.benchmark
    prompt = params.prompt
    width = params.width
    strength = params.strength
    height = params.height
    negative_prompt = params.prompt_negative
    steps = params.steps
    batch = params.count
    result_images = []
    cfg_scale = params.cfg_scale
    init_image = image
    init_image = init_image.resize((height, width))
    checkpoint = config.checkpoint_sd
    local_dir = "./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2"
    pipe = StableDiffusionAITPipeline.from_pretrained(
        local_dir,
        scheduler=EulerDiscreteScheduler.from_pretrained(
            local_dir, subfolder="scheduler"
        ),
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")
    
    with torch.autocast("cuda"):
        images = pipe(
            prompt=prompt, init_image=init_image, strength=strength, guidance_scale=cfg_scale
        ).images
        if benchmark:
            t = benchmark_torch_function(10, pipe, prompt, height=height, width=width)
            print(
                f"sd e2e: width={width}, height={height}, batchsize={batch}, latency={t} ms"
            )
    #image.save("example_ait.png")
    image = resize_image(image, params.width, params.height)
    result_images.append(image)
    return result_images

def ctrl2img_AIT(config: DiffusionModelConfig, params: Img2ImgParams, image: Image):
    batch = 1
    benchmark = params.benchmark
    prompt = params.prompt
    width = params.width
    strength = params.strength
    height = params.height
    negative_prompt = params.prompt_negative
    steps = params.steps
    batch = params.count
    result_images = []
    cfg_scale = params.cfg_scale
    init_image = image
    image = np.array(image)
    # get canny image
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    control_cond = prepare_image(
        canny_image,
        width,
        height,
        batch,
        1,
        "cuda",
        torch.float16,
        do_classifier_free_guidance=True,
    )    
    checkpoint = config.checkpoint_sd
    local_dir = "./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2"
    pipe = StableDiffusionAITPipeline.from_pretrained(
        local_dir,
        scheduler=EulerDiscreteScheduler.from_pretrained(
            local_dir, subfolder="scheduler"
        ),
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")
    
    with torch.autocast("cuda"):
        image = pipe(
            prompt=prompt,
            control_cond=control_cond,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
        ).images[0]
        if benchmark:
            t = benchmark_torch_function(10, pipe, prompt, height=height, width=width)
            print(
                f"sd e2e: width={width}, height={height}, batchsize={batch}, latency={t} ms"
            )
    #image.save("example_ait.png")
    image = resize_image(image, params.width, params.height)
    result_images.append(image)
    return result_images


def txt2img(config: DiffusionModelConfig, params: Txt2ImgParams):
    ctx = DiffusionContext(params=params)
    ctx.report_stage('init')
    
    result_images = []
    batch_size = 1
    prompt = params.prompt
    prompt_negative = params.prompt_negative
    seeds = [params.seed + x for x in range(params.count)]

    width = ceil(params.width / 64) * 64
    height = ceil(params.height / 64) * 64
    width_latent = width // 8
    height_latent = height // 8
    
    sampler = pick_sampler(params.sampler)
    model = load_stable_diffusion(config)
    cond = model.get_learned_conditioning([prompt] * params.count)
    uncond = model.get_learned_conditioning([prompt_negative] * params.count)

    sampler.use_model(model)

    with torch.no_grad(), torch.autocast('cuda'):
        denoising_steps = params.steps
        ctx.report_sampling_steps(denoising_steps)

        for i in range(0, params.count, batch_size):
            batch_seeds = seeds[i:i+batch_size]

            noise = create_random_tensors([4, height_latent, width_latent], seeds=batch_seeds)
            noise = noise.cuda()

            samples = sampler.sample(
                ctx=ctx,
                noise=noise, 
                cond=cond, 
                uncond=uncond, 
                steps=params.steps
            )

            ctx.report_stage('decode')

            images = latent_to_images(samples, model)

            for image in images:
                image = resize_image(image, params.width, params.height)
                result_images.append(image)


    ctx.finish()

    return result_images




def img2img(config: DiffusionModelConfig, params: Img2ImgParams, image: Image, mask: Image = None):
    assert 0. <= params.denoising_strength <= 1, 'denoising_strength must be between [0.0, 1.0]'

    ctx = DiffusionContext(params=params, image=image)
    ctx.report_stage('init')
    
    result_images = []
    batch_size = 1
    prompt = params.prompt
    prompt_negative = params.prompt_negative
    seeds = [params.seed + x for x in range(params.count)]

    if not params.width or not params.height:
        params.width = image.width
        params.height = image.height

    width = ceil(params.width / 64) * 64
    height = ceil(params.height / 64) * 64
    width_latent = width // 8
    height_latent = height // 8

    image = resize_image(image, width, height)
    image = image_to_tensor(image)

    if mask:
        mask = resize_image(mask, width, height)
        mask = mask_to_tensor(mask)


    sampler = pick_sampler(params.sampler)
    model = load_stable_diffusion(config)
    cond = model.get_learned_conditioning([prompt] * params.count)
    uncond = model.get_learned_conditioning([prompt_negative] * params.count)

    sampler.use_model(model)

    assert mask is not None if model.is_inpainting_model else True, 'the loaded model is an inpainting model and thus needs a mask'


    with torch.no_grad(), torch.autocast('cuda'):
        denoising_steps = int(
            min(params.denoising_strength, 0.999) 
            * params.steps
        )

        ctx.report_sampling_steps(denoising_steps)
        ctx.report_stage('encode')

        first_encoding = model.encode_first_stage(image)
        init_latent = model.get_first_stage_encoding(first_encoding)
        
        if model.is_inpainting_model:
            conditioning_mask = torch.round(mask)
            conditioning_image = torch.lerp(
                image,
                image * (1.0 - conditioning_mask),
                1 # todo: make parameter inpainting_mask_weight
            )

            conditioning_encoding = model.encode_first_stage(conditioning_image)
            conditioning_image = model.get_first_stage_encoding(conditioning_encoding)
            conditioning_mask = torch.nn.functional.interpolate(conditioning_mask, size=init_latent.shape[-2:])
            conditioning_mask = conditioning_mask.expand(conditioning_image.shape[0], -1, -1, -1)

            image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)
            image_conditioning = image_conditioning.cuda().half()
        else:
            image_conditioning = None


        for i in range(0, params.count, batch_size):
            batch_seeds = seeds[i:i+batch_size]

            noise = create_random_tensors([4, height_latent, width_latent], seeds=batch_seeds)
            noise = noise.cuda()

            samples = sampler.sample(
                ctx=ctx,
                noise=noise, 
                cond=cond, 
                uncond=uncond, 
                steps=denoising_steps, 
                init_latent=init_latent, 
                mask=mask,
                image_conditioning=image_conditioning
            )

            ctx.report_stage('decode')

            images = latent_to_images(samples, model)

            for image in images:
                image = resize_image(image, params.width, params.height)
                result_images.append(image)


    ctx.finish()

    return result_images




def ctrl2img(config: ControlDiffusionModelConfig, params: Ctrl2ImgParams, control: Image, mask: Image = None):
    ctx = DiffusionContext(params=params, image=control)
    ctx.report_stage('init')
    
    result_images = []
    batch_size = 1
    prompt = params.prompt
    prompt_negative = params.prompt_negative
    seeds = [params.seed + x for x in range(params.count)]

    if not params.width or not params.height:
        params.width = control.width
        params.height = control.height

    width = ceil(params.width / 64) * 64
    height = ceil(params.height / 64) * 64
    width_latent = width // 8
    height_latent = height // 8

    control = resize_image(control, width, height)
    control = image_to_tensor(control)

    if mask:
        mask = resize_image(mask, width, height)
        mask = mask_to_tensor(mask)


    sampler = pick_sampler(params.sampler)
    model = load_stable_diffusion(config, controlnet=True)
    cond = model.get_learned_conditioning([prompt] * params.count)
    uncond = model.get_learned_conditioning([prompt_negative] * params.count)

    sampler.use_model(model)

    assert mask is not None if model.is_inpainting_model else True, 'the loaded model is an inpainting model and thus needs a mask'


    with torch.no_grad(), torch.autocast('cuda'):
        ctx.report_sampling_steps(params.steps)
        ctx.report_stage('encode')

        model.control_scales = [params.ctrl_strength] * 13

        for i in range(0, params.count, batch_size):
            batch_seeds = seeds[i:i+batch_size]

            noise = create_random_tensors([4, height_latent, width_latent], seeds=batch_seeds)
            noise = noise.cuda()

            samples = sampler.sample(
                ctx=ctx,
                noise=noise, 
                cond=cond, 
                uncond=uncond, 
                steps=params.steps, 
                mask=mask,
                image_conditioning=control
            )

            ctx.report_stage('decode')

            images = latent_to_images(samples, model)

            for image in images:
                image = resize_image(image, params.width, params.height)
                result_images.append(image)


    ctx.finish()

    return result_images






