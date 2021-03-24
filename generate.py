""" Generate images  with CLIP """

import json
import os

import click
import numpy as np
import PIL.Image
import torch
import torchvision
from tqdm import tqdm

from structure.clip import load, tokenize, convert_weights
from structure.model import ImgBase
from structure.utils import Pipeline, Upscale, Pixelate, Dropper, Prod, SamplePatch, grad_sign, model_to_fp32

@click.command()
@click.pass_context

# First try at options.
@click.option('--text', help='Text prompt')
@click.option('--iter', type=int, help='Number of iterations')
@click.option('--seed', type=int, help='Random seed')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def main(ctx, **config_kwargs):
    """Generate images.
    Examples:
    """
    args = dict(**config_kwargs)
    # Print options.
    print()
    print('Generation options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args["outdir"]}')
    print(f'Prompt:             {args["text"]}')
    print()

    # Initialize
    device = torch.device('cuda')
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    os.makedirs(args['outdir'], exist_ok=True)

    #################
    # PARAMETERS
    #################

    # Training
    iterations = 2000
    grad_acc_steps = 1
    lr = 0.01
    betas = (0.99, 0.999)

    # Model
    image_size = 512
    weight_init = 0.05
    decolorize = 0.001
    darken = 0.01

    # General img options
    res_out = 224
    mode = 'area'

    # Pixelate pipeline
    px_no = 64
    px_patch_size_min = 256
    px_patch_size_max = 512
    px_size_min = 32
    px_size_max = 224
    px_drop = 0.3

    # Upscale pipeline
    up_no = 64
    up_patch_size_min = 64
    up_patch_size_max = 512
    up_drop = 0.3

    # Grow parameters
    grow_init_res = 32
    grow_step_size = 64
    grow_step = 20
    #################

    print("Loading clip.")
    perceptor, normalize_image = load('ViT-B/32', jit=False)
    txt_tok = tokenize(args["text"])
    text_latent = perceptor.encode_text(txt_tok.cuda()).detach()

    # Setting up image generation
    model = ImgBase(size=image_size,
                    weight_init=weight_init,
                    decolorize=decolorize,
                    darken=darken).cuda()
    normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                 (0.26862954, 0.26130258, 0.27577711))
    optimizer = torch.optim.Adam(model.parameters(), lr, betas=betas)

    grow_res = grow_init_res
    grow_pipeline = Pipeline(
        Pixelate(scale_size_min=grow_res, scale_size_max=grow_res),
        Upscale(res_out=image_size, mode=mode),
    )

    px_pipeline = Pipeline(
        SamplePatch(size_min=px_patch_size_min, size_max=px_patch_size_max),
        Dropper(drop=px_drop, drop2d=True),
        Pixelate(scale_size_min=px_size_min, scale_size_max=px_size_max),
        Upscale(res_out=res_out, mode=mode),
        Dropper(drop=px_drop, drop2d=True),
    )

    up_pipeline = Pipeline(
        SamplePatch(up_patch_size_min, up_patch_size_max),
        Dropper(drop=up_drop, drop2d=True),
        Upscale(res_out=res_out, mode=mode),
        Dropper(drop=up_drop, drop2d=True),
    )

    patches = [px_pipeline] * px_no + [up_pipeline] * up_no
    patches = Prod(*patches)

    # Image generation
    print('Generating image.')
    for i in tqdm(range(iterations)):
        optimizer.zero_grad()
        img = normalize(model())
        img = grow_pipeline(img)

        img_processed = torch.cat(patches(img), 0)
        img_latents = perceptor.encode_image(img_processed)

        loss = 10 * torch.cosine_similarity(text_latent, img_latents, dim=-1).mean().neg()
        loss.backward()

        if (i + 1) % grad_acc_steps == 0:
            model_to_fp32(perceptor.visual)
            model_to_fp32(model)
            grad_sign(model.w.grad)
            optimizer.step()
            convert_weights(perceptor.visual)
            convert_weights(model)

        # Post processing
        model.post_process()

        # Update grow resolution
        if (i + 1) % grow_step == 0:
            grow_res = min(image_size, grow_res + grow_step_size)
            grow_pipeline = Pipeline(
                Pixelate(scale_size_min=grow_res, scale_size_max=grow_res),
                Upscale(res_out=image_size, mode=mode),
            )

        # DEBUG
        if (i+1) % 20 == 0:
            with torch.no_grad():
                img = model()
                _img = (img.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(_img[0].cpu().numpy(), 'RGB').save(f'{args["outdir"]}/seed{args["seed"]:04d}.png')

    with torch.no_grad():
        img = model()
        _img = (img.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(_img[0].cpu().numpy(), 'RGB').save(f'{args["outdir"]}/seed{args["seed"]:04d}.png')


if __name__ == "__main__":
    main()
