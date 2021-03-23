""" Generate images  with CLIP """

import json
import os

import click
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from structure.clip import load, tokenize
from structure.model import ModelSmall
from structure.utils import process_img, grad_sign

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
    iterations = args["iter"]
    grad_acc_steps = 1
    lr = 0.02
    betas = (0.99, 0.999)

    # Model
    model_size = 512

    # Process image
    patches_no = 32
    downscaled_no = 32
    patch_size_min = 32
    patch_size_max = 224
    scale_size_min = 32
    scale_size_max = 224
    scale_patch_size_min = 32
    scale_patch_size_max = 224
    drop_patch = 0.
    drop_downscaled = 0.
    drop_patch_before_upscale = False
    drop_downscaled_before_upscale = False
    drop_patch_2d = True
    drop_downscale_2d = True
    res_out = 224
    mode = 'area'
    #################

    print("Loading clip.")
    perceptor, normalize_image = load('ViT-B/32', jit=False)
    txt_tok = tokenize(args["text"])
    text_latent = perceptor.encode_text(txt_tok.cuda()).detach()

    model = ModelSmall(size=model_size).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr, betas=betas)

    print('Generating image.')
    for i in tqdm(range(iterations)):
        optimizer.zero_grad()
        img = model()
        img_processed = process_img(img,
                                    patches_no=patches_no,
                                    downscaled_no=downscaled_no,
                                    patch_size_min=patch_size_min,
                                    patch_size_max=patch_size_max,
                                    scale_size_min=scale_size_min,
                                    scale_size_max=scale_size_max,
                                    scale_patch_size_min=scale_patch_size_min,
                                    scale_patch_size_max=scale_patch_size_max,
                                    drop_patch=drop_patch,
                                    drop_downscaled=drop_downscaled,
                                    drop_patch_before_upscale=drop_patch_before_upscale,
                                    drop_downscaled_before_upscale=drop_downscaled_before_upscale,
                                    drop_patch_2d=drop_patch_2d,
                                    drop_downscale_2d=drop_downscale_2d,
                                    res_out=res_out,
                                    mode=mode)
        img_latents = perceptor.encode_image(img_processed)
        loss = 10*torch.cosine_similarity(text_latent, img_latents, dim=-1).mean().neg()
        loss.backward()
        if (i+1) % grad_acc_steps == 0:
            grad_sign(model.w.grad)
            optimizer.step()

        with torch.no_grad():
            model.w.clamp_(0., 1.)

        # DEBUG
        if (i+1) % 20 == 0:
            with torch.no_grad():
                img = model.w.transpose(1, 2).transpose(2, 3).detach()
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{args["outdir"]}/seed{args["seed"]:04d}.png')

    with torch.no_grad():
        img = model.w.transpose(1, 2).transpose(2, 3).detach()
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{args["outdir"]}/seed{args["seed"]:04d}.png')


if __name__ == "__main__":
    main()