""" Generate images  with CLIP """

import os

import click
import numpy as np
import PIL.Image
import torch

@click.command()
@click.pass_context
@click.option('--text', type=str, help='Text prompt')
@click.option('--seed', type=int, help='Random seed')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(ctx=click.Context, outdir=str):
    """Generate images.
    Examples:
    """

    device = torch.device('cuda')

    os.makedirs(outdir, exist_ok=True)

    # Generate images.
    print('Generating image.')
    img = torch.randn(1, 3, 128, 128)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')


if __name__ == "__main__":
    generate_images()
