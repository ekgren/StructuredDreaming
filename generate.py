""" Generate images  with CLIP """

import os

import click
import numpy as np
import PIL.Image
import torch

from structure.clip import load, tokenize

@click.command()
@click.pass_context
@click.option('--text', help='Text prompt')
@click.option('--seed', type=int, help='Random seed')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(ctx, text, seed, outdir):
    """Generate images.
    Examples:
    """
    print(text)

    device = torch.device('cuda')
    os.makedirs(outdir, exist_ok=True)

    print("Loading clip.")
    perceptor, normalize_image = load('ViT-B/32', jit=False)

    # Generate images.
    print('Generating image.')
    img = torch.rand(1, 3, 512, 512).transpose(1, 2).transpose(2, 3)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')


if __name__ == "__main__":
    generate_images()