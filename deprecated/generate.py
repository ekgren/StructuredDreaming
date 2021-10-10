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
from deprecated.model import ImgBaseOld
from structure.utils import (
    Pipeline,
    Upscale,
    Pixelate,
    Dropper,
    Prod,
    SamplePatch,
    grad_sign,
    model_to_fp32,
    ArgDict,
)


@click.command()
@click.pass_context

# General options.
@click.option("--text", help="Text prompt", required=True)
@click.option("--seed", type=int, help="Random seed", default=0)
@click.option(
    "--outdir",
    help="Where to save the output images",
    type=str,
    required=True,
    metavar="DIR",
)

# Training.
@click.option(
    "--iterations",
    help="Number of iterations of generating image [default: 1000]",
    type=int,
    default=1000,
)
@click.option(
    "--grad_acc_steps",
    help="Gradient accumulation steps [default: 1]",
    type=int,
    default=1,
)
@click.option("--lr", help="Learning rate [default: 0.01]", type=float, default=0.01)
# TODO: add betas = (0.99, 0.999)

# Model.
@click.option("--image_size", help="Image size [default: 512]", type=int, default=512)
@click.option(
    "--weight_init", help="Image weight init [default: 0.05]", type=float, default=0.05
)
@click.option(
    "--decolorize", help="Image weight init [default: 0.001]", type=float, default=0.001
)
@click.option(
    "--darken", help="Image weight init [default: 0.005]", type=float, default=0.005
)

# General img options.
# TODO: add mode = 'area'

# Pixelate pipeline.
@click.option("--px_no", help="Number of patches [default: 32]", type=int, default=32)
@click.option(
    "--px_patch_size_min",
    help="Pixelate patch min size [default: 256]",
    type=int,
    default=256,
)
@click.option(
    "--px_patch_size_max",
    help="Pixelate patch max size [default: 512]",
    type=int,
    default=512,
)
@click.option(
    "--px_size_min", help="Pixelation min size [default: 32]", type=int, default=32
)
@click.option(
    "--px_size_max", help="Pixelation max size [default: 224]", type=int, default=224
)
@click.option(
    "--px_drop", help="Pixelation dropout [default: 0.3]", type=float, default=0.3
)

# Upscale pipeline.
@click.option("--up_no", help="Number of patches [default: 32]", type=int, default=32)
@click.option(
    "--up_patch_size_min",
    help="Upscale patch min size [default: 64]",
    type=int,
    default=64,
)
@click.option(
    "--up_patch_size_max",
    help="Upscale patch max size [default: 512]",
    type=int,
    default=512,
)
@click.option(
    "--up_drop", help="Pixelation dropout [default: 0.3]", type=float, default=0.3
)

# Grow parameters.
@click.option(
    "--grow_init_res", help="Number of patches [default: 32]", type=int, default=32
)
@click.option(
    "--grow_step_size", help="Number of patches [default: 64]", type=int, default=64
)
@click.option(
    "--grow_step", help="Number of patches [default: 20]", type=int, default=20
)
def main(ctx, **config_kwargs):
    """Generate images.
    Examples:
    """
    args = ArgDict(**config_kwargs)
    # Print options.
    print()
    print("Generation options:")
    print(json.dumps(args, indent=2))
    print()
    print(f"Output directory:   {args.outdir}")
    print(f"Prompt:             {args.text}")
    print()

    # Initialize
    device = torch.device("cuda")
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    os.makedirs(args["outdir"], exist_ok=True)

    #################
    # PARAMETERS
    #################

    # Training
    betas = (0.99, 0.999)

    # General img options
    res_out = 224
    mode = "area"
    #################

    print("Loading clip.")
    perceptor, normalize_image = load("ViT-B/32", jit=False)
    txt_tok = tokenize(args.text)
    text_latent = perceptor.encode_text(txt_tok.cuda()).detach()

    # Setting up image generation
    model = ImgBaseOld(
        size=args.image_size,
        weight_init=args.weight_init,
        decolorize=args.decolorize,
        darken=args.darken,
    ).cuda()
    normalize = torchvision.transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=betas)

    grow_res = args.grow_init_res
    grow_pipeline = Pipeline(
        Pixelate(scale_size_min=grow_res, scale_size_max=grow_res),
        Upscale(res_out=args.image_size, mode=mode),
    )

    px_pipeline = Pipeline(
        SamplePatch(size_min=args.px_patch_size_min, size_max=args.px_patch_size_max),
        Dropper(drop=args.px_drop, drop2d=True),
        Pixelate(scale_size_min=args.px_size_min, scale_size_max=args.px_size_max),
        Upscale(res_out=res_out, mode=mode),
        Dropper(drop=args.px_drop, drop2d=True),
    )

    up_pipeline = Pipeline(
        SamplePatch(args.up_patch_size_min, args.up_patch_size_max),
        Dropper(drop=args.up_drop, drop2d=True),
        Upscale(res_out=res_out, mode=mode),
        Dropper(drop=args.up_drop, drop2d=True),
    )

    patches = [px_pipeline] * args.px_no + [up_pipeline] * args.up_no
    patches = Prod(*patches)

    # Image generation
    print("Generating image.")
    for i in tqdm(range(args.iterations)):
        optimizer.zero_grad()
        img = normalize_image(model())
        img = grow_pipeline(img)

        img_processed = torch.cat(patches(img), 0)
        img_latents = perceptor.encode_image(img_processed)

        loss = (
            10 * torch.cosine_similarity(text_latent, img_latents, dim=-1).mean().neg()
        )
        loss.backward()

        if (i + 1) % args.grad_acc_steps == 0:
            model_to_fp32(perceptor.visual)
            model_to_fp32(model)
            grad_sign(model.w.grad)
            optimizer.step()
            convert_weights(perceptor.visual)
            convert_weights(model)

        # Post processing
        model.post_process()

        # Update grow resolution
        if (i + 1) % args.grow_step == 0:
            grow_res = min(args.image_size, grow_res + args.grow_step_size)
            grow_pipeline = Pipeline(
                Pixelate(scale_size_min=grow_res, scale_size_max=grow_res),
                Upscale(res_out=args.image_size, mode=mode),
            )

        # DEBUG
        if (i + 1) % 20 == 0:
            with torch.no_grad():
                img = model()
                _img = (img.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
                f_name = f'{args.outdir}/{args.text.replace(" ", "_")}__seed{args.seed:04d}.png'
                PIL.Image.fromarray(_img[0].cpu().numpy(), "RGB").save(f_name)

    with torch.no_grad():
        img = model()
        _img = (img.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
        f_name = f'{args.outdir}/{args.text.replace(" ", "_")}__seed{args.seed:04d}.png'
        PIL.Image.fromarray(_img[0].cpu().numpy(), "RGB").save(f_name)


if __name__ == "__main__":
    main()
