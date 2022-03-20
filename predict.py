import sys
sys.path.insert(0, 'stylegan2-ada-pytorch')
import tempfile
from pathlib import Path
import PIL
import torch
import cog
# StructuredDreaming imports
import structure
from structure import clip, sample, optim
# Stylegan imports
import dnnlib
import legacy


class Predictor(cog.Predictor):
    def setup(self):
        self.perceptor, self.normalize_image = structure.clip.load('ViT-B/16', jit=False)
        self.device = torch.device('cuda')

    @cog.input(
        "prompt",
        type=str,
        help="prompt for generating images",
    )
    @cog.input(
        "iterations",
        type=int,
        default=300,
        max=1000,
        min=1,
        help="iterations for generating images"
    )
    @cog.input(
        "display_frequency",
        type=int,
        default=30,
        min=1,
        help="display frequency for intermediate generated images"
    )
    def predict(self, prompt, iterations=300, display_frequency=30):
        with dnnlib.util.open_url('ffhq.pkl') as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self.device)  # type: ignore
        for p in G.parameters():
            p.requires_grad = True
        c = None
        # Training parameters
        grad_acc_steps = 1
        batch_size = 1
        lr = 5e-4
        loss_scale = 100.
        truncation_psi = 0.6
        clamp_val = 1e-30

        # Sampler
        kernel_min = 1
        kernel_max = 8

        iterations = int(iterations)
        display_frequency = int(min(iterations, display_frequency))
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        txt_tok = structure.clip.tokenize(prompt)
        text_latent = self.perceptor.encode_text(txt_tok.to(self.device)).detach()
        sampler = torch.jit.script(
            structure.sample.ImgSampleStylegan(kernel_min=kernel_min,
                                               kernel_max=kernel_max).to(self.device)
        )
        optimizer = structure.optim.ClampSGD(list(G.parameters()),
                                             lr=lr,
                                             clamp=clamp_val)

        im_no = 0
        print('Generating image.')
        for i in range(iterations):
            if i > 0 and i % display_frequency == 0:
                with torch.no_grad():
                    z = torch.randn([1, G.z_dim], device=self.device)
                    img = G(z, c, truncation_psi)
                    img = stylegan_to_rgb(img)
                    yield checkin(img, out_path, i, loss)
                    im_no += 1

            for _ in range(grad_acc_steps):
                optimizer.zero_grad()
                z = torch.randn([1, G.z_dim], device=self.device)
                img = G(z, c, truncation_psi)
                img = stylegan_to_rgb(img)
                img = sampler(img, size=224, bs=batch_size)
                img = self.normalize_image(img)
                img_latents = self.perceptor.encode_image(img)
                loss = torch.cosine_similarity(text_latent, img_latents, dim=-1).mean().neg() * loss_scale
                loss.backward()

            optimizer.step()

        z = torch.randn([1, G.z_dim], device=self.device)
        img = G(z, c, truncation_psi)
        img = stylegan_to_rgb(img)
        yield checkin(img, out_path)


@torch.no_grad()
def checkin(img, out_path, i=None, loss=None):
    if i is not None and i > 0:
        sys.stderr.write(f'step: {i}, loss: {loss.item()}, img.min: {img.min().item()}, img.max: {img.max().item()}\n')
    save_image(img, out_path)
    return out_path


def stylegan_to_rgb(input: torch.Tensor) -> torch.Tensor:
    return (input * 127.5 + 128) / 255


def save_image(input: torch.Tensor, out_path, size=1):
    """ Assumes tensor values in the range [0, 1] """
    with torch.no_grad():
        batch_size, num_channels, height, width = input.shape
        img = torch.nn.functional.interpolate(input, (int(size * height), int(size * width)), mode='area')
        img_show = img.cpu()[0].transpose(0, 1).transpose(1, 2)
        img_out = (img_show * 255).clamp(0, 255).to(torch.uint8)
        img = PIL.Image.fromarray(img_out.cpu().numpy(), 'RGB')
        img.save(out_path)
