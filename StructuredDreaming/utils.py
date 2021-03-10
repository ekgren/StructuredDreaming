import random

import torch


def get_random_patch(img, size_min, size_max):
    """ Get uniformly sampled patch from image. """
    bs, ch, h, w = img.shape
    assert h == w, "Assumes equal width and height of image."
    assert size_max <= h, "size_max has to be smaller than image height."
    assert size_min <= size_max, "size_min should be equal or smaller than size_max."
    assert size_min >= 2, "size_min can't be smaller than 2."
    if size_min == size_max:
        patch_size = size_max
    else:
        patch_size = random.randint(size_min, size_max)
    h_offset = random.randint(0, h - patch_size)
    w_offset = random.randint(0, w - patch_size)
    patch_h_s = h_offset
    patch_h_e = h_offset + patch_size
    patch_w_s = w_offset
    patch_w_e = w_offset + patch_size
    img_out = img[:, :, patch_h_s:patch_h_e, patch_w_s:patch_w_e]
    return img_out


def get_patch_upscaled(img, size_min, size_max, res_out=224, mode='area'):
    """ Get patch and scale it to target resolution. """
    img_out = get_random_patch(img, size_min, size_max)
    bs, ch, h, w = img_out.shape
    if h != 224:
        img_out = torch.nn.functional.interpolate(img_out, (res_out, res_out), mode=mode)
    return img_out


def get_downscaled(img, scale_size_min, scale_size_max, patch_size_min, patch_size_max, res_out=224, mode='area'):
    """ Get patch and downscale it before upscaling it to target resolution. """
    img_out = get_random_patch(img, patch_size_min, patch_size_max)
    bs, ch, h, w = img_out.shape
    downscale = random.randint(scale_size_min, scale_size_max)
    kernel_size = int(h / downscale)
    # TODO: Figure out better way to check that we don't get errors than try/catch
    if downscale < 224 and kernel_size > 0 and kernel_size < h:
        try:
            img_out = torch.nn.functional.avg_pool2d(img_out, kernel_size)
        except:
            pass
    img_out = torch.nn.functional.interpolate(img_out, (res_out, res_out), mode=mode)
    return img_out


def get_sub(img, steps=2):
    """ Get grid of image """
    bs, ch, h, w = img.shape
    outs = []
    for h_step in range(steps):
        for w_step in range(steps):
            h_ixs = torch.arange(h_step, h, steps)
            w_ixs = torch.arange(w_step, h, steps)
            _w = w_ixs.repeat(int(h/steps))
            _h = h_ixs.repeat_interleave(int(h/steps))
            outs.append(img[:, :, _h, _w].reshape(1, 3, int(h/steps), int(h/steps)))
    return outs
