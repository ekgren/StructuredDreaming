import random

from torch.nn.functional import dropout, dropout2d, interpolate, avg_pool2d
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


def get_patch_upscaled(img,
                       size_min,
                       size_max,
                       res_out=224,
                       mode='area',
                       drop=0.,
                       drop_before_upscale=False,
                       drop2d=True):
    """ Get patch and scale it to target resolution. """
    img_out = get_random_patch(img, size_min, size_max)
    bs, ch, h, w = img_out.shape

    if drop > 0. and drop_before_upscale:
        if not drop2d:
            img_out = dropout(img_out, p=drop)
        else:
            img_out = dropout2d(img_out, p=drop)

    if h != res_out:
        img_out = interpolate(img_out, (res_out, res_out), mode=mode)

    if drop > 0. and not drop_before_upscale:
        if not drop2d:
            img_out = dropout(img_out, p=drop)
        else:
            img_out = dropout2d(img_out, p=drop)

    return img_out


def get_downscaled(img,
                   scale_size_min,
                   scale_size_max,
                   patch_size_min,
                   patch_size_max,
                   res_out=224,
                   mode='area',
                   drop=0.,
                   drop_before_upscale=False,
                   drop2d=True):
    """
    Get patch and downscale it before upscaling it to target resolution.
    """
    img_out = get_random_patch(img, patch_size_min, patch_size_max)
    bs, ch, h, w = img_out.shape
    downscale = random.randint(scale_size_min, scale_size_max)
    kernel_size = int(h / downscale)
    # TODO: Figure out better way to check that we don't get errors than try/catch
    if downscale < res_out and kernel_size > 0 and kernel_size < h:
        try:
            img_out = avg_pool2d(img_out, kernel_size)
        except:
            pass

    if drop_before_upscale:
        if not drop2d:
            img_out = dropout(img_out, p=drop)
        else:
            img_out = dropout2d(img_out, p=drop)

    img_out = interpolate(img_out, (res_out, res_out), mode=mode)

    if not drop_before_upscale:
        if not drop2d:
            img_out = dropout(img_out, p=drop)
        else:
            img_out = dropout2d(img_out, p=drop)

    return img_out


def get_sub(img, steps=2):
    """ Get grid of image.
        Only support certain image sizes. """
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


def process_img(img,
                patches_no=32,
                downscaled_no=32,
                patch_size_min=32,
                patch_size_max=224,
                scale_size_min=32,
                scale_size_max=224,
                scale_patch_size_min=32,
                scale_patch_size_max=224,
                drop_patch=0.,
                drop_downscaled=0.,
                drop_patch_before_upscale=False,
                drop_downscaled_before_upscale=False,
                drop_patch_2d=True,
                drop_downscale_2d=True,
                res_out=224,
                mode='area'):
    """ Stuff happens here. """
    img_patches = []

    if patches_no > 0:
        for _ in range(patches_no):
            img_patches.append(get_patch_upscaled(img,
                                                  size_min=patch_size_min,
                                                  size_max=patch_size_max,
                                                  res_out=res_out,
                                                  mode=mode,
                                                  drop=drop_patch,
                                                  drop_before_upscale=drop_patch_before_upscale,
                                                  drop2d=drop_patch_2d))

    if downscaled_no > 0:
        for _ in range(downscaled_no):
            img_patches.append(get_downscaled(img,
                                              scale_size_min=scale_size_min,
                                              scale_size_max=scale_size_max,
                                              patch_size_min=scale_patch_size_min,
                                              patch_size_max=scale_patch_size_max,
                                              res_out=res_out,
                                              mode=mode,
                                              drop=drop_downscaled,
                                              drop_before_upscale=drop_downscaled_before_upscale,
                                              drop2d=drop_downscale_2d))

    if len(img_patches) == 0:
        img_patches.append(torch.nn.functional.interpolate(img, (res_out, res_out), mode=mode))

    return torch.cat(img_patches, 0)


def grad_drop(grad, drop=0., drop2d=True):
    """ Dropout on gradient. """
    with torch.no_grad():
        if not drop2d:
            grad += -grad + dropout(grad, drop)
        else:
            grad += -grad + dropout2d(grad, drop)


def grad_sign(grad):
    """ Convert gradient to signed gradient. """
    with torch.no_grad():
        grad += -grad + grad.sign()
