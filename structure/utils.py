import random

from torch.nn.functional import dropout, dropout2d, interpolate, avg_pool2d
import torch


class SamplePatch(object):
    def __init__(self, size_min, size_max):
        assert size_min <= size_max, "size_min should be equal or smaller than size_max."
        assert size_min >= 2, "size_min can't be smaller than 2."

        self.size_min = size_min
        self.size_max = size_max

    def __call__(self, img):
        """ Get uniformly sampled patch from image. """
        bs, ch, h, w = img.shape
        assert h == w, "Assumes equal width and height of image."
        assert self.size_max <= h, "size_max has to be smaller than or equal to image height."
        if self.size_min == self.size_max:
            patch_size = self.size_max
        else:
            patch_size = random.randint(self.size_min, self.size_max)
        h_offset = random.randint(0, h - patch_size)
        w_offset = random.randint(0, w - patch_size)
        patch_h_s = h_offset
        patch_h_e = h_offset + patch_size
        patch_w_s = w_offset
        patch_w_e = w_offset + patch_size
        img_out = img[:, :, patch_h_s:patch_h_e, patch_w_s:patch_w_e]
        return img_out


class Dropper(object):
    def __init__(self, drop=0., drop2d=False):
        self.drop = drop
        self.drop2d = drop2d

    def __call__(self, img):
        if not self.drop2d:
            return dropout(img, p=self.drop)
        else:
            return dropout2d(img, p=self.drop)


class Upscale(object):
    def __init__(self, res_out=224, mode='area'):
        self.res_out = res_out
        self.mode = mode

    def __call__(self, img):
        bs, ch, h, w = img.shape
        if h != self.res_out:
            return interpolate(img, (self.res_out, self.res_out), mode=self.mode)
        else:
            return img


class Pixelate(object):
    def __init__(self, scale_size_min, scale_size_max):
        self.scale_size_max = scale_size_max
        self.scale_size_min = scale_size_min

    def __call__(self, img):
        bs, ch, h, w = img.shape
        downscale = random.randint(self.scale_size_min, self.scale_size_max)
        kernel_size = int(max(1, h / downscale))
        return avg_pool2d(img, kernel_size)


class RandomPool(object):
    def __init__(self, kernel_min=1, kernel_max=8):
        self.kernel_min = kernel_min
        self.kernel_max = kernel_max

    def __call__(self, img, pad_val=None):
        pad_val = pad_val if pad_val is not None else random.random() * 2. - 1.
        # Double uniform sample different distribution
        kernel_size = random.randint(self.kernel_min,
                                     random.randint(self.kernel_min,
                                                    self.kernel_max))
        # kernel_size = random.randint(self.kernel_min, self.kernel_max)
        pad_size = random.randint(0, kernel_size - 1)
        action = random.randint(0, 3)
        if action == 0:
            pad = (pad_size, pad_size, 0, 0)
        elif action == 1:
            pad = (pad_size, 0, 0, pad_size)
        elif action == 2:
            pad = (0, pad_size, pad_size, 0)
        elif action == 3:
            pad = (0, 0, pad_size, pad_size)
        img = torch.nn.functional.pad(img,
                                      pad,
                                      mode='constant',
                                      value=pad_val)
        img = avg_pool2d(img, kernel_size=kernel_size, stride=None, padding=0)
        return img


class RandomMirror(object):
    def __init__(self, blend=False):
        # Blend not properly implemented
        self.modes = 2 if blend is False else 4

    def __call__(self, img):
        bs, ch, h, w = img.shape
        augmentation = random.randint(0, self.modes)
        if augmentation == 0:
            pass
        elif augmentation == 1:
            img_l = img[:, :, :, :w // 2]
            img_r = torch.flip(img[:, :, :, :w // 2], [3])
            img = torch.cat([img_l, img_r], dim=3)
        elif augmentation == 2:
            img_l = torch.flip(img[:, :, :, w // 2:], [3])
            img_r = img[:, :, :, w // 2:]
            img = torch.cat([img_l, img_r], dim=3)
        elif augmentation == 3:
            img_l = img[:, :, :, :w // 2]
            img_r = (img[:, :, :, w // 2:] + torch.flip(img[:, :, :, :w // 2], [3])) / 2
            img = torch.cat([img_l, img_r], dim=3)
        elif augmentation == 4:
            img_l = (img[:, :, :, :w // 2] + torch.flip(img[:, :, :, w // 2:], [3])) / 2
            img_r = img[:, :, :, w // 2:]
            img = torch.cat([img_l, img_r], dim=3)
        return img


class RandomPad(object):
    def __init__(self, pad_min=0, pad_max=224):
        self.pad_min = pad_min
        self.pad_max = pad_max

    def __call__(self, img, pad_val=None):
        pad_val = pad_val if pad_val is not None else random.random() * 2. - 1.
        pad = (random.randrange(self.pad_min, self.pad_max, 2),
               random.randrange(self.pad_min, self.pad_max, 2),
               random.randrange(self.pad_min, self.pad_max, 2),
               random.randrange(self.pad_min, self.pad_max, 2))
        return torch.nn.functional.pad(img, pad, mode='constant', value=pad_val)


class Flip(object):
    """ Horizontal random flip p=0.5 """
    def __init__(self):
        pass

    def __call__(self, img):
        if random.randint(0, 1) == 0:
            return img
        return torch.flip(img, [3])


class Pipeline(object):
    def __init__(self, *args):
        self.functions = args

    def __call__(self, img):
        tmp = img
        for f in self.functions:
            tmp = f(tmp)
        return tmp


class Prod(object):
    def __init__(self, *args):
        self.functions = args

    def __call__(self, x):
        return [f(x) for f in self.functions]


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


def grad_drop(grad, drop=0., drop2d=True):
    """ Dropout gradient. """
    with torch.no_grad():
        if not drop2d:
            grad += -grad + dropout(grad, drop)
        else:
            grad += -grad + dropout2d(grad, drop)


def grad_sign(grad):
    """ Convert gradient to signed gradient. """
    with torch.no_grad():
        grad += -grad + grad.sign()


def model_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


class ArgDict(dict):
    def __setattr__(self, name, value):
        self[name] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __delattr__(self, name):
        del self[name]
