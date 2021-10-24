import io
import requests

import numpy as np


def model_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def img_pil_to_opencv(input):
    """ Converts PIL image to numpy array with open cv formatting. """
    open_cv_image = np.array(input)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def get_img(url):
    """ Minimal function to download image from url. """
    response = requests.get(url)
    return io.BytesIO(response.content)


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
