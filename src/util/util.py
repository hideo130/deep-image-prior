"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from logging import getLogger, StreamHandler, DEBUG, basicConfig, Formatter, INFO
from pathlib import Path
import logzero


def make_mask(cfg, img):
    """numpy型のmask画像を作る．

    Args:
        cfg(hydra):
        img (float): numpy型の画像
    """
    mask = np.ones_like(img, dtype=float)
    # mask[0:160, 290:500, :] = 0
    img[0:50, 150:230, :] = 255
    return mask


def get_noisy_img(img, sigma):
    """画像にガウシアンノイズを加える.

    Args:
        img: 0から1の値を持つnumpy型の画像
        sigma: ノイズの標準分散
    """
    noisy_img = np.clip(
        img + np.random.normal(scale=sigma, size=img.shape), 0, 1)
    return noisy_img.astype(np.float32)


def setup_logger(log_filename):
    format_str = '%(asctime)s %(levelname)s %(message)s'
    basicConfig(filename=log_filename, level=DEBUG, format=format_str)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter(format_str))
    getLogger().addHandler(stream_handler)


def get_logger(log_dir, loglevel=INFO):
    from logzero import logger
    # format_str = '%(asctime)s %(levelname)s %(message)s'
    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True)
    logzero.loglevel(loglevel)
    logzero.logfile(log_dir + '/log.txt')

    return logger


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Args:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # convert it into a numpy array
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / \
            2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Args
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Args:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Args:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Args:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Args:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
