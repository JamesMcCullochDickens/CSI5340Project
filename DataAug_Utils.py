import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import PIL
import ShowImageUtils as s_utils
import math

im_height = 227
im_width = 227
im_tuple = (im_height, im_width)

# input shape [3, H, W]
def npy_random_resized_crop(im, is_depth=False):
    if is_depth:
        transform = T.RandomResizedCrop(size=im_tuple, scale=(0.60, 1.0), ratio=(0.75, 1.3333333333333333),
                                 interpolation=T.InterpolationMode.NEAREST)
    else:
        transform = T.RandomResizedCrop(size=im_tuple, scale=(0.60, 1.0), ratio=(0.75, 1.3333333333333333),
                                 interpolation=T.InterpolationMode.BILINEAR)
    im = transform(im)
    return np.asarray(im)

# input shape [3, H, W]
def random_resized_crop(im, is_depth=False):
    if is_depth:
        transform = T.RandomResizedCrop(size=im_tuple, scale=(0.60, 1.0), ratio=(0.75, 1.3333333333333333),
                                 interpolation=T.InterpolationMode.NEAREST)
    else:
        transform = T.RandomResizedCrop(size=im_tuple, scale=(0.60, 1.0), ratio=(0.75, 1.3333333333333333),
                                 interpolation=T.InterpolationMode.BILINEAR)
    im = transform(im)
    im = np.asarray(im)
    return torch.tensor(im)

"""
test_img = "F:/Datasets/MS_COCO/test2015/COCO_test2015_000000000028.jpg"
test_img = Image.open(test_img)
test_img = random_resized_crop(test_img)
test_img = np.asarray(test_img)
s_utils.showImage(test_img)
"""

def npy_random_equalize(im):
    transform = T.RandomEqualize(p=1.0)
    im = transform(im)
    return np.asarray(im)

def random_equalize(im):
    transform = T.RandomEqualize(p=1.0)
    im = transform(im)
    im = np.asarray(im)
    im = torch.tensor(im)
    return im

"""
test_img = "F:/Datasets/MS_COCO/test2015/COCO_test2015_000000000028.jpg"
test_img = Image.open(test_img)
test_img = npy_random_equalize(test_img)
test_img = np.asarray(test_img)
s_utils.showImage(test_img)
"""

def npy_gaussian_blur(im):
    transform = T.GaussianBlur(kernel_size=17, sigma=(9.0, 10.0))
    im = transform(im)
    im = np.asarray(im)
    return im

def gaussian_blur(im):
    transform = T.GaussianBlur(kernel_size=17, sigma=(9.0, 10.0))
    im = transform(im)
    im = np.asarray(im)
    return torch.tensor(im)

"""
test_img = "F:/Datasets/MS_COCO/test2015/COCO_test2015_000000000028.jpg"
test_img = Image.open(test_img)
test_img = npy_gaussian_blur(test_img)
test_img = np.asarray(test_img)
s_utils.showImage(test_img)
"""

# great!
def npy_solarize(im):
    transform = T.RandomSolarize(threshold=150, p=1.0)
    im = transform(im)
    im = np.asarray(im)
    return im

def solarize(im):
    transform = T.RandomSolarize(threshold=150, p=1.0)
    im = transform(im)
    im = np.asarray(im)
    return torch.tensor(im)

"""
test_img = "F:/Datasets/MS_COCO/test2015/COCO_test2015_000000000028.jpg"
test_depth_img = "F:/Datasets/NYUDv2/unlabelled_data/unlabelled_pairs/r-1294439279.105557-2125534748.ppm_raw_depth.png"
test_img = Image.open(test_depth_img)
test_img = npy_solarize(test_img)
test_img = np.asarray(test_img)
s_utils.showImage(test_img)
"""

def npy_random_erase(im):
    im = np.asarray(im)
    im_height = im.shape[0]
    im_width = im.shape[1]
    rect_width = int(im_width/4)
    rect_height = int(im_height/4)
    p = np.random.rand(2)
    p_1 = p[0]
    p_2 = p[1]
    if p_1 < 0.5:
        p_1 = -1
    else:
        p_1 = 1
    if p_2 < 0.5:
        p_2 = -1
    else:
        p_2 = 1

    x_min = int(im_width/2) + p_1*np.random.randint(low=0, high=int(im.shape[0]/2)-rect_width)
    y_min = int(im_height/2) + p_2*np.random.randint(low=0, high=int(im.shape[1]/2)-rect_height)
    x_max = min(im_width-1, x_min + rect_width)
    y_max = min(im_height-1, y_min + rect_height)
    im[x_min:x_max, y_min:y_max, :] = [0, 0, 0]
    return im

def random_erase(im):
    im = np.asarray(im)
    im_height = im.shape[0]
    im_width = im.shape[1]
    rect_width = int(im_width / 4)
    rect_height = int(im_height / 4)
    p = np.random.rand(2)
    p_1 = p[0]
    p_2 = p[1]
    if p_1 < 0.5:
        p_1 = -1
    else:
        p_1 = 1
    if p_2 < 0.5:
        p_2 = -1
    else:
        p_2 = 1

    x_min = int(im_width / 2) + p_1 * np.random.randint(low=0, high=int(im.shape[0] / 2) - rect_width)
    y_min = int(im_height / 2) + p_2 * np.random.randint(low=0, high=int(im.shape[1] / 2) - rect_height)
    x_max = min(im_width - 1, x_min + rect_width)
    y_max = min(im_height - 1, y_min + rect_height)
    im[x_min:x_max, y_min:y_max, :] = [0, 0, 0]
    return im

"""
test_img = "F:/Datasets/MS_COCO/test2015/COCO_test2015_000000000028.jpg"
#test_depth_img = "F:/Datasets/NYUDv2/unlabelled_data/unlabelled_pairs/r-1294439279.105557-2125534748.ppm_raw_depth.png"
test_img = Image.open(test_img)
test_img = npy_random_erase(test_img)
s_utils.showImage(test_img)
"""

def color_jitter(im):
    transform = T.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5)
    im = transform(im)
    im = np.asarray(im)
    return torch.Tensor(im)

def npy_color_jitter(im):
    transform = T.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5)
    im = transform(im)
    im = np.asarray(im)
    return im

def horizontal_flip(im):
    transform = T.RandomHorizontalFlip(p=1)
    im = transform(im)
    im = np.asarray(im)
    return torch.Tensor(im)

def npy_Rrndom_horizontal_flip(im):
    transform = T.RandomHorizontalFlip(p=1)
    im = transform(im)
    im = np.asarray(im)
    return im

def ertical_flip(im):
    transform = T.RandomVerticalFlip(p=1)
    im = transform(im)
    im = np.asarray(im)
    return torch.Tensor(im)

def npy_vertical_flip(im):
    transform = T.RandomVerticalFlip(p=1)
    im = transform(im)
    im = np.asarray(im)
    return im

def random_rotate(im):
    transform = T.RandomRotation(10)
    im = transform(im)
    im = np.asarray(im)
    return torch.Tensor(im)

def npy_random_rotate(im):
    transform = T.RandomRotation(10)
    im = transform(im)
    im = np.asarray(im)
    return im

def rgb_to_grayscale(im):
    transform = T.RandomGrayscale(p=1)
    im = transform(im)
    im = np.asarray(im)
    return torch.Tensor(im)

def npy_rgb_to_grayscale(im):
    transform = T.RandomGrayscale(p=1)
    im = transform(im)
    im = np.asarray(im)
    return im

def random_affine(im, isDepth=False):
    if isDepth:
        transform = T.RandomAffine(degrees=(1, 45), translate=(0, 0.1), shear=(1, 45),
                                   interpolation=T.InterpolationMode.NEAREST)
    else :
        transform = T.RandomAffine(degrees=(1, 45), translate=(0, 0.1), shear=(1, 45),
                                   interpolation=T.InterpolationMode.BILINEAR)
    im = transform(im)
    im = np.asarray(im)
    return torch.Tensor(im)

def npy_random_affine(im):
    transform = T.RandomGrayscale(p=1)
    im = transform(im)
    im = np.asarray(im)
    return im

def grayscale2RGB(im):
    transform = T.Grayscale(num_output_channels=3)
    im = transform(im)
    im = np.asarray(im)
    return torch.Tensor(im)

def npy_grayscale2RG(im):
    transform = T.Grayscale(num_output_channels=3)
    im = transform(im)
    im = np.asarray(im)
    return im

class multi_input_hflip(object):
    def __init__(self, threshold):
        self.threshold = threshold
        pass
    def __call__(self, ims):
        im1, im2 = ims
        random_val = random.random()
        if random_val > self.threshold:
            im1 = F.hflip(im1)
            im2 = F.hflip(im2)
        return im1, im2

class multi_input_vflip(object):
    def __init__(self, threshold):
        self.threshold = threshold
        pass
    def __call__(self, ims):
        im1, im2 = ims
        random_val = random.random()
        if random_val > self.threshold:
            im1 = F.vflip(im1)
            im2 = F.vflip(im2)
        return im1, im2