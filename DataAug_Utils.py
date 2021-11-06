import numpy as np
import random
from PIL import Image
import torchvision.transforms.functional as F

def random_horizontal_flip(im, sem_mask):
    rand_num = random.random()
    if rand_num > 0.5:
        im = np.flilr(im)
        sem_mask = np.fliplr(sem_mask)
    return im, sem_mask

def random_vertical_flip(im, sem_mask):
    rand_num = random.random()
    if rand_num > 0.5:
        im = np.flipud(im)
        sem_mask = np.flipud(sem_mask)
    return im, sem_mask

def random_color_jitter(im, sem_mask):
    rand_num = random.random()
    if rand_num < 0.3:
        rand_channel = random.randint(0, 3)
        jitter_val = np.random.randint(low=0, high=10, size=(im.shape[0], im.shape[1]))
        im[:, :, rand_channel] += jitter_val
        im = np.clip(im, 0, 255)
    return im, sem_mask

def random_crop_and_resize(im, sem_mask):
    rand_num = random.random()
    if rand_num < 0.2:
        im_h = im.shape[0]
        im_w = im.shape[1]
        crop_size_h = 224
        crop_size_w = 224
        x_min = random.randint(0, im_w-(crop_size_w+1))
        x_max = x_min + crop_size_w
        y_min = random.randint(0, im_h-(crop_size_h+1))
        y_max = y_min + crop_size_h
        im = im[y_min:y_max+1, x_min:x_max+1, :]
        sem_mask = sem_mask[y_min:y_max+1, x_min:x_max+1]
        im = np.asarray(Image.fromarray(im).resize((im_w, im_h)))
        sem_mask = np.asarray(Image.fromarray(sem_mask).resize((im_w, im_h), resample=PIL.Image.NEAREST))
    return im, sem_mask

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