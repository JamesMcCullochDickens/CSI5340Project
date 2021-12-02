import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import PIL
import ShowImageUtils as s_utils
import random

im_height = 224
im_width = 224
im_tuple = (im_height, im_width)
norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# input shape [3, H, W]
def random_resized_crop(im, is_depth=False):
    if is_depth:
        transform = T.RandomResizedCrop(size=im_tuple, scale=(0.4, 0.8),
                                 interpolation=T.InterpolationMode.NEAREST)
    else:
        transform = T.RandomResizedCrop(size=im_tuple, scale=(0.45, 0.75), ratio=(0.75, 1.3333333333333333),
                                 interpolation=T.InterpolationMode.BILINEAR)
    im = transform(im)
    im = np.asarray(im)
    return torch.tensor(im)

def npy_random_equalize(im):
    transform = T.RandomEqualize(p=0.5)
    im = transform(im)
    return np.asarray(im)

def random_equalize(im):
    transform = T.RandomEqualize(p=0.5)
    im = transform(im)
    im = np.asarray(im)
    im = torch.tensor(im)
    return im

def torch_channels_first_to_last(t):
    return torch.permute(t, (2, 0, 1))

def torch_channels_last_to_first(t):
    return torch.permute(t, (2, 0, 1))


def random_bool():
    rand = random.random()
    return rand > 0.6


def npy_gaussian_blur(im):
    transform = T.GaussianBlur((3, 3), (1.0, 2.0))
    im = transform(im)
    im = np.asarray(im)
    return im

def gaussian_blur(im):
    transform = T.GaussianBlur((7, 7), (5.0, 6.0))
    im = transform(im)
    im = np.asarray(im)
    return torch.tensor(im)

# great!
def npy_solarize(im):
    transform = T.RandomSolarize(threshold=220, p=1.0)
    im = transform(im)
    im = np.asarray(im)
    return im

def solarize(im):
    transform = T.RandomSolarize(threshold=220, p=1.0)
    im = transform(im)
    im = np.asarray(im)
    return torch.tensor(im)


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

def color_jitter(im):
    transform = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.3)
    im = transform(im)
    im = np.asarray(im)
    return torch.Tensor(im)

def npy_color_jitter(im):
    transform = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.3)
    im = transform(im)
    im = np.asarray(im)
    return im

def horizontal_flip(im):
    transform = T.RandomHorizontalFlip(p=1)
    im = transform(im)
    im = np.asarray(im)
    return torch.Tensor(im)

def npy_horizontal_flip(im):
    transform = T.RandomHorizontalFlip(p=1)
    im = transform(im)
    im = np.asarray(im)
    return im

def random_horizontal_flip(im):
    transform = T.RandomHorizontalFlip(p=0.5)
    im = transform(im)
    im = np.asarray(im)
    return torch.Tensor(im)

def npy_random_horizontal_flip(im):
    transform = T.RandomHorizontalFlip(p=0.5)
    im = transform(im)
    im = np.asarray(im)
    return im

def vertical_flip(im):
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

def random_rgb_to_grayscale(im):
    transform = T.RandomGrayscale(p=0.2)
    im = transform(im)
    im = np.asarray(im)
    return torch.Tensor(im)

def random_npy_rgb_to_grayscale(im):
    transform = T.RandomGrayscale(p=0.2)
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
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
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


# For BYOL, page 5, https://arxiv.org/pdf/2006.07733.pdf
# BYOL uses the same set of image augmentations as in SimCLR [8]. First, a random patch
# of the image is selected and resized to 224 × 224 with a random horizontal flip, followed by a color distortion,
# consisting of a random sequence of brightness, contrast, saturation, hue adjustments, and an optional grayscale
# conversion. Finally Gaussian blur and solarization are applied to the patches.


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = T.ToTensor()
        self.tensor_to_pil = T.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


color_jitter = T.ColorJitter(0.8 , 0.8 , 0.8 , 0.0)
data_transforms = T.Compose([T.RandomResizedCrop(size=224),
                                          T.RandomHorizontalFlip(),
                                          T.RandomApply([color_jitter], p=0.8),
                                          T.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=5),
                                          T.ToTensor()])


def depth_transform(depth_im):
    depth_im = np.expand_dims(depth_im, axis=-1)
    depth_im = np.repeat(depth_im, 3, axis=-1)
    depth_im = Image.fromarray(depth_im)
    depth_im = data_transforms(depth_im)

    """
    # sanity check
    depth_im *= 255.0
    depth_im = depth_im.numpy().astype(np.uint8)
    depth_im = s_utils.channelsFirstToLast(depth_im)
    s_utils.showImage(depth_im)
    debug = "debug"
    """
    depth_im = depth_im[0]
    depth_im = depth_im - 0.449  # average of ImageNet channel means
    depth_im = depth_im / 0.226  # average of ImageNet channels standard deviations
    depth_im = torch.unsqueeze(torch.unsqueeze(depth_im, dim=0), dim=0)
    return depth_im


def rgb_transform(rgb_im):
    rgb_im = Image.fromarray(rgb_im)
    rgb_im = color_jitter(rgb_im)
    rgb_im = random_resized_crop(torch_channels_first_to_last(rgb_im), is_depth=False)
    rgb_im = random_horizontal_flip(rgb_im)
    rgb_im = random_rgb_to_grayscale(rgb_im)
    rgb_im = gaussian_blur(rgb_im)
    rgb_im = rgb_im * (1/255.0)
    rgb_im = norm_transform(rgb_im)
    rgb_im = torch.unsqueeze(rgb_im, dim=0)
    return rgb_im


"""
# example depth aug
sample_depth_im = Image.open("F:/Datasets/SUN_RGBD/SUNRGBD/kv2/kinect2data/000009_2014-05-26_14-32-05_260595134347_rgbf000034-resize/depth_bfx/0000034.png")
sample_depth_im = Image.fromarray(s_utils.normalizeDepthImage(sample_depth_im))
s_utils.showImage(npy_grayscale2RG(sample_depth_im))
depth_transform(np.asarray(sample_depth_im))

sample_depth_im = Image.open("F:/Datasets/SUN_RGBD/SUNRGBD/kv2/kinect2data/000009_2014-05-26_14-32-05_260595134347_rgbf000034-resize/depth_bfx/0000034.png")
sample_depth_im = Image.fromarray(s_utils.normalizeDepthImage(sample_depth_im))
s_utils.showImage(npy_grayscale2RG(sample_depth_im))
depth_transform(np.asarray(sample_depth_im))
#s_utils.showImage(s_utils.channelsFirstToLast(transformed_depth_im))
"""

"""
# example rgb aug
sample_rgb_im = Image.open("F:/Datasets/SUN_RGBD/SUNRGBD/kv2/kinect2data/000009_2014-05-26_14-32-05_260595134347_rgbf000034-resize/image/0000034.jpg")
s_utils.showImage(np.asarray(sample_rgb_im))
transformed_depth_im = rgb_transform(sample_rgb_im).numpy().astype(np.uint8)
s_utils.showImage(s_utils.channelsFirstToLast(transformed_depth_im))

# example rgb aug
sample_rgb_im = Image.open("F:/Datasets/SUN_RGBD/SUNRGBD/kv2/kinect2data/000009_2014-05-26_14-32-05_260595134347_rgbf000034-resize/image/0000034.jpg")
s_utils.showImage(np.asarray(sample_rgb_im))
transformed_depth_im = rgb_transform(sample_rgb_im).numpy().astype(np.uint8)
s_utils.showImage(s_utils.channelsFirstToLast(transformed_depth_im))
"""