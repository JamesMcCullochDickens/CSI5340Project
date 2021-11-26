import torch
import numpy as np
from PIL import Image
import os
import PathGetter
import DataLoaders.SUNRGBD_dl as srgb_dl
import GenericDataloader as g_dl
import ShowImageUtils as s_utils
from functools import partial
import DataAug_Utils as da_utils

unlabeled_path = "unlabelled_data"
NYUDv2_outer_path = PathGetter.get_fp("NYUDv2")
unlabeled_pairs = "unlabelled_pairs"
unlabeled_pairs_nyudv2_path = os.path.join(NYUDv2_outer_path, unlabeled_path, unlabeled_pairs)
SUNRGBD_outer_path = PathGetter.get_fp("SUN_RGBD")
sun_rgbd_test_path = "SUNRGBDv2Test"
unlabeled_pairs_srgbd_path = os.path.join(SUNRGBD_outer_path, sun_rgbd_test_path)

resize_tuple = (600, 800)
num_pairs = 13947

def is_folder_empty(fp):
    dirContents = os.listdir(fp)
    return len(dirContents) == 0

"""
test = is_folder_empty("F:/Datasets/SUN_RGBD/SUNRGBDv2Test/11082015/2015-11-08T13.42.28.846-0000006517/")
debug = "debug"
"""

def get_srgbd_unlabeled_paths():
    fps = []
    for outer_fp in os.listdir(unlabeled_pairs_srgbd_path):
        inner_fp = os.path.join(unlabeled_pairs_srgbd_path, outer_fp)
        if not is_folder_empty(inner_fp):
            inner_fps = os.listdir(inner_fp)
            for inner_fp in inner_fps:
                fps.append(os.path.join(unlabeled_pairs_srgbd_path, outer_fp, inner_fp))
    return fps

def filter_unlabeled_nyudv2_pairs(paths):
    filtered_fps = []
    for path in paths:
        if "rgb" in path:
            filtered_fps.append(os.path.join(unlabeled_pairs_nyudv2_path, path))
    return filtered_fps

# this function we want to concatenate file paths for
# the NYUDv2 unlabeled RGBD pairs
# The SUN RGBD labelled RGBD pairs from the training set
# THE SUN RGBD unlabelled pairs
def get_all_fps():
    nyudv2_unlabeled = filter_unlabeled_nyudv2_pairs(os.listdir(unlabeled_pairs_nyudv2_path))
    srgbd_trainval = srgb_dl.get_test_train_split()["trainval_paths"]
    srgbd_unlabeled = get_srgbd_unlabeled_paths()
    all_fps = nyudv2_unlabeled + srgbd_trainval + srgbd_unlabeled
    return all_fps


"""
all_fps = get_all_fps()
debug = "debug"
"""


def unlabeled_iterator(start_value, skip_value, batch_size, depth_only=False, permutation=None, debug=False):
    all_fps = get_all_fps()
    if permutation is not None:
        all_fps = [all_fps[i] for i in permutation]
    for index, path in enumerate(all_fps):
        if skip_value != 0:
            if not g_dl.skipFunction(index, start_value, batch_size, skip_value):
                continue

        raw_data = {}

        # the nyudv2 unlabeled pairs
        if "unlabelled_pairs" in path:
            rgb_path = path
            depth_path = path[0:-7]+"inpainted_depth.png"
            requires_normalization = False

        # sun rgbd unlabeled pairs
        elif "SUNRGBDv2Test" in path:
            rgb_outer_path = os.path.join(path, "image")
            rgb_path = os.path.join(rgb_outer_path, os.listdir(rgb_outer_path)[0])
            depth_outer_path = os.path.join(path, "depth")
            depth_path = os.path.join(depth_outer_path, os.listdir(depth_outer_path)[0])
            requires_normalization = True

        # sun rgbd trainval pairs
        else:
            rgb_outer_path = os.path.join(path, "image")
            rgb_path = os.path.join(rgb_outer_path, os.listdir(rgb_outer_path)[0])
            depth_outer_path = os.path.join(path, "depth_bfx")
            depth_path = os.path.join(depth_outer_path, os.listdir(depth_outer_path)[0])
            requires_normalization = True

        if not depth_only:
            rgb_im = Image.open(rgb_path)
            rgb_im = rgb_im.resize((600, 800), resample=Image.BILINEAR)
            rgb_im = np.asarray(rgb_im, dtype=np.uint8)
            raw_data["rgb_im"] = rgb_im

        depth_im = Image.open(depth_path)
        depth_im = depth_im.resize((600, 800), resample=Image.NEAREST)
        depth_im = np.asarray(depth_im)
        if requires_normalization:
            depth_im = s_utils.normalizeDepthImage(depth_im)
        else:
            depth_im = depth_im.astype(np.uint8)
        raw_data["depth_im"] = depth_im

        """
        # sanity check for visualization
        s_utils.showImage(da_utils.npy_grayscale2RG(raw_data["depth_im"]))
        if not depth_only:
            s_utils.showImage(raw_data["rgb_im"])
        """

        yield raw_data

class unlabeled_iterable_dataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, depth_only=False, permutation=None):
        super(unlabeled_iterable_dataset).__init__()
        self.batch_size = batch_size
        self.permutation = permutation
        self.depth_only = depth_only
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_value = 0
            skip_value = 0
        else:
            start_value = worker_info.id * self.batch_size
            skip_value = (worker_info.num_workers - 1) * self.batch_size
        return iter(unlabeled_iterator(start_value, skip_value, self.batch_size, self.depth_only, self.permutation))


def collate(depth_only, rgb_transform, depth_transform, batch):
    rgb_ims_t1 = []
    rgb_ims_t2 = []
    depth_ims_t1 = []
    depth_ims_t2 = []
    for data in batch:
        if not depth_only:
            rgb_im = data["rgb_im"]
            rgb_im_t1 = rgb_transform(rgb_im.copy())
            rgb_im_t2 = rgb_transform(rgb_im.copy())
            rgb_ims_t1.append(rgb_im_t1)
            rgb_ims_t2.append(rgb_im_t2)
        depth_im = data["depth_im"]
        depth_im_t1 = depth_transform(depth_im.copy())
        depth_im_t2 = depth_transform(depth_im.copy())
        depth_ims_t1.append(depth_im_t1)
        depth_ims_t2.append(depth_im_t2)
    rgb_ims_t1.extend(rgb_ims_t2)
    depth_ims_t1.extend(depth_ims_t2)
    depth_ims = torch.cat([depth_im for depth_im in depth_ims_t1], dim=0)
    if not depth_only:
        rgb_ims = torch.cat([rgb_im for rgb_im in rgb_ims_t1], dim=0)
        return rgb_ims, depth_ims
    else:
        return depth_ims


def get_unlabeled_pair_dl(batch_size, num_workers, depth_only, rgb_transform=None, depth_transform=None):
    if rgb_transform is None:
        rgb_transform = da_utils.rgb_transform
    if depth_transform is None:
        depth_transform = da_utils.depth_transform
    coll = partial(collate, depth_only, rgb_transform, depth_transform)
    permutation = np.random.permutation(num_pairs).tolist()
    it = unlabeled_iterable_dataset(batch_size, depth_only, permutation=permutation)
    dl = torch.utils.data.DataLoader(it, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                     collate_fn=coll, persistent_workers=False, drop_last=True)
    return dl

"""
# testing
rgb_transform = da_utils.rgb_transform
depth_transform = da_utils.depth_transform
data_it = get_unlabeled_pair_dl(10, 0, False, rgb_transform, depth_transform)
for data in data_it:
    debug = "debug"
"""