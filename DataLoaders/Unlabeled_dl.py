import torch
import numpy as np
from PIL import Image
import os
import PathGetter
import DataLoaders.SUNRGBD_dl as srgb_dl
import ShowImageUtils as s_utils
from functools import partial
import DataAug_Utils as da_utils
import CacheDictUtils as cd_utils

unlabeled_path = "unlabelled_data"
NYUDv2_outer_path = PathGetter.get_fp("NYUDv2")
unlabeled_pairs = "unlabelled_pairs"
unlabeled_pairs_nyudv2_path = os.path.join(NYUDv2_outer_path, unlabeled_path, unlabeled_pairs)
SUNRGBD_outer_path = PathGetter.get_fp("SUN_RGBD")
sun_rgbd_test_path = "SUNRGBDv2Test"
unlabeled_pairs_srgbd_path = os.path.join(SUNRGBD_outer_path, sun_rgbd_test_path)

resize_tuple = (600, 800)
num_pairs = 13944

def is_folder_empty(fp):
    dirContents = os.listdir(fp)
    is_empty = len(dirContents) == 0
    return is_empty

# oddly there are empty folders in the SUN RGBD unlabeled dataset
def delete_empty_folders(fp):
    for path in os.listdir(fp):
        folder_path = os.path.join(path, fp)
        for sub_path in os.listdir(folder_path):
            to_check = os.path.join(folder_path, sub_path)
            if is_folder_empty(to_check):
                os.rmdir(to_check)

"""
delete_empty_folders("F:/Datasets/SUN_RGBD/SUNRGBDv2Test/11082015")
delete_empty_folders("F:/Datasets/SUN_RGBD/SUNRGBDv2Test/11092015")
delete_empty_folders("F:/Datasets/SUN_RGBD/SUNRGBDv2Test/11112015")
delete_empty_folders("F:/Datasets/SUN_RGBD/SUNRGBDv2Test/11122015")
delete_empty_folders("F:/Datasets/SUN_RGBD/SUNRGBDv2Test/11132015")
delete_empty_folders("F:/Datasets/SUN_RGBD/SUNRGBDv2Test/black_batch1")
delete_empty_folders("F:/Datasets/SUN_RGBD/SUNRGBDv2Test/black_batch2")
debug = "debug"
"""


def get_srgbd_unlabeled_paths():
    fps = []
    for outer_fp in os.listdir(unlabeled_pairs_srgbd_path):
        inner_fp = os.path.join(unlabeled_pairs_srgbd_path, outer_fp)
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

def get_all_fps_():
    rgb_paths = []
    for path in os.listdir("../ProjectData/Unlabelled_Images"):
        if "rgb" in path:
            rgb_paths.append(path)
    return rgb_paths


"""
all_fps = get_all_fps()
debug = "debug"
"""

def filter_func(fp):
    return "rgb" in fp

def write_all_unlabeled_files():
    all_fps = os.listdir()
    unlablled_ims_path = "C:/Users/james/PycharmProjects/CSI5340Project/Unlabelled_Images"
    for index, path in enumerate(all_fps):
        if "unlabelled_pairs" in path:
            rgb_path = path
            depth_path = path[0:-7] + "inpainted_depth.png"
        # sun rgbd unlabeled pairs
        elif "SUNRGBDv2Test" in path:
            rgb_outer_path = os.path.join(path, "image")
            rgb_path = os.path.join(rgb_outer_path, os.listdir(rgb_outer_path)[0])
            depth_outer_path = os.path.join(path, "depth")
            depth_path = os.path.join(depth_outer_path, os.listdir(depth_outer_path)[0])
        # sun rgbd trainval pairs
        else:
            rgb_outer_path = os.path.join(path, "image")
            rgb_path = os.path.join(rgb_outer_path, os.listdir(rgb_outer_path)[0])
            depth_outer_path = os.path.join(path, "depth_bfx")
            depth_path = os.path.join(depth_outer_path, os.listdir(depth_outer_path)[0])
        rgb_im = Image.open(rgb_path)
        depth_im = Image.open(depth_path)
        rgb_im.save(os.path.join(unlablled_ims_path, "rgb_im_"+str(index))+".png")
        depth_im.save(os.path.join(unlablled_ims_path, "depth_im_"+str(index))+".png")
    print("Finished writing files")

#write_all_unlabeled_files()

def requires_normalization(im):
    return (im > 255.0).any() > 0

def get_im_num(path):
    path = path.split("_")[-1]
    return path

def unlabeled_iterator(start_value, skip_value, batch_size, depth_only=False, permutation=None, debug=False):
    outer_path = "../ProjectData/Unlabelled_Images"
    all_fps = get_all_fps_()
    if permutation is not None:
        all_fps = [all_fps[i] for i in permutation]

    start_counter = 0
    skip_counter = 0
    current_batch_val = 0
    for index, path in enumerate(all_fps):
        if skip_value != 0:
            if index < start_value:
                start_counter += 1
                continue

                # skip the value
            if skip_counter != 0 and skip_counter < skip_value and current_batch_val == 0:
                skip_counter += 1
                continue

                # reset the skip counter and start skipping at the end of a batch
            if skip_counter == 0 and current_batch_val == batch_size:
                current_batch_val = 0
                skip_counter += 1
                continue

                # reset the skip counter and get the batch when the skip counter has reached its upper bound
            if skip_counter == skip_value and skip_value != 0:
                skip_counter = 0
                current_batch_val += 1

                # get more batch iterations
            elif skip_counter == 0 and current_batch_val < batch_size and skip_value != 0:
                current_batch_val += 1

        raw_data = {}
        im_num = get_im_num(path)
        depth_path = os.path.join(outer_path, "depth_im_"+im_num)


        """
        # the nyudv2 unlabeled pairs
        if "unlabelled_pairs" in path:
            rgb_path = path
            depth_path = path[0:-7]+"inpainted_depth.png"

        # sun rgbd unlabeled pairs
        elif "SUNRGBDv2Test" in path:
            rgb_outer_path = os.path.join(path, "image")
            rgb_path = os.path.join(rgb_outer_path, os.listdir(rgb_outer_path)[0])
            depth_outer_path = os.path.join(path, "depth")
            depth_path = os.path.join(depth_outer_path, os.listdir(depth_outer_path)[0])

        # sun rgbd trainval pairs
        else:
            rgb_outer_path = os.path.join(path, "image")
            rgb_path = os.path.join(rgb_outer_path, os.listdir(rgb_outer_path)[0])
            depth_outer_path = os.path.join(path, "depth_bfx")
            depth_path = os.path.join(depth_outer_path, os.listdir(depth_outer_path)[0])
        """

        if not depth_only:
            rgb_im = Image.open(os.path.join(outer_path, path))
            rgb_im = rgb_im.resize((600, 800), resample=Image.BILINEAR)
            rgb_im = np.asarray(rgb_im, dtype=np.uint8)
            raw_data["rgb_im"] = rgb_im

        depth_im = Image.open(depth_path)
        depth_im = depth_im.resize((600, 800), resample=Image.NEAREST)
        depth_im = np.asarray(depth_im)
        if np.all(depth_im == 0): # some depth images in the SUN RGBD TEST are totally black, i.e. depth values all 0
            continue
        if len(depth_im.shape) == 2:
            depth_im = np.expand_dims(depth_im, axis=-1)
            depth_im = np.repeat(depth_im, 3, axis=-1)
        if depth_im.shape[-1] == 4: # for the odd case where PIL gives a transparency channel
            depth_im = depth_im[0:3, :, :]
        if requires_normalization(depth_im):
            depth_im = s_utils.normalizeDepthImage(depth_im)

        raw_data["depth_im"] = depth_im

        """
        # sanity check for visualization
        s_utils.showImage(da_utils.npy_grayscale2RG(raw_data["depth_im"]))
        if not depth_only:
            s_utils.showImage(raw_data["rgb_im"])
        """

        yield raw_data


def generate_rotation_vals():
    test_files = srgb_dl.get_test_train_split()["test_paths"]
    rotation_vals = np.random.randint(0, 4, len(test_files)).tolist()
    write_fp = "./rotation_gts"
    gt_dict = {"gt_rotations": rotation_vals}
    cd_utils.writeReadableCachedDict(write_fp, gt_dict)

def get_gt_rotation_vals():
    fp = "C:/Users/james/PycharmProjects/CSI5340Project/DataLoaders/rotation_gts"
    return cd_utils.readReadableCachedDict(fp)["gt_rotations"]



def labeled_rotation_iterator(depth_only):
    test_files = srgb_dl.get_test_train_split()["test_paths"]
    rotations = get_gt_rotation_vals()
    for index, path in enumerate(test_files):
        raw_data = {}
        rotation_val = rotations[index]
        if not depth_only:
            rgb_outer_path = os.path.join(path, "image")
            rgb_path = os.path.join(rgb_outer_path, os.listdir(rgb_outer_path)[0])
        depth_outer_path = os.path.join(path, "depth_bfx")
        depth_path = os.path.join(depth_outer_path, os.listdir(depth_outer_path)[0])
        if not depth_only:
            rgb_im = Image.open(rgb_path)
            rgb_im = rgb_im.resize((600, 800), resample=Image.BILINEAR)
            rgb_im = np.asarray(rgb_im, dtype=np.uint8)
            raw_data["rgb_im"] = rgb_im

        depth_im = Image.open(depth_path)
        depth_im = depth_im.resize((600, 800), resample=Image.NEAREST)
        depth_im = np.asarray(depth_im)
        if np.all(depth_im == 0):  # some depth images in the SUN RGBD TEST are totally black, i.e. depth values all 0
            continue
        if len(depth_im.shape) == 2:
            depth_im = np.expand_dims(depth_im, axis=-1)
            depth_im = np.repeat(depth_im, 3, axis=-1)
        if depth_im.shape[-1] == 4:  # for the odd case where PIL gives a transparency channel
            depth_im = depth_im[0:3, :, :]
        if requires_normalization(depth_im):
            depth_im = s_utils.normalizeDepthImage(depth_im)

        raw_data["depth_im"] = depth_im
        raw_data["rotation_val"] = rotation_val
        yield raw_data

class rotation_test_dataset(torch.utils.data.IterableDataset):
    def __init__(self, depth_only=False):
        super(rotation_test_dataset).__init__()
        self.depth_only = depth_only
    def __iter__(self):
        return iter(labeled_rotation_iterator(self.depth_only))


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
    if not depth_only:
        rgb_ims = rgb_ims_t1 + rgb_ims_t2
    depth_ims = depth_ims_t1 + depth_ims_t2

    """
    # sanity check
    batch_size = int(len(depth_ims)/2)
    for index, im in enumerate(depth_ims):
        if index >= batch_size:
            break
        im = im[0]
        other_im = depth_ims[index+batch_size][0]
        im = im + 0.449
        im = im * 0.226
        im = im * 255.0
        im = im.repeat(3, 1, 1)
        im = im.numpy().astype(np.uint8)
        im = s_utils.channelsFirstToLast(im)
        other_im = other_im + 0.449
        other_im = other_im * 0.226
        other_im = other_im * 255.0
        other_im = other_im.repeat(3, 1, 1)
        other_im = other_im.numpy().astype(np.uint8)
        other_im = s_utils.channelsFirstToLast(other_im)
        to_display = s_utils.concatenateImagesHorizontally([im, other_im])
        s_utils.showImage(to_display)
    """


    depth_ims = torch.cat([depth_im for depth_im in depth_ims], dim=0)
    if not depth_only:
        rgb_ims = torch.cat([rgb_im for rgb_im in rgb_ims], dim=0)
        return rgb_ims, depth_ims
    else:
        return depth_ims


# 0 means no rotation
# 1 means horizontal flip
# 2 means vertical flip
# 3 horizontal and vertical flip

def rotation_collate(batch):
    rotated_depth_ims = []
    batch_size = len(batch)
    gt_labels = torch.randint(0, 4, (batch_size,))
    for index, data in enumerate(batch):
        depth_im = data["depth_im"]
        if gt_labels[index] == 0:
            depth_im = torch.tensor(depth_im)
        elif gt_labels[index] == 1:
            depth_im = da_utils.horizontal_flip(depth_im)
        elif gt_labels[index] == 2:
            depth_im = da_utils.vertical_flip(depth_im)
        elif gt_labels[index] == 3:
            depth_im = da_utils.horizontal_flip(depth_im).numpy().astype(np.uint8)
            depth_im = da_utils.vertical_flip(depth_im)
        depth_im = depth_im[:, :, 0]
        depth_im = depth_im*(1/255)
        depth_im = depth_im - 0.449
        depth_im = depth_im / 0.226
        rotated_depth_ims.append(torch.unsqueeze(torch.unsqueeze(depth_im, dim=0), dim=0))
    rotated_depth_ims = torch.cat([depth_im for depth_im in rotated_depth_ims], dim=0)
    gt_labels = torch.tensor(gt_labels, dtype=torch.long)
    return rotated_depth_ims, gt_labels

# TODO add code for rgb ims too
def rotation_collate_test(depth_only, batch):
    rotated_depth_ims = []
    gt_labels = []
    for index, data in enumerate(batch):
        depth_im = data["depth_im"]
        rotation_label = data["rotation_val"]
        gt_labels.append(rotation_label)
        if rotation_label == 0:
            depth_im = torch.tensor(depth_im)
        elif rotation_label == 1:
            depth_im = da_utils.horizontal_flip(depth_im)
        elif rotation_label == 2:
            depth_im = da_utils.vertical_flip(depth_im)
        elif rotation_label == 3:
            depth_im = da_utils.horizontal_flip(depth_im).numpy().astype(np.uint8)
            depth_im = da_utils.vertical_flip(depth_im)
        depth_im = depth_im[:, :, 0]
        depth_im = depth_im * (1 / 255)
        depth_im = depth_im - 0.449
        depth_im = depth_im / 0.226
        rotated_depth_ims.append(torch.unsqueeze(torch.unsqueeze(depth_im, dim=0), dim=0))
    rotated_depth_ims = torch.cat([depth_im for depth_im in rotated_depth_ims], dim=0)
    gt_labels = torch.tensor(gt_labels, dtype=torch.long)
    return rotated_depth_ims, gt_labels

def get_test_rotation_dl(batch_size, num_workers, depth_only):
    coll = partial(rotation_collate_test, depth_only)
    it = rotation_test_dataset(depth_only)
    dl = torch.utils.data.DataLoader(it, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                     collate_fn=coll, persistent_workers=False, drop_last=True)
    return dl

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

def get_unlabeled_rotation_dl(batch_size, num_workers, depth_only):
    coll = rotation_collate
    permutation = np.random.permutation(num_pairs).tolist()
    it = unlabeled_iterable_dataset(batch_size, depth_only, permutation=permutation)
    dl = torch.utils.data.DataLoader(it, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                     collate_fn=coll, persistent_workers=False, drop_last=True)
    return dl



"""
# testing
data_it = get_unlabeled_pair_dl(1, 0, True, None, None)
for data in data_it:
    debug = "debug"
"""