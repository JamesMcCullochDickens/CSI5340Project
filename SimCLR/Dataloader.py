import torch
import numpy as np
from PIL import Image
import os
import sys
import DataAug_Utils as da_utils
from functools import partial


data_path = "./ProjectData/Unlabelled_Images"
resize_tuple = (600, 800)
num_pairs = 13944

def get_all_rgb_fps():
    rgb_paths = []
    for path in os.listdir(data_path):
        if "rgb" in path:
            rgb_paths.append(path)
    return rgb_paths

"""
all_fps = get_all_rgb_fps()
print(len(all_fps))
"""

def get_im_num(path):
    path = path.split("_")[-1]
    return path

def unlabeled_iterator(start_value, skip_value, batch_size, depth_only=False, permutation=None, debug=False):
    all_fps = get_all_rgb_fps()
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
        # depth_path = "depth_im_"+im_num+".png"
        depth_path = os.path.join(data_path, "depth_im_{}".format(im_num))

        if not depth_only:
            rgb_im = Image.open(os.path.join(data_path, path))
            rgb_im = rgb_im.resize((600, 800), resample=Image.BILINEAR)
            # rgb_im = np.asarray(rgb_im, dtype=np.uint8)
            rgb_im = np.asarray(rgb_im)
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
        # if requires_normalization(depth_im):
        #     depth_im = s_utils.normalizeDepthImage(depth_im)

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

"""
def collate(depth_only, rgb_transform, depth_transform, batch):
#rgb-d
    rgb_ims_t = []
    depth_ims_t = []

    for data in batch:

        if not depth_only:
            rgb_im = data["rgb_im"]
            rgb_im_t = rgb_transform(rgb_im.copy())
            rgb_ims_t.append(rgb_im_t)

        depth_im = data["depth_im"]

        depth_im_t = depth_transform(depth_im.copy())

        depth_ims_t.append(depth_im_t)

    if not depth_only:
        rgb_ims = rgb_ims_t
    depth_ims = depth_ims_t

    depth_ims = torch.cat([depth_im for depth_im in depth_ims], dim=0)
    if not depth_only:
        rgb_ims = torch.cat([rgb_im for rgb_im in rgb_ims], dim=0)
        return rgb_ims, depth_ims
    else:
        return depth_ims, depth_ims
"""

def collate(depth_only, rgb_transform, depth_transform, batch):
    rgb_ims_t1 = []
    rgb_ims_t2 = []
    depth_ims_t1 = []
    depth_ims_t2 = []

    for data in batch:
        # print(data)
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

    depth_ims = torch.cat([depth_im for depth_im in depth_ims], dim=0)
    if not depth_only:
        rgb_ims = torch.cat([rgb_im for rgb_im in rgb_ims], dim=0)
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
data_it = get_unlabeled_pair_dl(1, 0, False, None, None)
for data in data_it:
    debug = "debug"
"""
