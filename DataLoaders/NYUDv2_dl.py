import os
from pathlib import Path
import torch
import math
import scipy.io
import imageio
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import PathGetter
import ShowImageUtils as s_utils
import InPainting as ip
import torchvision.transforms as T
import random
import DataAug_Utils as d_utils
from functools import partial
unlabeled_path = "unlabelled_data"
NYUDv2_outer_path = PathGetter.get_fp("NYUDv2")
unlabeled_pairs = "unlabelled_pairs"
train_test_splits = os.path.join(NYUDv2_outer_path, "eccv14-data", "benchmarkData", "metadata", "eccv14-splits.mat")
is_masks_fp = os.path.join(NYUDv2_outer_path, "eccv14-data", "data", "is_masks")
NYUDv2_instance_segmentation_categories_dict = {"unknown": 0, "bathtub": 1, "bed": 2, "bookshelf": 3, "box": 4, "chair": 5,
                                               "counter": 6, "desk": 7, "door": 8, "dresser": 9, "lamp": 10, "night-stand": 11,
                                               "pillow": 12, "sink": 13, "sofa": 14, "table": 15, "television": 16, "toilet": 17}


semantic_40_dict = {0: "unknown", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair", 6: "sofa", 7: "table", 8: "door", 9: "window", 10:"bookshelf",
                    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves", 16:"curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floor_mat",
                    21:"clothes", 22: "ceiling", 23:"books", 24: "fridge", 25: "tv", 26:"paper", 27: "towel", 28: "shower-curtain", 29: "box", 30: "whiteboard",
                    31:"person", 32: "night_stand", 33: "toilet", 34: "sink", 35: "lamp", 36:"bathtub", 37: "bag", 38:"other-struct", 39: "other-furniture", 40: "other-prop"}


# The low performing categories are: door (8), shelves, clothes, books, box, bag, other-furniture, other-struct

ce_weights = np.ones(41)
lower_performing_indices = [8, 15, 21, 23, 29, 37, 38, 39]
ce_weights[lower_performing_indices]=5.0
ce_weights = torch.tensor(ce_weights).float()

nyudv2_train_size = 795
nyudv2_test_size = 654
total_nyudv2_size = 1449

# TODO not quite implemented, get the unixEpoch
def get_timestamp_from_filename(fn):
    fn = fn[2:].split("-")
    millis = float(fn[1])
    unixEpoch = 0.0 # to be implemented
    matlabTime = millis/86400 + unixEpoch
    return matlabTime


def get_synched_frames(scene_dir):
    rgb_images = []
    frameList = {}
    accelRecs = []
    files = os.listdir(scene_dir)
    files.sort()
    numDepth = 0
    numRGB = 0
    numAccel = 0

    for f_ind, file in enumerate(files):
        if file[0:2] == "a-":
            accelRecs.append(file)
            numAccel += 1
        elif file[0:2] == "r-":
            rgb_images.append(file)
            numRGB += 1
        elif file[0:2] == "d-":
            frameList[numDepth] = {"depth_file": file}
            numDepth += 1

    rgb_pointer = 0
    accel_pointer = 0

    for i in range(numDepth):
        time_parts_depth = frameList[i]["depth_file"][2:].split("-")
        time_parts_rgb = rgb_images[rgb_pointer][2:].split("-")
        time_parts_accel = accelRecs[accel_pointer][2:].split("-")

        tDepth = float(time_parts_depth[0])
        tRgb = float(time_parts_rgb[0])
        tAccel = float(time_parts_accel[0])

        tDiff = math.fabs(tDepth-tRgb)

        while rgb_pointer < numRGB -1:
            time_parts_rgb = rgb_images[rgb_pointer+1][2:].split("-")
            tRGB = float(time_parts_rgb[0])
            tmpDiff = math.fabs(tDepth-tRGB)
            if tmpDiff > tDiff:
                break
            tDiff = tmpDiff
            rgb_pointer += 1

        tDiff = math.fabs(tDepth-tAccel)

        while accel_pointer < numAccel -1:
            time_parts_accel = accelRecs[accel_pointer+1][2:].split("-")
            tAccel = float(time_parts_accel[0])
            tmpDiff = math.fabs(tDepth-tAccel)
            if tmpDiff > tDiff:
                break
            tDiff = tmpDiff
            accel_pointer += 1

        frameList[i]["depth_file"] = os.path.join(frameList[i]["depth_file"])
        frameList[i]["rgb_file"] = os.path.join(rgb_images[rgb_pointer])
        frameList[i]["accel_file"] = os.path.join(accelRecs[accel_pointer])

    timed_framed_list = []
    for index in frameList.keys():
        if index % 60 == 0:
            timed_framed_list.append((frameList[index]["rgb_file"], frameList[index]["depth_file"]))

    return timed_framed_list

def read_ppm(fp):
    img = imageio.imread(fp)
    img = np.asarray(img)
    return img

#read_ppm(im_fp)

def read_pgm(fp):
    img = imageio.imread(fp)
    img = np.asarray(img)
    return img

def existsNoise(depth_im):
    return (depth_im == 0).any


def write_unlabelled_frames():
    unlabelled_dirs = os.path.join(NYUDv2_outer_path, unlabeled_path)
    for dir in os.listdir(unlabelled_dirs):
        if dir == "unlabelled_pairs":
            continue
        sub_path = os.path.join(unlabelled_dirs, dir)
        for sub_dir in os.listdir(sub_path):
            path = os.path.join(sub_path, sub_dir)
            synched_frames = get_synched_frames(path)
            for frame_pair in synched_frames:
                if not os.path.isfile(os.path.join(NYUDv2_outer_path, unlabeled_path, unlabeled_pairs, frame_pair[0] + "_rgb.png")):
                    rgb_im = read_ppm(os.path.join(NYUDv2_outer_path, unlabeled_path, dir, sub_dir, frame_pair[0]))
                    rgb_im = Image.fromarray(rgb_im)
                    rgb_im.save(os.path.join(NYUDv2_outer_path, unlabeled_path, unlabeled_pairs, frame_pair[0] + "_rgb.png"))

                if not os.path.isfile(os.path.join(NYUDv2_outer_path, unlabeled_path, unlabeled_pairs, frame_pair[0] + "_raw_depth.png")):
                    depth_im = read_pgm(os.path.join(NYUDv2_outer_path, unlabeled_path, dir, sub_dir, frame_pair[1]))
                    depth_im = s_utils.normalizeDepthImage(depth_im)
                    depth_im = Image.fromarray(depth_im)
                    depth_im.save(os.path.join(NYUDv2_outer_path, unlabeled_path, unlabeled_pairs, frame_pair[0] + "_raw_depth.png"))

                if not os.path.isfile(os.path.join(NYUDv2_outer_path, unlabeled_path, unlabeled_pairs, frame_pair[0] + "_inpainted_depth.png")):
                    depth_im = np.asarray(depth_im)
                    if existsNoise(depth_im):
                        inpainted_depth = Image.fromarray(ip.fill_depth_colorization(np.asarray(rgb_im)/255, np.asarray(depth_im), 1))
                    else:
                        inpainted_depth = Image.fromarray(depth_im)
                    inpainted_depth = Image.fromarray(s_utils.normalizeDepthImage(inpainted_depth))
                    inpainted_depth.save(os.path.join(NYUDv2_outer_path, unlabeled_path, unlabeled_pairs, frame_pair[0] + "_inpainted_depth.png"))

#write_unlabelled_frames()

def convert_im_num_to_srgbd_fmt(im_nums):
    return im_nums - 5000

def get_train_test_split():
    split = scipy.io.loadmat(train_test_splits)
    train_nums = split["trainval"]
    test_nums = split["test"]
    return train_nums, test_nums

def write_all_ims_and_seg_masks(train):
    if train:
        save_fp = "C:/Users/James/PycharmProjects/CSI5340Project/ProjectData/NYUDv2/Train"
        nums = get_train_test_split()[0]
    else:
        save_fp = "C:/Users/James/PycharmProjects/CSI5340Project/ProjectData/NYUDv2/Test"
        nums = get_train_test_split()[1]
    rgb_ims_paths = os.listdir("F:/Datasets/NYUDv2/eccv14-data/data/images")
    depth_ims_paths = os.listdir("F:/Datasets/NYUDv2/eccv14-data/data/depth")
    sem_ims_paths = os.listdir("F:/Datasets/NYUDv2/eccv14-data/data/semantic_segmentation_masks")

    im_counter = 0
    offset = 0
    if not train:
        offset = 795
    for index, (rgb_fp, depth_fp, sem_fp) in enumerate(zip(rgb_ims_paths, depth_ims_paths, sem_ims_paths)):
        file_num = int(rgb_fp[-8:-4])
        if file_num not in nums:
            continue
        else:
            im_counter += 1
        rgb_im = Image.open(os.path.join("F:/Datasets/NYUDv2/eccv14-data/data/images", rgb_fp))
        rgb_save_fp = os.path.join(save_fp, "rgb_im_"+str(im_counter+offset)+".png")
        rgb_im.save(rgb_save_fp)

        depth_im = Image.open(os.path.join("F:/Datasets/NYUDv2/eccv14-data/data/depth", depth_fp))
        depth_im_save_fp = os.path.join(save_fp, "depth_im_" + str(im_counter+offset) + ".png")
        depth_im.save(depth_im_save_fp)

        sem_im = Image.open(os.path.join("F:/Datasets/NYUDv2/eccv14-data/data/semantic_segmentation_masks", sem_fp))
        sem_save_fp = os.path.join(save_fp, "seg40_im_" + str(im_counter+offset) + ".png")
        sem_im.save(sem_save_fp)

#write_all_ims_and_seg_masks(True)
#write_all_ims_and_seg_masks(False)


def read_line_into_arr(line):
    arr = []
    line = line.split(",")
    for l in line:
        arr.append(int(l)-1)
    return arr


def write_is_masks():
    im_h = 425
    im_w = 560
    instance_num = 1
    for fp in os.listdir(is_masks_fp):
        instance_id_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        sem_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        for file in os.listdir(os.path.join(is_masks_fp, fp)):
            if file[-4:] == ".txt":
                class_str = file.split("_")[0]
                class_val = NYUDv2_instance_segmentation_categories_dict[class_str]
                with open(os.path.join(is_masks_fp, fp, file)) as f:
                    bbs = []
                    lines = f.readlines()
                    for line in lines:
                        bbs.append(read_line_into_arr(line[0:-1]))
                for i in range(1, len(bbs)+1):
                    bb = bbs[i-1]
                    mask = np.asarray(Image.open(os.path.join(is_masks_fp, fp, class_str+str(i)+".png")))
                    instance_mask = np.zeros_like(instance_id_mask, dtype=np.uint8)
                    instance_mask[bb[1]:bb[3]+1, bb[0]:bb[2]+1] = mask
                    instance_mask = np.where(instance_mask==1, instance_num, 0).astype(np.uint8)
                    instance_id_mask += instance_mask
                    sem_to_add = np.where(instance_mask==1, class_val, 0).astype(np.uint8)
                    sem_mask += sem_to_add
                    instance_num += 1
        instance_id_fp = os.path.join(is_masks_fp, fp, "is_ids.png")
        Image.fromarray(instance_id_mask).save(instance_id_fp)
        seg_id_fp = os.path.join(is_masks_fp, fp, "seg_ids.png")
        Image.fromarray(sem_mask).save(seg_id_fp)

#write_is_masks()

def get_is_masks(fp, original_image_fp=None):
    is_ids_fp = os.path.join(fp, "is_ids.png")
    seg_ids_fp = os.path.join(fp, "seg_ids.png")
    is_ids = np.asarray(Image.open(is_ids_fp))
    seg_ids = np.asarray(Image.open(seg_ids_fp))
    im = np.asarray(Image.open(original_image_fp))
    unique_ids = np.unique(is_ids)
    i_masks = []
    class_vals = []
    for unique_id in unique_ids:
        if unique_id != 0:
            instance_mask = np.where(is_ids == unique_id, 1, 0).astype(np.uint8)
            first_nonzero_ind = np.argwhere(instance_mask==1)[0]
            class_val = seg_ids[first_nonzero_ind[0]][first_nonzero_ind[1]]
            # sanity check code
            #s_utils.showSegmentationImage(instance_mask, im)
            i_masks.append(instance_mask)
            class_vals.append(class_val)
    return i_masks, class_vals

def get_is_masks(fp, original_image_fp=None):
    is_ids_fp = os.path.join(fp, "is_ids.png")
    seg_ids_fp = os.path.join(fp, "seg_ids.png")
    is_ids = np.asarray(Image.open(is_ids_fp))
    seg_ids = np.asarray(Image.open(seg_ids_fp))
    im = np.asarray(Image.open(original_image_fp))
    unique_ids = np.unique(is_ids)
    i_masks = []
    class_vals = []
    for unique_id in unique_ids:
        if unique_id != 0:
            instance_mask = np.where(is_ids == unique_id, 1, 0).astype(np.uint8)
            first_nonzero_ind = np.argwhere(instance_mask==1)[0]
            class_val = seg_ids[first_nonzero_ind[0]][first_nonzero_ind[1]]
            # sanity check code
            #s_utils.showSegmentationImage(instance_mask, im)
            i_masks.append(instance_mask)
            class_vals.append(class_val)
    return i_masks, class_vals

"""
original_image_fp = "F:/Datasets/NYUDv2/eccv14-data/data/images/img_5001.png"
fp = "F:/Datasets/NYUDv2/eccv14-data/data/is_masks/img_5001"
i_masks, c_vals = get_is_masks(fp, original_image_fp)
"""

def get_bb_from_is_mask(is_mask):
    val_tuples = np.argwhere(is_mask == 1)
    y_vals = val_tuples[:, 0:1]
    y_min = np.amin(y_vals)
    y_max = np.amax(y_vals)
    x_vals = val_tuples[:, 1:]
    x_min = np.amin(x_vals)
    x_max = np.amax(x_vals)
    bb = [x_min, y_min, x_max, y_max]
    return bb

def get_bbs(fp, original_image_fp=None):
    is_ids_fp = os.path.join(fp, "is_ids.png")
    seg_ids_fp = os.path.join(fp, "seg_ids.png")
    is_ids = np.asarray(Image.open(is_ids_fp))
    seg_ids = np.asarray(Image.open(seg_ids_fp))
    im = np.asarray(Image.open(original_image_fp))
    unique_ids = np.unique(is_ids)
    bbs = []
    class_vals = []
    for unique_id in unique_ids:
        if unique_id != 0:
            instance_mask = np.where(is_ids == unique_id, 1, 0).astype(np.uint8)
            bb = get_bb_from_is_mask(instance_mask)
            first_nonzero_ind = np.argwhere(instance_mask == 1)[0]
            class_val = seg_ids[first_nonzero_ind[0]][first_nonzero_ind[1]]
            bbs.append(bb)
            class_vals.append(class_val)
    # sanity check code
    s_utils.showImageWithBoundingBoxes(im, np.array(bbs))
    return bbs, class_vals

"""
original_image_fp = "F:/Datasets/NYUDv2/eccv14-data/data/images/img_5001.png"
fp = "F:/Datasets/NYUDv2/eccv14-data/data/is_masks/img_5001"
bbs, c_vals = get_bbs(fp, original_image_fp)
"""

def filter_path(path):
    return "rgb" in path

def get_im_num(path):
    path = path.split("_")[-1]
    return path

def requires_normalization(im):
    return (im > 255.0).any() > 0

def nyudv2_it(start_value, skip_value, batch_size, train, permutation, debug=False):
    if train:
        images_path = "../ProjectData/NYUDv2/train"
    else:
        images_path = "../ProjectData/NYUDv2/test"

    fps = list(filter(filter_path, os.listdir(images_path)))

    if permutation is not None:
        fps = [fps[i] for i in permutation]

    start_counter = 0
    skip_counter = 0
    current_batch_val = 0

    for index, fp in enumerate(fps):
        if skip_value != 0: # there is more than one worker
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

        if debug == True and index < 50:
            break

        #print(index)

        # rgb im
        rgb_im = Image.open(os.path.join(images_path, fp))
        rgb_im = rgb_im.resize((800, 600), resample=Image.BILINEAR)
        rgb_im = np.asarray(rgb_im)

        # depth im
        im_num = get_im_num(fp)
        depth_fp = images_path+"\depth_im_"+im_num
        depth_im = Image.open(os.path.join(depth_fp))
        depth_im = depth_im.resize((800, 600), resample=Image.NEAREST)
        depth_im = np.asarray(depth_im)
        if requires_normalization(depth_im):
            depth_im = s_utils.normalizeDepthImage(depth_im)

        # gt semantic segmentation mask
        sem_fp = images_path+"\seg40_im_"+im_num
        ss_mask = Image.open(os.path.join(sem_fp))
        original_ss_mask = np.asarray(ss_mask)
        resized_ss_mask = np.asarray(ss_mask.resize((800, 600), resample=Image.NEAREST))

        yield {"rgb_im": rgb_im, "depth_im": depth_im, "resized_ss_mask": resized_ss_mask, "original_ss_mask": original_ss_mask}


class nyudv2_iterable_dataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, train, permutation=None):
        super(nyudv2_iterable_dataset).__init__()
        self.batch_size = batch_size
        self.train = train
        self.permutation = permutation
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_value = 0
            skip_value = 0
        else:
            start_value = worker_info.id * self.batch_size
            skip_value = (worker_info.num_workers - 1) * self.batch_size
        return iter(nyudv2_it(start_value, skip_value, self.batch_size, self.train, self.permutation))


# transforms on rgb images
color_jitter = T.ColorJitter(0.5, 0.5, 0.1, 0.1) # brightness, contrast, saturation , hue
horizontal_flip = T.RandomHorizontalFlip(p=1.0)
gaussian_blur = d_utils.GaussianBlur(kernel_size=7)
rgb_normalization = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = T.ToTensor()

def nyu_collate(train, batch):
    rgb_ims = []
    depth_ims = []
    original_sem_masks = []
    resized_sem_masks = []
    for element in batch:
        rgb_im = to_tensor(element["rgb_im"])
        depth_im = torch.unsqueeze(torch.tensor(element["depth_im"]), dim=0)
        original_ss_mask = torch.unsqueeze(torch.tensor(element["original_ss_mask"]), dim=0)
        resized_ss_mask = torch.unsqueeze(torch.tensor(element["resized_ss_mask"]), dim=0)

        if train:
            if random.choice([True, False]):
                #rgb_im = color_jitter(rgb_im)
                rgb_im = horizontal_flip(rgb_im)
                depth_im = horizontal_flip(depth_im)
                resized_ss_mask = horizontal_flip(resized_ss_mask)

        rgb_im = rgb_normalization(rgb_im)
        rgb_ims.append(torch.unsqueeze(rgb_im, dim=0))
        depth_im = depth_im*(1/255)
        depth_im = (depth_im-0.449)/0.226

        """        
        # sanity check visualization code
        rgb_im_ = s_utils.unNormalizeImage(rgb_im)
        s_utils.showImage(rgb_im_)
        depth_im_ = (depth_im*0.226)+0.449
        depth_im_ *= 255
        depth_im_ = s_utils.channelsFirstToLast(depth_im_.numpy())
        depth_im_ = np.repeat(depth_im_, 3, axis=-1).astype(np.uint8)
        s_utils.showImage(depth_im_)
        #original_sem_mask_ = original_ss_mask.numpy()[0]
        resized_sem_mask_ = resized_ss_mask.numpy()[0]
        s_utils.showSegmentationImage(resized_sem_mask_, rgb_im_)
        """


        depth_ims.append(torch.unsqueeze(depth_im, dim=0))
        original_sem_masks.append(original_ss_mask)
        resized_sem_masks.append(resized_ss_mask)
    rgb_ims = torch.cat([rgb_im for rgb_im in rgb_ims], dim=0)
    depth_ims = torch.cat([depth_im for depth_im in depth_ims], dim=0)
    original_sem_masks = torch.cat([original_sem_mask for original_sem_mask in original_sem_masks], dim=0).type(torch.LongTensor)
    resized_ss_masks = torch.cat([resized_sem_mask for resized_sem_mask in resized_sem_masks], dim=0).type(torch.LongTensor)
    return rgb_ims, depth_ims, resized_ss_masks, original_sem_masks


def nyudv2_dl(batch_size=8, num_workers=0, train=True):
    coll = partial(nyu_collate, train)
    permutation = None
    if train:
        permutation = np.random.permutation(795).tolist()
    it = nyudv2_iterable_dataset(batch_size=batch_size, train=train, permutation=permutation)
    dl = torch.utils.data.DataLoader(it, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                     collate_fn=coll,  persistent_workers=False, drop_last=True)
    return dl


"""
if __name__ == "__main__":
    test_dl = nyudv2_dl(batch_size=1, num_workers=0, train=False)
    for index, l in enumerate(test_dl):
        pass
    print(index)
"""
