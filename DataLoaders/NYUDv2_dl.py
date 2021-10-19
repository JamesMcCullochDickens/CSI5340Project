import os
from pathlib import Path
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