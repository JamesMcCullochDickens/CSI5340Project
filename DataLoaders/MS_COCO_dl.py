import PIL.Image
import pycocotools.coco as pycoco
import OriginalCocoClasses
import PathGetter as pg
import os
from PIL import Image
import numpy as np
import ShowImageUtils as s_utils
import pprint as pp
import json
import GenericDataloader as g_dl
import random

outer_path = pg.get_fp("MS_COCO")
train_2017_fp = os.path.join(outer_path, "train2017")
val_2017_fp = os.path.join(outer_path, "val2017")
annotations_folder_fp = "annotations"
stuff_train_annotations_fp = "stuff_train2017.json"
instance_train_annotations_fp = "instances_train2017.json"
stuff_val_annotations_fp = "stuff_val2017.json"
instance_val_annotations_fp = "instances_val2017.json"
ps_train_annotations_fp = "panoptic_train2017.json"
ps_val_annotations_fp = "panoptic_val2017.json"
ps_png_train_fp = "panoptic_train2017"
ps_png_val_fp = "panoptic_val2017"
original_coco_classes = OriginalCocoClasses.class_dict
exclude_things = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
stuff_start_index = 92
include_stuff = [93, 101, 102, 103, 105, 108, 109, 110, 114, 115, 116, 117, 118, 123, 130, 133, 152, 156, 165, 168, 171, 172, 173, 174, 175, 176, 177, 180, 181]
stuff_range = ["ceiling", "floor", "wall", "window"] # classes that we will merge the stuff classes for (i.e. all types of floor going to one generic floor class)
total_num_classes = 182

def create_sem_dict():
    sem_dict = {}
    map_dict = {}
    counter = 0
    is_Ranged = False
    isFirst = True
    for i in range(total_num_classes+1):
        is_Ranged = False
        if i < stuff_start_index and i not in exclude_things:
            sem_dict[counter] = (original_coco_classes[i])
            map_dict[i] = counter
            counter += 1
            continue
        if i > stuff_start_index:
            if i in include_stuff:
                for stuff in stuff_range:
                    if stuff in original_coco_classes[i]:
                        is_Ranged = True
                        if isFirst:
                            sem_dict[counter] = stuff
                            map_dict[i] = counter
                            counter += 1
                            isFirst = False
                            break
                        else:
                            map_dict[i] = counter-1
                            break
                if not is_Ranged:
                    sem_dict[counter] = original_coco_classes[i]
                    map_dict[i] = counter
                    counter += 1
                    isFirst = True
    #pp.pprint(sem_dict)
    #pp.pprint(map_dict)
    return sem_dict, map_dict

# Writes the semantic segmentation masks to the same folder as train2017 images
def write_ss_masks(val=False):
    sem_dict, map_dict = create_sem_dict()
    if not val:
        things_coco = pycoco.COCO(os.path.join(outer_path, annotations_folder_fp, instance_train_annotations_fp))
        stuff_coco = pycoco.COCO(os.path.join(outer_path, annotations_folder_fp, stuff_train_annotations_fp))
    else:
        things_coco = pycoco.COCO(os.path.join(outer_path, annotations_folder_fp, instance_val_annotations_fp))
        stuff_coco = pycoco.COCO(os.path.join(outer_path, annotations_folder_fp, stuff_val_annotations_fp))
    for index, image_info in enumerate(things_coco.dataset["images"]):
        if index % 1000 == 0 and index != 0:
            print("Written " + str(index) + " ground truth semantic segmentation masks")
        image_name = image_info["file_name"]
        image_id = image_info["id"]
        if not val:
            image_fp = os.path.join(train_2017_fp, image_name)
        else:
            image_fp = os.path.join(val_2017_fp, image_name)
        im_array = np.asarray(Image.open(image_fp))
        im_height = im_array.shape[0]
        im_width = im_array.shape[1]
        sem_mask = np.zeros((im_height, im_width), dtype=np.uint8)

        # adding thing elements
        anns = things_coco.imgToAnns[image_id]
        for ann in anns:
            id = ann['category_id']
            if id not in map_dict:
                continue
            id = map_dict[id]
            i_mask = things_coco.annToMask(ann)
            i_mask = np.where(i_mask == 1, id, 0)
            sem_mask = np.maximum(sem_mask, i_mask)

        # adding stuff elements
        anns = stuff_coco.imgToAnns[image_id]
        for ann in anns:
            id = ann['category_id']
            if id not in map_dict:
                continue
            id = map_dict[id]
            i_mask = things_coco.annToMask(ann)
            i_mask = np.where(i_mask == 1, id, 0)
            sem_mask = np.maximum(sem_mask, i_mask)

        if np.sum(sem_mask) == 0: #don't write gt segmentation masks
            continue

        fp_to_save = image_fp[:-4] + "_seg.png"
        sem_mask = Image.fromarray(sem_mask)
        sem_mask.save(fp_to_save)

#write_ss_masks()
#write_ss_masks(val=True)

def write_ps_masks(val=False):
    if not val:
        path = os.path.join(outer_path, annotations_folder_fp, ps_train_annotations_fp)
    else:
        path = os.path.join(outer_path, annotations_folder_fp, ps_val_annotations_fp)
    ps_dict = json.load(open(path, 'r'))
    #pp.pprint(ps_dict["categories"])
    for index, ann in enumerate(ps_dict["annotations"]):
        if index % 1000 == 0 and index != 0:
            print("Written " + str(index) + " ground truth semantic segmentation masks")
        im_name = ann["file_name"]
        if not val:
            path = os.path.join(outer_path, annotations_folder_fp, ps_png_train_fp, im_name)
            im = np.asarray(Image.open(path))
        else:
            path = os.path.join(outer_path, annotations_folder_fp, ps_png_val_fp, im_name)
            im = np.asarray(Image.open(path))

        id_mask = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
        original_ids = [0]
        id_map = {0: 0}
        counter = 1
        unique_colors = get_unique_colors(im)
        for unique_color in unique_colors:
            if not (np.sum(unique_color == [0,0,0]) == 3):
                id = unique_color[0]+unique_color[1]*256+unique_color[2]*(256**2)
                original_ids.append(id)
                id_map[id] = counter
                id_mask = np.where(im == unique_color, [counter, counter, counter], id_mask)
                counter += 1
            else:
                id_mask = np.where(im == unique_color, [0,0,0], id_mask)

        id_mask = id_mask[:, :, 0]

        sem_mask = np.zeros_like(id_mask, dtype=np.uint8)
        for id in original_ids:
            for segment_info in ann["segments_info"]:
                if id == segment_info["id"]:
                    sem_mask = np.where(id_mask == id_map[id], segment_info["category_id"], sem_mask)

        id_mask = Image.fromarray(id_mask)
        id_mask.save(path[0:-4]+"_id_mask.png")
        sem_mask = Image.fromarray(sem_mask)
        sem_mask.save(path[0:-4]+"_sem_mask.png")

def get_unique_colors(img):
    unique_colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    return unique_colors

#write_ps_masks()
#write_ps_masks(val=True)

def filterSeg(path):
    if "_seg" in path:
        return False
    else:
        return True

def coco_ss_iterator(start_value, skip_value, batch_size, val):
    if not val:
        images_path = train_2017_fp
    else:
        images_path = val_2017_fp
    image_names = list(filter(filterSeg, os.listdir(images_path)))
    for index, image_name in enumerate(image_names):
        if not g_dl.skipFunction(index, start_value, batch_size, skip_value):
            continue
        im_array = np.asarray(Image.open(os.path.join(images_path, image_name)), dtype=np.uint8)
        seg_array = np.asarray(Image.open(os.path.join(images_path, image_name[0:-4]+"_seg.png")), dtype=np.uint8)
        # sanity check code
        s_utils.showSegmentationImage(seg_array, im_array)
        yield {"image": im_array, "ss_mask": seg_array}

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


example_im_path = "F:/Datasets/MS_COCO/train2017/000000000113.jpg"
im = np.asarray(Image.open(example_im_path))
seg = np.asarray(Image.open(example_im_path[0:-4]+"_seg.png"))
s_utils.showSegmentationImage(seg, im)
im, seg = random_crop_and_resize(im, seg)
s_utils.showSegmentationImage(seg, im)


"""
def collate_func(batch):
    images = []
    sem_masks = []
    for batch_elem in batch:
"""
