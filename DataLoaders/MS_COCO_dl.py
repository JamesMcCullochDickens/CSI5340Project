import OriginalCocoClasses
import PathGetter as pg
import os
from PIL import Image
import numpy as np
import json
import GenericDataloader as g_dl
import torch
from torchvision import transforms
import DataAug_Utils as d_utils
from pathlib import Path
import COCOStuffDict as cs_dict
import torchvision.transforms.functional as TF
from functools import partial

# paths
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
saved_models_path = os.path.join(Path(os.getcwd()).parent, "Models/Trained_Models")
num_train_ss_imgs = 118287 # after filtering out the odd gray-scale image that for some reason is included
num_val_ss_imgs = 5000


# coco stuff class info
original_coco_classes = OriginalCocoClasses.class_dict
coco_classes = cs_dict.coco_stuff_dict
exclude_things = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
stuff_start_index = 92
include_stuff = [93, 101, 102, 103, 105, 108, 109, 110, 114, 115, 116, 117, 118, 123, 130, 133, 152, 156, 165, 168, 171, 172, 173, 174, 175, 176, 177, 180, 181]
stuff_range = ["ceiling", "floor", "wall", "window"] # classes that we will merge the stuff classes for (i.e. all types of floor going to one generic floor class)
total_num_classes = 182

# transforms
crop_tuple = (500, 500)
transform_to_tensor = transforms.ToTensor()
normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image_mean_mean = 0.449
image_std_mean = 0.226
random_crop_transform = transforms.RandomCrop((513, 513))
hflip_transform = d_utils.multi_input_hflip(threshold=0.4)
resize_tuple = (800, 600)


def create_sem_dict():
    sem_dict = {}
    map_dict = {}
    counter = 0
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

#create_sem_dict()

"""
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
"""

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

def coco_ss_iterator(start_value, skip_value, batch_size, val, permutation, debug=False):
    if not val:
        images_path = train_2017_fp
    else:
        images_path = val_2017_fp
    image_names = list(filter(filterSeg, os.listdir(images_path)))
    if permutation is not None:
        image_names = [image_names[i] for i in permutation]
    for index, image_name in enumerate(image_names):
        if skip_value != 0: # there is more than one worker
            if not g_dl.skipFunction(index, start_value, batch_size, skip_value):
                continue

        # not all images in train or val have a seg mask
        seg_path = os.path.join(images_path, image_name[0:-4] + "_seg.png")
        if not os.path.isfile(seg_path):
            continue

        if debug and index >= 24: # DEBUG (set debug false by default obv)
            return

        im = Image.open(os.path.join(images_path, image_name))
        im = im.resize((600, 800), resample=Image.BILINEAR)
        im = np.asarray(im, dtype=np.uint8)
        if len(im.shape) == 2: # ignore the grey-scale images
            continue

        ss_mask = Image.open(os.path.join(images_path, image_name[0:-4]+"_seg.png"))
        original_ss_mask = np.asarray(ss_mask, dtype=np.uint8)
        ss_mask = ss_mask.resize((600, 800), resample=Image.NEAREST)
        ss_mask = np.asarray(ss_mask, dtype=np.uint8)

        """
        # sanity check code to visualize mask
        cats = np.unique(ss_mask)
        for cat in cats:
            print(coco_classes[cat])
        s_utils.showSegmentationImage(ss_mask, im)
        """


        yield {"image": im, "resized_ss_mask": ss_mask, "original_ss_mask": original_ss_mask}

class COCO_iterable_dataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, val, permutation=None):
        super(COCO_iterable_dataset).__init__()
        self.batch_size = batch_size
        self.val = val
        self.permutation = permutation
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_value = 0
            skip_value = 0
        else:
            start_value = worker_info.id * self.batch_size
            skip_value = (worker_info.num_workers - 1) * self.batch_size
        return iter(coco_ss_iterator(start_value, skip_value, self.batch_size, self.val, self.permutation))


def my_collate(batch, val):
    ims = []
    ss_masks = []
    original_ss_masks = []
    for batch_dict in batch:
        im = transform_to_tensor(batch_dict["image"])
        ss_mask = torch.tensor(batch_dict["resized_ss_mask"])
        if not val: # don't crop during testing
            i, j, h, w = random_crop_transform.get_params(im, crop_tuple)
            im = TF.crop(im, i, j, h, w)
            ss_mask = TF.crop(ss_mask, i, j, h, w)
        im = torch.unsqueeze(normalize_transform(im), 0)
        ss = torch.unsqueeze(ss_mask, 0)
        ims.append(im)
        ss_masks.append(ss)
        original_ss_masks.append(torch.tensor(batch_dict["original_ss_mask"]))
    ims = torch.cat([im for im in ims], dim=0)
    ss_masks = torch.cat([ss_mask for ss_mask in ss_masks], dim=0)
    ss_masks = torch.squeeze(ss_masks, dim=1).type(torch.LongTensor)
    return ims, ss_masks, original_ss_masks

def my_collate_grayscale(batch, val):
    ims = []
    ss_masks = []
    original_ss_masks = []
    for batch_dict in batch:
        im = transform_to_tensor(batch_dict["image"])
        ss_mask = torch.tensor(batch_dict["resized_ss_mask"])
        if not val: # don't crop during testing
            i, j, h, w = random_crop_transform.get_params(im, crop_tuple)
            im = TF.crop(im, i, j, h, w)
            ss_mask = TF.crop(ss_mask, i, j, h, w)
        im = torch.unsqueeze(normalize_transform(im), 0)
        im = torch.unsqueeze(torch.mean(im, dim=1), 1)
        ss = torch.unsqueeze(ss_mask, 0)
        ims.append(im)
        ss_masks.append(ss)
        original_ss_masks.append(torch.tensor(batch_dict["original_ss_mask"]))
    ims = torch.cat([im for im in ims], dim=0)
    ss_masks = torch.cat([ss_mask for ss_mask in ss_masks], dim=0)
    ss_masks = torch.squeeze(ss_masks, dim=1).type(torch.LongTensor)
    return ims, ss_masks, original_ss_masks


def get_mscoco_stuff_train_it(batch_size=8, num_workers=8, pin_memory=False):
    coll = partial(my_collate, val=False)
    permutation = np.random.permutation(num_train_ss_imgs).tolist()
    it = COCO_iterable_dataset(batch_size, val=False, permutation=permutation)
    dl = torch.utils.data.DataLoader(it, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                     collate_fn=coll, persistent_workers=False, drop_last=True)
    return dl

def get_mscoco_stuff_val_it(batch_size=12, num_workers=8, pin_memory=False):
    coll = partial(my_collate, val=True)
    permutation = None
    it = COCO_iterable_dataset(batch_size, val=True, permutation=permutation)
    dl = torch.utils.data.DataLoader(it, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                     collate_fn=coll, persistent_workers=False, drop_last=True)
    return dl


def get_mscoco_stuff_grayscale_train_it(batch_size=8, num_workers=8, pin_memory=False):
    coll = partial(my_collate_grayscale, val=False)
    permutation = np.random.permutation(num_train_ss_imgs).tolist()
    it = COCO_iterable_dataset(batch_size, val=False, permutation=permutation)
    dl = torch.utils.data.DataLoader(it, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                     collate_fn=coll, persistent_workers=False, drop_last=True)
    return dl

def get_mscoco_stuff_grayscale_val_it(batch_size=12, num_workers=8, pin_memory=False):
    coll = partial(my_collate_grayscale, val=True)
    permutation = None
    it = COCO_iterable_dataset(batch_size, val=True, permutation=permutation)
    dl = torch.utils.data.DataLoader(it, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                                     collate_fn=coll, persistent_workers=False, drop_last=True)
    return dl

"""
if __name__ == "__main__":
    dl = get_mscoco_stuff_grayscale_train_it(num_workers=0, batch_size=1)
    for index, t in enumerate(dl):
        debug = "debug"
"""
