import CacheDictUtils
import PathGetter
import scipy.io
import os
import mat73 # requires install with pip install mat73
from PIL import Image
import numpy as np
import CategoryFilter
import CategoryInfo
import torch
from functools import partial
import ShowImageUtils as s_utils
import torchvision.transforms as T

outer_path = PathGetter.get_fp("SUN_RGBD")
train_test_path = "traintestSUNRGBD"
train_test_split = "allsplit.mat"
toolbox_path = "SUNRGBDtoolbox"
toolbox_metadata_path = "Metadata"
seg37list_path = "seg37list.mat" # just the 37 file categories
seg_metadata_path = "SUNRGBD2Dseg.mat" # contains the seg masks for 37 and all, but not the file names
sunrgbd_meta_path = "SUNRGBDMeta.mat" # the file names
corrected_v2_info = os.path.join(outer_path, "SUNRGBDMeta2DBB_v2.mat")
outer_toolbox_path = os.path.join(outer_path, toolbox_path, toolbox_metadata_path)

instance_segmentation_categories_dict = CategoryInfo.SUN_RGBD_instance_segmentation_categories_dict
instance_segmentation_categories_dict_reverse = {v: k for k, v in instance_segmentation_categories_dict.items()}
instance_segmentation_keys = list(instance_segmentation_categories_dict.keys())

def get_SUNRGBD_file_list():
    sunrgbd_meta = scipy.io.loadmat(os.path.join(outer_toolbox_path, sunrgbd_meta_path))
    file_list = sunrgbd_meta["SUNRGBDMeta"][0]
    return file_list

def write_semantic_seg_masks():
    #seg37_list = scipy.io.loadmat(os.path.join(outer_toolbox_path, seg37list_path))
    #print(seg37_list)
    seg_metadata = mat73.loadmat(os.path.join(outer_toolbox_path, seg_metadata_path))
    file_list = get_SUNRGBD_file_list()
    for index, fp in enumerate(file_list):
        fp = fp[0][0]
        path = os.path.join(outer_path, fp)
        semantic_segmentation_mask = seg_metadata["SUNRGBD2Dseg"]["seglabel"][index].astype(np.uint8)
        semantic_segmentation_mask = Image.fromarray(semantic_segmentation_mask)
        semantic_segmentation_mask.save(path+"/seg37.png")

#write_semantic_seg_masks()

def get_test_train_split():
    path = os.path.join(outer_path, toolbox_path, train_test_path, train_test_split)
    split = scipy.io.loadmat(path)
    test_file_paths = split["alltest"][0]
    temp_test_file_paths = []
    for fp in test_file_paths:
        temp_test_file_paths.append(fp[0])
    test_file_paths = temp_test_file_paths

    train_val_split = split["trainvalsplit"][0][0]
    train_paths = []
    val_paths = []
    for fp in train_val_split[0]:
        train_paths.append(fp[0][0])
    for fp in train_val_split[1]:
        val_paths.append(fp[0][0])

    trainval_paths = train_paths+val_paths
    dataset_list = {"train_paths": train_paths, "trainval_paths": trainval_paths, "val_paths": val_paths, "test_paths": test_file_paths}
    for type in dataset_list.keys():
        for index, val in enumerate(dataset_list[type]):
            if val[-1] == "/":
                val = val[:-1]  # in case the last letter is a slash
            dataset_list[type][index] = outer_path + "/" + val[17:]
    return dataset_list

#get_test_train_split()

def write_all_to_folder(train=True):
    if train:
        save_fp = "C:/Users/James/PycharmProjects/CSI5340Project/ProjectData/SUN_RGBD/Train"
        fps = get_test_train_split()["trainval_paths"]
    else:
        save_fp = "C:/Users/James/PycharmProjects/CSI5340Project/ProjectData/SUN_RGBD/Test"
        fps = get_test_train_split()["test_paths"]
    for index, fp in enumerate(fps):
        if not train:
            counter = index + 5285
        else:
            counter = 0
        rgb_folder_path = fp + "/image"
        rgb_file_name = os.listdir(rgb_folder_path)[0]
        rgb_im = Image.open(os.path.join(rgb_folder_path, rgb_file_name))
        rgb_save_path = os.path.join(save_fp, "rgb_im_"+str(index+counter)+".png")
        rgb_im.save(rgb_save_path)

        depth_folder_path = fp + "/depth_bfx"
        depth_file_name = os.listdir(depth_folder_path)[0]
        depth_im = Image.open(os.path.join(depth_folder_path, depth_file_name))
        depth_im_save_path = os.path.join(save_fp, "depth_im_" + str(index+counter) + ".png")
        depth_im.save(depth_im_save_path)

        seg_path = fp+"/seg37.png"
        seg_im = Image.open(seg_path)
        seg_im_save_path = os.path.join(save_fp, "seg_im_" + str(index+counter) + ".png")
        seg_im.save(seg_im_save_path)

#write_all_to_folder(True)
#write_all_to_folder(False)

def get_bounding_boxes():
    data = scipy.io.loadmat(corrected_v2_info)
    bounding_boxes = data["SUNRGBDMeta2DBB"][0]
    image_name_bounding_box_dict = {}
    for image in bounding_boxes:
        image_name = image[0][0]
        image_path = os.path.join(outer_path, image_name)
        if len(image[1]) != 0:
            bounding_boxes = image[1][0]
            bounding_box_object_name_tuples = []
            for i in range(len(bounding_boxes)):
                bounding_box = bounding_boxes[i][1][0].tolist()
                object_type = bounding_boxes[i][2][0]
                bounding_box_object_name_tuples.append((bounding_box, object_type))
            image_name_bounding_box_dict[image_path] = bounding_box_object_name_tuples
        else:
            bounding_box_object_name_tuples = []
            image_name_bounding_box_dict[image_path] = []  # no bounding boxes for a given image
        bb_dict = {"bbs:" : bounding_box_object_name_tuples}
        # writing the bounding boxes to the same folder as the image as a dict
        #CacheDictUtils.writeReadableCachedDict(image_path+"/bbs", bb_dict)
    return image_name_bounding_box_dict

#bb_dict = get_bounding_boxes()


def getInstanceInfo(folder_path):
    if "NYU" not in folder_path:
        return getInstanceInfo_(folder_path)
    else:
        return getInstanceInfoNYUDv2(folder_path)

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

def getInstanceInfo_(folder_path):
    global object_detection_keys

    # load seg data from matlab file
    seg_path = folder_path + "/seg"
    seg_data = scipy.io.loadmat(seg_path)

    # not an NYUDv2 image
    instance_mask = seg_data["seglabel"]
    seg_names = seg_data["names"][0]

    labels = []
    instance_masks = []
    bbs = []
    label_nums = []
    for index, name in enumerate(seg_names):
        filtered_name = CategoryFilter.filter_type_instance_segmentation_sun_rgbd(name[0])
        if filtered_name in instance_segmentation_keys and filtered_name != "unknown":
            name_number = index+1 #0 is always unknown

            # create a binary instance mask
            mask = np.where(instance_mask == name_number, 1, 0)
            if np.count_nonzero(mask) == 0: # sometimes a given name is not included in the overall mask strangely
                continue
            instance_masks.append(mask)

            # get the enclosing bounding box
            bb = get_bb_from_is_mask(mask)
            bbs.append(bb)

            labels.append(filtered_name)

            label_nums.append(instance_segmentation_categories_dict[filtered_name])

    return {"instance_masks": instance_masks, "bbs": bbs, "labels": labels, "label_nums": label_nums}


def getInstanceInfoNYUDv2(folder_path):
    global object_detection_keys

    #image = Image.open(folder_path + "/image/NYU0486.jpg")
    #image = np.array(image)

    # load seg data from matlab file
    seg_path = folder_path + "/seg"
    seg_data = scipy.io.loadmat(seg_path)
    seg_names = seg_data["names"]

    # using all the labels in the NYUDv2 instance segmentation set
    label_map = seg_data["seglabel"]
    instance_map = seg_data["seginstances"]
    #instance_map_values = np.unique(instance_map)
    label_map_values = np.unique(label_map)
    instance_masks = []
    labels = []
    label_nums = []
    bbs = []


    # basically the way this works is you have an overall semantic label map
    # for a unique label you filter the semantic label map according to the label to get a binary image
    # then within that binary image you find the 1s, and project this on to the instance lab map
    # from the instance label map you get the unique values, and for each unique value
    # you have an instance mask for that particular instance


    for label_map_value in label_map_values:
        if label_map_value == 0:
            continue
        label_name = seg_names[label_map_value-1][0][0]
        label_name = CategoryFilter.filter_type_instance_segmentation_sun_rgbd(label_name)
        if label_name in instance_segmentation_keys and label_name != "unknown":

            # get the segmentation label values where the label map is equal to a given value
            overall_binary_mask = np.where(label_map == label_map_value, 1, 0)

            # for the 1 values of the binary mask, get the corresponding values of the instance map
            filtered_instance_map = np.where(overall_binary_mask == 1, instance_map, 0)

            # the unique non-zero values of the instance map
            unique_filtered_instance_nums = np.unique(filtered_instance_map)[1:]

            for unique_filtered_instance_num in unique_filtered_instance_nums:
                instance_mask = np.where(filtered_instance_map == unique_filtered_instance_num, 1, 0)
                instance_masks.append(instance_mask)
                labels.append(label_name)
                bb = get_bb_from_is_mask(instance_mask)
                bbs.append(bb)
                label_nums.append(instance_segmentation_categories_dict[label_name])
                #S_utils.showBinaryInstanceMask(image, instance_mask, bb=[bb])

    return {"instance_masks": instance_masks, "bbs": bbs, "labels": labels, "label_nums": label_nums}

"""
# example usage of getInstanceInfo
sample_fp1 = "F:/Datasets/SUN_RGBD/SUNRGBD/kv1/b3dodata/img_0065" # not NYUDv2
sample_fp2 = "F:/Datasets/SUN_RGBD/SUNRGBD/kv1/NYUdata" # NYUDv2
instance_info1 = getInstanceInfo(sample_fp1)
instance_info2 = getInstanceInfo(sample_fp2)
debug = "debug"
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
        images_path = "../ProjectData/SUN_RGBD/train"
    else:
        images_path = "../ProjectData/SUN_RGBD/test"

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


class sun_rgbd_iterable_dataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, train, permutation=None):
        super(sun_rgbd_iterable_dataset).__init__()
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
        depth_im_ = depth_im*255.0
        depth_im_ = (depth_im_*0.226)+0.449
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


def sun_rgbd_dl(batch_size=8, num_workers=0, train=True):
    coll = partial(nyu_collate, train)
    permutation = None
    if train:
        permutation = np.random.permutation(5285).tolist()
    it = sun_rgbd_iterable_dataset(batch_size=batch_size, train=train, permutation=permutation)
    dl = torch.utils.data.DataLoader(it, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                     collate_fn=coll,  persistent_workers=False, drop_last=True)
    return dl












