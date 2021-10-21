import CacheDictUtils
import PathGetter
import scipy.io
import os
import mat73 # requires install with pip install mat73
from PIL import Image
import numpy as np
import CategoryFilter
import CategoryInfo

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