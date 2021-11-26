import torch
import numpy as np
import pprint
import CacheDictUtils

def computeMeanIU(model, test_data, cat_map, num_categories, with_depth=False, save_fp=None):
    model.eval()
    model.cuda()
    IU_dict = {}
    for i in range(num_categories+1):
        if i != 0:
            IU_dict[i] = {"intersection": 0, "union": 0, "total_count": 0}

    for batch_number, batch in enumerate(test_data):
        if batch_number % 100 == 0 and batch_number != 0:
            print("batch number: " + str(batch_number))

        # get images and bbs from the dataloader and send them to the gpu
        images = batch[0].cuda(non_blocking=True)
        if with_depth:
            depth_images = batch[1].cuda(non_blocking=True)
            segmentation_masks = batch[2].cuda(non_blocking=True)
            gt_seg_mask = batch[3][0].cuda(non_blocking=True)
        else:
            segmentation_masks = batch[1].cuda(non_blocking=True)
            gt_seg_mask = batch[2][0].cuda(non_blocking=True)

        hw_tuple = (gt_seg_mask.shape[0], gt_seg_mask.shape[1])

        with torch.no_grad():
            with torch.cuda.amp.autocast():  # 16 bit precision
                if not with_depth:
                    seg_mask, _ = model(images, segmentation_masks, hw_tuple)
                else:
                    seg_mask, _ = model(images, depth_images, segmentation_masks, hw_tuple)

        # gets counts of each category in the gt seg mask
        gt_categories, gt_counts = torch.unique(gt_seg_mask, return_counts=True)
        gt_categories_dict = {}
        for index, cat in enumerate(gt_categories):
            gt_categories_dict[cat.item()] = gt_counts[index].item()

        # gets counts of each category in the predicted seg mask
        pred_seg_mask = seg_mask[0]
        pred_categories, pred_counts = torch.unique(pred_seg_mask, return_counts=True)
        pred_categories_dict = {}
        for index, cat in enumerate(pred_categories):
            pred_categories_dict[cat.item()] = pred_counts[index].item()

        for cat in pred_categories_dict.keys():
            if cat != 0:
                if cat in gt_categories_dict:
                    num_pixels_per_cat = gt_categories_dict[cat]
                else:
                    num_pixels_per_cat = 0

                if num_pixels_per_cat != 0:
                    intersection = torch.count_nonzero(torch.logical_and(pred_seg_mask == cat, gt_seg_mask == cat)).item()
                    union = num_pixels_per_cat + torch.count_nonzero(torch.logical_and(pred_seg_mask == cat,
                                                                                       torch.logical_and(gt_seg_mask != cat, gt_seg_mask != 0))).item()
                else:
                    intersection = 0
                    union = torch.count_nonzero(torch.logical_and(pred_seg_mask == cat,
                                                                                       torch.logical_and(gt_seg_mask != cat, gt_seg_mask != 0))).item()

                IU_dict[cat]["intersection"] += int(intersection)
                IU_dict[cat]["union"] += int(union)
                IU_dict[cat]["total_count"] += int(num_pixels_per_cat) # the total number of actual pixels of this category in the ground truth dataset

        for cat in gt_categories_dict.keys():
            if cat not in pred_categories_dict.keys():
                if cat != 0:
                    num_pixels = int(gt_categories_dict[cat])
                    IU_dict[cat]["union"] += num_pixels
                    IU_dict[cat]["total_count"] += num_pixels

    print("Evaluation before mean IOU computation complete")

    # mean IU computation
    mean_IU = 0.0
    zero_cats = 0
    for cat in IU_dict.keys():
        if IU_dict[cat]["union"] != 0:
            cat_mean_IU = IU_dict[cat]["intersection"]/IU_dict[cat]["union"]
            IU_dict[cat]["mean_IU"] = round(cat_mean_IU, 3)
            mean_IU += cat_mean_IU
        else:
            zero_cats += 1
    mean_IU /= (num_categories-zero_cats)

    # numbers to labels
    IU_dict_ = {}
    for cat in IU_dict.keys():
        IU_dict_[cat_map[int(cat)]] = IU_dict[cat]
    IU_dict = IU_dict_

    pprint.pprint(IU_dict)
    print("The mean IU is " + str(round(mean_IU, 3)))

    # weighted mean IU computation
    total_count = 0
    for cat in IU_dict.keys():
        total_count += IU_dict[cat]["total_count"]

    weighted_mean_IU = 0.0
    for cat in IU_dict.keys():
        if IU_dict[cat]["total_count"] != 0:
            weighted_mean_IU += (IU_dict[cat]["total_count"]/total_count) * (IU_dict[cat]["mean_IU"])

    IU_dict["mean_IU"] = mean_IU
    IU_dict["weighted_mean_IU"] = weighted_mean_IU

    print("The weighted mean IU is " + str(round(weighted_mean_IU, 3)))
    if save_fp is not None:
        CacheDictUtils.writeReadableCachedDict(save_fp, IU_dict)
    return IU_dict


def computeMeanPixelAccuracy(model, test_data, cat_map, num_categories, with_depth=False, save_fp=None):
    model.eval()
    model.cuda()
    accuracy_per_cat_dict = {}
    for i in range(num_categories + 1):
        if i != 0:
            accuracy_per_cat_dict[i] = {"count": 0, "correct": 0}
    total_num_pix = 0
    for batch_number, batch in enumerate(test_data):
        if batch_number % 100 == 0 and batch_number != 0:
            print("batch number: " + str(batch_number))

        # get images and bbs from the dataloader and send them to the gpu
        images = batch[0].cuda(non_blocking=True)
        if with_depth:
            depth_images = batch[1].cuda(non_blocking=True)
            segmentation_masks = batch[2].cuda(non_blocking=True)
            gt_seg_mask = batch[3][0].cuda(non_blocking=True)
        else:
            segmentation_masks = batch[1].cuda(non_blocking=True)
            gt_seg_mask = batch[2][0].cuda(non_blocking=True)

        hw_tuple = (gt_seg_mask.shape[0], gt_seg_mask.shape[1])

        with torch.no_grad():
            with torch.cuda.amp.autocast():  # 16 bit precision
                if not with_depth:
                    seg_mask, _ = model(images, segmentation_masks, hw_tuple)
                else:
                    seg_mask, _ = model(images, depth_images, segmentation_masks, hw_tuple)

        # gets counts of each category in the gt seg mask
        gt_categories, gt_counts = torch.unique(gt_seg_mask, return_counts=True)
        gt_categories_dict = {}
        for index, cat in enumerate(gt_categories):
            gt_categories_dict[cat.item()] = gt_counts[index].item()

        pred_seg_mask = seg_mask[0]
        for cat in gt_categories_dict.keys():
            if cat != 0:
                num_pixels_per_cat = gt_categories_dict[cat]
                total_num_pix += num_pixels_per_cat
                correct = torch.count_nonzero(torch.logical_and(pred_seg_mask == cat, gt_seg_mask == cat)).item()
                accuracy_per_cat_dict[cat]["count"] += int(num_pixels_per_cat)
                accuracy_per_cat_dict[cat]["correct"] += int(correct)

    # numbers to labels
    accuracy_per_cat_dict_ = {}
    for cat in accuracy_per_cat_dict.keys():
        accuracy_per_cat_dict_[cat_map[int(cat)]] = accuracy_per_cat_dict[cat]
    accuracy_per_cat_dict = accuracy_per_cat_dict_

    correct_count = 0
    total_count = 0
    num_zero_cats = 0
    average_accuracy = 0

    for cat in accuracy_per_cat_dict.keys():
        if cat != 0:
            if accuracy_per_cat_dict[cat]["count"] != 0:
                total_count += accuracy_per_cat_dict[cat]["count"]
                correct_count += accuracy_per_cat_dict[cat]["correct"]
                accuracy_per_cat_dict[cat]["accuracy"] = accuracy_per_cat_dict[cat]["correct"]/accuracy_per_cat_dict[cat]["count"]
                average_accuracy += accuracy_per_cat_dict[cat]["accuracy"]
            else:
                num_zero_cats += 1

    overall_accuracy = correct_count/total_count
    average_accuracy = average_accuracy/(len(list(accuracy_per_cat_dict.keys()))-num_zero_cats)
    accuracy_per_cat_dict["overall_accuracy"] = overall_accuracy
    accuracy_per_cat_dict["average_accuracy"] = average_accuracy

    print("The accuracy is " + str(round(overall_accuracy, 3)))
    print("The average accuracy is " + str(round(average_accuracy, 3)))
    if save_fp is not None:
        CacheDictUtils.writeReadableCachedDict(save_fp, accuracy_per_cat_dict)
    return accuracy_per_cat_dict