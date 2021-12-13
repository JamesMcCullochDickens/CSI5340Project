import torch
import pprint
import os
from pathlib import Path
import time
import ShowImageUtils as s_utils
saved_models_path = os.path.join(os.getcwd(), "Trained_Models")
saved_plots_path = os.path.join(os.getcwd(), "Saved_Plots")
saved_eval_results_path = os.path.join(Path(os.getcwd()).parent, "Evaluation/Eval_Results")
import Models.DeepLabV3Plus as dlv3p
import PlotUtils.PlotUtils as plot_utils
import Evaluation.Evaluate_SS as eval_ss
import COCOStuffDict as stuff_dict
import logging
import Models.DeepLabv3 as dlv3
import numpy as np
import DataLoaders.NYUDv2_dl as nyudv2_dl
import Backbones.ResNet as rn
logging.captureWarnings(True)

def saveModel(epoch, model, optimizer, lr_scheduler, loss_dict, save_path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict':lr_scheduler.state_dict(),
        'loss_dict': loss_dict
    }, save_path)

def loadModel(model, optimizer, lr_scheduler, save_path, model_only=False):
    try:
        checkpoint = torch.load(save_path)
    except:
        print("No model file exists")
    model.load_state_dict(checkpoint['model_state_dict'])
    if model_only:
        return model
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    loss_dict = checkpoint["loss_dict"]
    return epoch, model, optimizer, lr_scheduler, loss_dict

def train_rgb_ss_model(model, optimizer, lr_scheduler, num_epochs, dataloader, save_path, load_path=None):
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    model.train()
    model.cuda()
    if load_path is not None:
        epoch, model, optimizer, lr_scheduler, loss_dict = loadModel(model, optimizer, lr_scheduler, load_path)
    else:
        epoch = None
        loss_dict = {"epoch_losses": [], "iteration_losses":[]}
    if epoch is None:
        epoch_min = 1
    else:
        epoch_min = epoch
    for epoch_num in range(epoch_min, num_epochs+1):
        epoch_loss = 0.0
        iteration_losses = []
        for batch_num, data in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            if batch_num % 100 == 0 and batch_num != 0:
                print("Batch num " + str(batch_num))
            ims = data[0].to('cuda:0', non_blocking=True)
            seg_masks = data[1].to('cuda:0', non_blocking=True)
            with torch.cuda.amp.autocast():  # 16 bit precision = faster and less memory
                _, loss = model(ims, seg_masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss = loss.item()
            epoch_loss += loss
            iteration_losses.append(loss)
        lr_scheduler.step()
        loss_dict["epoch_losses"].extend([epoch_loss])
        loss_dict["iteration_losses"].extend(iteration_losses)
        print("\nFinished training epoch " + str(epoch_num) + " with loss: " + str(round(epoch_loss, 2)))
        saveModel(epoch_num, model, optimizer, lr_scheduler, loss_dict, save_path)
    return {"epoch_losses": loss_dict["epoch_losses"], "iteration_losses": loss_dict["iteration_losses"]}

def train_ss_model(model, optimizer, lr_scheduler, num_epochs, train_dl, test_dl,
                         eval_dict, save_path, depth_only=False, rgb_only=False, load_path=None):
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    model.train()
    model.cuda()
    if load_path is not None:
        epoch, model, optimizer, lr_scheduler, loss_dict = loadModel(model, optimizer, lr_scheduler, load_path)
    else:
        epoch = None
        loss_dict = {"epoch_losses": [], "iteration_losses": []}
    if epoch is None:
        epoch_min = 1
    else:
        epoch_min = epoch
    best_mIoU = -1
    for epoch_num in range(epoch_min, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        iteration_losses = []
        for batch_num, data in enumerate(train_dl):
            optimizer.zero_grad(set_to_none=True)
            if batch_num % 100 == 0 and batch_num != 0:
                print("Batch num " + str(batch_num))

            seg_masks = data[2].to('cuda:0')
            if rgb_only:
                ims = data[0].to('cuda:0')
            elif depth_only:
                ims = data[1].to('cuda:0')
            if depth_only or rgb_only:
                with torch.cuda.amp.autocast():  # 16 bit precision = faster and less memory
                    _, loss = model(ims, seg_masks)
            else:
                rgb_ims = data[0].to('cuda:0')
                depth_ims = data[1].to('cuda:0')
                with torch.cuda.amp.autocast():  # 16 bit precision = faster and less memory
                    _, loss = model(rgb_ims, depth_ims, seg_masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss = loss.item()
            epoch_loss += loss
            iteration_losses.append(loss)
        lr_scheduler.step()
        loss_dict["epoch_losses"].extend([epoch_loss])
        loss_dict["iteration_losses"].extend(iteration_losses)
        print("Finished training epoch " + str(epoch_num) + " with loss: " + str(round(epoch_loss, 2)) + "\n")

        # eval check
        num_categories = len(list(eval_dict.keys()))-1 # -1 for the unknown class
        current_mIoU = eval_ss.computeMeanIU(model, test_dl, eval_dict, num_categories,
                                             rgb_dataset=False, depth_dataset=True, rgb_only=rgb_only, depth_only=depth_only, save_fp=None)["mean_IU"]
        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            saveModel(epoch_num, model, optimizer, lr_scheduler, loss_dict, save_path)
    return {"epoch_losses": loss_dict["epoch_losses"], "iteration_losses": loss_dict["iteration_losses"]}


def plot_train_dict(train_dict, model_name, save_fp):
    save_fp_epochs = save_fp + "/" + model_name + "_EpochLosses"
    save_fp_iterations = save_fp + "/" + model_name + "_Iterations"
    epoch_losses = train_dict["epoch_losses"]
    plot_utils.plot(x_vals=range(1, len(epoch_losses) + 1), y_vals=epoch_losses,
                    title=model_name + "\n Epochs vs. Loss", x_label="Epochs", y_label="Loss", save_fp=save_fp_epochs)
    iteration_losses = train_dict["iteration_losses"]
    plot_utils.plot(x_vals=range(1, len(iteration_losses) + 1), y_vals=iteration_losses,
                     title=model_name + "\n Iterations vs. Loss", x_label="Iterations", y_label="Loss",
                     save_fp=save_fp_iterations)


def visualize_ss_masks(model, dataloader, save_fp, with_depth=False, num_ims=50):
    model.eval()
    model.cuda()
    for batch_number, batch in enumerate(dataloader):
        if batch_number == num_ims:
            return

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

        # file paths for saving the predicted and ground truth seg masks
        save_fp_ = save_fp + "im"+str(batch_number+1)

        # un-normalize the original image

        # for grayscale ims
        if images.shape[1] == 1:
            images = torch.clamp(images, min=-1, max=1)
            image_mean_mean = 0.449
            image_std_mean = 0.226
            images *= image_std_mean
            images -= image_mean_mean
            images *= 255.0
            images = images.repeat(1, 3, 1, 1)
            images = images.cpu().numpy().astype(np.uint8)
            original_image = s_utils.channelsFirstToLast(images[0])
        else:
            original_image = s_utils.unNormalizeImage(images[0])

        original_image = s_utils.resize_npy_img(original_image, hw_tuple)

        pred_seg_mask = seg_mask[0].cpu().numpy()
        original_seg_mask = gt_seg_mask.cpu().numpy()
        pred_seg_mask = np.where(original_seg_mask == 0, 0, pred_seg_mask)

        pred_seg_mask_overlay = s_utils.showSegmentationImage(pred_seg_mask, original_image)
        original_seg_mask_overlay = s_utils.showSegmentationImage(original_seg_mask, original_image)
        overall_image = s_utils.concatenateImagesHorizontally([pred_seg_mask_overlay, original_seg_mask_overlay])
        s_utils.saveImage(save_fp_, overall_image)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # training
    num_epochs = 40
    num_classes = 41

    model_name = "dlv3_Depth_Grayscale_nyudv2"
    save_path = os.path.join(saved_models_path, model_name)
    load_path = os.path.join(saved_models_path, model_name)
    train_dl = nyudv2_dl.nyudv2_dl(batch_size=4, num_workers=8, train=True)
    test_dl = nyudv2_dl.nyudv2_dl(batch_size=1, num_workers=0, train=False)

    grayscale_backbone = rn.DeepLabV3PlusBackbone(rn.get_grayscale_rn50_backbone())
    model = dlv3p.DeepLabHeadV3Plus(num_classes=97, backbone=grayscale_backbone) # the pretrained coco model
    pre_trained_model_path = "C:/Users/james/PycharmProjects/CSI5340Project/Training/Trained_Models/dlv3+_ResNet50_grayscale_COCO_train"
    model = loadModel(model, None, None, pre_trained_model_path, model_only=True)
    model.backbone.out_channels = 2048
    model = dlv3.DeepLabv3(num_classes=41, backbone=model.backbone, penalize_zero=False)
    #model.freeze_weights()
    eval_dict = nyudv2_dl.semantic_40_dict
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.009, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8, verbose=False)

    train_dict = train_ss_model(model, optimizer, lr_scheduler, num_epochs, train_dl, test_dl,
                         eval_dict, save_path, depth_only=True, rgb_only=False, load_path=None)
    plot_train_dict(train_dict, model_name, saved_plots_path)

    # testing
    save_fp_miou = saved_eval_results_path+"/"+model_name+"_mIOU"
    save_fp_accuracy = saved_eval_results_path+"/"+model_name+"_pixel_accuracy"
    cat_map = stuff_dict.coco_stuff_dict
    model = loadModel(model, None, None, load_path, save_path=load_path)
    eval_ss.computeMeanIU(model, test_dl, cat_map, num_classes-1, depth_dataset=True, rgb_dataset=False, rgb_only=False, depth_only=True, save_fp=save_fp_miou)
    eval_ss.computeMeanPixelAccuracy(model, test_dl, cat_map, num_classes-1, depth_dataset=True, rgb_dataset=False, rgb_only=False, depth_only=True, save_fp=save_fp_accuracy)
