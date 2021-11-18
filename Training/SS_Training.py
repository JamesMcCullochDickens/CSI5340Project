import torch
import pprint
import os
from pathlib import Path

import CacheDictUtils
import DataLoaders.MS_COCO_dl as ms_coco_dl
saved_models_path = os.path.join(os.getcwd(), "Trained_Models")
saved_plots_path = os.path.join(os.getcwd(), "Saved_Plots")
saved_eval_results_path = os.path.join(Path(os.getcwd()).parent, "Evaluation/Eval_Results")
import Models.DeepLabV3Plus as dlv3
import Training.Optimizers as opt_utils
import PlotUtils.PlotUtils as plot_utils
import Evaluation.Evaluate_SS as eval_ss
import COCOStuffDict as stuff_dict

def saveModel(epoch, model, optimizer, lr_scheduler, save_path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict':lr_scheduler.state_dict()
    }, save_path)

def loadModel(model, optimizer, lr_scheduler, save_path):
    try:
        checkpoint = torch.load(save_path)
    except:
        print("No model file exists")
    epoch = checkpoint(['epoch'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    return epoch, model, optimizer, lr_scheduler

def train_rgb_ss_model(model, optimizer, lr_scheduler, num_epochs, dataloader, save_path):
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    epoch_losses = []
    iteration_losses = []
    model.train()
    model.cuda()
    for epoch_num in range(1, num_epochs+1):
        epoch_loss = 0.0
        for batch_num, data in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            if batch_num % 500 == 0 and batch_num != 0:
                print("Batch num" + str(batch_num))
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
        epoch_losses.append(epoch_loss)
        print("\nFinished training epoch " + str(epoch_num) + " with loss: " + str(round(epoch_loss, 2)))
        saveModel(epoch_num, model, optimizer, lr_scheduler, save_path)
    return {"epoch_losses": epoch_losses, "iteration_losses": iteration_losses}

def plot_train_dict(train_dict, model_name, save_fp):
    save_fp_epochs = save_fp + "/_EpochLosses"
    save_fp_iterations = save_fp + "/_Iterations"
    epoch_losses = train_dict["epoch_losses"]
    plot_utils.plot(x_vals=range(1, len(epoch_losses) + 1), y_vals=epoch_losses,
                    title=model_name + "\n Epochs vs. Loss", x_label="Epochs", y_label="Loss", save_fp=save_fp_epochs)
    iteration_losses = train_dict["iteration_losses"]
    plot_utils.plot(x_vals=range(1, len(iteration_losses) + 1), y_vals=iteration_losses,
                     title=model_name + "\n Iterations vs. Loss", x_label="Iterations", y_label="Loss",
                     save_fp=save_fp_iterations)


if __name__ == "__main__":
    # training
    num_epochs = 15
    num_classes = 97
    model_name = "dlv3+_ResNet101_COCO_train"
    save_path = os.path.join(saved_models_path, model_name)
    dataloader = ms_coco_dl.get_mscoco_stuff_train_it(batch_size=6, num_workers=8)
    model = dlv3.DeepLabHeadV3Plus(num_classes=num_classes)
    optimizer = opt_utils.getMaskRCNNOptimizer(model, learning_rate=0.003)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8, verbose=True)
    train_dict = train_rgb_ss_model(model, optimizer, lr_scheduler, num_epochs, dataloader, save_path)
    plot_train_dict(train_dict, model_name, saved_plots_path)

    # testing
    test_dl = ms_coco_dl.get_mscoco_stuff_val_it(batch_size=1, num_workers=1)
    save_fp_miou = saved_eval_results_path+"/"+model_name+"_mIOU"
    save_fp_accuracy = saved_eval_results_path+"/"+model_name+"_pixel_accuracy"
    cat_map = stuff_dict.coco_stuff_dict
    eval_ss.computeMeanIU(model, test_dl, cat_map, num_classes-1, with_depth=False, save_fp=save_fp_miou)
    eval_ss.computeMeanPixelAccuracy(model, test_dl, cat_map, num_classes-1, with_depth=False, save_fp=save_fp_accuracy)