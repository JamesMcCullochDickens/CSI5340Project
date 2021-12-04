import torch
import numpy as np
from pathlib import Path
import os
import PlotUtils.PlotUtils as plot_utils
import DataLoaders.Unlabeled_dl as dl
import Backbones.ResNet as rn
import logging
import Models.Rotation_Predictor as rp
import Visualization.VisualizeConvFilters as vcf
logging.captureWarnings(True)

saved_models_path = os.path.join(os.getcwd(), "Trained_Models")
saved_plots_path = os.path.join(os.getcwd(), "Saved_Plots")
saved_eval_results_path = os.path.join(Path(os.getcwd()).parent, "Evaluation/Eval_Results")

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




def train(model, optimizer, lr_scheduler, dataloader, num_epochs, save_path, load_path):
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
    for epoch_num in range(epoch_min, num_epochs + 1):
        epoch_loss = 0.0
        iteration_losses = []
        for batch_num, data in enumerate(dataloader):
            if batch_num % 100 == 0 and batch_num != 0:
                print("Batch num " + str(batch_num))
            ims = data[0].to('cuda:0', non_blocking=True)
            gt_labels = data[1].to('cuda:0', non_blocking=True)
            with torch.cuda.amp.autocast():  # mixed precision = faster and less memory
                loss, _ = model(ims, gt_labels)
            loss.backward()
            loss = loss.item()
            epoch_loss += loss
            iteration_losses.append(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
        loss_dict["epoch_losses"].extend([epoch_loss])
        loss_dict["iteration_losses"].extend(iteration_losses)
        print("\nFinished training epoch " + str(epoch_num) + " with loss: " + str(round(epoch_loss, 2)) + "\n")
        saveModel(epoch_num, model, optimizer, lr_scheduler, loss_dict, save_path)
    return {"epoch_losses": loss_dict["epoch_losses"], "iteration_losses": loss_dict["iteration_losses"]}


# dl must have batch size 1
def eval_concise(model, dl):
    model.eval()
    model.cuda()
    correct = 0.0
    total = 0.0
    for index, data in enumerate(dl):
        ims = data[0].cuda()
        rotation_labels = data[1].cuda()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                _, pred = model(ims, rotation_labels)
        pred_top1 = torch.topk(pred, dim=1, k=1)[1]
        for index, label in enumerate(rotation_labels):
            total += 1
            label = int(label)
            pred_top_1 = int(pred_top1[index].cpu().item())
            if pred_top_1 == label:
                correct +=1
    if total == 0:
        print("No examples ??")
        return 0.0
    else:
        accuracy = round((correct/total), 3)
        print("The accuracy is " + str(accuracy))
        return accuracy


if __name__ == "__main__":
    num_epochs = 25
    model_name = "Rotation_Exp4"
    save_path = os.path.join(saved_models_path, model_name)
    load_path = os.path.join(saved_models_path, model_name)
    dataloader = dl.get_unlabeled_rotation_dl(batch_size=16, num_workers=0, depth_only=True)
    rn_50 = rn.get_grayscale_rn50_backbone(pre_trained=False, with_pooling=True, dilation_vals=[False, True, True])
    model = rp.Rotation_Predictor(backbone=rn_50)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9, weight_decay=0.00005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
    train(model, optimizer, lr_scheduler, dataloader, num_epochs, save_path, load_path=None)

    # testing
    model = loadModel(model, None, None, save_path, model_only=True)
    test_dl = dl.get_test_rotation_dl(batch_size=16, num_workers=0, depth_only=True)
    vcf.visualize_conv_filters_backbone(model)
    accuracy = eval_concise(model, test_dl)

