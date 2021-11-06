import torch
import pprint

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
    loss_dict = {}
    model.cuda()
    for epoch_num in range(1, num_epochs):
        epoch_loss = 0.0
        for batch_num, (ims, seg_masks) in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            if batch_num % 250 == 0 and batch_num != 0:
                print("Batch num" + str(batch_num))
            ims = ims.cuda()
            seg_masks = seg_masks.cuda()
            with torch.cuda.amp.autocast():  # 16 bit precision = faster and less memory
                _, loss = model(ims, seg_masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        lr_scheduler.step()
        loss_dict[epoch_num] = epoch_loss
        print("\nFinished training epoch " + str(epoch_num) + " with loss: " + str(round(epoch_loss, 2)))
        saveModel(epoch_num, model, optimizer, lr_scheduler, save_path)
    print("Training complete")
    pprint(loss_dict)
