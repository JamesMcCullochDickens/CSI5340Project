import torch

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

