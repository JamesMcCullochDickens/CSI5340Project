import torch

def getFasterRCNNOptimzer(model, learning_rate=0.001):
    #without filter
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=0.0005)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    return optimizer

def getMaskRCNNOptimizer(model, learning_rate=0.02):
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    return optimizer

def getDepthPredictionOptimizer(model, learning_rate=0.001):
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9,
                                weight_decay=0.0001)
    return optimizer

# this caused a memory error for some reason, watchout!
def changeOptimizerLearningRate(optimizer, new_rate):
    for g in optimizer.param_groups:
        g['lr'] = new_rate
    return optimizer


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
