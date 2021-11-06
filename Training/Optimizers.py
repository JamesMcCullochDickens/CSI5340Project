import torch

def getFasterRCNNOptimzer(model, learning_rate=0.001):
    #without filter
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=0.0005)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    return optimizer

def getMaskRCNNOptimizer(model, learning_rate=0.02):
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    return optimizer