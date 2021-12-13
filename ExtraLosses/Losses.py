import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma=1, alpha=0.25, with_ignore_index=False):
        nn.Module.__init__(self)
        self.gamma = gamma
        self.alpha=alpha
        if with_ignore_index:
            self.loss = torch.nn.NLLLoss(ignore_index=0)
        else:
            self.loss = torch.nn.NLLLoss()

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return self.alpha*F.nll_loss(((1 - prob) ** self.gamma) * log_prob, target_tensor)
