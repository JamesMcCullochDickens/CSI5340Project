import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import Backbones.ResNet as ResNet

class BYOL(nn.Module):
    def __init__(self, input_dim, backbone=None, projection_dim=256, hidden_dim=4096, target_decay=0.99):
        super(BYOL, self).__init__()
        self.input_dim = input_dim
        if backbone is not None:
            self.online_backbone = backbone
        else: # a default backbone
            self.online_backbone = ResNet.remove_head(ResNet.resnet50(pretrained=False, dilation_vals=[False, True, True]))
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.target_decay = target_decay
        self.online_projector = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                 nn.BatchNorm1d(self.hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, projection_dim))
        self.target_projector = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                       nn.BatchNorm1d(self.hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_dim, projection_dim))
        self.predictor = nn.Sequential(nn.Linear(self.projection_dim, self.hidden_dim),
                                 nn.BatchNorm1d(self.hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, projection_dim))
        self.target_backbone = copy.deepcopy(self.online_backbone)

        for param in self.target_backbone:
            param.requires_grad = False

        for param in self.target_projector:
            param.requires_grad = False

    def forward(self, view1, view2):
        v1 = self.online_backbone(view1)
        v1 = torch.flatten(v1, 1)
        v1 = self.online_projector(v1)
        v1 = self.predictor(v1)

        v2 = self.online_backbone(view2)
        v2 = torch.flatten(v2, 1)
        v2 = self.online_projector(v2)
        v2 = self.predictor(v2)

        with torch.no_grad():
            view1_target = self.target_backbone(view1)
            view1_target = torch.flatten(view1_target, 1)
            view1_target = self.target_projector(view1_target)
            view2_target = self.target_backbone(view2)
            view2_target = torch.flatten(view2_target, 1)
            view2_target = self.target_projector(view2_target)
            view1_target.detach_()
            view2_target.detach_()

        loss = self.loss(v1, view2_target, v2, view1_target)
        return loss

    def update_moving_average(self):
        for online_param, target_param in \
                zip(self.online_backbone.parameters(), self.target_backbone.parameters()):
            online_param_ = online_param.data
            target_param_ = target_param.data
            new_param = target_param_ * self.target_decay + (1 - self.target_decay) * online_param_
            target_param.data = new_param

        for online_param, target_param in \
                zip(self.online_projector.parameters(), self.target_projector.parameters()):
            online_param_ = online_param.data
            target_param_ = target_param.data
            new_param = target_param_ * self.target_decay + (1 - self.target_decay) * online_param_
            target_param.data = new_param

    def loss(self, v1_online_pred, v2_target_proj, v2_online_pred, v1_target_proj):
        view1_ = F.normalize(v1_online_pred, p=2, dim=-1)
        view2_ = F.normalize(v2_target_proj, p=2, dim=-1)
        loss1 = 2 - 2 * (view1_ * view2_).sum(dim=-1)
        view1 = F.normalize(v2_online_pred, p=2, dim=-1)
        view2 = F.normalize(v1_target_proj, p=2, dim=-1)
        loss2 = 2 - 2 * (view1 * view2).sum(dim=-1)
        overall_loss = loss1+loss2
        return overall_loss.mean()


"""
# debugging BYOL
rn_50 = ResNet.remove_classification(ResNet.resnet50(pretrained=True, dilation_vals=[False, True, True]))
batch_1 = torch.rand(6, 3, 600, 600).cuda()
batch_2 = torch.rand(6, 3, 600, 600).cuda()
# 2048 for ResNet 50 and 101, 512 for ResNet 18
model = BYOL(input_dim=2048, backbone=rn_50, projection_dim=256, hidden_dim=4096).cuda()
optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
output = model(batch_1, batch_2)
output.backward()
optim.step()
model.update_moving_average()
"""