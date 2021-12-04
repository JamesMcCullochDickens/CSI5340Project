import torch

class Rotation_Predictor(torch.nn.Module):
    def __init__(self, backbone):
        super(Rotation_Predictor, self).__init__()
        assert backbone is not None
        self.backbone = backbone
        self.fc = torch.nn.Linear(2048, 4)
        self.ce_loss = torch.nn.CrossEntropyLoss()
    def forward(self, ims, gt):
        x = self.backbone(ims)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.training:
            loss = self.ce_loss(x, gt)
        else:
            loss = None
        pred = x
        return loss, pred
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False






