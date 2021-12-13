import torch
import torchvision.models.segmentation as SegModels
import torchvision.models.segmentation.deeplabv3 as deepLab
import torch.nn.functional as F

class DeepLabv3(torch.nn.Module):
    def __init__(self, backbone=None, pretrained=True, num_classes=38, penalize_zero=False, ce_loss_weights=None):
        super(DeepLabv3, self).__init__()
        if backbone is None:
            resnet_pretrained = SegModels.deeplabv3_resnet101(pretrained=pretrained)
            self.backbone = resnet_pretrained.backbone
            self.backbone_output_channels = 2048
        else:
            self.backbone_output_channels = backbone.out_channels
            self.backbone = backbone

        self.DeepLab_Head = deepLab.DeepLabHead(self.backbone_output_channels, num_classes)
        if penalize_zero:
            self.loss = torch.nn.CrossEntropyLoss(weight=ce_loss_weights)
        else:
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=0, weight=ce_loss_weights)
    def freeze_weights(self):
        for param in self.backbone.parameters():
            param.requires_grad=False

    def forward(self, rgb_ims, seg_masks, size_tuple=None):
        im_shape = rgb_ims.shape[-2:]
        output = self.backbone(rgb_ims)
        output = self.DeepLab_Head(output["out"])
        if self.training:
            output = F.interpolate(output, im_shape, mode='bilinear', align_corners=False)
            loss = self.loss(output, seg_masks)
            return None, loss
        else:
            predicted_seg_mask = F.interpolate(output, (size_tuple[0], size_tuple[1]), mode='bilinear', align_corners=False)
            predicted_seg_mask = torch.argmax(predicted_seg_mask, dim=1)
            return predicted_seg_mask, None