import torchvision.models.segmentation.deeplabv3 as deepLab
import torch
import torchvision.models.segmentation as SegModels
from torch.nn import functional as F

class DA_Concat_DeepLabv3(torch.nn.Module):
    def __init__(self, rgb_backbone=None, depth_backbone=None, rgb_pretrained=True, depth_pretrained=True,
                 num_classes=40, penalize_zero=True):
        super(DA_Concat_DeepLabv3, self).__init__()
        if rgb_backbone is None:
            self.rgb_backbone = SegModels.deeplabv3_resnet101(pretrained=rgb_pretrained).backbone
        else:
            self.rgb_backbone = rgb_backbone
        if depth_backbone is None:
            self.depth_backbone = SegModels.deeplabv3_resnet101(pretrained=depth_pretrained).backbone
        else:
            self.depth_backbone = depth_backbone

        self.backbone_output_channels = 2048
        self.DeepLab_Head = deepLab.DeepLabHead(self.backbone_output_channels, num_classes)
        self.one_by_one = torch.nn.Conv2d(in_channels=2*self.backbone_output_channels, out_channels=self.backbone_output_channels, kernel_size=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(num_features=self.backbone_output_channels, momentum=0.1, affine=True, track_running_stats=True)

        # for regular semantic segmentation
        if penalize_zero:
            self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        else:
            self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, rgb_ims, depth_ims, seg_masks, hw_tuple=None):
        im_shape = rgb_ims.shape[-2:]
        rgb_output = self.rgb_backbone(rgb_ims)["out"]
        depth_output = self.depth_backbone(depth_ims)["out"]
        output = torch.cat([rgb_output, depth_output], dim=1)
        output = self.one_by_one(output)
        output = self.bn(output)
        output = self.DeepLab_Head(output)
        if self.training:
            predicted_seg_mask = F.interpolate(output, im_shape, mode='bilinear', align_corners=False)
            loss = self.ce_loss(predicted_seg_mask, seg_masks)
            return None, loss
        else:
            predicted_seg_mask = F.interpolate(output, (hw_tuple[0], hw_tuple[1]), mode='bilinear',
                                               align_corners=False)
            predicted_seg_mask = torch.argmax(predicted_seg_mask, dim=1)
            return predicted_seg_mask, None

