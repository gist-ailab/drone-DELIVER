import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
from semseg.models.heads import LightHamHead
from semseg.models.heads import UPerHead
from fvcore.nn import flop_count_table, FlopCountAnalysis

import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class CMNeXtBackbone(BaseModel):
    def __init__(self, backbone: str = 'CMNeXt-B2', modals: list = ['img', 'depth', 'event', 'lidar']):
        super().__init__(backbone, num_classes=None, modals=modals)
        self.out_channels = 256 if 'B0' in backbone or 'B1' in backbone else 512

    def forward(self, x):
        features = self.backbone(x)
        # Select 4 features from MiT stages (C1-C4)
        return {
            '0': features[0],  # 1/4
            '1': features[1],  # 1/8
            '2': features[2],  # 1/16
            '3': features[3],  # 1/32
        }
    


class CMNeXtFasterRCNN(nn.Module):
    def __init__(self, backbone_name='CMNeXt-B2', num_classes=25, modals=['img', 'depth', 'event', 'lidar']):
        super().__init__()
        self.backbone = CMNeXtBackbone(backbone_name, modals)

        # Set the out_channels (e.g., 512) for the FPN
        out_channels = self.backbone.out_channels

        # Define anchor generator for RPN
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # Define ROI Align for detection head
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        self.model = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)


if __name__ == '__main__':
    modals = ['img', 'depth', 'event', 'lidar']
    model = CMNeXtFasterRCNN('CMNeXt-B2', 25, modals)
    model.init_pretrained('checkpoints/pretrained/segformer/mit_b2.pth')
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    y = model(x)
    print(y.shape)
