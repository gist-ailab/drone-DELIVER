
import torch
import torch.nn as nn
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign, FeaturePyramidNetwork
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from semseg.models.base import BaseModel
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou, sigmoid_focal_loss

class CMNeXtBackbone(BaseModel):
    def __init__(self, backbone: str = 'CMNeXt-B2', modals: list = ['image', 'depth', 'event', 'lidar']):
        super().__init__(backbone, num_classes=None, modals=modals)
        self.out_channels = 256 if 'B0' in backbone or 'B1' in backbone else 512

    def forward(self, x):
        features = self.backbone(x)
        # Select 4 features from MiT stages (C1-C4)
        return {
            '0': features[0],  # 1/4  : 64
            '1': features[1],  # 1/8  : 128
            '2': features[2],  # 1/16 : 320
            '3': features[3],  # 1/32 : 512
        }
    