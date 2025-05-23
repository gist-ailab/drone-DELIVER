import os
from semseg.models.cmnext_backbone import CMNeXtBackbone

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
import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
from semseg.models.heads import LightHamHead
from semseg.models.heads import UPerHead
from fvcore.nn import flop_count_table, FlopCountAnalysis



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
    

class RetinaNetHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        conv_layers = []
        for _ in range(4):
            conv_layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            conv_layers.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*conv_layers)
        self.bbox_subnet = nn.Sequential(*conv_layers)

        self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)

        # Initialization as in RetinaNet paper
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        logits = []
        bbox_reg = []
        for feat in features:
            cls_feat = self.cls_subnet(feat)
            bbox_feat = self.bbox_subnet(feat)
            logits.append(self.cls_score(cls_feat))
            bbox_reg.append(self.bbox_pred(bbox_feat))
        return logits, bbox_reg
    
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import nms


class CMNeXtRetinaNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone

        in_channels_list = [64, 128, 320, 512]
        out_channels = 256

        self.fpn = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])

        self.extra_fpn_blocks = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        ])

        self.head = RetinaNetHead(out_channels, num_anchors=9, num_classes=num_classes)

        # self.anchor_generator = AnchorGenerator(
        #     sizes=((32,), (64,), (128,), (256,), (512,), (1024,)),  # 6 levels!
        #     aspect_ratios=((0.5, 1.0, 2.0),) * 6
        # )

        self.anchor_generator = AnchorGenerator(
            sizes=((32, 45, 64), (64, 90, 128), (128, 180, 256),
                (256, 360, 512), (512, 724, 1024), (1024, 1448, 2048)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 6   # 3 ratio × 3 scale = 9
        )  
        self.num_classes = num_classes

    def forward(self, images, targets=None, score_thresh=0.05, nms_thresh=0.5):
        rgb_images = images[0]
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in rgb_images]
        rgb_image_list = ImageList(rgb_images, image_sizes)

        features = self.backbone(images)
        feats = [self.fpn[i](features[str(i)]) for i in range(4)]

        last_feat = feats[-1]
        for block in self.extra_fpn_blocks:
            last_feat = block(last_feat)
            feats.append(last_feat)

        logits, bbox_reg = self.head(feats)
        anchors_list = self.anchor_generator(rgb_image_list, feats)  # len = batch_size
        anchors = torch.stack(anchors_list, dim=0)

        if self.training:
            return self.compute_loss(logits, bbox_reg, anchors, targets)
        else:
            return self.inference(logits, bbox_reg, anchors, score_thresh, nms_thresh)

    def inference(self, logits, bbox_reg, anchors, score_thresh, nms_thresh):
        results = []
        cls_preds, box_preds = [], []

        for cls_out, box_out in zip(logits, bbox_reg):
            B, _, H, W = cls_out.shape
            cls_out = cls_out.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            box_out = box_out.permute(0, 2, 3, 1).reshape(B, -1, 4)
            cls_preds.append(cls_out)
            box_preds.append(box_out)

        cls_preds = torch.cat(cls_preds, dim=1)  # [B, N, C]
        box_preds = torch.cat(box_preds, dim=1)  # [B, N, 4]
        # anchors = torch.cat(anchors, dim=0).unsqueeze(0).expand(cls_preds.size(0), -1, 4)
        # anchors = anchors.unsqueeze(0).expand(cls_preds.size(0), -1, 4)

        for b in range(cls_preds.size(0)):
            scores = cls_preds[b].sigmoid()
            boxes = self.decode_boxes(anchors[b], box_preds[b])  # [N, 4]

            # Top-k filtering before NMS (optional)
            max_scores, labels = scores.max(dim=1)
            keep = max_scores > score_thresh

            boxes = boxes[keep]
            scores = max_scores[keep]
            labels = labels[keep]

            if boxes.numel() == 0:
                results.append({'boxes': torch.empty((0, 4)), 'scores': torch.empty((0,)), 'labels': torch.empty((0,), dtype=torch.long)})
                continue

            keep_idx = nms(boxes, scores, nms_thresh)
            results.append({
                'boxes': boxes[keep_idx],
                'scores': scores[keep_idx],
                'labels': labels[keep_idx]
            })
        return results

    def decode_boxes(self, anchors, deltas):
        # anchors: [N, 4] (x1, y1, x2, y2)
        # deltas: [N, 4] (dx, dy, dw, dh)

        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx, dy, dw, dh = deltas.unbind(-1)

        # Clip the deltas to avoid large transformations
        dw = torch.clamp(dw, min=-1.0, max=1.0)
        dh = torch.clamp(dh, min=-1.0, max=1.0)

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        # Clip to avoid generating extremely large boxes
        pred_w = torch.clamp(pred_w, min=0, max=1000)  # Max width (you can adjust this value)
        pred_h = torch.clamp(pred_h, min=0, max=1000)  # Max height (you can adjust this value)

        pred_boxes = torch.stack([
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ], dim=-1)

        pred_boxes = torch.nan_to_num(pred_boxes, nan=0.0, posinf=0.0, neginf=0.0)
        return pred_boxes
    
    def compute_loss(self, cls_outputs, box_outputs, anchors, targets, alpha: float = 0.25, gamma: float = 2.0):
        device = cls_outputs[0].device
        batch_size = len(targets)

        # flatten pyramid outputs
        cls_preds_all, box_preds_all = [], []
        for cls_out, box_out in zip(cls_outputs, box_outputs):
            B, A_C, H, W = cls_out.shape
            cls_preds_all.append(cls_out.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))
            box_preds_all.append(box_out.permute(0, 2, 3, 1).reshape(B, -1, 4))

        cls_preds_cat = torch.cat(cls_preds_all, dim=1)
        box_preds_cat = torch.cat(box_preds_all, dim=1)
        anchors_cat = anchors  # [B, N, 4]

        # ---> patched: keep losses as tensors even when 0
        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            gt_boxes = targets[b]['boxes']
            gt_labels = targets[b]['labels']
            num_anchors = anchors_cat.shape[1]
            if gt_boxes.numel() == 0:
                cls_target = torch.zeros((num_anchors, self.num_classes), device=device)
                total_cls_loss += sigmoid_focal_loss(cls_preds_cat[b], cls_target,
                                                     reduction='sum', alpha=alpha, gamma=gamma)
                continue

            ious = box_iou(anchors_cat[b], gt_boxes)
            max_ious, matched_idx = ious.max(dim=1)
            pos_mask = max_ious >= 0.5
            neg_mask = max_ious < 0.4
            ignore_mask = ~(pos_mask | neg_mask)

            cls_target = torch.zeros((num_anchors, self.num_classes), device=device)
            cls_target[pos_mask, gt_labels[matched_idx[pos_mask]]] = 1.0

            cls_preds_b = cls_preds_cat[b].clone()
            cls_preds_b[ignore_mask] *= 0  # zero-out ignored

            total_cls_loss += sigmoid_focal_loss(cls_preds_b, cls_target,
                                                 reduction='sum', alpha=alpha, gamma=gamma)

            if pos_mask.any():
                total_box_loss += F.smooth_l1_loss(
                    box_preds_cat[b][pos_mask],
                    gt_boxes[matched_idx[pos_mask]],
                    reduction='sum'
                )

        normalizer = max(sum(len(t['labels']) for t in targets), 1)
        return {
            'loss_cls': total_cls_loss / normalizer,
            'loss_bbox': total_box_loss / normalizer,
        }