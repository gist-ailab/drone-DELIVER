import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# import torch
# from torch import Tensor
# from torch.nn import functional as F
# from semseg.models.base import BaseModel
# from semseg.models.heads import SegFormerHead
# from semseg.models.heads import LightHamHead
# from semseg.models.heads import UPerHead
# from fvcore.nn import flop_count_table, FlopCountAnalysis

# import torch.nn as nn
# from torchvision.models.detection import FasterRCNN, FastRCNNPredictor, TwoMLPHead
# from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
# from torchvision.models.detection.backbone_utils import BackboneWithFPN
# from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.models.detection.roi_heads import RoIHeads
# from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
# from torchvision.models.detection.transform import GeneralizedRCNNTransform
# from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

# from typing import Dict, List, Optional, Tuple, Union
# from collections import OrderedDict
# import warnings
# from torchvision.models.detection.image_list import ImageList


# class CMNeXtBackbone(BaseModel):
#     def __init__(self, backbone: str = 'CMNeXt-B2', modals: list = ['image', 'depth', 'event', 'lidar']):
#         super().__init__(backbone, num_classes=None, modals=modals)
#         self.out_channels = 256 if 'B0' in backbone or 'B1' in backbone else 512

#     def forward(self, x):
#         features = self.backbone(x)
#         # Select 4 features from MiT stages (C1-C4)
#         return {
#             '0': features[0],  # 1/4  : 64
#             '1': features[1],  # 1/8  : 128
#             '2': features[2],  # 1/16 : 320
#             '3': features[3],  # 1/32 : 512
#         }
    

# def _default_anchorgen():
#     anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
#     aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
#     return AnchorGenerator(anchor_sizes, aspect_ratios)

# class CMNeXtFasterRCNN(GeneralizedRCNN):
#     def __init__(
#         self,
#         backbone_name='CMNeXt-B2',
#         num_classes=2,
#         modals=['img', 'depth', 'event', 'lidar'],
#         rpn_anchor_generator=None,
#         rpn_head=None,
#         rpn_pre_nms_top_n_train=2000,
#         rpn_pre_nms_top_n_test=1000,
#         rpn_post_nms_top_n_train=2000,
#         rpn_post_nms_top_n_test=1000,
#         rpn_nms_thresh=0.7,
#         rpn_fg_iou_thresh=0.7,
#         rpn_bg_iou_thresh=0.3,
#         rpn_batch_size_per_image=256,
#         rpn_positive_fraction=0.5,
#         rpn_score_thresh=0.0,
#         box_roi_pool=None,
#         box_head=None,
#         box_predictor=None,
#         box_score_thresh=0.05,
#         box_nms_thresh=0.5,
#         box_detections_per_img=100,
#         box_fg_iou_thresh=0.5,
#         box_bg_iou_thresh=0.5,
#         box_batch_size_per_image=512,
#         box_positive_fraction=0.25,
#         bbox_reg_weights=None,
#         transform=None,
#         min_size=800,
#         max_size=1333,
#         image_mean=None,
#         image_std=None,
#         target_format='xywh',
#         **kwargs,
#     ):
#         # Step 1: Initialize CMNeXtBackbone
#         backbone_model = CMNeXtBackbone(backbone_name, modals)

#         if 'B0' in backbone_name:
#             in_channels_list = [32, 64, 160, 256]
#         elif 'B1' in backbone_name:
#             in_channels_list = [64, 128, 320, 512]
#         else:
#             in_channels_list = [64, 128, 320, 512]

#         fpn_out_channels = 512

#         class BackboneWithCustomFPN(nn.Module):
#             def __init__(self, body, in_channels_list, out_channels):
#                 super().__init__()
#                 self.body = body
#                 self.fpn_projections = nn.ModuleDict({
#                     f'{i}': nn.Conv2d(in_channels, out_channels, kernel_size=1)
#                     for i, in_channels in enumerate(in_channels_list)
#                 })
#                 self.out_channels = out_channels

#             def forward(self, x):
#                 features = self.body(x)
#                 return {
#                     str(k): self.fpn_projections[str(k)](v)
#                     for k, v in features.items()
#                 }
#         backbone = BackboneWithCustomFPN(backbone_model, in_channels_list, fpn_out_channels)

#         if rpn_anchor_generator is None:
#             anchor_sizes = ((32,), (64,), (128,), (256,))
#             aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
#             rpn_anchor_generator = AnchorGenerator(
#                 sizes=anchor_sizes,
#                 aspect_ratios=aspect_ratios
#             )

#         if rpn_head is None:
#             rpn_head = RPNHead(
#                 fpn_out_channels,
#                 rpn_anchor_generator.num_anchors_per_location()[0]
#             )

#         rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
#         rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

#         rpn = RegionProposalNetwork(
#             rpn_anchor_generator,
#             rpn_head,
#             rpn_fg_iou_thresh,
#             rpn_bg_iou_thresh,
#             rpn_batch_size_per_image,
#             rpn_positive_fraction,
#             rpn_pre_nms_top_n,
#             rpn_post_nms_top_n,
#             rpn_nms_thresh,
#             score_thresh=rpn_score_thresh,
#         )

#         if box_roi_pool is None:
#             box_roi_pool = MultiScaleRoIAlign(
#                 featmap_names=['0', '1', '2', '3'],
#                 output_size=7,
#                 sampling_ratio=2
#             )

#         if box_head is None:
#             resolution = box_roi_pool.output_size[0]
#             representation_size = 1024
#             box_head = TwoMLPHead(
#                 fpn_out_channels * resolution**2,
#                 representation_size
#             )

#         if box_predictor is None:
#             representation_size = 1024
#             box_predictor = FastRCNNPredictor(representation_size, num_classes)

#         roi_heads = RoIHeads(
#             box_roi_pool,
#             box_head,
#             box_predictor,
#             box_fg_iou_thresh,
#             box_bg_iou_thresh,
#             box_batch_size_per_image,
#             box_positive_fraction,
#             bbox_reg_weights,
#             box_score_thresh,
#             box_nms_thresh,
#             box_detections_per_img,
#         )

#         if transform is None:
#             if image_mean is None:
#                 image_mean = [0.485, 0.456, 0.406]
#             if image_std is None:
#                 image_std = [0.229, 0.224, 0.225]

#             transform = GeneralizedRCNNTransform(
#                 min_size=min_size,
#                 max_size=max_size,
#                 image_mean=image_mean,
#                 image_std=image_std
#             )

#         super().__init__(backbone, rpn, roi_heads, transform)
#         self.modals = modals
#         self.target_format = target_format
#         self._has_warned = False

#     def _filter_empty_targets(self, targets):
#         if targets is None:
#             return None
#         filtered_targets = []
#         for target in targets:
#             target_copy = {
#                 k: v.clone() if isinstance(v, torch.Tensor) else v
#                 for k, v in target.items()
#             }
#             if 'boxes' in target_copy and target_copy['boxes'].numel() == 0:
#                 device = target_copy['boxes'].device
#                 dummy_box = torch.tensor([[-10.0, -10.0, -9.0, -9.0]], device=device)
#                 target_copy['boxes'] = dummy_box
#                 if 'labels' in target_copy and target_copy['labels'].numel() == 0:
#                     target_copy['labels'] = torch.tensor([0], device=device, dtype=torch.int64)
#             filtered_targets.append(target_copy)
#         return filtered_targets
    
#     def _convert_xywh_to_xyxy(self, boxes):
#         """
#         Convert boxes from [x, y, width, height] format to [x1, y1, x2, y2] format
        
#         Args:
#             boxes (Tensor): Boxes in [x, y, width, height] format
            
#         Returns:
#             Tensor: Boxes in [x1, y1, x2, y2] format
#         """
#         if boxes.numel() == 0:
#             return boxes  # 빈 텐서는 변환 없이 반환
            
#         x, y, w, h = boxes.unbind(1)
#         x1 = x
#         y1 = y
#         x2 = x + w
#         y2 = y + h
#         return torch.stack((x1, y1, x2, y2), dim=1)
    
#     def _convert_xyxy_to_xywh(self, boxes):
#         """
#         Convert boxes from [x1, y1, x2, y2] format to [x, y, width, height] format
        
#         Args:
#             boxes (Tensor): Boxes in [x1, y1, x2, y2] format
            
#         Returns:
#             Tensor: Boxes in [x, y, width, height] format
#         """
#         if boxes.numel() == 0:
#             return boxes  # 빈 텐서는 변환 없이 반환
            
#         x1, y1, x2, y2 = boxes.unbind(1)
#         x = x1
#         y = y1
#         w = x2 - x1
#         h = y2 - y1
#         return torch.stack((x, y, w, h), dim=1)

#     def forward(self, images, targets=None):
#         """
#         Args:
#             images (list): images to be processed, expected to be a list of tensors for different modalities
#             targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

#         Returns:
#             result (list[BoxList] or dict[Tensor]): the output from the model.
#                 During training, it returns a dict[Tensor] which contains the losses.
#                 During testing, it returns list[BoxList] contains additional fields
#                 like `scores`, `labels` and `mask` (for Mask R-CNN models).
#         """
#         if self.training and targets is not None:
#             target_shapes = []
#             for t in targets:
#                 box_shape = t['boxes'].shape if 'boxes' in t else None
#                 label_shape = t['labels'].shape if 'labels' in t else None
#                 target_shapes.append((box_shape, label_shape))
#         if targets is not None and self.target_format == 'xywh':
#             targets_converted = []
#             for target in targets:
#                 target_copy = {
#                     k: v.clone() if isinstance(v, torch.Tensor) else v
#                     for k, v in target.items()
#                 }
#                 if 'boxes' in target_copy and target_copy['boxes'].numel() > 0:
#                     target_copy["boxes"] = self._convert_xywh_to_xyxy(target_copy["boxes"])
#                 targets_converted.append(target_copy)
#             targets = targets_converted

#         if targets is not None:
#             for target_idx, target in enumerate(targets):
#                 if 'boxes' not in target or target['boxes'].numel() == 0:
#                     continue
#                 boxes = target["boxes"]
#                 if isinstance(boxes, torch.Tensor):
#                     if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
#                         raise ValueError(
#                             f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}."
#                         )
#                 else:
#                     raise ValueError(
#                         f"Expected target boxes to be of type Tensor, got {type(boxes)}."
#                     )

#                 degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
#                 if degenerate_boxes.any():
#                     print(f"Warning: Found degenerate boxes in target {target_idx}. Fixing...")
#                     fixed_boxes = boxes.clone()
#                     for b_idx in torch.where(degenerate_boxes.any(dim=1))[0]:
#                         fixed_boxes[b_idx, 2:] = fixed_boxes[b_idx, :2] + 1.0
#                     target["boxes"] = fixed_boxes

#         if isinstance(images, list) and len(images) > 0:
#             rgb_images = images[0]
#         else:
#             rgb_images = images

#         # images_transformed, targets_transformed = self.transform(rgb_images, targets)
        
#         features = self.backbone(images)
#         proposals, proposal_losses = self.rpn(images_transformed, features, targets_transformed)
#         detections, detector_losses = self.roi_heads(features, proposals, images_transformed.image_sizes, targets_transformed)

#         if not self.training:
#             detections = self.transform.postprocess(
#                 detections,
#                 images_transformed.image_sizes,
#                 [(img.shape[1], img.shape[2]) for img in rgb_images]
#             )

#             if self.target_format == 'xywh':
#                 for detection in detections:
#                     if detection["boxes"].numel() > 0:
#                         detection["boxes"] = self._convert_xyxy_to_xywh(detection["boxes"])

#         losses = {}
#         losses.update(detector_losses)
#         losses.update(proposal_losses)

#         if self.training:
#             return losses
#         return detections



# if __name__ == '__main__':
#     modals = ['img', 'depth', 'event', 'lidar']
#     model = CMNeXtFasterRCNN('CMNeXt-B2', 25, modals)
#     model.init_pretrained('checkpoints/pretrained/segformer/mit_b2.pth')
#     x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
#     y = model(x)
#     print(y.shape)





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
    

class BackboneWithFPN(nn.Module):
    """
    CMNeXtBackbone 출력을 1×1 conv 로 채널 맞춘 뒤
    torchvision.ops.FeaturePyramidNetwork 로 top-down fusion.
    """
    def __init__(self,
                 body: CMNeXtBackbone,
                 in_channels_list=(64, 128, 320, 512),
                 out_channels=256):
        super().__init__()
        self.body = body
        # 1×1 채널 정합
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c_in, out_channels, 1) for c_in in in_channels_list
        ])
        # top-down + 3×3 smooth
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[out_channels]*4,
            out_channels=out_channels,
            extra_blocks=None
        )
        self.out_channels = out_channels

    def forward(self, x_list):
        # -------- ① list(B,C,H,W) 중 RGB(first)만 백본에 입력 --------
        # rgb = x_list[0] if isinstance(x_list, list) else x_list
        feats_mit = self.body(x_list)                    # dict: '0'~'3'
        # dict → Ordered (stage idx ascending)
        feats = [feats_mit[str(i)] for i in range(4)]
        feats = [lat(f) for lat, f in zip(self.lateral_convs, feats)]
        # FeaturePyramidNetwork expects OrderedDict
        fpn_in = {str(i): f for i, f in enumerate(feats)}
        fpn_out = self.fpn(fpn_in)                    # keys 그대로 '0'...'3'
        return fpn_out
    

class CMNeXtFasterRCNN(GeneralizedRCNN):
    def __init__(self,
                 backbone_name='CMNeXt-B2',
                 num_classes=2,
                 modals=('image', 'depth', 'event', 'lidar'),
                 img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225),
                 min_size=800,
                 max_size=1333):
        # 1) backbone + FPN
        body = CMNeXtBackbone(backbone_name, modals)
        in_ch = [64, 128, 320, 512] if 'B0' not in backbone_name else [32, 64, 160, 256]
        backbone = BackboneWithFPN(body, in_ch, out_channels=256)

        # 2) RPN
        # anchor_gen = AnchorGenerator(
        #     sizes=((32, 64), (64, 128), (128, 256), (256, 512), (512,)),
        #     aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        # )
        anchor_gen = AnchorGenerator(
            sizes=((32, 64), (64, 128), (128, 256), (256, 512)),  # 각 피쳐맵 크기 별로 앵커 크기를 설정
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,  # Aspect ratio는 동일하게 설정
        )
        rpn_head = RPNHead(backbone.out_channels,
                           anchor_gen.num_anchors_per_location()[0])
        rpn = RegionProposalNetwork(
            anchor_gen, rpn_head,
            fg_iou_thresh=0.7, bg_iou_thresh=0.3,
            batch_size_per_image=256, positive_fraction=0.5,
            pre_nms_top_n={'training': 2000, 'testing': 1000},
            post_nms_top_n={'training': 2000, 'testing': 1000},
            nms_thresh=0.7
        )

        # 3) RoI Heads
        roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                      output_size=7, sampling_ratio=2)
        head = TwoMLPHead(backbone.out_channels * 7 * 7, 1024)
        predictor = FastRCNNPredictor(1024, num_classes)
        roi_heads = RoIHeads(
            roi_pool, head, predictor,
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05, nms_thresh=0.5,
            detections_per_img=100
        )

        # 4) Transform (변환 작업만, augmentation 없이 크기 조정만)
        transform = GeneralizedRCNNTransform(
            min_size=min_size, max_size=max_size,
            image_mean=img_mean, image_std=img_std
        )

        super().__init__(backbone, rpn, roi_heads, transform)
        # -------- 저장용 정보 --------
        self.modals = modals

        
    def forward(self, images, targets=None):        
        # 각 modality 별로 이미지를 분리 (rgb, depth, event, lidar 등)
        rgb = images[0] if isinstance(images, list) else images  # B x C x H x W
        # torchvision 내부 transform 호출
        rgb_transformed, targets = self.transform(rgb, targets)
        filtered_targets = [target for target in targets if target['boxes'].size(0) > 0]
        if len(filtered_targets) == 0:
            return {}
        # 4개의 모달리티에 대해 모두 변환을 진행해야 한다.
        # 각 모달리티에서 변환한 후 백본에 전달
        features = self.backbone(images)  # 4개의 modality를 모두 사용하여 features 추출

        # 5) RPN
        proposals, rpn_losses = self.rpn(rgb_transformed, features, targets)

        # 6) RoI Heads
        detections, roi_losses = self.roi_heads(features, proposals,
                                                rgb_transformed.image_sizes, targets)


        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses

        detections = self.transform.postprocess(
            detections, images.image_sizes,
            [(img.shape[-2], img.shape[-1]) for img in rgb] )
        return detections
    





import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
# from semseg.models.deformable_transformer import DeformableTransformer  # Corrected import path


# class HungarianMatcher(nn.Module):
#     def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
#         super().__init__()
#         self.cost_class = cost_class
#         self.cost_bbox = cost_bbox
#         self.cost_giou = cost_giou
#         assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "All costs can't be 0"

#     @torch.no_grad()
#     def forward(self, outputs, targets):
#         bs, num_queries = outputs['pred_logits'].shape[:2]

#         out_prob = outputs['pred_logits'].softmax(-1)  # [bs, num_queries, num_classes+1]
#         out_bbox = outputs['pred_boxes']              # [bs, num_queries, 4]

#         indices = []
#         for b in range(bs):
#             tgt_ids = targets[b]['labels']
#             tgt_bbox = targets[b]['boxes']

#             cost_class = -out_prob[b][:, tgt_ids]
#             cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
#             cost_giou = -generalized_box_iou(
#                 box_cxcywh_to_xyxy(out_bbox[b]),
#                 box_cxcywh_to_xyxy(tgt_bbox)
#             )

#             C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
#             C = C.cpu()
#             indices_b = linear_sum_assignment(C)
#             indices.append((torch.as_tensor(indices_b[0], dtype=torch.int64),
#                             torch.as_tensor(indices_b[1], dtype=torch.int64)))
#         return indices


# class SetCriterion(nn.Module):
#     def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
#         super().__init__()
#         self.num_classes = num_classes
#         self.matcher = matcher
#         self.weight_dict = weight_dict
#         self.eos_coef = eos_coef
#         self.losses = losses

#         empty_weight = torch.ones(self.num_classes + 1)
#         empty_weight[-1] = self.eos_coef
#         self.register_buffer('empty_weight', empty_weight)

#     def loss_labels(self, outputs, targets, indices, num_boxes):
#         src_logits = outputs['pred_logits']

#         idx = self._get_src_permutation_idx(indices)
#         target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                     dtype=torch.int64, device=src_logits.device)
#         target_classes[idx] = target_classes_o

#         loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
#         return {'loss_ce': loss_ce}

#     def loss_boxes(self, outputs, targets, indices, num_boxes):
#         idx = self._get_src_permutation_idx(indices)
#         src_boxes = outputs['pred_boxes'][idx]
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
#         losses = {}
#         losses['loss_bbox'] = loss_bbox.sum() / num_boxes
#         losses['loss_giou'] = 1 - torch.diag(generalized_box_iou(
#             box_cxcywh_to_xyxy(src_boxes),
#             box_cxcywh_to_xyxy(target_boxes)
#         )).sum() / num_boxes
#         return losses

#     def _get_src_permutation_idx(self, indices):
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def forward(self, outputs, targets):
#         indices = self.matcher(outputs, targets)

#         num_boxes = sum(len(t['labels']) for t in targets)
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
#         num_boxes = torch.clamp(num_boxes, min=1).item()

#         losses = {}
#         for loss in self.losses:
#             losses.update(getattr(self, f'loss_{loss}')(outputs, targets, indices, num_boxes))
#         total_loss = sum(self.weight_dict[k] * losses[k] for k in losses.keys() if k in self.weight_dict)
#         losses['total_loss'] = total_loss
#         return losses

# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=-1)

# def generalized_box_iou(boxes1, boxes2):
#     # Placeholder: use torchvision.ops.generalized_box_iou in practice
#     from torchvision.ops import generalized_box_iou as giou
#     return giou(boxes1, boxes2)


# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         layers = []
#         for i in range(num_layers):
#             in_dim = input_dim if i == 0 else hidden_dim
#             out_dim = output_dim if i == num_layers - 1 else hidden_dim
#             layers.append(nn.Linear(in_dim, out_dim))
#         self.mlp = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.mlp(x)

# class CMNeXtDETR(nn.Module):
#     def __init__(self, backbone, num_classes=91, num_queries=300, hidden_dim=256, num_feature_levels=4, criterion_cfg=None):
#         super().__init__()
#         self.backbone = backbone
#         self.input_proj = nn.ModuleList([
#             nn.Conv2d(c, hidden_dim, kernel_size=1)
#             for c in [64, 128, 320, 512][:num_feature_levels]
#         ])
#         self.transformer = DeformableTransformer(
#             d_model=hidden_dim,
#             nhead=8,
#             num_encoder_layers=6,
#             num_decoder_layers=6,
#             dim_feedforward=2048,
#             dropout=0.1,
#             activation="relu",
#             num_feature_levels=num_feature_levels,
#             dec_n_points=4,
#             enc_n_points=4
#         )
#         self.query_embed = nn.Embedding(num_queries, hidden_dim)
#         self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
#         self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
#         self.num_feature_levels = num_feature_levels

#         matcher = HungarianMatcher()
#         weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
#         losses = ["labels", "boxes"]

#         if criterion_cfg is None:
#             criterion_cfg = {

#             }
#         matcher = HungarianMatcher(
#             cost_class=criterion_cfg.get("COST_CLASS", 1.0),
#             cost_bbox=criterion_cfg.get("COST_BBOX", 5.0),
#             cost_giou=criterion_cfg.get("COST_GIOU", 2.0)
#         )
#         weight_dict = {
#             "loss_ce": criterion_cfg.get("COST_CLASS", 1.0),
#             "loss_bbox": criterion_cfg.get("COST_BBOX", 5.0),
#             "loss_giou": criterion_cfg.get("COST_GIOU", 2.0)
#         }
#         losses = criterion_cfg.get("LOSSES", ["labels", "boxes"])
#         eos_coef = criterion_cfg.get("EOS_COEF", 0.1)

#         self.criterion = SetCriterion(
#             num_classes=num_classes,
#             matcher=matcher,
#             weight_dict=weight_dict,
#             eos_coef=eos_coef,
#             losses=losses
#         )

#         self.criterion = SetCriterion(num_classes, matcher, weight_dict, eos_coef=0.1, losses=losses)

#     def forward(self, x: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]] = None):
#         features: Dict[str, torch.Tensor] = self.backbone(x)

#         srcs = []
#         masks = []
#         pos_embeds = []

#         for i in range(self.num_feature_levels):
#             src = self.input_proj[i](features[str(i)])
#             mask = torch.zeros((src.size(0), src.size(2), src.size(3)), dtype=torch.bool, device=src.device)
#             srcs.append(src)
#             masks.append(mask)
#             pos_embeds.append(self._get_positional_encoding(src))

#         hs, init_reference, inter_references, _, _ = self.transformer(
#             srcs, masks, pos_embeds, self.query_embed.weight
#         )
#         outputs_class = self.class_embed(hs[-1])
#         outputs_coord = self.bbox_embed(hs[-1]).sigmoid()

#         outputs = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

#         if self.training:
#             assert targets is not None
#             return self.criterion(outputs, targets)

#         return outputs

#     def _get_positional_encoding(self, src):
#         N, C, H, W = src.shape
#         y_embed = torch.linspace(0, 1, H, device=src.device).view(1, H, 1).expand(1, H, W)
#         x_embed = torch.linspace(0, 1, W, device=src.device).view(1, 1, W).expand(1, H, W)
#         pos = torch.cat([
#             x_embed.unsqueeze(1).repeat(N, C // 2, 1, 1),
#             y_embed.unsqueeze(1).repeat(N, C // 2, 1, 1)
#         ], dim=1)
#         return pos
    

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

    # def decode_boxes(self, anchors, deltas):
    #     # anchors: [N, 4] (x1, y1, x2, y2)
    #     # deltas: [N, 4] (dx, dy, dw, dh)
    #     widths = anchors[:, 2] - anchors[:, 0]
    #     heights = anchors[:, 3] - anchors[:, 1]
    #     ctr_x = anchors[:, 0] + 0.5 * widths
    #     ctr_y = anchors[:, 1] + 0.5 * heights

    #     dx, dy, dw, dh = deltas.unbind(-1)
    #     pred_ctr_x = dx * widths + ctr_x
    #     pred_ctr_y = dy * heights + ctr_y
    #     pred_w = torch.exp(dw) * widths
    #     pred_h = torch.exp(dh) * heights

    #     pred_boxes = torch.stack([
    #         pred_ctr_x - 0.5 * pred_w,
    #         pred_ctr_y - 0.5 * pred_h,
    #         pred_ctr_x + 0.5 * pred_w,
    #         pred_ctr_y + 0.5 * pred_h,
    #     ], dim=-1)
    #     return pred_boxes

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

        # Check for any NaNs or Inf values in the predicted boxes and remove them
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