import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
from semseg.models.heads import LightHamHead
from semseg.models.heads import UPerHead
from fvcore.nn import flop_count_table, FlopCountAnalysis

import torch.nn as nn
from torchvision.models.detection import FasterRCNN, FastRCNNPredictor, TwoMLPHead
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
# from torchvision.models.detection import GeneralizedRCNNTransform
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import warnings
from torchvision.models.detection.image_list import ImageList


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
    

def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)

'''
class CMNeXtFasterRCNN(nn.Module):
    def __init__(self, backbone_name='CMNeXt-B2', num_classes=25, modals=['img', 'depth', 'event', 'lidar']):
        super().__init__()
        cmnext_settings = {
            'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
            'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
            'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
            'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
            'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
            'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
        }
        backbone_suffix = backbone_name.split('-')[-1]
        if backbone_suffix in cmnext_settings:
            out_channels = cmnext_settings[backbone_suffix][0]  # 마지막 값이 out_channels에 해당
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Step 1: Get raw multi-scale features
        self.body = CMNeXtBackbone(backbone_name, modals)
        fpn_in_channels = {
            '0': out_channels[0],
            '1': out_channels[1],
            '2': out_channels[2],
            '3': out_channels[3],
        }

        # Step 2: FPN to unify feature dimensions
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(fpn_in_channels.values()),
            out_channels=out_channels[-1],
            extra_blocks=None
        )

        # Step 3: Combine backbone and FPN
        class BackboneWithCustomFPN(nn.Module):
            def __init__(self, body, fpn):
                super().__init__()
                self.body = body
                self.fpn = fpn
                self.out_channels = out_channels[3]

            def forward(self, x):
                features = self.body(x)
                keys = features.keys()
                out = self.fpn(features)
                new_dict = {key: list(value) for key, value in out.items()}

                return out
        
        backbone = BackboneWithCustomFPN(self.body, self.fpn)
        self.backbone = backbone
        # Step 4: RPN anchor generator and ROI align
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        # Step 5: Final Faster R-CNN model
        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

    def forward(self, images, targets=None):
        features  =self.backbone(images)
        return self.model(images, targets)

'''
'''
class CMNeXtFasterRCNN(FasterRCNN):
    def __init__(self, backbone_name='CMNeXt-B2', num_classes=2, modals=['img', 'depth', 'event', 'lidar'],
                    # RPN parameters
                    rpn_anchor_generator=None,
                    rpn_head=None,
                    rpn_pre_nms_top_n_train=2000,
                    rpn_pre_nms_top_n_test=1000,
                    rpn_post_nms_top_n_train=2000,
                    rpn_post_nms_top_n_test=1000,
                    rpn_nms_thresh=0.7,
                    rpn_fg_iou_thresh=0.7,
                    rpn_bg_iou_thresh=0.3,
                    rpn_batch_size_per_image=256,
                    rpn_positive_fraction=0.5,
                    rpn_score_thresh=0.0,
                    # Box parameters
                    box_roi_pool=None,
                    box_head=None,
                    box_predictor=None,
                    box_score_thresh=0.05,
                    box_nms_thresh=0.5,
                    box_detections_per_img=100,
                    box_fg_iou_thresh=0.5,
                    box_bg_iou_thresh=0.5,
                    box_batch_size_per_image=512,
                    box_positive_fraction=0.25,
                    bbox_reg_weights=None,
                    **kwargs,
                    ):
                 
        # CMNeXt 설정 (B0, B1, B2, B3 등)
        cmnext_settings = {
            'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
            'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
            'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
            'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
            'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
            'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
        }

        backbone_suffix = backbone_name.split('-')[-1]
        if backbone_suffix in cmnext_settings:
            out_channels = cmnext_settings[backbone_suffix][0]  # 마지막 값이 out_channels에 해당
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Step 1: Initialize CMNeXtBackbone
        body = CMNeXtBackbone(backbone_name, modals)
        fpn_in_channels = {
            '0': out_channels[0],
            '1': out_channels[1],
            '2': out_channels[2],
            '3': out_channels[3],
        }

        # Step 2: Initialize FPN
        fpn = FeaturePyramidNetwork(
            in_channels_list=list(fpn_in_channels.values()),
            out_channels=out_channels[-1],
            extra_blocks=None
        )

        # Step 3: Combine backbone and FPN into a custom model
        class BackboneWithCustomFPN(nn.Module):
            def __init__(self, body, fpn):
                super().__init__()
                self.body = body
                self.fpn = fpn
                self.out_channels = out_channels[3]

            def forward(self, x):
                features = self.body(x)
                out = self.fpn(features)
                return out

        backbone = BackboneWithCustomFPN(body, fpn)

        # Step 4: Define Anchor Generator and ROI Pooler
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        box_roi_pool = roi_pooler

        # Step 5: Define Box Head and Box Predictor
        resolution = roi_pooler.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(out_channels[-1] * resolution**2, representation_size)
        box_predictor = FastRCNNPredictor(representation_size, num_classes)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        # Step 6: Define ROI Heads
        roi_heads = RoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )
        super().__init__(backbone, rpn, roi_heads, transform)
'''
class CMNeXtFasterRCNN(GeneralizedRCNN):
    def __init__(self, backbone_name='CMNeXt-B2', num_classes=2, modals=['img', 'depth', 'event', 'lidar'],
                 # RPN parameters
                 rpn_anchor_generator=None,
                 rpn_head=None,
                 rpn_pre_nms_top_n_train=2000,
                 rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000,
                 rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7,
                 rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256,
                 rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None,
                 box_head=None,
                 box_predictor=None,
                 box_score_thresh=0.05,
                 box_nms_thresh=0.5,
                 box_detections_per_img=100,
                 box_fg_iou_thresh=0.5,
                 box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512,
                 box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 transform=None,
                 min_size=800,
                 max_size=1333,
                 image_mean=None,
                 image_std=None,
                 target_format='xywh',  # COCO 형식 지정
                 **kwargs,
                 ):
        
        # Step 1: Initialize CMNeXtBackbone
        backbone_model = CMNeXtBackbone(backbone_name, modals)
        
        # Determine out_channels based on the backbone
        if 'B0' in backbone_name or 'B1' in backbone_name:
            backbone_out_channels = 256
        else:
            backbone_out_channels = 512
            
        # Use a fixed out_channels for FPN (this will be used for all feature maps)
        fpn_out_channels = 512
            
        # Step 2: Create FPN to normalize all features to the same channel dimension
        in_channels_list = []
        if 'B0' in backbone_name:
            in_channels_list = [32, 64, 160, 256]
        elif 'B1' in backbone_name:
            in_channels_list = [64, 128, 320, 512]
        else:  # B2, B3, B4, B5
            in_channels_list = [64, 128, 320, 512]
        
        # Create a custom backbone with FPN that ensures all output features have the same channel count
        class BackboneWithCustomFPN(nn.Module):
            def __init__(self, body, in_channels_list, out_channels):
                super().__init__()
                self.body = body
                
                # Create projection layers for each feature level to normalize channels
                self.fpn_projections = nn.ModuleDict({
                    f'{i}': nn.Conv2d(in_channels, out_channels, kernel_size=1)
                    for i, in_channels in enumerate(in_channels_list)
                })
                
                self.out_channels = out_channels
                
            def forward(self, x):
                features = self.body(x)
                
                # Apply projection to each feature level to normalize channels
                return {
                    str(k): self.fpn_projections[str(k)](v)
                    for k, v in features.items()
                }
        
        # Combine backbone and custom FPN
        backbone = BackboneWithCustomFPN(backbone_model, in_channels_list, fpn_out_channels)
            
        # Step 3: Initialize the RPN anchor generator if not provided
        if rpn_anchor_generator is None:
            # 각 feature map별로 적절한 anchor 크기 지정
            anchor_sizes = ((32,), (64,), (128,), (256,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                sizes=anchor_sizes,
                aspect_ratios=aspect_ratios
            )
            
        # Step 4: Initialize RPN head if not provided
        if rpn_head is None:
            rpn_head = RPNHead(
                fpn_out_channels,  # Now all feature maps have this channel count
                rpn_anchor_generator.num_anchors_per_location()[0]
            )
            
        # Step 5: Set up RPN parameters
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        
        # Step 6: Initialize RPN
        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )
        
        # Step 7: Initialize ROI pooler if not provided
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2
            )
            
        # Step 8: Initialize box head if not provided
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                fpn_out_channels * resolution**2,  # Use the normalized channel count
                representation_size
            )
            
        # Step 9: Initialize box predictor if not provided
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)
            
        # Step 10: Initialize ROI heads
        roi_heads = RoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )
        
        # Step 11: Initialize transform if needed
        if transform is None:
            if image_mean is None:
                image_mean = [0.485, 0.456, 0.406]
            if image_std is None:
                image_std = [0.229, 0.224, 0.225]
                
            from torchvision.models.detection.transform import GeneralizedRCNNTransform
            transform = GeneralizedRCNNTransform(
                min_size=min_size,
                max_size=max_size,
                image_mean=image_mean,
                image_std=image_std
            )
        
        # Step 12: Initialize the GeneralizedRCNN parent class with all components
        super().__init__(backbone, rpn, roi_heads, transform)
        
        # Save parameters for later use
        self.modals = modals
        self.target_format = target_format
        self._has_warned = False
    
    def _convert_xywh_to_xyxy(self, boxes):
        """
        Convert boxes from [x, y, width, height] format to [x1, y1, x2, y2] format
        
        Args:
            boxes (Tensor): Boxes in [x, y, width, height] format
            
        Returns:
            Tensor: Boxes in [x1, y1, x2, y2] format
        """
        if boxes.numel() == 0:
            return boxes  # 빈 텐서는 변환 없이 반환
            
        x, y, w, h = boxes.unbind(1)
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        return torch.stack((x1, y1, x2, y2), dim=1)
    
    def _convert_xyxy_to_xywh(self, boxes):
        """
        Convert boxes from [x1, y1, x2, y2] format to [x, y, width, height] format
        
        Args:
            boxes (Tensor): Boxes in [x1, y1, x2, y2] format
            
        Returns:
            Tensor: Boxes in [x, y, width, height] format
        """
        if boxes.numel() == 0:
            return boxes  # 빈 텐서는 변환 없이 반환
            
        x1, y1, x2, y2 = boxes.unbind(1)
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        return torch.stack((x, y, w, h), dim=1)
    
    def _filter_empty_targets(self, targets):
        """
        빈 타겟을 필터링하고 더미 박스를 추가하는 함수
        """
        if targets is None:
            return None
            
        filtered_targets = []
        for target in targets:
            target_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
            
            # 박스가 없는 경우 더미 박스 추가
            if 'boxes' in target_copy and target_copy['boxes'].numel() == 0:
                device = target_copy['boxes'].device
                # 이미지 밖에 위치한 작은 더미 박스 추가 - 학습에 영향을 주지 않음
                dummy_box = torch.tensor([[-10.0, -10.0, -9.0, -9.0]], device=device)
                target_copy['boxes'] = dummy_box
                
                # 해당 더미 박스의 라벨도 추가
                if 'labels' in target_copy and target_copy['labels'].numel() == 0:
                    # 배경 클래스(일반적으로 0) 추가
                    target_copy['labels'] = torch.tensor([0], device=device, dtype=torch.int64)
            
            filtered_targets.append(target_copy)
        
        return filtered_targets
        
    def forward(self, images, targets=None):
        """
        Args:
            images (list): images to be processed, expected to be a list of tensors for different modalities
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        # 디버깅을 위한 정보 출력
        if self.training and targets is not None:
            target_shapes = []
            for t in targets:
                box_shape = t['boxes'].shape if 'boxes' in t else None
                label_shape = t['labels'].shape if 'labels' in t else None
                target_shapes.append((box_shape, label_shape))
            print(f"Input targets: {target_shapes}")
            
            for i, t in enumerate(targets):
                if 'boxes' in t:
                    print(f"Target {i} boxes device: {t['boxes'].device}, dtype: {t['boxes'].dtype}")
                    if t['boxes'].numel() > 0:  # 빈 텐서 체크
                        print(f"Target {i} boxes min/max: {t['boxes'].min().item()}, {t['boxes'].max().item()}")
                    else:
                        print(f"Target {i} boxes is empty")
                if 'labels' in t:
                    print(f"Target {i} labels device: {t['labels'].device}, dtype: {t['labels'].dtype}")
                    if t['labels'].numel() > 0:  # 빈 텐서 체크
                        print(f"Target {i} labels unique values: {t['labels'].unique()}")
                    else:
                        print(f"Target {i} labels is empty")
        
        # Check if we're in training mode
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            
            # 빈 타겟 처리
            has_empty_targets = any(t.get('boxes', torch.tensor([])).numel() == 0 for t in targets)
            if has_empty_targets:
                # 빈 타겟이 있는 경우, 필터링 또는 더미 추가
                targets = self._filter_empty_targets(targets)
        
        # Convert target box format if needed
        if targets is not None and self.target_format == 'xywh':
            targets_converted = []
            for target in targets:
                target_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                if 'boxes' in target_copy and target_copy['boxes'].numel() > 0:
                    target_copy["boxes"] = self._convert_xywh_to_xyxy(target_copy["boxes"])
                targets_converted.append(target_copy)
            targets = targets_converted
        
        # Validate target boxes if provided
        if targets is not None:
            for target_idx, target in enumerate(targets):
                if 'boxes' not in target or target['boxes'].numel() == 0:
                    continue  # 빈 타겟은 건너뜀
                    
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")
                
                # Fix degenerate boxes instead of raising an error
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    print(f"Warning: Found degenerate boxes in target {target_idx}. Fixing...")
                    fixed_boxes = boxes.clone()
                    # Add a small offset to ensure boxes have positive area
                    for b_idx in torch.where(degenerate_boxes.any(dim=1))[0]:
                        fixed_boxes[b_idx, 2:] = fixed_boxes[b_idx, :2] + 1.0
                    target["boxes"] = fixed_boxes
        
        # Handle multi-modal input - extract RGB images for RPN
        if isinstance(images, list) and len(images) > 0:
            rgb_images = images[0]  # First modality is RGB
        else:
            rgb_images = images
        
        # Apply transform to RGB images
        # try:
        images_transformed, targets_transformed = self.transform(rgb_images, targets)
        # except Exception as e:
        #     print(f"Error in transform: {e}")
        #     # 훈련 시 오류 발생하면 더미 손실 반환
        #     if self.training:
        #         return {"loss_classifier": torch.tensor(0.0, device=rgb_images[0].device, requires_grad=True),
        #                 "loss_box_reg": torch.tensor(0.0, device=rgb_images[0].device, requires_grad=True),
        #                 "loss_objectness": torch.tensor(0.0, device=rgb_images[0].device, requires_grad=True),
        #                 "loss_rpn_box_reg": torch.tensor(0.0, device=rgb_images[0].device, requires_grad=True)}
        #     else:
        #         return []
        
        try:
            # Extract features from all modalities using the backbone
            features = self.backbone(images)
            
            # Get RPN proposals
            proposals, proposal_losses = self.rpn(images_transformed, features, targets_transformed)
            
            # Get ROI detections and losses
            detections, detector_losses = self.roi_heads(features, proposals, images_transformed.image_sizes, targets_transformed)
            
            # Postprocess detections if not in training mode
            if not self.training:
                detections = self.transform.postprocess(detections, images_transformed.image_sizes, 
                                                     [(img.shape[1], img.shape[2]) for img in rgb_images])
            
                # Convert output box format back to xywh if necessary
                if self.target_format == 'xywh':
                    for detection in detections:
                        if detection["boxes"].numel() > 0:
                            detection["boxes"] = self._convert_xyxy_to_xywh(detection["boxes"])
            
            # Aggregate losses
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            
        # except Exception as e:
        #     print(f"Error during forward pass: {e}")
        #     # 훈련 시 오류 발생하면 더미 손실 반환
        #     if self.training:
        #         return {"loss_classifier": torch.tensor(0.0, device=rgb_images[0].device, requires_grad=True),
        #                 "loss_box_reg": torch.tensor(0.0, device=rgb_images[0].device, requires_grad=True),
        #                 "loss_objectness": torch.tensor(0.0, device=rgb_images[0].device, requires_grad=True),
        #                 "loss_rpn_box_reg": torch.tensor(0.0, device=rgb_images[0].device, requires_grad=True)}
        #     else:
                # 추론 시 빈 결과 반환
            batch_size = len(rgb_images)
            return [
                {
                    "boxes": torch.zeros((0, 4), device=rgb_images[0].device),
                    "labels": torch.zeros(0, dtype=torch.int64, device=rgb_images[0].device),
                    "scores": torch.zeros(0, device=rgb_images[0].device)
                }
                for _ in range(batch_size)
            ]
        
        # Return based on mode (training or inference)
        if self.training:
            return losses
        
        return detections

if __name__ == '__main__':
    modals = ['img', 'depth', 'event', 'lidar']
    model = CMNeXtFasterRCNN('CMNeXt-B2', 25, modals)
    model.init_pretrained('checkpoints/pretrained/segformer/mit_b2.pth')
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    y = model(x)
    print(y.shape)






'''
# class CMNeXtFasterRCNN(nn.Module):
#     def __init__(self, backbone_name='CMNeXt-B2', num_classes=25, modals=['img', 'depth', 'event', 'lidar']):
#         super().__init__()
#         self.backbone = CMNeXtBackbone(backbone_name, modals)

#         # Set the out_channels (e.g., 512) for the FPN
#         out_channels = self.backbone.out_channels

#         # Define anchor generator for RPN
#         anchor_generator = AnchorGenerator(
#             sizes=((32, 64, 128, 256, 512),),
#             aspect_ratios=((0.5, 1.0, 2.0),)
#         )

#         # Define ROI Align for detection head
#         roi_pooler = MultiScaleRoIAlign(
#             featmap_names=['0', '1', '2', '3'],
#             output_size=7,
#             sampling_ratio=2
#         )

#         self.model = FasterRCNN(
#             backbone=self.backbone,
#             num_classes=num_classes,
#             rpn_anchor_generator=anchor_generator,
#             box_roi_pool=roi_pooler
#         )

#     def forward(self, images, targets=None):
#         return self.model(images, targets)
'''


