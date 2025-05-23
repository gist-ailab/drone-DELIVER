

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
from semseg.models.cmnext_backbone import CMNeXtBackbone

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
    