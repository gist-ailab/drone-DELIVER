import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 
from typing import Tuple, List, Union, Tuple, Optional, Dict


def get_train_augmentation(img_size, 
                           format='pascal_voc',
                           additional_targets: Dict[str, str] = None):
    if additional_targets is None:
        additional_targets = {}
    h, w = img_size[0], img_size[1]
    albumentation_aug = A.Compose([
        A.RandomCrop(width = w, height=h, p=1.0),  # 예시용 crop
        A.ColorJitter(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(3, 3), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format=format, label_fields=['labels']),
    additional_targets=additional_targets)
    return albumentation_aug


def get_val_augmentation(img_size, 
                           format='pascal_voc',
                           additional_targets: Dict[str, str] = None):
    h = img_size[0]
    w = img_size[1]
    if additional_targets is None:
        additional_targets = {}

    albumentation_aug = A.Compose([
        A.Resize(width = w, height = h),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format=format, label_fields=['labels']),
    additional_targets=additional_targets)
    return albumentation_aug