import os
import torch
import numpy as np
from torch.utils.data import Dataset
# torchvision.transforms.functional is no longer directly needed here for main transforms
# from torchvision import io # Replaced by PIL or OpenCV for image loading
from PIL import Image # Using PIL for image loading
import cv2 # OpenCV can also be used, PIL is often simpler for basic loading
import torchvision.transforms.functional as TF 

from pathlib import Path
from typing import Tuple, List, Dict, Union # Added Dict, Union
# import glob # Not used
# import einops # Not used
from torch.utils.data import DataLoader
from semseg.augmentations_detection_mm2 import get_train_augmentation, get_val_augmentation
import json

def coco_bbox_to_pascal_voc(bbox):
    """Converts COCO bbox [x, y, w, h] to Pascal VOC [xmin, ymin, xmax, ymax]."""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

class DELIVERCOCO(Dataset):
    """
    DELIVER Dataset for Object Detection, compatible with COCO format annotations.
    Uses Albumentations for transformations.
    """
    CLASSES = ["Human", "Car"] # Example, adjust if different based on coco.json categories

    def __init__(self, root: str = 'data/DELIVER', split: str = 'train', 
                 transform=None, modals: List[str] = ['img'], case: str = None,
                 target_img_size: Union[int, Tuple[int, int]] = (1024, 1024)):
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.root = Path(root)
        self.split = split
        self.modals = modals # e.g., ['img', 'depth']
        self.case = case # Could be used to select sub-folders or specific conditions
        self.target_img_size = target_img_size # Used for get_val_augmentation

        # Initialize augmentations if not provided (e.g., for train/val splits)
        if transform is None:
            additional_targets_setup = {}
            if split == 'train':
                self.transform = get_train_augmentation(self.target_img_size, additional_targets=additional_targets_setup)
            else: # val or test
                self.transform = get_val_augmentation(self.target_img_size, additional_targets=additional_targets_setup)
        else:
            self.transform = transform


        ann_path = os.path.join(self.root , f'coco_{split}.json')
        with open(ann_path, 'r') as f:
            coco_data = json.load(f)

        coco_cat_ids = [cat['id'] for cat in sorted(coco_data['categories'], key=lambda x: x['id'])]
        self.cat_id_to_idx = {cid: idx for idx, cid in enumerate(coco_cat_ids)}
        self.CLASSES = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['id'])]

        self.image_id_to_filename = {img_info['id']: img_info['file_name'] for img_info in coco_data['images']}
        self.img_ids = list(self.image_id_to_filename.keys())
        
        # Update CLASSES from coco_data if available and consistent
        if 'categories' in coco_data:
            self.CLASSES = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['id'])]

        self.annotations = {}
        for ann in coco_data['annotations']:
            if ann.get('iscrowd', 0): # Handle missing 'iscrowd' key, default to not crowd
                continue
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        print(f"Loaded {len(self.img_ids)} images for {split} split from {self.root}")
        print(f"Dataset classes: {self.CLASSES}")


    def __len__(self):
        return len(self.img_ids)
    
    def _open_img(self, file):
        # Using PIL to open images
        img = np.array(Image.open(file))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)  # Convert grayscale to RGB
        elif img.shape[-1] == 4:
            img = img[..., :3]  # Discard alpha channel if present

        # all image is shape (H, W, C)
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.img_ids[idx]
        file_name = self.image_id_to_filename[img_id] # e.g., "rgb/image_0001.png"

        rgb = os.path.join(self.root,file_name)
        x1 = rgb.replace('/img', '/hha').replace('_rgb', '_depth')
        x2 = rgb.replace('/img', '/lidar').replace('_rgb', '_lidar')
        x3 = rgb.replace('/img', '/event').replace('_rgb', '_event')

        sample = {}
        sample['image'] = self._open_img(rgb)  # e.g., (H, W, 3)
        H, W = sample['image'].shape[:2]

        if 'depth' in self.modals:
            dimg = self._open_img(x1)
            sample['depth'] = cv2.resize(dimg, (W, H), interpolation=cv2.INTER_NEAREST)

        if 'lidar' in self.modals:
            limg = self._open_img(x2)
            sample['lidar'] = cv2.resize(limg, (W, H), interpolation=cv2.INTER_NEAREST)

        if 'event' in self.modals:
            eimg = self._open_img(x3)
            sample['event'] = cv2.resize(eimg, (W, H), interpolation=cv2.INTER_NEAREST)

        # --- Load Annotations ---
        anns = self.annotations.get(img_id, [])
        bboxes_coco = [] # List of [x, y, w, h]
        labels_list = []

        anns = self.annotations.get(img_id, [])
        bboxes_coco = [ann['bbox'] for ann in anns]
        labels_list = [self.cat_id_to_idx[ann['category_id']] for ann in anns]  # ★ 0-based 변환
        # ids = [ann['category_id'] for ann in self.annotations[self.img_ids[0]]]  # 임의 한 이미지
        # print('DEBUG::::raw category_id sample:', ids[:10])  # 아마 [1] 또는 [1,2] ...


        sample['bboxes'] = bboxes_coco
        sample['labels'] = labels_list
        if self.transform:
            transformed = self.transform(**sample)
            sample['image'] = transformed['image']
            if 'depth' in self.modals:
                sample['depth'] = transformed['depth']
            if 'lidar' in self.modals:      
                sample['lidar'] = transformed['lidar']
            if 'event' in self.modals:
                sample['event'] = transformed['event']

            return_list = [sample[k] for k in self.modals]
            target = {
                'boxes': torch.tensor(transformed['bboxes'], dtype=torch.float32),
                'labels': torch.tensor(transformed['labels'], dtype=torch.int64),
                'image_id': torch.tensor(img_id, dtype=torch.int64)
            }
        
            return return_list, target
        else:
            # 변환이 없는 경우 처리
            assert False, "Transform is None, but no transform was provided in __init__."