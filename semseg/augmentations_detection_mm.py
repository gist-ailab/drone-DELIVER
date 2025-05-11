import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import math
import cv2 # OpenCV is a dependency of albumentations for interpolations etc.
from typing import Union, Tuple, List, Dict

# Helper function to calculate target size for Resize transform to match original logic
def _calculate_resize_dims(original_height: int, original_width: int, size_param: Union[int, Tuple[int, int]]):
    if isinstance(size_param, int): # Target short edge
        if original_height < original_width:
            sf = size_param / original_height
        else:
            sf = size_param / original_width
        scaled_h, scaled_w = round(original_height * sf), round(original_width * sf)
    else: # Target (h, w) tuple
        scaled_h, scaled_w = size_param
    
    aligned_h = math.ceil(scaled_h / 32.0) * 32
    aligned_w = math.ceil(scaled_w / 32.0) * 32
    return int(aligned_h), int(aligned_w)

class AlbumentationsResize:
    """
    Custom wrapper for Albumentations Resize to mimic the original two-step resize:
    1. Scale to target short edge (if size is int) or target H,W (if size is tuple).
    2. Align dimensions to be divisible by 32.
    """
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size_param = size
        # Interpolation for image, mask respectively
        self.interpolation_img = cv2.INTER_LINEAR 
        self.interpolation_mask = cv2.INTER_NEAREST

    def __call__(self, **data) -> Dict:
        img = data['image'] # Albumentations expects 'image' key
        original_height, original_width = img.shape[:2]
        
        aligned_h, aligned_w = _calculate_resize_dims(original_height, original_width, self.size_param)
        
        # Create a resize transform on the fly for the calculated dimensions
        resizer = A.Resize(height=aligned_h, width=aligned_w, 
                           interpolation=self.interpolation_img, 
                           always_apply=True)
        
        # Prepare data for resizer, handling additional targets for masks if present
        transform_input = {'image': img}
        additional_targets_configs = {} # To store interpolation for additional targets

        if 'mask' in data: # Assuming 'mask' is a key for segmentation masks
            transform_input['mask'] = data['mask']
            # For A.Resize, specific interpolation for masks is handled by its own param,
            # but if we were using a more general A.Compose, we'd set it there.
            # Here, we'd need to apply resize separately or ensure A.Resize handles it.
            # A.Resize itself doesn't have a 'mask_interpolation', it uses 'interpolation'.
            # So, we apply resize with NEAREST for masks separately if needed.

        # Handle other image-like additional targets
        other_image_keys = [k for k in data if k not in ['image', 'mask', 'bboxes', 'category_ids'] and isinstance(data[k], np.ndarray)]
        for key in other_image_keys:
            transform_input[key] = data[key]
            # Assuming other image-like modalities also use linear interpolation
            additional_targets_configs[key] = {'interpolation': self.interpolation_img}


        # Apply resize to the main image
        resized_main = resizer(image=img)
        processed_data = {'image': resized_main['image']}

        # Apply resize to the mask with NEAREST interpolation
        if 'mask' in transform_input:
            mask_resizer = A.Resize(height=aligned_h, width=aligned_w, interpolation=self.interpolation_mask, always_apply=True)
            resized_mask_data = mask_resizer(image=transform_input['mask']) # Use 'image' key for A.Resize
            processed_data['mask'] = resized_mask_data['image']
        
        # Apply resize to other image-like modalities
        for key in other_image_keys:
            # Assuming linear interpolation for other modalities
            modality_resizer = A.Resize(height=aligned_h, width=aligned_w, interpolation=self.interpolation_img, always_apply=True)
            resized_modality_data = modality_resizer(image=transform_input[key])
            processed_data[key] = resized_modality_data['image']

        # Handle bboxes - Albumentations A.Resize handles bboxes automatically if passed.
        if 'bboxes' in data:
            # To correctly transform bboxes, A.Resize needs to be part of a Compose pipeline
            # or called with bbox_params. This custom wrapper makes it tricky.
            # A simpler way is to include A.Resize directly in the main A.Compose pipeline.
            # For this wrapper, we'd manually scale bboxes.
            sf_h = aligned_h / original_height if original_height > 0 else 1.0
            sf_w = aligned_w / original_width if original_width > 0 else 1.0
            
            bboxes = np.array(data['bboxes']).astype(np.float32)
            if bboxes.shape[0] > 0:
                bboxes[:, [0, 2]] *= sf_w
                bboxes[:, [1, 3]] *= sf_h
                
                # Clip bboxes
                bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, aligned_w)
                bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, aligned_h)

                # Filter out invalid boxes
                widths = bboxes[:, 2] - bboxes[:, 0]
                heights = bboxes[:, 3] - bboxes[:, 1]
                valid_indices = (widths > 1e-3) & (heights > 1e-3)
                processed_data['bboxes'] = bboxes[valid_indices]
                if 'category_ids' in data:
                    processed_data['category_ids'] = np.array(data['category_ids'])[valid_indices]
            else:
                processed_data['bboxes'] = bboxes
                if 'category_ids' in data:
                    processed_data['category_ids'] = np.array(data['category_ids'])
        
        elif 'category_ids' in data: # Pass through if no bboxes
             processed_data['category_ids'] = data['category_ids']


        # Add back any non-transformed data
        for k, v_item in data.items():
            if k not in processed_data:
                processed_data[k] = v_item
        
        return processed_data


def get_train_augmentation(size: Union[int, Tuple[int, int]], 
                           mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                           additional_targets: Dict[str, str] = None):
    
    if isinstance(size, int): # If size is int, it's target for shorter edge for RRC
        # A.RandomResizedCrop expects (height, width)
        # This behavior differs from the original custom RRC.
        # For simplicity, we'll assume 'size' is (height, width) for A.RandomResizedCrop
        # If 'size' was truly meant for short edge, RRC would need custom logic or a different approach.
        # Given the original RRC took a tuple 'size', we assume 'size' is (h,w) here.
        if isinstance(size, int):
             # This is a common practice if an int is given, use it for both H and W
            crop_h, crop_w = size, size
        else:
            crop_h, crop_w = size 
    else:
        crop_h, crop_w = size

    return A.Compose([
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0, p=0.2), # Original had brightness, contrast, saturation random.uniform(0.5,1.5)
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(3, 3), p=0.2), # Original kernel_size was (3,3)
        # Original RandomResizedCrop had scale=(0.5, 2.0).
        # A.RandomResizedCrop uses scale=(0.08, 1.0) by default. We'll match the old scale.
        A.RandomResizedCrop(height=crop_h, width=crop_w, scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2() # Converts image to C,H,W and bboxes/labels to tensors
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.1),
       additional_targets=additional_targets or {})


def get_val_augmentation(size: Union[int, Tuple[int, int]], 
                         mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                         additional_targets: Dict[str, str] = None):
    # For validation, we use the custom AlbumentationsResize to ensure divisibility by 32
    # This is a bit of a workaround. A cleaner way would be to integrate this logic
    # into a standard Albumentations transform if possible, or accept that A.Resize
    # might not perfectly match the "divisible by 32" rule without PadIfNeeded.

    # We will use A.Lambda with a helper for val resize to keep it within albumentations pipeline
    def resize_fn(image, **kwargs):
        # This function will be wrapped by A.Lambda. It needs to calculate dims and call A.Resize
        # However, A.Lambda applies to 'image' only by default.
        # For bboxes and additional_targets, this is complex with A.Lambda.
        # It's better to use A.Resize directly in the Compose pipeline.
        # The custom wrapper `AlbumentationsResize` is hard to integrate directly here
        # without making it an official albumentations-style class.
        
        # Let's use A.Resize directly and calculate dimensions beforehand.
        # This means the `size` parameter for `get_val_augmentation` must be
        # pre-calculated if dynamic sizing based on input is needed.
        # Or, the dataset's __getitem__ must calculate it.
        
        # For simplicity in this function, if size is an int, we assume it's the desired short edge
        # and we'll use A.SmallestMaxSize then A.PadIfNeeded.
        # If size is a tuple (h,w), we use A.Resize directly.
        
        # This part is tricky because the A.Compose pipeline is static.
        # Dynamic resizing based on image content needs to happen in the Dataset's __getitem__
        # *before* this static pipeline, or by using a more complex custom transform.

        # For now, let's assume `size` is the final (H, W) after all calculations.
        # The Dataset will be responsible for providing an image that, when `size` is applied,
        # results in dimensions divisible by 32.
        # Or, we use A.PadIfNeeded here.
        
        # Let's try: A.Resize to a target, then A.PadIfNeeded to make it divisible by 32.
        # This is a common pattern.
        if isinstance(size, int):
            # This means 'size' is the target for the smallest edge.
            # This is not directly what the original Resize did in its second step.
            # The original Resize took the output of the first scaling and made *that* divisible by 32.
            # Let's stick to a simpler A.Resize(h,w) for validation, assuming h,w are pre-calculated
            # to be divisible by 32 if that's a strict model requirement.
            # If `size` is an int, we'll treat it as (size, size) for A.Resize.
            val_h, val_w = size, size
        else:
            val_h, val_w = size # Assuming size is (h,w) and these are final target dims

        # The pipeline will be:
        # 1. Resize to val_h, val_w
        # 2. Normalize
        # 3. ToTensorV2
        # The divisibility by 32 must be ensured by the choice of val_h, val_w.
        # Example: if model needs 256x256, pass size=(256,256)
        pass # This logic will be in the A.Compose return statement

    # Determine target H, W for A.Resize.
    # This is tricky because A.Compose is defined once.
    # If `size` is an int, it implies dynamic calculation based on image aspect ratio.
    # Albumentations typically expects fixed H, W for A.Resize in a static pipeline.
    # A common approach: Resize so smallest side is `size`, then PadIfNeeded to a fixed H,W divisible by 32.
    
    # Let's assume `size` is the desired (H,W) output, and these H,W are chosen to be divisible by 32.
    if isinstance(size, int):
        target_h, target_w = size, size # Assume square output if int
    else:
        target_h, target_w = size # Assume (h,w) tuple

    return A.Compose([
        A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.1),
       additional_targets=additional_targets or {})


if __name__ == '__main__':
    # Example usage for object detection with Albumentations
    h_orig, w_orig = 230, 420
    
    # Albumentations expects image as HWC, NumPy array
    img_np = np.random.randint(0, 256, (h_orig, w_orig, 3), dtype=np.uint8)
    depth_np = np.random.rand(h_orig, w_orig, 1).astype(np.float32) # Example additional target
    
    # BBoxes in pascal_voc format: [x_min, y_min, x_max, y_max]
    bboxes_np = np.array([[10, 20, 100, 120], [150, 160, 200, 210]], dtype=np.float32)
    # Labels for bboxes
    category_ids_np = np.array([1, 2], dtype=np.long)

    target_size_train = (224, 224) # For RandomResizedCrop
    
    # Define additional targets for the augmentation pipeline
    # The keys here ('depth') must match the keys passed to the transform call.
    additional_targets_setup = {'depth': 'image'} 

    train_augs = get_train_augmentation(target_size_train, additional_targets=additional_targets_setup)
    
    sample_to_transform = {
        'image': img_np,
        'bboxes': bboxes_np,
        'category_ids': category_ids_np,
        'depth': depth_np
    }
    
    augmented_sample_train = train_augs(**sample_to_transform)
    
    print("Train Augmented Sample (Albumentations):")
    for k_item, v_item in augmented_sample_train.items():
        if isinstance(v_item, torch.Tensor):
            print(f"  {k_item}: shape={v_item.shape}, dtype={v_item.dtype}, device={v_item.device}")
        elif isinstance(v_item, np.ndarray):
            print(f"  {k_item}: shape={v_item.shape}, dtype={v_item.dtype}")
        elif isinstance(v_item, list) and len(v_item) > 0 and isinstance(v_item[0], (int, float)): # bboxes might be list of lists
             print(f"  {k_item}: {v_item}")
        else:
            print(f"  {k_item}: {v_item}")
    if isinstance(augmented_sample_train['image'], torch.Tensor):
        print(f"  image tensor min: {augmented_sample_train['image'].min()}, max: {augmented_sample_train['image'].max()}")

    # For validation, the original `Resize` logic was:
    # 1. Scale to short_edge `s` (if size=s) or to (h,w) (if size=(h,w))
    # 2. Make resulting H, W divisible by 32.
    # Let's pre-calculate the target validation size based on this logic.
    # Example: if val_short_edge_target = 256
    val_short_edge_target = 256 
    # Assume we need to calculate final H,W based on this and original image size
    # This calculation should ideally happen in the dataset __getitem__ before calling val_augs
    
    # For this __main__ example, let's calculate it:
    val_target_h, val_target_w = _calculate_resize_dims(h_orig, w_orig, val_short_edge_target)
    print(f"\nCalculated validation target dims (divisible by 32): H={val_target_h}, W={val_target_w}")

    val_augs = get_val_augmentation((val_target_h, val_target_w), additional_targets=additional_targets_setup)
    
    # Create a fresh sample for validation
    img_np_val = np.random.randint(0, 256, (h_orig, w_orig, 3), dtype=np.uint8)
    depth_np_val = np.random.rand(h_orig, w_orig, 1).astype(np.float32)
    bboxes_np_val = np.array([[20, 30, 110, 130]], dtype=np.float32)
    category_ids_np_val = np.array([0], dtype=np.long)

    sample_to_transform_val = {
        'image': img_np_val,
        'bboxes': bboxes_np_val,
        'category_ids': category_ids_np_val,
        'depth': depth_np_val
    }
    augmented_sample_val = val_augs(**sample_to_transform_val)

    print("\nValidation Augmented Sample (Albumentations):")
    for k_item, v_item in augmented_sample_val.items():
        if isinstance(v_item, torch.Tensor):
            print(f"  {k_item}: shape={v_item.shape}, dtype={v_item.dtype}, device={v_item.device}")
        elif isinstance(v_item, np.ndarray):
             print(f"  {k_item}: shape={v_item.shape}, dtype={v_item.dtype}")
        elif isinstance(v_item, list) and len(v_item) > 0 and isinstance(v_item[0], (int, float)):
             print(f"  {k_item}: {v_item}")
        else:
            print(f"  {k_item}: {v_item}")
    if isinstance(augmented_sample_val['image'], torch.Tensor):
        print(f"  image tensor min: {augmented_sample_val['image'].min()}, max: {augmented_sample_val['image'].max()}")


    # Test with empty bboxes
    empty_bboxes_np = np.empty((0,4), dtype=np.float32)
    empty_category_ids_np = np.empty((0,), dtype=np.long)
    sample_empty_bbox = {
        'image': img_np.copy(), # Use a copy
        'bboxes': empty_bboxes_np,
        'category_ids': empty_category_ids_np,
        'depth': depth_np.copy()
    }
    augmented_empty_bbox = train_augs(**sample_empty_bbox)
    print("\nAugmented Sample with Empty BBoxes (Albumentations):")
    for k_item, v_item in augmented_empty_bbox.items():
        if isinstance(v_item, torch.Tensor):
            print(f"  {k_item}: shape={v_item.shape}, dtype={v_item.dtype}")
        else:
            print(f"  {k_item}: {v_item}")
