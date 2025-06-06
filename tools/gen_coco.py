import json
from pycocotools.coco import COCO
import os
import shutil
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
"""
num_classes: 25
"""

CLASSES = ["background", "Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road", "SideWalk", "Vegetation", 
            "Cars", "Wall", "TrafficSign", "Sky", "Ground", "Bridge", "RailTrack", "GroundRail", 
            "TrafficLight", "Static", "Dynamic", "Water", "Terrain", "TwoWheeler", "Bus", "Truck"]

PALETTE = np.array([[70, 70, 70],
        [100, 40, 40],
        [55, 90, 80],
        [220, 20, 60],
        [153, 153, 153],
        [157, 234, 50],
        [128, 64, 128],
        [244, 35, 232],
        [107, 142, 35],
        [0, 0, 142],
        [102, 102, 156],
        [220, 220, 0],
        [70, 130, 180],
        [81, 0, 81],
        [150, 100, 100],
        [230, 150, 140],
        [180, 165, 180],
        [250, 170, 30],
        [110, 190, 160],
        [170, 120, 50],
        [45, 60, 150],
        [145, 170, 100],
        [  0,  0, 230], 
        [  0, 60, 100],
        [  0,  0, 70],
        ])

def mask_process(img_pth):
    mask_pth  = img_pth.replace('rgb_', 'semantic_').replace('img', 'semantic')
    mask = cv2.imread(mask_pth)
    boxes_4 = extract_boxes(mask, None, 4)
    boxes_10 = extract_boxes(mask, None, 10)
    boxes_23 = extract_boxes(mask, None, 23)
    boxes_24 = extract_boxes(mask, None, 24)
    boxes_25 = extract_boxes(mask, None, 25)
    box_dict={
        4: boxes_4,
        10: boxes_10,
        23: boxes_23,
        24: boxes_24,
        25: boxes_25
    }
    return box_dict, 


def extract_boxes(mask, depth, id):
    binary_mask = mask[:,:, 2]
    empty = np.zeros_like(binary_mask)
    empty[binary_mask == id] = 255
    contours, _ = cv2.findContours(empty, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes =[]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, x + w, y + h])
    return boxes


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return image

def gen_img_list(root_dir, mode=['train', 'val', 'test']):
    '''
    returns a list of relative paths to the images from the root_dir
    '''
    img_dir = os.path.join(root_dir,'img', "*", mode)
    img_lists =glob.glob(os.path.join(img_dir, "**/*.png"), recursive=True)
    img_lists = [os.path.relpath(img, root_dir) for img in img_lists]
    return img_lists

def create_coco_json(root_dir, img_lists, mode, box_type='xywh'):
    '''
    creates a coco json file from the image lists
    '''
    coco_json = {
        "info": {
            "description": "Detection Deliver Dataset",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "maengjemo",
            "date_created": "2025-05-01"}
    }

    coco_json["categories"] =[
        {
            "id": 1,
            "name": "Human",
            "supercategory": "none",
            "color": [100, 40, 40]
        },
        {
            "id" : 2,
            "name" : "Vehicle",
            "supercategory": "none",
            "color": [110, 190, 160]
        } ]

    coco_images = []
    coco_annotations = []
    img_paths = gen_img_list(root_dir, mode)
    img_id = 1
    annotation_id = 1
    example_img = cv2.imread(os.path.join(root_dir, img_paths[0]))
    height, width, _ = example_img.shape

    for img_path in tqdm(img_paths):
        img = cv2.imread(os.path.join(root_dir, img_path))
        height, width, _ = img.shape
        coco_images.append({
            "id": int(img_id),
            "width": width,
            "height": height,
            "file_name": img_path,
            "depth_path" : img_path.replace('img', 'depth').replace('rgb_', 'depth_'),
            "lidar_path" : img_path.replace('img', 'lidar').replace('rgb_', 'lidar_'),
            "event_path" : img_path.replace('img', 'event').replace('rgb_', 'event_'),
        })
        
        calculated_boxes= mask_process(os.path.join(root_dir, img_path))            
        for idx in range(len(calculated_boxes[0].keys())):
            cat_id_source = list(calculated_boxes[0].keys())[idx]
            boxes = calculated_boxes[0][cat_id_source]
            if cat_id_source == 4:
                cat_id_edited = 1
            elif cat_id_source == 10 or cat_id_source == 23 or cat_id_source == 24 or cat_id_source == 25:
                cat_id_edited = 2
            for box in boxes:
                w = box[2] - box[0]
                h = box[3] - box[1]
                
                if w * h < 144:      #Filter small boxes under 12 x 12
                    continue
                if box_type == 'xywh':
                    box = [box[0], box[1], w, h]
                elif box_type == 'xyxy':
                    box = [box[0], box[1], box[2], box[3]]

                coco_annotations.append({
                    "id": int(annotation_id),
                    "image_id": int(img_id),
                    "category_id": cat_id_edited,
                    "bbox": box,
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [],
                    "keypoints": [],
                    "num_keypoints": 0,
                    "iscrowd": 0
                })
                annotation_id += 1
        img_id +=1
    
    coco_json["images"] = coco_images
    coco_json["annotations"] = coco_annotations
    return coco_json

def save_coco_json(coco_json, output_dir, mode):
    '''
    saves the coco json file to the output directory
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"coco_{mode}.json")
    with open(output_path, 'w') as f:
        json.dump(coco_json, f)


def main():
    parser = argparse.ArgumentParser(description='Generate COCO JSON file')
    parser.add_argument('--root_dir', type=str, default = '/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-DELIVER/data/DELIVER', help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default='/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-DELIVER/data/DELIVER', help='Output directory for the COCO JSON file')
    parser.add_argument('--mode', type=str, default='train', help='Mode of the dataset (train, val, test)')
    parser.add_argument('--root_dir', type=str, default = '/media/jemo/HDD1/Workspace/dset/DELIVER', help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default='/media/jemo/HDD1/Workspace/dset/DELIVER', help='Output directory for the COCO JSON file')
    parser.add_argument('--mode', type=str, default='test', help='Mode of the dataset (train, val, test)')
    args = parser.parse_args()
    root_dir = args.root_dir
    output_dir = args.output_dir
    mode = args.mode
    img_lists = gen_img_list(root_dir, mode)
    coco_json = create_coco_json(root_dir, img_lists, mode, box_type='xyxy')
    save_coco_json(coco_json, output_dir, mode)


if __name__ == '__main__':
    main()






