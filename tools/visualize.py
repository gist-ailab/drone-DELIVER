import cv2
import json
import os
import numpy as np
from pycocotools.coco import COCO
import random

# === 설정 ===
IMAGE_DIR = "/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-DELIVER/data/DELIVER"              # 이미지 폴더 경로
ANNOTATION_FILE = "/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-DELIVER/data/DELIVER/coco_train.json"  # COCO json 경로

# === COCO 불러오기 ===
coco = COCO(ANNOTATION_FILE)
image_ids = coco.getImgIds()
category_ids = coco.getCatIds()
categories = coco.loadCats(category_ids)
category_id_to_name = {cat['id']: cat['name'] for cat in categories}

# 클래스별 색상 지정
category_colors = {cat['id']: (random.randint(0, 255),
                               random.randint(0, 255),
                               random.randint(0, 255)) for cat in categories}

# === 이미지 인덱스 관리 ===
current_index = 0

def draw_annotations(img, annotations):
    for ann in annotations:
        bbox = ann['bbox']
        cat_id = ann['category_id']
        color = category_colors[cat_id]
        class_name = category_id_to_name[cat_id]
        
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

while True:
    image_info = coco.loadImgs(image_ids[current_index])[0]
    image_path = os.path.join(IMAGE_DIR, image_info['file_name'])
    image = cv2.imread(image_path)
    
    ann_ids = coco.getAnnIds(imgIds=image_info['id'])
    anns = coco.loadAnns(ann_ids)
    
    vis_image = image.copy()
    vis_image = draw_annotations(vis_image, anns)
    
    display_text = f"[{current_index+1}/{len(image_ids)}] {image_info['file_name']}"
    cv2.putText(vis_image, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("COCO Viewer", vis_image)
    key = cv2.waitKey(0)

    # 방향키: ← (81), → (83), ESC (27)
    if key == 81 and current_index > 0:
        current_index -= 1
    elif key == 83 and current_index < len(image_ids) - 1:
        current_index += 1
    elif key == 27:
        break

cv2.destroyAllWindows()
