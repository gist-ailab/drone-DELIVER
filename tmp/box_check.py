

import cv2
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

path ="/media/jemo/HDD1/Workspace/dset/DELIVER/img/night/train/MAP_3_point9/341800_rgb_front.png"

rgb = cv2.imread(path)
lidar = cv2.imread(path.replace('img', 'lidar').replace('_rgb', '_lidar'))
depth = cv2.imread(path.replace('img', 'hha').replace('_rgb', '_depth'))
event = cv2.imread(path.replace('img', 'event').replace('_rgb', '_event'))

# 주어진 바운딩 박스 리스트
annotations = [
    {"id": 10419, "image_id": 2152, "category_id": 1, "bbox": [457, 530, 465, 553]},
    {"id": 10420, "image_id": 2152, "category_id": 1, "bbox": [177, 530, 187, 552]},
    {"id": 10421, "image_id": 2152, "category_id": 1, "bbox": [440, 529, 453, 557]},
    {"id": 10422, "image_id": 2152, "category_id": 1, "bbox": [416, 529, 431, 551]},
    {"id": 10423, "image_id": 2152, "category_id": 2, "bbox": [340, 541, 382, 576]},
    {"id": 10424, "image_id": 2152, "category_id": 2, "bbox": [57, 538, 72, 557]},
    {"id": 10425, "image_id": 2152, "category_id": 2, "bbox": [375, 531, 416, 545]},
    {"id": 10426, "image_id": 2152, "category_id": 2, "bbox": [521, 530, 568, 545]},
    {"id": 10427, "image_id": 2152, "category_id": 2, "bbox": [461, 526, 492, 545]},
    {"id": 10428, "image_id": 2152, "category_id": 2, "bbox": [287, 522, 313, 533]},
    {"id": 10429, "image_id": 2152, "category_id": 2, "bbox": [0, 521, 66, 561]}
]


# 바운딩 박스와 클래스 라벨 분리
boxes = []
class_labels = []
for ann in annotations:
    x1, y1, x2, y2 = map(int, ann["bbox"])
    boxes.append([x1, y1, x2, y2])
    class_labels.append(ann["category_id"])  # <- 반드시 label_fields에 명시한 이름과 동일해야 함


additional_targets = {
    'depth': 'image',
    'lidar': 'image',
    'event': 'image'
}

# 변환 적용 시 추가 타겟도 전달
train_transform = A.Compose([
    A.RandomCrop(width=800, height=800, p=1.0),  # 예시용 crop
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['class_labels']  # <- 여기와 동일한 이름의 변수 필요
), additional_targets=additional_targets)


print("class labels", class_labels) 

# 변환 적용 시 class_labels도 전달
augmented = train_transform(
    image=rgb,
    depth=depth,
    lidar=lidar,
    event=event,
    bboxes=boxes,
    class_labels=class_labels
)

def post_process(a):
    # tensor = tensor[0]
    a = a.numpy().transpose(1,2,0)
    a = np.abs(a)*100
    a= a.astype(np.uint8)
    return a


augmented_image = augmented['image']
augmented_depth = augmented['depth']
augmented_lidar = augmented['lidar']
augmented_event = augmented['event']
augmented_boxes = augmented['bboxes']

augmented_image = post_process(augmented_image)
augmented_depth = post_process(augmented_depth)
augmented_lidar = post_process(augmented_lidar)
augmented_event = post_process(augmented_event)


revert_list = []
for i in return_list:
    tmp = post_process(i)
    revert_list.append(tmp)


for box in target['boxes']:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(revert_list[0], (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.rectangle(revert_list[1], (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.rectangle(revert_list[2], (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.rectangle(revert_list[3], (x1, y1), (x2, y2), (255, 0, 0), 2)



# 박스 그리기
for box in augmented_boxes:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(augmented_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.rectangle(augmented_depth, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.rectangle(augmented_lidar, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.rectangle(augmented_event, (x1, y1), (x2, y2), (255, 255, 0), 2)

cv2.imwrite('augmented_image3.png', cv2.hconcat([augmented_image, augmented_depth, augmented_lidar, augmented_event]))

# 박스 그리기
for ann in annotations:
    x1, y1, x2, y2 = map(int, ann["bbox"])
    color = (255, 255, 0) if ann["category_id"] == 1 else (0, 255, 0)
    rgb = cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
cv2.imwrite('box_example.png', rgb)


'''
color=(255,0,0)
for box in boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(b, (x1, y1), (x2, y2), color, 2)
'''