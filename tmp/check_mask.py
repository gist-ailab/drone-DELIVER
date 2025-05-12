import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

img_pth = '/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-DELIVER/data/DELIVER/semantic/cloud/train/MAP_1_point102/110350_semantic_front.png'

mask = cv2.imread(img_pth)
mask = mask[: ,:,2]
plt.imshow(mask)
plt.show()