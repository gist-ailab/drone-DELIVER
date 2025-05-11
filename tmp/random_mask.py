import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
# Path to the image
image_path = "/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/ros_bag_api/out_single/drone_2025-03-10-17-11-42_2/rgb_image_raw"
image_name = "frame_1741594308.744017.png"  # Replace with the actual image name
full_image_path = os.path.join(image_path, image_name)

# Load the image
image = cv2.imread(full_image_path)

if image is None:
    print(f"Failed to load image from {full_image_path}")
else:
    # Get image dimensions
    height, width, _ = image.shape

    # Define a random rectangle for masking
    x1 = np.random.randint(0, width // 2)
    y1 = np.random.randint(0, height // 2)
    x2 = np.random.randint(width // 2, width)
    y2 = np.random.randint(height // 2, height)

    # Apply the mask (black rectangle)
    masked_image = image.copy()
    # cv2.rectangle(masked_image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    # Generate Gaussian noise
    noise = np.random.normal(0, 25, (y2 - y1, x2 - x1, 3)).astype(np.uint8)

    # Add noise to the selected rectangle area
    masked_image[y1:y2, x1:x2] = cv2.add(masked_image[y1:y2, x1:x2], noise)

    # Save or display the result
    output_path = "/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-DELIVER/tmp/gaussian.jpg"
    cv2.imwrite(output_path, masked_image)
    print(f"Masked image saved to {output_path}")