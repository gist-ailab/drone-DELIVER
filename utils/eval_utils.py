#evala_utils.py

import torch
import tqdm
import os
from pathlib import Path
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import cv2

# COCO Evaluation function
def evaluate_detection(model, dataloader, device, coco_gt_path, logger_instance):
    logger_instance.info("Starting COCO evaluation...")
    model.eval()
    coco_results = []
    img_ids_processed = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            sample = [img.to(device) for img in images]                                                                      # images are already stacked by collate_fn
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]       # targets is a list of dicts, each dict's tensors need to be moved to device
            outputs = model(sample) # Model in eval mode returns list of dicts (boxes, labels, scores)
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item() # Get original image_id
                img_ids_processed.append(image_id)
                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                for box, label, score in zip(boxes, labels, scores):
                    if score < 0.05:
                        continue        
                    x1, y1, x2, y2 = box
                    # coco_box = [x1, y1, x2 - x1, y2 - y1]
                    coco_box = [x1,y1,x2,y2] # pascal voc format
                    coco_results.append({
                        'image_id': image_id,
                        'category_id': int(label) +1,   # 0 based to 1 ased
                        'bbox': [round(float(c), 2) for c in coco_box],
                        'score': round(float(score), 3)
                    })

    if not coco_results:
        logger_instance.warning("No detections found to evaluate.")
        return 0.0, {} # mAP, all_stats
    logger_instance.info(f"Generated {len(coco_results)} detections for {len(set(img_ids_processed))} images.")
    # dt_results_path = 
    dt_results_path = "./output_detection/detection_results.json" # Temporary path for COCO results

    with open(dt_results_path, 'w') as f:
        json.dump(coco_results, f)
    temp_gt_path = None
    try:
        with open(coco_gt_path, 'r') as f:
            gt_data = json.load(f)
        coco_gt = COCO(coco_gt_path)
        coco_dt = coco_gt.loadRes(str(dt_results_path)) # Load detection results
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox') # Use 'bbox' for bounding box evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize() # This prints the summary to stdout by default
        
        # stats is a numpy array with 12 numbers (mAP [IoU=0.50:0.95], AP50, AP75, etc.)
        stats = coco_eval.stats
        mAP = stats[0] # AP @ IoU=0.50:0.95 | area=all | maxDets=100
        mAP50 = stats[1] # AP @ IoU=0.50
        all_stats_dict = {
            "mAP": mAP,
            "mAP@.50": mAP50,
            "mAP@.75": stats[2],
            "mAP (small)": stats[3],
            "mAP (medium)": stats[4],
            "mAP (large)": stats[5],
            "AR@1": stats[6],
            "AR@10": stats[7],
            "AR@100": stats[8],
            "AR (small)": stats[9],
            "AR (medium)": stats[10],
            "AR (large)": stats[11],
        }
        logger_instance.info(f"COCO Evaluation Summary: mAP={mAP:.4f}, mAP@.50={mAP50:.4f}")
        for k, v in all_stats_dict.items():
            logger_instance.info(f"  {k}: {v:.4f}")

    except Exception as e:
        logger_instance.error(f"Error during COCO evaluation: {e}")
        import traceback
        logger_instance.error(traceback.format_exc())
        mAP = 0.0
        all_stats_dict = {}
    finally:
        if os.path.exists(dt_results_path):
            os.remove(dt_results_path) # Clean up temporary detection results file
        if temp_gt_path and os.path.exists(temp_gt_path):
            os.remove(temp_gt_path) # Clean up temporary converted GT file

    vis_coco(coco_gt_path, dt_results_path, save_dir='/media/jemo/HDD1/Workspace/src/Project/Drone24/detection/drone-DELIVER/tmp/pred_check')
    return mAP, all_stats_dict




def vis_coco(gt_coco_path, pred_coco_path, save_dir ='/media/jemo/HDD1/Workspace/src/Project/Drone24/detection/drone-DELIVER/tmp/pred_check'):
    os.makedirs(save_dir, exist_ok=True)
    COLORS = [[0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
    coco_gt = COCO(gt_coco_path)
    img_dir = os.path.dirname(gt_coco_path)
    with open(pred_coco_path, 'r') as f:
        preds = json.load(f)
    image_ids = set([pred['image_id'] for pred in preds])
    for image_id in tqdm(image_ids):
        img_info = coco_gt.loadImgs(image_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
        pred_anns = [pred for pred in preds if pred['image_id'] == image_id]
        
        # Draw GT boxes
        for ann in gt_anns:
            bbox = ann['bbox']
            x1, y1, w, h = map(int, bbox)
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[0], 2)
        
        # Draw Pred boxes
        for ann in pred_anns:
            bbox = ann['bbox']
            x1, y1, w, h = map(int, bbox)
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[1], 2)
        
        save_path = os.path.join(save_dir, f"{image_id}.png")
        cv2.imwrite(save_path, img)
