# train_detection_mm.py
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json

# Detection specific imports
from semseg.models.cmnext_detection import CMNeXtFasterRCNN # Adjusted import
from semseg.datasets.deliver_detection import DELIVERCOCO # Adjusted import

from semseg.augmentations_detection_mm2 import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss # Loss is handled by FasterRCNN
from semseg.schedulers import get_scheduler # Re-evaluate if needed for detection
from semseg.optimizers import get_optimizer # Re-evaluate if needed for detection
from semseg.schedulers import get_scheduler
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops # print_iou removed
from utils.eval_utils import evaluate_detection # Adjusted import for detection evaluation
from utils.train_utils import get_model_from_config



def main(cfg, gpu, save_dir):
    start = time.time()
    best_mAP = 0.0 # Changed from best_mIoU to best_mAP
    best_epoch = 0
    num_workers = cfg['TRAIN'].get('NUM_WORKERS', 8) # Get from cfg or default
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    optim_cfg, sched_cfg = cfg['OPTIMIZER'], cfg['SCHEDULER'] # Loss_cfg removed
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    resume_path = cfg['MODEL'].get('RESUME', None) # Use .get for safer access
    gpus = int(os.environ.get('WORLD_SIZE', 1))

    active_modals = dataset_cfg['MODALS']    
    additional_targets_setup = {}
    if 'depth' in active_modals:
        additional_targets_setup['depth'] = 'image'
    if 'event' in active_modals:
        additional_targets_setup['event'] = 'image'
    if 'lidar' in active_modals:
        additional_targets_setup['lidar'] = 'image'
    train_augs = get_train_augmentation(train_cfg['IMAGE_SIZE'], additional_targets=additional_targets_setup)
    val_augs = get_val_augmentation(eval_cfg['IMAGE_SIZE'], additional_targets=additional_targets_setup)


    train_json= dataset_cfg['TRAIN_JSON']
    val_json= dataset_cfg['VAL_JSON']
    
    trainset = DELIVERCOCO(root=dataset_cfg['ROOT'], ann_path = train_json, transform=train_augs, modals=dataset_cfg['MODALS'], target_img_size = train_cfg['IMAGE_SIZE'])
    valset = DELIVERCOCO(root=dataset_cfg['ROOT'], ann_path=val_json, transform=val_augs, modals=dataset_cfg['MODALS'],target_img_size = eval_cfg['IMAGE_SIZE'])
    
    
    # Path to the ground truth COCO annotation file for validation
    coco_val_gt_path = os.path.join(dataset_cfg['ROOT'], 'coco_val.json')

    # num_classes for detection model should come from dataset_cfg or be explicit
    num_detection_classes = len(trainset.CLASSES) 
    class_names = trainset.CLASSES # For logging/evaluation

    # Model
    # CMNeXtFasterRCNN expects backbone_name, num_classes, modals
    # model = CMNeXtFasterRCNN(backbone_name=model_cfg['BACKBONE'], num_classes=num_detection_classes, modals=dataset_cfg['MODALS'])
    model = get_model_from_config(cfg, num_detection_classes)

    
    resume_checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        resume_checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        msg = model.load_state_dict(resume_checkpoint['model_state_dict'])
        logger.info(f"Resumed model from: {resume_path}, msg: {msg}")
    else:
        if model_cfg.get('PRETRAINED'):
            logger.info(f"Attempting to load pretrained weights for backbone from: {model_cfg['PRETRAINED']}")
    model = model.to(device) 
    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE'] // gpus
    start_epoch = 0
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    # Scheduler might need adjustment based on detection training practices
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, int((epochs+1)*iters_per_epoch), sched_cfg.get('POWER', 0.9), 
                              iters_per_epoch * sched_cfg.get('WARMUP_EPOCHS', 0), sched_cfg.get('WARMUP_RATIO', 0.1))


    if train_cfg.get('DDP', False):
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = DistributedSampler(valset, dist.get_world_size(), dist.get_rank(), shuffle=False) # Usually not shuffled for val
        model = DDP(model, device_ids=[gpu], find_unused_parameters=model_cfg.get('DDP_FIND_UNUSED_PARAMS', True)) # find_unused_parameters might be needed
    else:
        sampler = RandomSampler(trainset)
        sampler_val = RandomSampler(valset) # Or SequentialSampler for validation
    
    if resume_checkpoint:
        start_epoch = resume_checkpoint.get('epoch', 0) # Use .get for safety
        if 'optimizer_state_dict' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        loss = resume_checkpoint.get('loss', 0) # Loss is a dict for FasterRCNN
        best_mAP = resume_checkpoint.get('best_mAP', 0.0) # Changed to best_mAP
        logger.info(f"Resumed training from epoch {start_epoch}, best_mAP: {best_mAP}")

    def detection_collate_fn(batch):
        """
        batch: list[ tuple(modal_tensors_list, target_dict) ]

        - 빈 타겟(박스 0개) 이미지도 그대로 유지
        - modal 별 tensor shape: (C, H, W)
        - 최종 return:
            inputs_transposed : list[Tensor]  # len = #modal, shape (B, C, H, W)
            targets           : list[Dict]    # len = B
        """
        # ❶ DataLoader 가 drop_last=True 이므로 마지막 미니배치가 작아도 안전
        inputs, targets = zip(*batch)                         # tuple → tuple

        # ❷ modal 차원 먼저 transpose 해서 모달별 스택 (ex. [rgb_list, depth_list, …])
        inputs_transposed = [torch.stack(mod_list, dim=0)     # (B, C, H, W)
                            for mod_list in zip(*inputs)]

        return inputs_transposed, list(targets)

    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=sampler, collate_fn=detection_collate_fn)
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=True, sampler=sampler_val, collate_fn=detection_collate_fn)
    scaler = GradScaler(enabled=train_cfg.get('AMP', False))
    
    if (not train_cfg.get('DDP', False) or torch.distributed.get_rank() == 0):
        writer = SummaryWriter(str(save_dir))
        logger.info('================== model structure =====================')
        logger.info(model) # This might be very verbose for FasterRCNN
        logger.info('================== training config =====================')
        logger.info(cfg)

    for epoch in range(start_epoch, epochs):
        model.train()
        if train_cfg.get('DDP', False): sampler.set_epoch(epoch)
        train_total_loss = 0.0
        # Individual losses (optional, for more detailed logging)        
        current_lr = optimizer.param_groups[0]['lr']
        pbar_desc = f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {current_lr:.8f} Loss: {0:.8f}"
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=pbar_desc, disable=(train_cfg.get('DDP', False) and torch.distributed.get_rank() != 0))

        for iter_num, (sample, targets) in pbar:
            sample = [img.to(device) for img in sample]                                                                      # images are already stacked by collate_fn
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]       # targets is a list of dicts, each dict's tensors need to be moved to device
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=train_cfg.get('AMP', False)):
                # FasterRCNN returns a dict of losses during training when targets are provided
                loss_dict = model(sample, targets)
                loss = sum(l for l in loss_dict.values())

            scaler.scale(loss).backward()
            
            if train_cfg.get('CLIP_GRAD_NORM', 0) > 0:      # Gradient clipping (optional, but often useful for detection)          
                scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['CLIP_GRAD_NORM'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() # Step scheduler
            current_lr = optimizer.param_groups[0]['lr'] # Get current LR after scheduler step
            train_total_loss += loss.item()
            loss_items_str = ' | '.join([f"{k}:{v.item():.4f}" for k, v in loss_dict.items()])
            pbar.set_description(
                f"Epoch: [{epoch+1}/{epochs}] | Iter: [{iter_num+1}/{iters_per_epoch}] | LR: {current_lr:.6f} | {loss_items_str}"
            )
        
        train_total_loss /= (iter_num + 1) # Average loss over iterations
        if (not train_cfg.get('DDP', False) or torch.distributed.get_rank() == 0):
            writer.add_scalar('train/total_loss', train_total_loss, epoch)
            # writer.add_scalar('train/lr', current_lr, epoch)
        torch.cuda.empty_cache()

        if (epoch + 1) % 10 == 0 and (not train_cfg.get('DDP', False) or dist.get_rank() == 0):
            ckpt_name = f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}"\
                        f"_epoch{epoch+1:03d}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if train_cfg.get('DDP', False) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mAP': best_mAP,
            }, save_dir / ckpt_name)
            logger.info(f"Epoch {epoch+1}: periodic checkpoint saved → {ckpt_name}")

        # Evaluation
        if ((epoch+1) % train_cfg.get('EVAL_INTERVAL', 1) == 0) or (epoch+1) == epochs:
            # if (not train_cfg.get('DDP', False) or torch.distributed.get_rank() == 0):
            current_mAP, all_coco_stats = evaluate_detection(model.module if train_cfg.get('DDP', False) else model, 
                                                                valloader, 
                                                                device, 
                                                                coco_val_gt_path,
                                                                logger) # Pass the actual model and logger
            writer.add_scalar('val/mAP', current_mAP, epoch)
            for stat_name, stat_val in all_coco_stats.items():
                writer.add_scalar(f'val_coco_stats/{stat_name.replace(" ", "_")}', stat_val, epoch)
            if current_mAP > best_mAP:
                # Clean up previous best checkpoint names
                prev_best_ckp_name = f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mAP:.4f}_checkpoint.pth"
                prev_best_name = f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mAP:.4f}.pth"
                prev_best_ckp_path = save_dir / prev_best_ckp_name
                prev_best_path = save_dir / prev_best_name
                if os.path.isfile(prev_best_path): os.remove(prev_best_path)
                if os.path.isfile(prev_best_ckp_path): os.remove(prev_best_ckp_path)
                best_mAP = current_mAP
                best_epoch = epoch+1
                cur_best_ckp_name = f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mAP:.4f}_checkpoint.pth"
                cur_best_name = f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mAP:.4f}.pth"
                torch.save(model.module.state_dict() if train_cfg.get('DDP', False) else model.state_dict(), save_dir / cur_best_name)
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.module.state_dict() if train_cfg.get('DDP', False) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mAP': best_mAP, # Changed to best_mAP
                    # 'loss_dict': loss_dict # Can save last loss_dict if needed
                }, save_dir / cur_best_ckp_name)
                logger.info(f"Epoch {epoch+1}: New best model saved with mAP: {best_mAP:.4f}")
            logger.info(f"Current epoch:{epoch+1} mAP: {current_mAP:.4f} Best mAP: {best_mAP:.4f}") # Changed "Placeholder mAP" to "mAP"

    if (not train_cfg.get('DDP', False) or torch.distributed.get_rank() == 0):
        writer.close()
    pbar.close()
    end_time_struct = time.gmtime(time.time() - start)

    table = [
        ['Best mAP', f"{best_mAP:.4f}"], # Changed to mAP
        ['Total Training Time', time.strftime("%H:%M:%S", end_time_struct)]
    ]
    if (not train_cfg.get('DDP', False) or torch.distributed.get_rank() == 0): # Ensure logger is used by rank 0 only in DDP
        logger.info(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Default config might need to be a detection-specific one
    parser.add_argument('--cfg', type=str, default='configs/levine-deliver_detection_rgbdl_retinanet.yaml', help='Configuration file to use') # Example new config name
    parser.add_argument('--gpu_ids', type=str, default='4', help='GPU IDs to use (comma-separated)')
    args = parser.parse_args()


    # GPU setup
    gpu_ids = args.gpu_ids.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
    os.environ['PYTHONHASHSEED'] = str(3407) # Set hash seed for reproducibility
    os.environ['OMP_NUM_THREADS'] = '1' # Set OMP threads to 1 for reproducibility

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(cfg.get('SEED', 3407)) # Get seed from cfg or default
    setup_cudnn()
    
    # DDP setup
    if cfg['TRAIN'].get('DDP', False):
        gpu = setup_ddp()
    else:
        gpu = cfg.get('GPU_ID', 0) # Use a single GPU if DDP is false
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


    modals_str = ''.join([m[0] for m in cfg['DATASET']['MODALS']]) if 'MODALS' in cfg['DATASET'] else 'rgb'
    # Construct experiment name
    exp_name_parts = [
        cfg['DATASET']['NAME'],
        cfg['MODEL']['BACKBONE'],
        modals_str
    ]
    # Add a tag if specified in config
    if cfg.get('TAG'):
        exp_name_parts.append(cfg['TAG'])
    exp_name = '_'.join(exp_name_parts)
    save_dir_base = cfg.get('SAVE_DIR_BASE', 'output_detection') # Base directory for detection outputs
    save_dir = Path(save_dir_base) / exp_name
    
    # Handle resume path for save_dir
    resume_path_cfg = cfg['MODEL'].get('RESUME', None)
    if resume_path_cfg and os.path.isfile(resume_path_cfg):
        # If resuming, save to the same directory as the checkpoint
        save_dir = Path(os.path.dirname(resume_path_cfg))
    os.makedirs(save_dir, exist_ok=True)
    
    # Logger setup: ensure logger is created only by rank 0 in DDP
    if not cfg['TRAIN'].get('DDP', False) or torch.distributed.get_rank() == 0:
        logger = get_logger(save_dir / 'train_detection.log') # Changed log file name
        logger.info(f"Saving results to {save_dir}")
    else: # For other DDP processes, create a dummy logger or disable logging
        class DummyLogger:
            def info(self, msg): pass
            def warning(self, msg): pass
            def error(self, msg): pass
        logger = DummyLogger()

    try:
        main(cfg, gpu, save_dir)
    except Exception as e:
        if not cfg['TRAIN'].get('DDP', False) or torch.distributed.get_rank() == 0: # Log exception only on rank 0
            logger.error(f"Training failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        raise e # Re-raise the exception
    finally:
        if cfg['TRAIN'].get('DDP', False):
            cleanup_ddp()
