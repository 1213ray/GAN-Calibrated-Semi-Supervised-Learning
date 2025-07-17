#!/usr/bin/env python3
"""
Enhanced CGAN training script with improved loss functions and stability
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import numpy as np
import wandb

# Setup path
project_root = Path(__file__).parent

from models import GeneratorUNet, GeneratorSimpleRegressor, Discriminator, weights_init_normal
from dataset import CalibratorDataset
from losses import HybridLoss, apply_delta_to_bbox, iou_metric

def get_generator(generator_type, delta_scale):
    """根據配置選擇生成器類型"""
    if generator_type == "simple":
        return GeneratorSimpleRegressor(delta_scale=delta_scale)
    else:
        return GeneratorUNet(delta_scale=delta_scale)

def get_refined_patch_batch(original_image_paths, pred_bboxes, deltas_pred, img_size, device, fallback_patches=None):
    """Generate refined patches using predicted deltas with proper fallback"""
    refined_patches = []
    from torchvision import transforms
    from PIL import Image, ImageOps
    
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    fallback_count = 0
    
    for i in range(len(original_image_paths)):
        img_path = original_image_paths[i]
        pred_box = pred_bboxes[i]
        delta = deltas_pred[i]
        
        # 限制delta範圍防止數值爆炸 (使用推論模式以確保穩定性)
        refined_box = apply_delta_to_bbox(pred_box.unsqueeze(0).cpu(), delta.unsqueeze(0).cpu(), training=False).squeeze(0)

        try:
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            cx, cy, w, h = refined_box
            
            # 確保邊界框在合理範圍內
            cx = max(0.1, min(0.9, float(cx)))
            cy = max(0.1, min(0.9, float(cy)))
            w = max(0.05, min(0.8, float(w)))
            h = max(0.05, min(0.8, float(h)))
            
            px, py, pw, ph = cx * W, cy * H, w * W, h * H
            x1, y1 = max(0, px - pw / 2), max(0, py - ph / 2)
            x2, y2 = min(W, px + pw / 2), min(H, py + ph / 2)
            
            # 檢查邊界框是否有效
            if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                # 使用原始pred_box作為fallback
                orig_cx, orig_cy, orig_w, orig_h = pred_box.cpu()
                orig_px, orig_py, orig_pw, orig_ph = float(orig_cx) * W, float(orig_cy) * H, float(orig_w) * W, float(orig_h) * H
                orig_x1, orig_y1 = max(0, orig_px - orig_pw / 2), max(0, orig_py - orig_ph / 2)
                orig_x2, orig_y2 = min(W, orig_px + orig_pw / 2), min(H, orig_py + orig_ph / 2)
                crop = img.crop((int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)))
                fallback_count += 1
            else:
                crop = img.crop((int(x1), int(y1), int(x2), int(y2)))

            # 標準化patch處理
            pad_w = max(crop.height - crop.width, 0)
            pad_h = max(crop.width - crop.height, 0)
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            crop_square = ImageOps.expand(crop, padding, fill=(128, 128, 128))
            crop_resized = crop_square.resize((img_size, img_size), Image.BICUBIC)
            
            refined_patches.append(transform_to_tensor(crop_resized))
            
        except Exception as e:
            # 使用原始patch作為fallback而不是零張量
            if fallback_patches is not None:
                refined_patches.append(fallback_patches[i])
            else:
                # 創建一個中性的patch
                neutral_patch = torch.ones(3, img_size, img_size) * 0.5
                refined_patches.append(neutral_patch)
            fallback_count += 1
    
    if fallback_count > 0:
        print(f"Warning: {fallback_count}/{len(original_image_paths)} patches used fallback")

    return torch.stack(refined_patches).to(device)

def main():
    # 載入配置檔案
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=config['data_dir'])
    parser.add_argument("--img_size", type=int, default=config['img_size'])
    parser.add_argument("--batch_size", type=int, default=config['batch_size'])
    parser.add_argument("--n_epochs", type=int, default=config['n_epochs'])
    parser.add_argument("--lr", type=float, default=config['lr'])
    parser.add_argument("--beta1", type=float, default=config['beta1'])
    parser.add_argument("--beta2", type=float, default=config['beta2'])
    parser.add_argument("--lambda_l1", type=float, default=config['lambda_l1'])
    parser.add_argument("--lambda_iou", type=float, default=config['lambda_iou'])
    parser.add_argument("--use_giou", action='store_true', default=config['use_giou'])
    parser.add_argument("--use_hybrid_loss", action='store_true', default=config['use_hybrid_loss'])
    parser.add_argument("--spectral_norm", action='store_true', default=config['spectral_norm'])
    parser.add_argument("--delta_scale", type=float, default=config['delta_scale'])
    parser.add_argument("--generator_type", type=str, default=config['generator_type'])
    parser.add_argument("--patience", type=int, default=config['early_stop']['patience'])
    parser.add_argument("--min_delta", type=float, default=config['early_stop']['min_delta'])
    parser.add_argument("--train_split", type=float, default=config['train_split'])
    parser.add_argument("--val_split", type=float, default=config['val_split'])
    parser.add_argument("--save_dir", type=str, default=config['save_dir'])
    parser.add_argument("--seed", type=int, default=config['seed'])
    parser.add_argument("--d_train_ratio", type=int, default=config['d_train_ratio'])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Generator type: {args.generator_type}")
    print(f"Using hybrid loss: {args.use_hybrid_loss}")
    print(f"Using GIoU: {args.use_giou}")

    # Initialize W&B
    wandb_config = config.get('wandb', {})
    if wandb_config.get('enabled', False):
        wandb.init(
            project=wandb_config.get('project', 'cgan-calibration'),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('name'),
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', ''),
            config={
                'img_size': args.img_size,
                'batch_size': args.batch_size,
                'n_epochs': args.n_epochs,
                'lr': args.lr,
                'beta1': args.beta1,
                'beta2': args.beta2,
                'lambda_l1': args.lambda_l1,
                'lambda_iou': args.lambda_iou,
                'use_giou': args.use_giou,
                'use_hybrid_loss': args.use_hybrid_loss,
                'spectral_norm': args.spectral_norm,
                'delta_scale': args.delta_scale,
                'generator_type': args.generator_type,
                'patience': args.patience,
                'min_delta': args.min_delta,
                'train_split': args.train_split,
                'val_split': args.val_split,
                'seed': args.seed,
                'd_train_ratio': args.d_train_ratio,
                'data_dir': args.data_dir,
                'device': str(device)
            }
        )
        print("W&B initialized successfully!")
    else:
        print("W&B disabled in config")

    # Load dataset
    full_dataset = CalibratorDataset(args.data_dir, img_size=args.img_size)
    val_len = max(1, int(args.val_split * len(full_dataset)))
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_set)}, Validation samples: {len(val_set)}")

    # Initialize models
    netG = get_generator(args.generator_type, args.delta_scale).to(device)
    netD = Discriminator(spectral_norm=args.spectral_norm).to(device)
    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)
    
    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")

    # Log model info to W&B
    if wandb_config.get('enabled', False):
        wandb.log({
            'model/generator_params': sum(p.numel() for p in netG.parameters()),
            'model/discriminator_params': sum(p.numel() for p in netD.parameters()),
            'dataset/train_samples': len(train_set),
            'dataset/val_samples': len(val_set),
            'dataset/total_samples': len(full_dataset)
        })
        # Watch models for gradient and parameter tracking
        wandb.watch(netG, log='all', log_freq=100)
        wandb.watch(netD, log='all', log_freq=100)

    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    
    if args.use_hybrid_loss:
        criterion_hybrid = HybridLoss(
            lambda_iou=args.lambda_iou,
            lambda_l1=args.lambda_l1,
            lambda_focal=0.5,
            use_giou=args.use_giou
        )
    else:
        criterion_L1 = nn.SmoothL1Loss()

    # Optimizers
    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='max', factor=0.5, patience=5)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='max', factor=0.5, patience=5)

    # Output directory
    out_root = Path(args.save_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "samples").mkdir(exist_ok=True)
    ckpt_best = out_root / "G_best.pth"

    best_val_iou, epochs_no_improve = -1.0, 0
    training_history = []
    
    print("Starting enhanced training...")
    for epoch in range(1, args.n_epochs + 1):
        netG.train()
        netD.train()
        
        epoch_stats = {
            'loss_G': 0.0,
            'loss_D': 0.0,
            'loss_iou': 0.0,
            'loss_l1': 0.0,
            'loss_gan': 0.0,
            'fallback_count': 0
        }
        
        for i, (pred_patch, gt_patch, delta_true, pred_box, img_path) in enumerate(train_loader):
            
            pred_patch = pred_patch.to(device)
            gt_patch = gt_patch.to(device)
            delta_true = delta_true.to(device)
            pred_box = pred_box.to(device)

            batch_size = pred_patch.size(0)
            
            # Labels for discriminator
            patch_size = netD(pred_patch, gt_patch).size()
            valid = torch.ones(patch_size, device=device) * 0.9  # Label smoothing
            fake = torch.zeros(patch_size, device=device) + 0.1  # Label smoothing

            # Train Discriminator
            if i % args.d_train_ratio == 0:
                optimizer_D.zero_grad()

                # Real loss
                pred_real = netD(pred_patch, gt_patch)
                loss_real = criterion_GAN(pred_real, valid)

                # Fake loss
                delta_pred_detached = netG(pred_patch).detach()
                refined_patch = get_refined_patch_batch(
                    img_path, pred_box, delta_pred_detached, args.img_size, device, fallback_patches=pred_patch
                )
                pred_fake = netD(pred_patch, refined_patch)
                loss_fake = criterion_GAN(pred_fake, fake)
                
                loss_D = (loss_real + loss_fake) * 0.5
                loss_D.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
                optimizer_D.step()
                epoch_stats['loss_D'] += loss_D.item()

            # Train Generator
            optimizer_G.zero_grad()
            
            delta_pred = netG(pred_patch)
            
            # Calculate calibrated boxes (使用訓練模式)
            calibrated_boxes = apply_delta_to_bbox(pred_box, delta_pred, training=True)
            gt_boxes = apply_delta_to_bbox(pred_box, delta_true, training=True)
            
            if args.use_hybrid_loss:
                # Use enhanced hybrid loss
                result = criterion_hybrid(
                    delta_pred, delta_true, calibrated_boxes, gt_boxes
                )
                if len(result) == 4:
                    loss_G_reg, loss_iou, loss_l1, loss_focal = result
                    epoch_stats['loss_focal'] = epoch_stats.get('loss_focal', 0) + loss_focal.item()
                else:
                    loss_G_reg, loss_iou, loss_l1 = result
                    epoch_stats['loss_focal'] = epoch_stats.get('loss_focal', 0)
                
                epoch_stats['loss_iou'] += loss_iou.item()
                epoch_stats['loss_l1'] += loss_l1.item()
            else:
                # Use traditional L1 loss
                loss_G_reg = criterion_L1(delta_pred, delta_true) * args.lambda_l1
                epoch_stats['loss_l1'] += loss_G_reg.item()
                epoch_stats['loss_focal'] = epoch_stats.get('loss_focal', 0)

            # Adversarial loss
            refined_patch_for_G = get_refined_patch_batch(
                img_path, pred_box, delta_pred, args.img_size, device, fallback_patches=pred_patch
            )
            pred_fake_for_G = netD(pred_patch, refined_patch_for_G)
            loss_GAN_G = criterion_GAN(pred_fake_for_G, valid)

            loss_G = loss_G_reg + loss_GAN_G
            loss_G.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optimizer_G.step()

            epoch_stats['loss_G'] += loss_G.item()
            epoch_stats['loss_gan'] += loss_GAN_G.item()

            # Save sample images occasionally
            if i == 0 and epoch % 10 == 1:
                try:
                    img_sample = torch.cat((pred_patch.data[:4], refined_patch_for_G.data[:4], gt_patch.data[:4]), -2)
                    sample_path = out_root / f"samples/epoch_{epoch}.png"
                    save_image(img_sample, sample_path, nrow=4, normalize=True)
                    
                    # Log sample images to W&B
                    if wandb_config.get('enabled', False):
                        wandb.log({
                            "samples/training_images": wandb.Image(
                                str(sample_path),
                                caption=f"Epoch {epoch}: Pred | Refined | GT"
                            )
                        })
                except Exception as e:
                    print(f"Warning: Could not save sample image: {e}")

        # Validation
        netG.eval()
        with torch.no_grad():
            iou_sum_before, iou_sum_after = 0.0, 0.0
            n_val_samples = 0
            for pred_patch_val, _, delta_true_val, pred_box_val, _ in val_loader:
                batch_size_val = pred_patch_val.size(0)
                pred_patch_val = pred_patch_val.to(device)
                delta_true_val = delta_true_val.to(device)
                pred_box_val = pred_box_val.to(device)
                
                delta_pred_val = netG(pred_patch_val)

                # 驗證時使用推論模式
                calibrated_boxes = apply_delta_to_bbox(pred_box_val, delta_pred_val, training=False)
                gt_boxes = apply_delta_to_bbox(pred_box_val, delta_true_val, training=False)

                iou_before = iou_metric(pred_box_val, gt_boxes)
                iou_after = iou_metric(calibrated_boxes, gt_boxes)

                iou_sum_before += iou_before.sum().item()
                iou_sum_after += iou_after.sum().item()
                n_val_samples += batch_size_val
            
            mean_iou_before = iou_sum_before / max(1, n_val_samples)
            mean_iou_after = iou_sum_after / max(1, n_val_samples)
            delta_iou = mean_iou_after - mean_iou_before

        # Calculate average losses
        for key in epoch_stats:
            if key == 'loss_D':
                epoch_stats[key] /= max(1, len(train_loader) // args.d_train_ratio)
            else:
                epoch_stats[key] /= len(train_loader)
        
        # Update learning rate
        scheduler_G.step(delta_iou)
        scheduler_D.step(delta_iou)
        
        # Store training history
        training_history.append({
            'epoch': epoch,
            'delta_iou': delta_iou,
            'mean_iou_before': mean_iou_before,
            'mean_iou_after': mean_iou_after,
            **epoch_stats
        })
        
        print(f"[Epoch {epoch}/{args.n_epochs}] "
              f"G: {epoch_stats['loss_G']:.3f} "
              f"D: {epoch_stats['loss_D']:.3f} "
              f"IoU: {epoch_stats['loss_iou']:.3f} "
              f"L1: {epoch_stats['loss_l1']:.3f} "
              f"GAN: {epoch_stats['loss_gan']:.3f} "
              f"ΔIoU: {delta_iou:.4f} "
              f"Before: {mean_iou_before:.4f} "
              f"After: {mean_iou_after:.4f}")

        # Log metrics to W&B
        if wandb_config.get('enabled', False):
            log_dict = {
                'epoch': epoch,
                'train/loss_G': epoch_stats['loss_G'],
                'train/loss_D': epoch_stats['loss_D'],
                'train/loss_iou': epoch_stats['loss_iou'],
                'train/loss_l1': epoch_stats['loss_l1'],
                'train/loss_gan': epoch_stats['loss_gan'],
                'val/delta_iou': delta_iou,
                'val/mean_iou_before': mean_iou_before,
                'val/mean_iou_after': mean_iou_after,
                'val/improvement': delta_iou,
                'learning_rate/generator': optimizer_G.param_groups[0]['lr'],
                'learning_rate/discriminator': optimizer_D.param_groups[0]['lr']
            }
            
            # Add focal loss if available
            if 'loss_focal' in epoch_stats:
                log_dict['train/loss_focal'] = epoch_stats['loss_focal']
            
            wandb.log(log_dict)
        
        # 檢查是否有數值問題
        loss_tensor = torch.tensor([epoch_stats['loss_G'], epoch_stats['loss_D']], dtype=torch.float32)
        if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
            print(f"Warning: NaN or Inf detected in losses! G: {epoch_stats['loss_G']:.6f}, D: {epoch_stats['loss_D']:.6f}")
            print("Stopping training to prevent further instability.")
            break

        # Early stopping with improved criteria
        if delta_iou > best_val_iou + args.min_delta:
            best_val_iou = delta_iou
            torch.save({
                'generator': netG.state_dict(),
                'discriminator': netD.state_dict(),
                'epoch': epoch,
                'delta_iou': delta_iou,
                'config': config
            }, ckpt_best)
            epochs_no_improve = 0
            print(f"New best model saved! Delta IoU = {best_val_iou:.4f}")
            
            # Log best model to W&B
            if wandb_config.get('enabled', False):
                wandb.log({
                    'best_model/delta_iou': best_val_iou,
                    'best_model/epoch': epoch
                })
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered.")
                break

    # Save training history
    import json
    with open(out_root / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"Training complete. Best Delta IoU = {best_val_iou:.4f}")
    print(f"Model saved to: {ckpt_best}")

    # Final W&B logging and artifact saving
    if wandb_config.get('enabled', False):
        # Log final metrics
        wandb.log({
            'final/best_delta_iou': best_val_iou,
            'final/total_epochs': epoch,
            'final/early_stopped': epochs_no_improve >= args.patience
        })
        
        # Save model artifacts
        model_artifact = wandb.Artifact(
            name=f"cgan_model_{wandb.run.id}",
            type="model",
            description="Best CGAN model for pseudo-label calibration"
        )
        model_artifact.add_file(str(ckpt_best))
        wandb.log_artifact(model_artifact)
        
        # Save training history
        history_artifact = wandb.Artifact(
            name=f"training_history_{wandb.run.id}",
            type="dataset",
            description="Training history and metrics"
        )
        history_artifact.add_file(str(out_root / "training_history.json"))
        wandb.log_artifact(history_artifact)
        
        # Save sample images
        if (out_root / "samples").exists():
            samples_artifact = wandb.Artifact(
                name=f"training_samples_{wandb.run.id}",
                type="dataset",
                description="Sample images during training"
            )
            samples_artifact.add_dir(str(out_root / "samples"))
            wandb.log_artifact(samples_artifact)
        
        print("W&B artifacts saved successfully!")
        wandb.finish()

if __name__ == "__main__":
    main()