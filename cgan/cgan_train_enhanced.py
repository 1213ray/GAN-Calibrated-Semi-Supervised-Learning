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
from losses import HybridLoss, apply_delta_to_bbox, iou_metric, compute_gradient_penalty

def get_generator(generator_type, delta_scale):
    """根據配置選擇生成器類型"""
    if generator_type == "simple":
        return GeneratorSimpleRegressor(delta_scale=delta_scale)
    else:
        return GeneratorUNet(delta_scale=delta_scale)

# 全域圖像緩存
_IMAGE_CACHE = {}
_CACHE_MAX_SIZE = 100  # 最多緩存100張圖像

def get_refined_patch_batch(original_image_paths, pred_bboxes, deltas_pred, img_size, device, fallback_patches=None):
    """
    Generate refined patches using predicted deltas with image caching for better performance
    添加了圖像緩存機制和性能優化
    """
    from torchvision import transforms
    from PIL import Image, ImageOps
    
    # 預定義變換，避免重複創建
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    refined_patches = []
    fallback_count = 0
    
    # 批量處理 delta 應用，提高效率
    pred_bboxes_cpu = pred_bboxes.cpu()
    deltas_pred_cpu = deltas_pred.cpu()
    refined_boxes = apply_delta_to_bbox(pred_bboxes_cpu, deltas_pred_cpu, training=False)
    
    for i in range(len(original_image_paths)):
        img_path = str(original_image_paths[i])
        pred_box = pred_bboxes_cpu[i]
        refined_box = refined_boxes[i]
        
        try:
            # 使用緩存機制加載圖像
            if img_path in _IMAGE_CACHE:
                img = _IMAGE_CACHE[img_path]
            else:
                img = Image.open(img_path).convert("RGB")
                # 簡單的LRU緩存實現
                if len(_IMAGE_CACHE) >= _CACHE_MAX_SIZE:
                    # 移除最舊的條目
                    oldest_key = next(iter(_IMAGE_CACHE))
                    del _IMAGE_CACHE[oldest_key]
                _IMAGE_CACHE[img_path] = img
            
            W, H = img.size
            cx, cy, w, h = refined_box
            
            # 邊界檢查和修正
            cx = float(torch.clamp(cx, 0.1, 0.9))
            cy = float(torch.clamp(cy, 0.1, 0.9))
            w = float(torch.clamp(w, 0.05, 0.8))
            h = float(torch.clamp(h, 0.05, 0.8))
            
            # 計算像素座標
            px, py, pw, ph = cx * W, cy * H, w * W, h * H
            x1, y1 = max(0, px - pw / 2), max(0, py - ph / 2)
            x2, y2 = min(W, px + pw / 2), min(H, py + ph / 2)
            
            # 檢查邊界框有效性
            if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                # 使用原始 pred_box 作為 fallback
                orig_cx, orig_cy, orig_w, orig_h = pred_box
                orig_px, orig_py = float(orig_cx) * W, float(orig_cy) * H
                orig_pw, orig_ph = float(orig_w) * W, float(orig_h) * H
                orig_x1, orig_y1 = max(0, orig_px - orig_pw / 2), max(0, orig_py - orig_ph / 2)
                orig_x2, orig_y2 = min(W, orig_px + orig_pw / 2), min(H, orig_py + orig_ph / 2)
                crop = img.crop((int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)))
                fallback_count += 1
            else:
                crop = img.crop((int(x1), int(y1), int(x2), int(y2)))

            # 優化的patch處理：使用更快的方法
            if crop.width != crop.height:
                # 只在需要時進行padding
                pad_w = max(crop.height - crop.width, 0)
                pad_h = max(crop.width - crop.height, 0)
                padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
                crop = ImageOps.expand(crop, padding, fill=(128, 128, 128))
            
            # 調整大小
            if crop.size != (img_size, img_size):
                crop = crop.resize((img_size, img_size), Image.BICUBIC)
            
            refined_patches.append(transform_to_tensor(crop))
            
        except Exception as e:
            # 錯誤處理：使用fallback
            if fallback_patches is not None:
                refined_patches.append(fallback_patches[i].cpu())
            else:
                # 創建中性patch
                neutral_patch = torch.zeros(3, img_size, img_size)
                refined_patches.append(neutral_patch)
            fallback_count += 1
    
    # 批量移動到設備，減少GPU傳輸次數
    if refined_patches:
        result = torch.stack(refined_patches).to(device, non_blocking=True)
    else:
        result = torch.zeros(len(original_image_paths), 3, img_size, img_size, device=device)
    
    if fallback_count > 0:
        print(f"Warning: {fallback_count}/{len(original_image_paths)} patches used fallback")

    return result

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
    parser.add_argument("--lambda_iou", type=float, default=config['lambda_iou'])
    # Pure EIoU configuration (simplified)
    parser.add_argument("--use_eiou", action='store_true', default=config.get('use_eiou', True))
    parser.add_argument("--pure_eiou", action='store_true', default=config.get('pure_eiou', True))
    parser.add_argument("--spectral_norm", action='store_true', default=config['spectral_norm'])
    parser.add_argument("--delta_scale", type=float, default=config['delta_scale'])
    parser.add_argument("--generator_type", type=str, default=config['generator_type'])
    parser.add_argument("--patience", type=int, default=config['early_stop']['patience'])
    parser.add_argument("--min_delta", type=float, default=config['early_stop']['min_delta'])
    parser.add_argument("--train_split", type=float, default=config['train_split'])
    parser.add_argument("--val_split", type=float, default=config['val_split'])
    parser.add_argument("--save_dir", type=str, default=config['save_dir'])
    parser.add_argument("--seed", type=int, default=config['seed'])
    parser.add_argument("--lambda_gp", type=float, default=config.get('lambda_gp', 10.0))
    parser.add_argument("--n_critic", type=int, default=config.get('n_critic', 5))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Generator type: {args.generator_type}")
    print(f"Using Pure EIoU Loss: {args.use_eiou}")
    print(f"Loss weight - EIoU: {args.lambda_iou}")
    print(f"Delta scale: {args.delta_scale}")
    print(f"WGAN-GP lambda: {args.lambda_gp}")
    print(f"N-Critic: {args.n_critic}")

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
                'lambda_iou': args.lambda_iou,
                # Removed complex constraint parameters
                'use_eiou': args.use_eiou,
                'use_pure_eiou': args.pure_eiou,
                'spectral_norm': args.spectral_norm,
                'delta_scale': args.delta_scale,
                'generator_type': args.generator_type,
                'patience': args.patience,
                'min_delta': args.min_delta,
                'train_split': args.train_split,
                'val_split': args.val_split,
                'seed': args.seed,
                'lambda_gp': args.lambda_gp,
                'n_critic': args.n_critic,
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

    # WGAN-GP 不需要 criterion_GAN，使用 Wasserstein 距離
    
    # Pure EIoU loss system
    criterion_hybrid = HybridLoss(lambda_iou=args.lambda_iou)

    # Optimizers - WGAN-GP 通常使用 Adam 或 RMSprop
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
        
        # Pure EIoU loss doesn't need epoch setting
        
        epoch_stats = {
            'loss_G': 0.0,
            'loss_D': 0.0,
            'loss_iou': 0.0,
            'loss_wgan': 0.0,
            'loss_gp': 0.0,
            'wasserstein_distance': 0.0
        }
        
        for i, (pred_patch, gt_patch, delta_true, pred_box, img_path) in enumerate(train_loader):
            
            pred_patch = pred_patch.to(device)
            gt_patch = gt_patch.to(device)
            delta_true = delta_true.to(device)
            pred_box = pred_box.to(device)

            batch_size = pred_patch.size(0)

            # ===== WGAN-GP 訓練循環：先訓練判別器 n_critic 次，再訓練生成器一次 =====
            
            # 1. 訓練判別器 (Critic) n_critic 次
            d_loss_epoch = 0.0
            gp_loss_epoch = 0.0
            wd_epoch = 0.0
            
            for critic_step in range(args.n_critic):
                optimizer_D.zero_grad()

                # 真實樣本 (pred_patch + gt_patch)
                real_validity = netD(pred_patch, gt_patch)
                
                # 生成假樣本 (pred_patch + refined_patch)
                with torch.no_grad():  # 生成器參數不參與判別器訓練
                    delta_pred_detached = netG(pred_patch)
                    refined_patch = get_refined_patch_batch(
                        img_path, pred_box, delta_pred_detached, args.img_size, device, fallback_patches=pred_patch
                    )
                fake_validity = netD(pred_patch, refined_patch)
                
                # 計算梯度懲罰
                gradient_penalty = compute_gradient_penalty(
                    netD, 
                    (pred_patch, gt_patch), 
                    (pred_patch, refined_patch), 
                    device
                )
                
                # WGAN-GP 損失：W_distance + λ * GP
                wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)
                d_loss = -wasserstein_distance + args.lambda_gp * gradient_penalty
                
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
                optimizer_D.step()
                
                # 累積損失用於記錄
                d_loss_epoch += d_loss.item()
                gp_loss_epoch += gradient_penalty.item()
                wd_epoch += wasserstein_distance.item()
            
            # 記錄平均判別器損失
            epoch_stats['loss_D'] += d_loss_epoch / args.n_critic
            epoch_stats['loss_gp'] += gp_loss_epoch / args.n_critic
            epoch_stats['wasserstein_distance'] += wd_epoch / args.n_critic

            # 2. 訓練生成器一次
            optimizer_G.zero_grad()
            
            # 重新前向傳播生成器（現在需要梯度）
            delta_pred = netG(pred_patch)
            
            # 計算 EIoU 回歸損失
            calibrated_boxes = apply_delta_to_bbox(pred_box, delta_pred, training=True)
            gt_boxes = apply_delta_to_bbox(pred_box, delta_true, training=True)
            loss_G_reg, loss_iou = criterion_hybrid(
                delta_pred, delta_true, calibrated_boxes, gt_boxes
            )
            
            # 計算對抗損失：生成器希望判別器給假樣本高分
            refined_patch_for_G = get_refined_patch_batch(
                img_path, pred_box, delta_pred, args.img_size, device, fallback_patches=pred_patch
            )
            fake_validity_for_G = netD(pred_patch, refined_patch_for_G)
            loss_WGAN_G = -torch.mean(fake_validity_for_G)  # 生成器希望最大化判別器輸出

            # 總生成器損失
            loss_G = loss_G_reg + loss_WGAN_G
            loss_G.backward()
            
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optimizer_G.step()

            # 記錄生成器損失
            epoch_stats['loss_G'] += loss_G.item()
            epoch_stats['loss_iou'] += loss_iou.item()
            epoch_stats['loss_wgan'] += loss_WGAN_G.item()

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
              f"EIoU: {epoch_stats['loss_iou']:.3f} "
              f"WGAN: {epoch_stats['loss_wgan']:.3f} "
              f"GP: {epoch_stats['loss_gp']:.3f} "
              f"WD: {epoch_stats['wasserstein_distance']:.3f} "
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
                'train/loss_wgan': epoch_stats['loss_wgan'],
                'train/loss_gp': epoch_stats['loss_gp'],
                'train/wasserstein_distance': epoch_stats['wasserstein_distance'],
                'val/delta_iou': delta_iou,
                'val/mean_iou_before': mean_iou_before,
                'val/mean_iou_after': mean_iou_after,
                'val/improvement': delta_iou,
                'learning_rate/generator': optimizer_G.param_groups[0]['lr'],
                'learning_rate/discriminator': optimizer_D.param_groups[0]['lr']
            }
            
            # Add focal loss if available
            # Simplified - no additional losses to track
            
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