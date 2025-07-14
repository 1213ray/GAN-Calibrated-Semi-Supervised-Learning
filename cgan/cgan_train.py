#!/usr/bin/env python3
"""
Fixed CGAN training script with improved hyperparameters
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

# Setup path
project_root = Path(__file__).parent

from models import GeneratorUNet, Discriminator, weights_init_normal
from dataset import CalibratorDataset

def apply_delta_to_bbox(bbox, delta):
    """Apply predicted deltas to original bboxes"""
    cx = bbox[..., 0] + delta[..., 0] * bbox[..., 2]
    cy = bbox[..., 1] + delta[..., 1] * bbox[..., 3]
    w = bbox[..., 2] * torch.exp(delta[..., 2])
    h = bbox[..., 3] * torch.exp(delta[..., 3])
    return torch.stack([cx, cy, w, h], dim=-1)

def get_refined_patch_batch(original_image_paths, pred_bboxes, deltas_pred, img_size, device):
    """Generate refined patches using predicted deltas"""
    refined_patches = []
    from torchvision import transforms
    from PIL import Image, ImageOps
    
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    for i in range(len(original_image_paths)):
        img_path = original_image_paths[i]
        pred_box = pred_bboxes[i]
        delta = deltas_pred[i]

        refined_box = apply_delta_to_bbox(pred_box.cpu(), delta.cpu())

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        cx, cy, w, h = refined_box
        px, py, pw, ph = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        x1, y1 = max(0, px - pw / 2), max(0, py - ph / 2)
        x2, y2 = min(W, px + pw / 2), min(H, py + ph / 2)
        
        if x2 <= x1 or y2 <= y1:
            crop = Image.new('RGB', (1, 1), (128, 128, 128))
        else:
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))

        pad_w = max(crop.height - crop.width, 0)
        pad_h = max(crop.width - crop.height, 0)
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        crop_square = ImageOps.expand(crop, padding, fill=(128, 128, 128))
        crop_resized = crop_square.resize((img_size, img_size), Image.BICUBIC)
        
        refined_patches.append(transform_to_tensor(crop_resized))

    return torch.stack(refined_patches).to(device)

def iou_xywh_batch(boxes1, boxes2):
    """Calculate IoU for batch of bboxes"""
    b1_x1, b1_y1 = boxes1[:, 0] - boxes1[:, 2] / 2, boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2, b1_y2 = boxes1[:, 0] + boxes1[:, 2] / 2, boxes1[:, 1] + boxes1[:, 3] / 2
    b2_x1, b2_y1 = boxes2[:, 0] - boxes2[:, 2] / 2, boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2, b2_y2 = boxes2[:, 0] + boxes2[:, 2] / 2, boxes2[:, 1] + boxes2[:, 3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    return inter_area / (union_area + 1e-6)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--img_size", type=int, default=128, help="Image size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr_g", type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--lambda_l1", type=float, default=20.0, help="L1 loss weight (reduced from 100)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=1e-5, help="Early stopping min delta")
    parser.add_argument("--out_dir", type=str, default="runs_fixed", help="Output directory")
    parser.add_argument("--delta_scale", type=float, default=0.1, help="Delta scale factor")
    parser.add_argument("--d_train_ratio", type=int, default=2, help="Train D every N batches")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    full_dataset = CalibratorDataset(args.data_dir, img_size=args.img_size)
    val_len = max(1, int(0.1 * len(full_dataset)))
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_set)}, Validation samples: {len(val_set)}")

    # Initialize models with improved delta_scale
    netG = GeneratorUNet(delta_scale=args.delta_scale).to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.SmoothL1Loss()

    # Optimizers with different learning rates
    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    # Output directory
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "samples").mkdir(exist_ok=True)
    ckpt_best = out_root / "G_best.pth"

    best_val_iou, epochs_no_improve = -1.0, 0  # Changed initial value
    
    print("Starting training...")
    for epoch in range(1, args.n_epochs + 1):
        netG.train()
        netD.train()
        loss_G_epoch, loss_D_epoch = 0.0, 0.0
        
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

            # Train Discriminator less frequently
            if i % args.d_train_ratio == 0:
                optimizer_D.zero_grad()

                # Real loss
                pred_real = netD(pred_patch, gt_patch)
                loss_real = criterion_GAN(pred_real, valid)

                # Fake loss
                delta_pred_detached = netG(pred_patch).detach()
                try:
                    refined_patch = get_refined_patch_batch(
                        img_path, pred_box, delta_pred_detached, args.img_size, device
                    )
                    pred_fake = netD(pred_patch, refined_patch)
                    loss_fake = criterion_GAN(pred_fake, fake)
                except:
                    # Fallback if refined patch generation fails
                    pred_fake = netD(pred_patch, pred_patch.detach())
                    loss_fake = criterion_GAN(pred_fake, fake)
                
                loss_D = (loss_real + loss_fake) * 0.5
                loss_D.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
                optimizer_D.step()
                loss_D_epoch += loss_D.item()

            # Train Generator
            optimizer_G.zero_grad()
            
            delta_pred = netG(pred_patch)
            
            # L1 loss
            loss_L1 = criterion_L1(delta_pred, delta_true)

            # Adversarial loss
            try:
                refined_patch_for_G = get_refined_patch_batch(
                    img_path, pred_box, delta_pred, args.img_size, device
                )
                pred_fake_for_G = netD(pred_patch, refined_patch_for_G)
                loss_GAN_G = criterion_GAN(pred_fake_for_G, valid)
            except:
                # Fallback
                pred_fake_for_G = netD(pred_patch, pred_patch)
                loss_GAN_G = criterion_GAN(pred_fake_for_G, valid)

            loss_G = loss_GAN_G + args.lambda_l1 * loss_L1
            loss_G.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optimizer_G.step()

            loss_G_epoch += loss_G.item()

            # Save sample images occasionally
            if i == 0 and epoch % 10 == 1:
                try:
                    img_sample = torch.cat((pred_patch.data[:4], refined_patch_for_G.data[:4], gt_patch.data[:4]), -2)
                    save_image(img_sample, out_root / f"samples/epoch_{epoch}.png", nrow=4, normalize=True)
                except:
                    pass

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

                calibrated_boxes = apply_delta_to_bbox(pred_box_val, delta_pred_val)
                gt_boxes = apply_delta_to_bbox(pred_box_val, delta_true_val)

                iou_before = iou_xywh_batch(pred_box_val, gt_boxes)
                iou_after = iou_xywh_batch(calibrated_boxes, gt_boxes)

                iou_sum_before += iou_before.sum().item()
                iou_sum_after += iou_after.sum().item()
                n_val_samples += batch_size_val
            
            mean_iou_before = iou_sum_before / max(1, n_val_samples)
            mean_iou_after = iou_sum_after / max(1, n_val_samples)
            delta_iou = mean_iou_after - mean_iou_before

        avg_loss_G = loss_G_epoch / len(train_loader)
        avg_loss_D = loss_D_epoch / max(1, len(train_loader) // args.d_train_ratio)
        
        print(f"[Epoch {epoch}/{args.n_epochs}]  loss_G: {avg_loss_G:.3f}  loss_D: {avg_loss_D:.3f}  ΔIoU: {delta_iou:.4f}")

        # Early stopping with improved criteria
        if delta_iou > best_val_iou + args.min_delta:
            best_val_iou = delta_iou
            torch.save(netG.state_dict(), ckpt_best)
            epochs_no_improve = 0
            print(f"New best model saved! Delta IoU = {best_val_iou:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered.")
                break

    print(f"Training complete. Best Delta IoU = {best_val_iou:.4f}")

if __name__ == "__main__":
    main()