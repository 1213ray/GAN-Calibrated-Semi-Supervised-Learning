from __future__ import annotations
import argparse, math, os, random, time
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from PIL import Image, ImageOps
from torchvision import transforms

# Local imports
from models import GeneratorUNet, Discriminator, weights_init_normal
from dataset import CalibratorDataset

# ---------------------  Utils  ---------------------

def apply_delta_to_bbox(bbox: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    # Applies predicted deltas to original bboxes.
    cx = bbox[..., 0] + delta[..., 0] * bbox[..., 2]
    cy = bbox[..., 1] + delta[..., 1] * bbox[..., 3]
    w  = bbox[..., 2] * torch.exp(delta[..., 2])
    h  = bbox[..., 3] * torch.exp(delta[..., 3])
    return torch.stack([cx, cy, w, h], dim=-1)

def get_refined_patch_batch(
    original_image_paths: List[str],
    pred_bboxes: torch.Tensor,
    deltas_pred: torch.Tensor,
    img_size: int,
    device: torch.device
) -> torch.Tensor:
    # Dynamically generates a batch of refined_patches based on predicted deltas.
    refined_patches = []
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
        px, py, pw, ph = cx * W, cy * H, w * W, h * H
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

def _iou_xywh_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # Calculates IoU for a batch of bboxes.
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

# ------------------  Main Routine  ------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--img_size", type=int, default=128, help="Image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=120, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="L1 loss weight")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Early stopping min delta")
    parser.add_argument("--out_dir", type=str, default="runs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = CalibratorDataset(args.data_dir, img_size=args.img_size)
    val_len = max(1, int(0.1 * len(full_dataset)))
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    netG = GeneratorUNet().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1  = nn.SmoothL1Loss()

    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "samples").mkdir(exist_ok=True)
    ckpt_best = out_root / "G_best.pth"

    best_val_iou, epochs_no_improve = 0.0, 0
    for epoch in range(1, args.n_epochs + 1):
        netG.train(); netD.train()
        loss_G_epoch, loss_D_epoch = 0.0, 0.0
        for i, (pred_patch, gt_patch, delta_true, pred_box, img_path) in enumerate(train_loader):
            
            pred_patch = pred_patch.to(device)
            gt_patch   = gt_patch.to(device)
            delta_true = delta_true.to(device)

            patch_size = netD(pred_patch, gt_patch).size()
            valid = torch.ones(patch_size, device=device)
            fake = torch.zeros(patch_size, device=device)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            pred_real = netD(pred_patch, gt_patch)
            loss_real = criterion_GAN(pred_real, valid)

            delta_pred_detached = netG(pred_patch).detach()
            refined_patch = get_refined_patch_batch(
                img_path, pred_box, delta_pred_detached, args.img_size, device
            )
            pred_fake = netD(pred_patch, refined_patch)
            loss_fake = criterion_GAN(pred_fake, fake)
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            
            delta_pred = netG(pred_patch)

            loss_L1 = criterion_L1(delta_pred, delta_true)

            refined_patch_for_G = get_refined_patch_batch(
                img_path, pred_box, delta_pred, args.img_size, device
            )
            pred_fake_for_G = netD(pred_patch, refined_patch_for_G)
            loss_GAN_G = criterion_GAN(pred_fake_for_G, valid)

            loss_G = loss_GAN_G + args.lambda_l1 * loss_L1
            loss_G.backward()
            optimizer_G.step()

            loss_G_epoch += loss_G.item()
            loss_D_epoch += loss_D.item()

            if i == 0:
                img_sample = torch.cat((pred_patch.data, refined_patch_for_G.data, gt_patch.data), -2)
                save_image(img_sample, out_root / f"samples/epoch_{epoch}.png", nrow=8, normalize=True)


        # --- Validation ---
        netG.eval()
        with torch.no_grad():
            iou_sum_before, iou_sum_after = 0.0, 0.0
            n_val_samples = 0
            for pred_patch_val, _, delta_true_val, pred_box_val, _ in val_loader:
                batch_size_val = pred_patch_val.size(0)
                pred_patch_val = pred_patch_val.to(device)
                delta_true_val = delta_true_val.to(device)
                pred_box_val   = pred_box_val.to(device)
                
                delta_pred_val = netG(pred_patch_val)

                calibrated_boxes = apply_delta_to_bbox(pred_box_val, delta_pred_val)
                gt_boxes = apply_delta_to_bbox(pred_box_val, delta_true_val)

                iou_before = _iou_xywh_batch(pred_box_val, gt_boxes)
                iou_after = _iou_xywh_batch(calibrated_boxes, gt_boxes)

                iou_sum_before += iou_before.sum().item()
                iou_sum_after  += iou_after.sum().item()
                n_val_samples += batch_size_val
            
            mean_iou_before = iou_sum_before / max(1, n_val_samples)
            mean_iou_after  = iou_sum_after  / max(1, n_val_samples)
            delta_iou = mean_iou_after - mean_iou_before

        print(f"[Epoch {epoch}/{args.n_epochs}]  loss_G: {loss_G_epoch/len(train_loader):.3f}  " +
              f"loss_D: {loss_D_epoch/len(train_loader):.3f}  ΔIoU: {delta_iou:.4f}")

        if delta_iou > best_val_iou + args.min_delta:
            best_val_iou = delta_iou
            torch.save(netG.state_dict(), ckpt_best)
            epochs_no_improve = 0
            print(f"💡 New best model saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"🚫 Early stopping triggered.")
                break

    print(f"✅ Training complete. Best ΔIoU = {best_val_iou:.4f}")

if __name__ == "__main__":
    main()