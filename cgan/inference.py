
"""cGAN Patch-Calibrator  —  程式碼庫 (步驟 1-4/5)
=================================================
此畫布現在包含 **四個 Python 模組**：
  • *models.py*          — 生成器 + 判別器 (步驟 1)
  • *cgan_datasets.py*   — 產生訓練元組的資料集 (步驟 2)
  • *train_cgan.py*      — 完整訓練迴圈 (步驟 3)
  • *inference.py*       — 單圖像校準器 (步驟 4)  ← **新增**
剩餘步驟：
  • README.md / config   — 使用文件與超參數 (步驟 5)

所有模組都保留在單一畫布中，以便於內聯編輯，但可以一對一地另存為單獨的 .py 檔案。
"""

# =================================================
# models.py  —  Generator & PatchGAN Discriminator
# =================================================
# [... 现有 models.py 内容保持不变 ...]

# =================================================
# cgan_datasets.py  —  CalibratorDataset & utils
# =================================================
# [... 现有 cgan_datasets.py 内容保持不变 ...]

# =================================================
# train_cgan.py  —  Training script
# =================================================
# [... 现有 train_cgan.py 内容保持不变 ...]

# =================================================
# inference.py  —  Single-image calibration utility
# =================================================
"""使用訓練好的生成器校準單張 JPEG/PNG 圖像上的偽標籤。

範例
-------
$ python inference.py \
      --weights    runs/G_best.pth \
      --image      demo.jpg \
      --pred_txt   demo_pred.txt \
      --out_txt    demo_calib.txt

* `pred_txt` 為 YOLO v5/8 txt：每行 `cls cx cy w h` (0-1 norm)。
* `out_txt` 將寫入校準後 bbox（同格式）。
"""
from __future__ import annotations
import argparse, math, os
from pathlib import Path
from typing import List

import torch
from torchvision import transforms
from PIL import Image, ImageOps

# 同資料夾匯入
from models import GeneratorUNet

# ------------------  helpers  ------------------

def load_yolo_txt(txt_path: Path) -> List[List[float]]:
    """載入 YOLO 格式的 txt 檔案。"""
    res = []
    if not txt_path.exists():
        return res
    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            # 處理可能包含類別和信心的情況
            res.append([float(x) for x in parts])
    return res

def save_yolo_txt(txt_path: Path, rows: List[List[float]]):
    """儲存 YOLO 格式的 txt 檔案。"""
    with open(txt_path, "w") as f:
        for r in rows:
            # 確保類別是整數
            r[0] = int(r[0])
            f.write(" ".join(map(str, r)) + "\n")

def letterbox(img: Image.Image, out_size: int) -> Image.Image:
    """將圖像填充成正方形並調整大小。"""
    pad_w = max(img.height - img.width, 0)
    pad_h = max(img.width - img.height, 0)
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    crop_square = ImageOps.expand(img, padding, fill=(128, 128, 128))
    return crop_square.resize((out_size, out_size), Image.BICUBIC)

def crop_patch(img: Image.Image, bbox: List[float]) -> Image.Image:
    """bbox = (cx,cy,w,h) 歸一化座標。返回裁剪後的 PIL.Image。"""
    W, H = img.size
    # 忽略類別和信心（如果存在）
    cx, cy, w, h = bbox[:4]
    px, py, pw, ph = cx * W, cy * H, w * W, h * H
    x1, y1 = max(0, px - pw / 2), max(0, py - ph / 2)
    x2, y2 = min(W, px + pw / 2), min(H, py + ph / 2)
    return img.crop((x1, y1, x2, y2))

def apply_delta_to_bbox_inference(bbox: List[float], delta: torch.Tensor) -> List[float]:
    """將預測的 delta 應用於原始 bbox。"""
    cx, cy, w, h = bbox[:4]
    delta_np = delta.numpy()
    cx_new = cx + delta_np[0] * w
    cy_new = cy + delta_np[1] * h
    w_new  = w * math.exp(delta_np[2])
    h_new  = h * math.exp(delta_np[3])
    return [cx_new, cy_new, w_new, h_new]

# ------------------  main  ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="訓練好的生成器權重路徑")
    parser.add_argument("--image",   type=str, required=True, help="輸入圖像路徑")
    parser.add_argument("--pred_txt", type=str, required=True, help="預測邊界框的 txt 檔案路徑")
    parser.add_argument("--out_txt",  type=str, required=True, help="校準後邊界框的輸出 txt 檔案路徑")
    parser.add_argument("--img_size", type=int, default=128, help="用於裁切 patch 的圖像大小")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入模型
    # 從權重檔名中自動推斷 delta_scale (如果存在)
    try:
        delta_scale = float(Path(args.weights).stem.split('=')[-1])
    except (ValueError, IndexError):
        delta_scale = 0.25 # 預設值
        print(f"警告：無法從檔名推斷 delta_scale，使用預設值 {delta_scale}")

    netG = GeneratorUNet(delta_scale=delta_scale).to(device)
    netG.load_state_dict(torch.load(args.weights, map_location=device))
    netG.eval()

    # 載入圖像和預測
    img = Image.open(args.image).convert("RGB")
    preds = load_yolo_txt(Path(args.pred_txt))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    calibrated_rows: List[List[float]] = []
    for pred_row in preds:
        cls = int(pred_row[0])
        bbox = pred_row[1:5]

        # 裁切、填充、轉換
        patch_pil = crop_patch(img, bbox)
        patch_pil_letterboxed = letterbox(patch_pil, args.img_size)
        patch_tensor = transform(patch_pil_letterboxed).unsqueeze(0).to(device)

        with torch.no_grad():
            delta = netG(patch_tensor)[0].cpu()
        
        # 反推校準後的邊界框
        calibrated_bbox = apply_delta_to_bbox_inference(bbox, delta)
        
        # 保留原始類別和信心（如果存在）
        new_row = [cls] + calibrated_bbox + pred_row[5:]
        calibrated_rows.append(new_row)

    save_yolo_txt(Path(args.out_txt), calibrated_rows)
    print(f"✅  已儲存 {len(calibrated_rows)} 個校準後的邊界框 → {args.out_txt}")


if __name__ == "__main__":
    main()

