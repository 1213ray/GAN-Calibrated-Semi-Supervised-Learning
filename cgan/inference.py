
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
from models import GeneratorUNet, GeneratorSimpleRegressor

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
    
    # 使用正確的歸一化方法（與dataset.py中一致）
    norm_factor = max(w, h, 0.1)
    cx_new = cx + delta_np[0] * norm_factor
    cy_new = cy + delta_np[1] * norm_factor
    w_new  = w * math.exp(delta_np[2])
    h_new  = h * math.exp(delta_np[3])
    
    # 限制結果在合理範圍內
    cx_new = max(0.05, min(0.95, cx_new))
    cy_new = max(0.05, min(0.95, cy_new))
    w_new = max(0.01, min(0.9, w_new))
    h_new = max(0.01, min(0.9, h_new))
    
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
    # 嘗試從模型檔案中載入完整的配置資訊
    try:
        checkpoint = torch.load(args.weights, map_location=device)
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            # 從模型檔案中載入配置
            model_config = checkpoint['config']
            delta_scale = model_config.get('delta_scale', 0.25)
            generator_type = model_config.get('generator_type', 'unet')
            print(f"從模型檔案載入配置: delta_scale={delta_scale}, generator_type={generator_type}")
        else:
            # 嘗試從檔名推斷 delta_scale
            try:
                delta_scale = float(Path(args.weights).stem.split('=')[-1])
                generator_type = 'unet'
                print(f"從檔名推斷 delta_scale: {delta_scale}")
            except (ValueError, IndexError):
                delta_scale = 0.25
                generator_type = 'unet'
                print(f"警告：無法推斷配置，使用預設值 delta_scale={delta_scale}")
    except Exception as e:
        print(f"警告：載入模型檔案時出錯: {e}")
        delta_scale = 0.25
        generator_type = 'unet'
        checkpoint = None

    # 創建模型
    if generator_type == 'simple':
        from models import GeneratorSimpleRegressor
        netG = GeneratorSimpleRegressor(delta_scale=delta_scale).to(device)
    else:
        netG = GeneratorUNet(delta_scale=delta_scale).to(device)
    
    # 載入模型權重
    try:
        if isinstance(checkpoint, dict) and 'generator' in checkpoint:
            netG.load_state_dict(checkpoint['generator'])
        elif isinstance(checkpoint, dict):
            netG.load_state_dict(checkpoint)
        else:
            netG.load_state_dict(torch.load(args.weights, map_location=device))
    except Exception as e:
        print(f"錯誤：載入模型權重時出錯: {e}")
        return
    
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

