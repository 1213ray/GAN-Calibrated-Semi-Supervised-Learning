
"""使用訓練好的生成器校準偽標籤。

支持兩種模式：
1.  **單張圖像模式**：
    ```bash
    python cgan/inference.py \
        --weights    runs/G_best.pth \
        --image      path/to/your/image.jpg \
        --pred_txt   path/to/your/prediction.txt \
        --out_txt    path/to/your/output.txt
    ```

2.  **批量處理模式** (圖像和標籤位於不同資料夾):
    ```bash
    python cgan/inference.py \
        --weights    runs/G_best.pth \
        --source     path/to/images_folder \
        --labels     path/to/labels_folder \
        --output     path/to/output_folder
    ```
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import List

import torch
from torchvision import transforms
from PIL import Image, ImageOps

# 同資料夾匯入
from models import GeneratorUNet, GeneratorSimpleRegressor

# ------------------  Helper Functions  ------------------

def load_yolo_txt(txt_path: Path) -> List[List[float]]:
    """載入 YOLO 格式的 txt 檔案。"""
    if not txt_path.exists():
        return []
    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        return [[float(x) for x in line.split()] for line in lines]

def save_yolo_txt(txt_path: Path, rows: List[List[float]]):
    """儲存 YOLO 格式的 txt 檔案。"""
    with open(txt_path, "w") as f:
        for r in rows:
            r[0] = int(r[0])  # 確保類別是整數
            f.write(" ".join(map(str, r)) + "\n")

def letterbox(img: Image.Image, out_size: int) -> Image.Image:
    """將圖像填充成正方形並調整大小。"""
    w, h = img.size
    pad_w = max(h - w, 0)
    pad_h = max(w - h, 0)
    padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))
    crop_square = ImageOps.expand(img, padding, fill=(128, 128, 128))
    return crop_square.resize((out_size, out_size), Image.BICUBIC)

def crop_patch(img: Image.Image, bbox: List[float]) -> Image.Image | None:
    """從圖像中裁剪出由 bbox 指定的區域，如果 bbox 無效則返回 None。"""
    W, H = img.size
    cx, cy, w, h = bbox[:4]
    
    # 檢查寬高是否為正
    if w <= 0 or h <= 0:
        return None
        
    px, py, pw, ph = cx * W, cy * H, w * W, h * H
    x1, y1 = max(0, px - pw / 2), max(0, py - ph / 2)
    x2, y2 = min(W, px + pw / 2), min(H, py + ph / 2)

    # 再次檢查裁剪後的寬高是否有效
    if x2 <= x1 or y2 <= y1:
        return None
        
    return img.crop((x1, y1, x2, y2))

def apply_delta_to_bbox(bbox: List[float], delta: torch.Tensor) -> List[float]:
    """將預測的修正量 delta 應用於原始 bbox。"""
    cx, cy, w, h = bbox[:4]
    delta_np = delta.cpu().numpy()
    
    # 與訓練過程一致地應用修正量
    cx_new = cx + delta_np[0] * w
    cy_new = cy + delta_np[1] * h
    w_new = w * math.exp(delta_np[2])
    h_new = h * math.exp(delta_np[3])
    
    # 限制匡線在合理範圍內
    cx_new = max(0.0, min(1.0, cx_new))
    cy_new = max(0.0, min(1.0, cy_new))
    w_new = max(0.01, min(1.0, w_new))
    h_new = max(0.01, min(1.0, h_new))
    
    return [cx_new, cy_new, w_new, h_new]

# ------------------  Core Processing Logic  ------------------

def process_single_image(netG, transform, image_path, pred_txt_path, out_txt_path, img_size, device):
    """處理單張圖像。"""
    img = Image.open(image_path).convert("RGB")
    preds = load_yolo_txt(Path(pred_txt_path))
    
    if not preds:
        print(f"警告：在 {pred_txt_path} 中沒有找到預測框。")
        return 0

    calibrated_rows = []
    skipped_boxes = 0
    for pred_row in preds:
        cls, bbox = int(pred_row[0]), pred_row[1:5]
        patch_pil = crop_patch(img, bbox)
        if patch_pil is None:
            skipped_boxes += 1
            continue
        
        patch_pil_letterboxed = letterbox(patch_pil, img_size)
        patch_tensor = transform(patch_pil_letterboxed).unsqueeze(0).to(device)

        with torch.no_grad():
            delta = netG(patch_tensor)[0]
        
        calibrated_bbox = apply_delta_to_bbox(bbox, delta)
        calibrated_rows.append([cls] + calibrated_bbox + pred_row[5:])

    if calibrated_rows:
        save_yolo_txt(Path(out_txt_path), calibrated_rows)
    
    if skipped_boxes > 0:
        print(f"警告：因座標無效，跳過了 {skipped_boxes} 個邊界框。")
        
    return len(calibrated_rows)

def process_batch(netG, transform, source_dir, labels_dir, output_dir, img_size, device):
    """批量處理整個資料夾。"""
    source_path, labels_path, output_path = Path(source_dir), Path(labels_dir), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [p for p in source_path.iterdir() if p.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"錯誤：在 '{source_dir}' 中找不到任何支援的圖像檔案。")
        return
    
    processed_count, total_boxes, skipped_boxes = 0, 0, 0
    for img_file in image_files:
        txt_file = labels_path / f"{img_file.stem}.txt"
        if not txt_file.exists():
            continue
        
        try:
            img = Image.open(img_file).convert("RGB")
            preds = load_yolo_txt(txt_file)
            if not preds:
                continue

            calibrated_rows = []
            for pred_row in preds:
                cls, bbox = int(pred_row[0]), pred_row[1:5]
                patch_pil = crop_patch(img, bbox)
                if patch_pil is None:
                    skipped_boxes += 1
                    continue

                patch_pil_letterboxed = letterbox(patch_pil, img_size)
                patch_tensor = transform(patch_pil_letterboxed).unsqueeze(0).to(device)

                with torch.no_grad():
                    delta = netG(patch_tensor)[0]
                
                calibrated_bbox = apply_delta_to_bbox(bbox, delta)
                calibrated_rows.append([cls] + calibrated_bbox + pred_row[5:])
            
            if calibrated_rows:
                save_yolo_txt(output_path / f"{img_file.stem}.txt", calibrated_rows)
                processed_count += 1
                total_boxes += len(calibrated_rows)
        except Exception as e:
            print(f"錯誤：處理 {img_file.name} 時發生錯誤: {e}")
            continue
    
    print(f"\n✅ 批量處理完成！")
    if skipped_boxes > 0:
        print(f"   -> 警告：因座標無效，共跳過了 {skipped_boxes} 個邊界框。")
    print(f"   -> 成功處理圖像數量: {processed_count}")
    print(f"   -> 成功校準邊界框總數: {total_boxes}")
    print(f"   -> 輸出目錄: {output_dir}")

def load_model(weights_path, device):
    """從檔案載入生成器模型。"""
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
        generator_type = config.get('generator_type', 'unet')
        
        netG = GeneratorUNet().to(device) if generator_type != 'simple' else GeneratorSimpleRegressor().to(device)
        
        if isinstance(checkpoint, dict) and 'generator' in checkpoint:
            netG.load_state_dict(checkpoint['generator'])
        else:
            netG.load_state_dict(checkpoint)
            
        netG.eval()
        print(f"成功從 {weights_path} 載入模型。")
        return netG
    except Exception as e:
        print(f"錯誤：載入模型權重時失敗: {e}")
        raise

# ------------------  Entry Point  ------------------

def main():
    parser = argparse.ArgumentParser(description="使用 CGAN 校準 YOLO 偽標籤。")
    parser.add_argument("--weights", type=str, required=True, help="訓練好的生成器權重路徑。")
    
    # 模式參數
    parser.add_argument("--source", type=str, help="輸入圖像或圖像資料夾的路徑。")
    parser.add_argument("--labels", type=str, help="標籤資料夾的路徑 (僅批量模式需要)。")
    parser.add_argument("--output", type=str, help="輸出檔案或資料夾的路徑。")
    
    # 舊版單張圖像參數 (相容用)
    parser.add_argument("--image", type=str, help="[已棄用] 請使用 --source。")
    parser.add_argument("--pred_txt", type=str, help="[已棄用] 請使用 --labels。")
    parser.add_argument("--out_txt", type=str, help="[已棄用] 請使用 --output。")

    # 通用參數
    parser.add_argument("--img-size", type=int, default=128, help="裁切 patch 的圖像大小。")
    parser.add_argument("--device", type=str, default="auto", help="計算設備 (cuda/cpu/auto)。")
    
    args = parser.parse_args()

    # 處理棄用參數
    source = args.source or args.image
    output = args.output or args.out_txt
    if not source or not output:
        parser.error("--source 和 --output 是必需的。")

    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu")
    print(f"使用設備: {device}")

    # 載入模型和轉換
    netG = load_model(args.weights, device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 判斷模式
    source_path = Path(source)
    if source_path.is_dir():
        print("📂 批量處理模式")
        if not args.labels:
            parser.error("批量處理模式下 --labels 資料夾是必需的。")
        process_batch(netG, transform, source, args.labels, output, args.img_size, device)
    elif source_path.is_file():
        print("📸 單張圖像模式")
        labels_path = args.labels or args.pred_txt
        if not labels_path:
            parser.error("單張圖像模式下需要標籤檔案 (透過 --labels 或 --pred_txt 指定)。")
        count = process_single_image(netG, transform, source, labels_path, output, args.img_size, device)
        print(f"✅ 已儲存 {count} 個校準後的邊界框 → {output}")
    else:
        parser.error(f"指定的 --source 路徑無效: {source}")

if __name__ == "__main__":
    main()

