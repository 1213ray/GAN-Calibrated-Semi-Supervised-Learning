
#!/usr/bin/env python3
"""
K-Fold 偽標籤與真實標籤配對並生成對比圖的腳本
-------------------------------------------------
功能：
1. 讀取 K-Fold 交叉驗證生成的所有偽標籤。
2. 讀取原始的真實世界 (Ground Truth) 標籤。
3. 對於每一張圖片，使用匈牙利配對演算法，將偽標籤與真實標籤進行一對一的最佳配對。
4. 對於每一對成功配對的 (偽標籤, 真實標籤)：
   a. 從原始影像中裁切出對應的 patch。
   b. 使用 letterbox 函式將 patch 處理成固定大小的正方形。
   c. 將偽標籤 patch (左) 和真實標籤 patch (右) 水平拼接成一張對比圖。
5. 將所有拼接好的圖片儲存到指定的輸出資料夾，方便直觀地檢查 CGAN 的訓練資料品質。
"""

import sys
import shutil
from pathlib import Path
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

# --- 設定區 ---
# K-Fold 產生的偽標籤資料夾
PSEUDO_LABEL_DIR = Path(r"C:\Users\alian\PycharmProjects\yolov8\datasets\500_100_100\cgan\labels_pred")

# 包含對應原始影像的資料夾
IMAGE_DIR = Path(r"C:\Users\alian\PycharmProjects\yolov8\datasets\500_100_100\cgan\images")

# 包含真實標籤的資料夾
GT_LABEL_DIR = Path(r"C:\Users\alian\PycharmProjects\yolov8\datasets\500_100_100\cgan\labels_gt")

# 拼接後的 patch 圖片輸出目錄
OUTPUT_DIR = Path(r"C:\Users\alian\PycharmProjects\yolov8\datasets\500_100_100\cgan_paired_patches")

# CGAN 相關設定
IMG_SIZE = 128  # 裁切 patch 後的目標大小
IOU_THRESHOLD = 0.25 # 匈牙利配對時的 IoU 閾值

# --- 將 cgan 資料夾加入 sys.path 以便導入函式 ---
# 這樣我們就可以直接重用 dataset.py 中的函式，而無需複製程式碼
cgan_path = Path(__file__).parent / 'cgan'
if str(cgan_path) not in sys.path:
    sys.path.insert(0, str(cgan_path))

# 從 cgan.dataset 導入所需的函式
# 注意：這假設 cgan/dataset.py 檔案存在且包含這些函式
try:
    from dataset import CalibratorDataset
except ImportError as e:
    print(f"無法從 'cgan/dataset.py' 導入函式。請確保該檔案存在且路徑正確。")
    print(f"錯誤訊息: {e}")
    sys.exit(1)

# --- 輔助函式 ---

def load_boxes(txt_path: Path, has_conf=False) -> torch.Tensor:
    """從 YOLO txt 檔案載入邊界框。"""
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        return torch.empty((0, 4))
    
    boxes = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            # 偽標籤通常有 class, cx, cy, w, h, conf (6個部分)
            # 真實標籤通常有 class, cx, cy, w, h (5個部分)
            min_parts = 6 if has_conf else 5
            if len(parts) >= min_parts:
                boxes.append([float(x) for x in parts[1:5]])
    
    return torch.tensor(boxes) if boxes else torch.empty((0, 4))

def combine_images_horizontally(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """將兩張 PIL 圖片水平拼接。"""
    dst = Image.new('RGB', (img1.width + img2.width, img1.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width, 0))
    return dst

# --- 新增的匹配函式 ---

def greedy_many_to_one(pred_boxes: torch.Tensor,
                       gt_boxes:   torch.Tensor,
                       iou_thr: float,
                       bbox_iou_func
                      ) -> list[tuple[int, int]]:
    """
    多對一簡化匹配：每個 pred 獨立選 IoU 最大的 gt（允許一個 gt 被多次匹配）。
    """
    from typing import List, Tuple
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []

    # 1. 計算 IoU 矩陣 (Np, Ng)
    Np, Ng = len(pred_boxes), len(gt_boxes)
    iou_matrix = torch.zeros((Np, Ng), device=pred_boxes.device)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou_matrix[i, j] = bbox_iou_func(pb, gb)

    # 2. 對每個 pred，找 IoU 最大的 gt
    #    best_iou: (Np,), best_gt_idx: (Np,)
    best_iou, best_gt_idx = iou_matrix.max(dim=1)

    # 3. 按閾值篩選，允許同一個 gt 被多次選中
    matches = []
    for i in range(Np):
        if best_iou[i] >= iou_thr:
            matches.append((i, int(best_gt_idx[i])))

    return matches

# --- 主程式 ---

def main():
    """主執行函式"""
    # 從設定區直接使用路徑變數
    pseudo_label_dir = PSEUDO_LABEL_DIR
    image_dir = IMAGE_DIR
    gt_label_dir = GT_LABEL_DIR

    if not pseudo_label_dir.exists() or not image_dir.exists() or not gt_label_dir.exists():
        print("錯誤：一個或多個必要的資料夾不存在。請檢查以下路徑：")
        print(f"  - 偽標籤資料夾: {pseudo_label_dir}")
        print(f"  - 影像資料夾: {image_dir}")
        print(f"  - 真實標籤資料夾: {gt_label_dir}")
        return

    # 建立輸出資料夾
    if OUTPUT_DIR.exists():
        print(f"警告：輸出資料夾 {OUTPUT_DIR} 已存在，將會被清空。")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    print("開始處理偽標籤與真實標籤的配對...")

    # 獲取所有偽標籤檔案
    pseudo_label_files = list(pseudo_label_dir.glob("*.txt"))
    
    # 實例化一個 CalibratorDataset 以便使用它的內部方法
    # 我們傳入一個假的 root 路徑，因為我們不會真的用它來載入資料
    # 我們只需要它的 _bbox_iou 和 _letterbox 方法
    helper_dataset = CalibratorDataset(root=".", iou_thr=IOU_THRESHOLD, img_size=IMG_SIZE)

    paired_count = 0
    for pseudo_txt_path in tqdm(pseudo_label_files, desc="處理影像"):
        file_stem = pseudo_txt_path.stem
        
        # 找到對應的影像和真實標籤
        image_path = image_dir / f"{file_stem}.jpg"
        if not image_path.exists():
            image_path = image_dir / f"{file_stem}.png" # 嘗試其他副檔名
        gt_txt_path = gt_label_dir / f"{file_stem}.txt"

        if not image_path.exists() or not gt_txt_path.exists():
            continue

        # 載入邊界框
        pred_boxes = load_boxes(pseudo_txt_path, has_conf=True)
        gt_boxes = load_boxes(gt_txt_path, has_conf=False)

        if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
            continue

        # 使用多對一的貪婪匹配策略
        try:
            # 我們需要 CalibratorDataset._bbox_iou 方法來計算 IoU
            iou_func = helper_dataset._bbox_iou
            matches = greedy_many_to_one(pred_boxes, gt_boxes, IOU_THRESHOLD, iou_func)
        except Exception as e:
            print(f"\n在檔案 {file_stem} 上執行貪婪匹配時出錯: {e}")
            continue

        if not matches:
            continue

        # 開啟原始影像
        try:
            original_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"\n無法開啟影像 {image_path}: {e}")
            continue

        # 處理每一對配對
        for i, (pred_idx, gt_idx) in enumerate(matches):
            pred_box = pred_boxes[pred_idx]
            gt_box = gt_boxes[gt_idx]

            # 裁切並 letterbox 處理 patch
            # 同樣，我們直接呼叫 _letterbox 方法
            pred_patch = helper_dataset._letterbox(original_img, pred_box, IMG_SIZE)
            gt_patch = helper_dataset._letterbox(original_img, gt_box, IMG_SIZE)

            # 拼接圖片
            combined_img = combine_images_horizontally(pred_patch, gt_patch)

            # 儲存拼接後的圖片
            output_filename = f"{file_stem}_pair_{i}.png"
            combined_img.save(OUTPUT_DIR / output_filename)
            paired_count += 1

    print("\n處理完成！")
    print(f"總共生成了 {paired_count} 張配對的 patch 圖片。")
    print(f"輸出目錄: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
