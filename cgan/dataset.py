"""
CGAN Dataset Module - 簡化和優化版本
用於加載和預處理 YOLO 格式的訓練數據
"""

import math
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps
import yaml

class CalibratorDataset(Dataset):
    """動態構建 (pred_patch, gt_patch, Δ_true, pred_bbox, img_path) 元組。

    預期在 *root* 下的目錄結構 (YOLO 風格):
      images/        *.jpg / *.png  (原始解析度)
      labels_gt/     *.txt          (類別 cx cy w h)
      labels_pred/   *.txt          (偽標籤 cx cy w h conf)
    所有座標都已正規化 (YOLO)。
    """

    def __init__(self, root: str | os.PathLike, img_size: int = None, iou_thr: float = None, transform: transforms.Compose | None = None):
        super().__init__()
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.gt_dir = self.root / "labels_gt"
        self.pred_dir = self.root / "labels_pred"
        
        # 載入預設配置
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if img_size is None:
            img_size = config['img_size']
        if iou_thr is None:
            iou_thr = config['iou_threshold']
        
        self.img_size = int(img_size)
        self.iou_thr = float(iou_thr)

        self.samples: List[Tuple[Path, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []  # 添加gt_box
        self._prepare_index()

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 轉換到 (-1,1)
        ])

    # ---------------------  utils ---------------------

    @staticmethod
    def _bbox_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
        """計算 [cx,cy,w,h] 正規化座標的邊界框 IoU。"""
        b1_x1 = box1[0] - box1[2] / 2
        b1_y1 = box1[1] - box1[3] / 2
        b1_x2 = box1[0] + box1[2] / 2
        b1_y2 = box1[1] + box1[3] / 2
        b2_x1 = box2[0] - box2[2] / 2
        b2_y1 = box2[1] - box2[3] / 2
        b2_x2 = box2[0] + box2[2] / 2
        b2_y2 = box2[1] + box2[3] / 2

        inter = max(0, min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) * max(0, min(b1_y2, b2_y2) - max(b1_y1, b2_y1))
        union = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _bbox2delta(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        計算穩定的 Δ = (dx_rel, dy_rel, log dw, log dh)
        改進的歸一化方法，增強數值穩定性
        """
        # 使用固定的歸一化因子，避免小框產生過大的 delta 值
        # 改用 pred 框的幾何平均尺寸作為歸一化因子
        pred_area = float(pred[2]) * float(pred[3])
        norm_factor = max(math.sqrt(pred_area), 0.05)  # 設置最小歸一化因子
        
        dx = (float(gt[0]) - float(pred[0])) / norm_factor
        dy = (float(gt[1]) - float(pred[1])) / norm_factor
        
        # 對數比值計算，加強數值保護
        eps = 1e-6
        gt_w = max(float(gt[2]), eps)
        gt_h = max(float(gt[3]), eps)
        pred_w = max(float(pred[2]), eps)
        pred_h = max(float(pred[3]), eps)
        
        # 限制比值範圍，避免極端情況
        w_ratio = max(0.1, min(10.0, gt_w / pred_w))
        h_ratio = max(0.1, min(10.0, gt_h / pred_h))
        
        dw = math.log(w_ratio)
        dh = math.log(h_ratio)
        
        return torch.tensor([dx, dy, dw, dh], dtype=torch.float32)

    @staticmethod
    def _letterbox(img: Image.Image, bbox_xywh: torch.Tensor, out_size: int) -> Image.Image:
        """裁剪邊界框區域 → 填充成正方形 → 調整大小 (out_size×out_size)。"""
        W, H = img.size
        cx, cy, w, h = bbox_xywh
        # 轉換為像素座標並確保是標量值
        px = float(cx) * W
        py = float(cy) * H
        pw = float(w) * W
        ph = float(h) * H
        x1 = max(0, px - pw / 2)
        y1 = max(0, py - ph / 2)
        x2 = min(W, px + pw / 2)
        y2 = min(H, py + ph / 2)
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        # 填充成正方形
        pad_w = max(crop.height - crop.width, 0)
        pad_h = max(crop.width - crop.height, 0)
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        crop_square = ImageOps.expand(crop, padding, fill=(128, 128, 128))
        crop_resized = crop_square.resize((out_size, out_size), Image.BICUBIC)
        return crop_resized

    # ----------------  index samples ----------------

    def _prepare_index(self) -> None:
        """使用多對一貪婪匹配策略構建索引列表。"""
        for txt_pred in sorted(self.pred_dir.glob("*.txt")):
            name = txt_pred.stem
            txt_gt = self.gt_dir / f"{name}.txt"
            img_path = self.img_dir / f"{name}.jpg"
            if not txt_gt.exists() or not img_path.exists():
                continue
            
            # 載入GT和預測框
            gt_boxes = self._load_boxes(txt_gt)
            pred_boxes = self._load_pred_boxes(txt_pred)
            
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                continue
            
            # 使用改進的一對一匹配
            matches = self._hungarian_matching(pred_boxes, gt_boxes)
            
            for pred_idx, gt_idx in matches:
                pred_box = pred_boxes[pred_idx]
                gt_box = gt_boxes[gt_idx]
                
                # 計算delta使用統一的方法
                delta = self._bbox2delta(gt_box, pred_box)
                self.samples.append((img_path, 0, pred_box, delta, gt_box))
    
    def _load_boxes(self, txt_path: Path) -> torch.Tensor:
        """載入邊界框"""
        if txt_path.stat().st_size == 0:
            return torch.empty((0, 4))
        
        boxes = []
        for line in txt_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append([float(x) for x in parts[1:5]])
        
        return torch.tensor(boxes) if boxes else torch.empty((0, 4))
    
    def _load_pred_boxes(self, txt_path: Path) -> torch.Tensor:
        """載入預測框"""
        if txt_path.stat().st_size == 0:
            return torch.empty((0, 4))
        
        boxes = []
        for line in txt_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 6:  # 預測框通常有confidence
                boxes.append([float(x) for x in parts[1:5]])
        
        return torch.tensor(boxes) if boxes else torch.empty((0, 4))
    
    def _hungarian_matching(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> List[Tuple[int, int]]:
        """
        多對一簡化匹配：每個 pred 獨立選 IoU 最大的 gt（允許一個 gt 被多次匹配）。
        """
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return []

        # 1. 計算 IoU 矩陣 (Np, Ng)
        Np, Ng = len(pred_boxes), len(gt_boxes)
        iou_matrix = torch.zeros((Np, Ng), device=pred_boxes.device)
        for i, pb in enumerate(pred_boxes):
            for j, gb in enumerate(gt_boxes):
                iou_matrix[i, j] = self._bbox_iou(pb, gb)

        # 2. 對每個 pred，找 IoU 最大的 gt
        #    best_iou: (Np,), best_gt_idx: (Np,)
        best_iou, best_gt_idx = iou_matrix.max(dim=1)

        # 3. 按閾值篩選，允許同一個 gt 被多次選中
        matches = []
        for i in range(Np):
            if best_iou[i] >= self.iou_thr:
                matches.append((i, int(best_gt_idx[i])))

        return matches
    

    @staticmethod
    def _apply_delta_to_bbox(bbox: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """將 delta 應用於 bbox 以獲得新框。"""
        cx = bbox[0] + delta[0] * bbox[2]
        cy = bbox[1] + delta[1] * bbox[3]
        w  = bbox[2] * torch.exp(delta[2])
        h  = bbox[3] * torch.exp(delta[3])
        return torch.tensor([cx, cy, w, h], dtype=torch.float32)

    # -----------------  Dataset API ----------------

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        img_path, _, pred_box, delta_true, gt_box = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # 使用存儲的 delta 確保數據一致性
        # 裁切 patch
        gt_patch = self._letterbox(img, gt_box, self.img_size)
        pred_patch = self._letterbox(img, pred_box, self.img_size)

        # 轉換為張量
        pred_patch = self.transform(pred_patch)
        gt_patch = self.transform(gt_patch)
        
        # 返回存儲的 delta，確保訓練一致性
        return pred_patch, gt_patch, delta_true, pred_box, str(img_path)