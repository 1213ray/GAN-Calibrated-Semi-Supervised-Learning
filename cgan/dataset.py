"""cGAN Patch-Calibrator  —  程式碼庫 (步驟 1-2/5)
=================================================
此畫布現在包含 **兩個 Python 模組**：
  • *models.py*          — 生成器 + 判別器 (已審閱)
  • *cgan_datasets.py*   — PyTorch 資料集，從 YOLO txt 標籤動態構建訓練元組。
接下來的步驟將新增：
  • train_cgan.py        — 完整訓練迴圈 (步驟 3)
  • inference.py         — 單圖像校準器 (步驟 4)
  • README.md / config   — 使用文件 (步驟 5)

所有模組都保留在單一畫布中，以便於內聯編輯，但可以一對一地另存為單獨的 .py 檔案。
"""

# =================================================
# models.py  —  Generator & PatchGAN Discriminator
# =================================================
from __future__ import annotations
import math, random, os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps
import yaml

__all__ = [
    "weights_init_normal", "GeneratorUNet", "Discriminator",
    "CalibratorDataset",
]

# -------------------------------------------------
#  Helper: weight initialisation (pix2pix style)
# -------------------------------------------------

def weights_init_normal(m: nn.Module) -> None:
    """初始化 `Conv` 權重 ~ N(0,0.02) & Batch/InstanceNorm gamma ~ N(1,0.02)。"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1 or classname.find("InstanceNorm") != -1:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

# -------------------------------------------------
#  Building blocks for U-Net
# -------------------------------------------------

class UNetDown(nn.Module):
    def __init__(self, in_size: int, out_size: int, normalize: bool = True, dropout: float | None = None):
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size: int, out_size: int, dropout: float | None = None):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

# -------------------------------------------------
#  Generator  (U-Net backbone → Δ vector)
# -------------------------------------------------

class GeneratorUNet(nn.Module):
    """4層下採樣 / 4層上採樣的U-Net，輸出4維的邊界框校正Δ。"""

    def __init__(self, delta_scale: float = 0.25):
        super().__init__()
        self.delta_scale = float(delta_scale)

        # 編碼器
        self.down1 = UNetDown(3,   64, normalize=False)  # (128→64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)

        # 解碼器
        self.up1   = UNetUp(512, 256, dropout=0.5)
        self.up2   = UNetUp(512, 128, dropout=0.5)
        self.up3   = UNetUp(256,  64)
        self.up4   = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
        )

        # 全局池化 → Δ (4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_delta = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 4),
            nn.Tanh(),  # (-1,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3)
        pooled = self.avg_pool(u4)
        delta_raw = self.fc_delta(pooled)
        return delta_raw * self.delta_scale

# -------------------------------------------------
#  Discriminator  (70×70 PatchGAN)
# -------------------------------------------------

class Discriminator(nn.Module):
    """PatchGAN，用於判斷 (pred_patch, other_patch) 對。"""

    def __init__(self, spectral_norm: bool = False):
        super().__init__()

        def conv_block(in_ch: int, out_ch: int, norm: bool = True):
            conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
            if spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers = [conv]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*[
            *conv_block(6,   64, norm=False),
            *conv_block(64, 128),
            *conv_block(128,256),
            *conv_block(256,512),
            nn.Conv2d(512, 1, 4, stride=1, padding=1, bias=False)
        ])

    def forward(self, pred_patch: torch.Tensor, other_patch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = torch.cat([pred_patch, other_patch], dim=1)
        return self.model(x)

# =================================================
# cgan_datasets.py  —  CalibratorDataset
# =================================================

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

        self.samples: List[Tuple[Path, int, torch.Tensor, torch.Tensor]] = []
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
        """計算 Δ = (dx_rel, dy_rel, log dw, log dh)。"""
        dx = (gt[0] - pred[0]) / (pred[2] + 1e-6)
        dy = (gt[1] - pred[1]) / (pred[3] + 1e-6)
        dw = math.log((gt[2] + 1e-6) / (pred[2] + 1e-6))
        dh = math.log((gt[3] + 1e-6) / (pred[3] + 1e-6))
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
        """遍歷所有預測標籤並與真實標籤匹配以構建索引列表。"""
        for txt_pred in sorted(self.pred_dir.glob("*.txt")):
            name = txt_pred.stem
            txt_gt = self.gt_dir / f"{name}.txt"
            img_path = self.img_dir / f"{name}.jpg"
            if not txt_gt.exists() or not img_path.exists():
                continue
            gt_boxes = torch.tensor([[float(x) for x in l.split()[1:5]]
                                      for l in txt_gt.read_text().strip().splitlines()]) if txt_gt.stat().st_size > 0 else torch.empty((0,4))
            for line in txt_pred.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                pred_box = torch.tensor([float(x) for x in parts[1:5]])
                # 匹配真實標籤
                if len(gt_boxes):
                    ious = torch.tensor([self._bbox_iou(pred_box, g) for g in gt_boxes])
                    idx = torch.argmax(ious).item()
                    if ious[idx] < self.iou_thr:
                        continue  # 太遠，跳過樣本
                    gt_box = gt_boxes[idx]
                else:
                    continue  # 圖像中沒有真實標籤
                delta = self._bbox2delta(gt_box, pred_box)
                self.samples.append((img_path, 0, pred_box, delta))  # cls=0 存根

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
        img_path, _, pred_box, delta_true = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # 正確計算 gt_box 的座標
        gt_box = self._apply_delta_to_bbox(pred_box, delta_true)

        # 根據正確的座標裁切 patch
        gt_patch = self._letterbox(img, gt_box, self.img_size)
        pred_patch = self._letterbox(img, pred_box, self.img_size)

        # 轉換為張量
        pred_patch = self.transform(pred_patch)
        gt_patch = self.transform(gt_patch)
        
        return pred_patch, gt_patch, delta_true, pred_box, str(img_path)