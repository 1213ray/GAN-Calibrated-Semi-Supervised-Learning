"""models.py  —  cGAN Patch-Calibrator  (步驟 1/5)
=================================================
生成器 (U-Net → Δ) 和 PatchGAN 判別器
‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
此模組定義了校準 cGAN 使用的兩個神經網路類別：
  • **GeneratorUNet**  — 接收一個 3 通道的預測圖像塊，並回歸一個
    4 維偏移向量 Δ = (dx_rel, dy_rel, log dw, log dh)，用於校正
    邊界框位置。其架構為：4 × 下採樣塊
    + 4 × 上採樣塊，帶有跳躍連接（一個輕量級的 U-Net）。
  • **Discriminator**  — 70×70 PatchGAN。接收一個連接的張量
    (pred_patch, other_patch)，其中 *other_patch* 是 GT
    圖像塊（真實）或 G 修正後的圖像塊（偽造）。輸出一個分數圖
    在 [0,1] 之間。
兩個模型都使用通用的 `weights_init_normal` 輔助函數，因此我們可以輕鬆地
應用 pix2pix 中看到的 *N(0, 0.02)* 卷積初始化。

使用範例（健全性檢查）：
>>> from models import GeneratorUNet, Discriminator, weights_init_normal
>>> G = GeneratorUNet().apply(weights_init_normal)
>>> D = Discriminator().apply(weights_init_normal)
>>> x = torch.randn(2, 3, 128, 128)  # 預測圖像塊的批次
>>> delta = G(x)                     # → (2,4)
>>> fake_score = D(x, x)             # → (2,1,14,14)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pathlib import Path

# -------------------------------------------------
#  Helper: 權重初始化 (pix2pix 風格)
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
#  U-Net 的構建塊
# -------------------------------------------------

class UNetDown(nn.Module):
    def __init__(self, in_size: int, out_size: int, normalize: bool = True, dropout: float | None = None):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
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
#  生成器 (U-Net 骨幹 → Δ 向量)
# -------------------------------------------------

class GeneratorUNet(nn.Module):
    """4層下採樣 / 4層上採樣的 U-Net，輸出 4 維的邊界框校正 Δ。"""

    def __init__(self, delta_scale: float = None):
        super().__init__()
        if delta_scale is None:
            # 載入預設配置
            config_path = Path(__file__).parent / "config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            delta_scale = config['delta_scale']
        self.delta_scale = float(delta_scale)

        # 編碼器
        self.down1 = UNetDown(3,   64, normalize=False)  # (128→64)
        self.down2 = UNetDown(64, 128)                   # 64→32
        self.down3 = UNetDown(128, 256)                  # 32→16
        self.down4 = UNetDown(256, 512, dropout=0.5)     # 16→8

        # 解碼器
        self.up1   = UNetUp(512, 256, dropout=0.5)
        self.up2   = UNetUp(512, 128, dropout=0.5)  # 256+256=512 輸入通道
        self.up3   = UNetUp(256,  64)              # 128+128=256
        self.up4   = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),  # cat 後 64+64=128 輸入通道
            nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
        )  # 輸出大小回到 128×128

        # 全局池化 → Δ (4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_delta = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 4),
            nn.Tanh(),       # (-1,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # 編碼器
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # 帶跳躍連接的解碼器
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3)          # 最後一層沒有跳躍連接以保持通道數最小

        # 全局池化 → Δ
        pooled = self.avg_pool(u4)  # (B, 64, 1, 1)
        delta_raw = self.fc_delta(pooled)  # (B,4) 在 (-1,1) 之間
        return delta_raw * self.delta_scale

# -------------------------------------------------
#  簡化的生成器 (更適合回歸任務)
# -------------------------------------------------

class GeneratorSimpleRegressor(nn.Module):
    """簡化的CNN回歸器，專門用於邊界框校正。"""
    
    def __init__(self, delta_scale: float = None):
        super().__init__()
        if delta_scale is None:
            # 載入預設配置
            config_path = Path(__file__).parent / "config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            delta_scale = config['delta_scale']
        self.delta_scale = float(delta_scale)
        
        # 特徵提取器
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )
        
        # 回歸頭
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 4),
            nn.Tanh()  # (-1, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        delta_raw = self.regressor(features)
        return delta_raw * self.delta_scale

# -------------------------------------------------
#  判別器 (70×70 PatchGAN)
# -------------------------------------------------

class Discriminator(nn.Module):
    """PatchGAN，用於判斷 (pred_patch, other_patch) 對。"""

    def __init__(self, spectral_norm: bool = None):
        super().__init__()
        
        if spectral_norm is None:
            # 載入預設配置
            config_path = Path(__file__).parent / "config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            spectral_norm = config['spectral_norm']

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
            *conv_block(6,   64, norm=False),  # (pred+other) 連接 → C=6
            *conv_block(64, 128),
            *conv_block(128,256),
            *conv_block(256,512),
            # 最終卷積 (stride=1) → 對於 128×128 輸入，分數圖約為 14×14
            nn.Conv2d(512, 1, 4, stride=1, padding=1, bias=False)
        ])

    def forward(self, pred_patch: torch.Tensor, other_patch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # 沿通道維度連接 (B,6,H,W)
        x = torch.cat([pred_patch, other_patch], dim=1)
        return self.model(x)
