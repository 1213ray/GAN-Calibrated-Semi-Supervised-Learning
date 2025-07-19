#!/usr/bin/env python3
"""
Pure EIoU Loss Functions for CGAN Calibration with WGAN-GP
只包含EIoU相關的損失函數和工具函數，使用 WGAN-GP 對抗損失
"""

import torch
import torch.nn as nn

class EIoULoss(nn.Module):
    """
    Efficient IoU Loss for bounding box regression
    EIoU = IoU - ρ²(b,b^gt)/c² - ρ²(w,w^gt)/c_w² - ρ²(h,h^gt)/c_h²
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: (N, 4) [cx, cy, w, h] normalized
            target_boxes: (N, 4) [cx, cy, w, h] normalized
        """
        # Convert to corners for IoU calculation
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        
        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
        
        # Calculate intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + self.eps)
        
        # Calculate enclosing box for EIoU
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        enclose_c = enclose_w ** 2 + enclose_h ** 2  # diagonal²
        
        # Center point distance
        center_distance = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2
        
        # Width and height differences
        w_distance = (pred_boxes[:, 2] - target_boxes[:, 2]) ** 2
        h_distance = (pred_boxes[:, 3] - target_boxes[:, 3]) ** 2
        
        # EIoU calculation
        eiou = iou - center_distance / (enclose_c + self.eps) - w_distance / (enclose_w ** 2 + self.eps) - h_distance / (enclose_h ** 2 + self.eps)
        
        # Return 1 - EIoU as loss (lower is better)
        return 1 - eiou.mean()

class HybridLoss(nn.Module):
    """
    Pure EIoU Loss for CGAN training
    """
    def __init__(self, lambda_iou=1.0, **kwargs):
        super().__init__()
        self.lambda_iou = lambda_iou
        self.iou_loss = EIoULoss()
        print(f"Using Pure EIoU Loss (lambda_iou={lambda_iou})")
    
    def forward(self, pred_deltas, target_deltas, pred_boxes, target_boxes):
        """
        Args:
            pred_deltas: (N, 4) predicted delta values (unused)
            target_deltas: (N, 4) target delta values (unused)
            pred_boxes: (N, 4) predicted boxes after applying deltas
            target_boxes: (N, 4) target boxes
        """
        # Pure EIoU loss - only geometric constraints
        iou_loss = self.iou_loss(pred_boxes, target_boxes)
        total_loss = self.lambda_iou * iou_loss
        
        return total_loss, iou_loss

def smooth_clamp(x, min_val, max_val, temperature=0.5):
    """
    Smooth clamp function that maintains gradients
    使用較大的溫度值以避免梯度消失問題
    """
    center = (min_val + max_val) / 2
    normalized = (x - center) / temperature
    return min_val + (max_val - min_val) * torch.sigmoid(normalized)

def apply_delta_to_bbox(bbox, delta, training=True):
    """
    Apply predicted deltas to bounding boxes with enhanced numerical stability
    Args:
        bbox: (N, 4) [cx, cy, w, h] normalized coordinates
        delta: (N, 4) [dx_rel, dy_rel, log_dw, log_dh] regression deltas
        training: bool, whether in training mode (affects gradient handling)
    Returns:
        calibrated_bbox: (N, 4) [cx, cy, w, h] calibrated coordinates
    """
    # 更保守的 delta 範圍限制，避免過度校正
    delta_clamp_range = 1.5  # 從 2.0 降低到 1.5
    
    if training:
        # 訓練時使用 smooth clamp 保持梯度連續性
        delta_clamped = smooth_clamp(delta, -delta_clamp_range, delta_clamp_range)
    else:
        # 推論時使用硬截斷確保穩定性
        delta_clamped = torch.clamp(delta, -delta_clamp_range, delta_clamp_range)
    
    # 中心點偏移：使用相對於邊界框尺寸的偏移
    cx = bbox[:, 0] + delta_clamped[:, 0] * bbox[:, 2]
    cy = bbox[:, 1] + delta_clamped[:, 1] * bbox[:, 3]
    
    # 尺寸縮放：加入數值穩定性保護
    w_scale = torch.exp(torch.clamp(delta_clamped[:, 2], -1.0, 1.0))  # 限制縮放範圍
    h_scale = torch.exp(torch.clamp(delta_clamped[:, 3], -1.0, 1.0))
    w = bbox[:, 2] * w_scale
    h = bbox[:, 3] * h_scale
    
    # 邊界約束：確保結果在有效範圍內
    if training:
        cx = smooth_clamp(cx, 0.05, 0.95)
        cy = smooth_clamp(cy, 0.05, 0.95)
        w = smooth_clamp(w, 0.02, 0.8)   # 稍微放寬最小值限制
        h = smooth_clamp(h, 0.02, 0.8)
    else:
        cx = torch.clamp(cx, 0.05, 0.95)
        cy = torch.clamp(cy, 0.05, 0.95)
        w = torch.clamp(w, 0.02, 0.8)
        h = torch.clamp(h, 0.02, 0.8)
    
    return torch.stack([cx, cy, w, h], dim=-1)

def iou_metric(pred_boxes, target_boxes, eps=1e-6):
    """
    Calculate IoU for evaluation
    """
    # Convert to corners
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    
    target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
    
    # Calculate intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + eps)
    
    return iou

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    計算 WGAN-GP 的梯度懲罰項
    Args:
        discriminator: 判別器模型
        real_samples: 真實樣本 (pred_patch, gt_patch)
        fake_samples: 生成樣本 (pred_patch, refined_patch)
        device: 計算設備
    Returns:
        gradient_penalty: 梯度懲罰項
    """
    batch_size = real_samples[0].size(0)
    
    # 隨機插值係數
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples[0])
    
    # 插值樣本
    interpolated_pred = (alpha * real_samples[0] + (1 - alpha) * fake_samples[0]).detach()
    interpolated_other = (alpha * real_samples[1] + (1 - alpha) * fake_samples[1]).detach()
    
    interpolated_pred.requires_grad_(True)
    interpolated_other.requires_grad_(True)
    
    # 計算判別器對插值樣本的輸出
    d_interpolated = discriminator(interpolated_pred, interpolated_other)
    
    # 計算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=[interpolated_pred, interpolated_other],
        grad_outputs=torch.ones_like(d_interpolated, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )
    
    # 計算梯度的範數
    gradients_pred = gradients[0].view(batch_size, -1)
    gradients_other = gradients[1].view(batch_size, -1)
    gradients_norm = torch.sqrt(
        torch.sum(gradients_pred ** 2, dim=1) + 
        torch.sum(gradients_other ** 2, dim=1) + 1e-12
    )
    
    # 梯度懲罰 (目標梯度範數為 1)
    gradient_penalty = torch.mean((gradients_norm - 1) ** 2)
    
    return gradient_penalty