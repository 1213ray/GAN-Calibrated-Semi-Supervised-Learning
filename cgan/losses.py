#!/usr/bin/env python3
"""
Pure EIoU Loss Functions for CGAN Calibration
只包含EIoU相關的損失函數和工具函數
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

def smooth_clamp(x, min_val, max_val, temperature=0.1):
    """Smooth clamp function that maintains gradients"""
    return min_val + (max_val - min_val) * torch.sigmoid((x - (min_val + max_val) / 2) / temperature)

def apply_delta_to_bbox(bbox, delta, training=True):
    """
    Apply predicted deltas to bounding boxes with consistent clamping behavior
    Args:
        bbox: (N, 4) [cx, cy, w, h]
        delta: (N, 4) [dx_rel, dy_rel, log_dw, log_dh]
        training: bool, whether in training mode (affects gradient handling)
    Returns:
        calibrated_bbox: (N, 4) [cx, cy, w, h]
    """
    if training:
        # 訓練時使用smooth clamp保持梯度連續性
        delta_clamped = smooth_clamp(delta, -2, 2)
    else:
        # 推論時使用硬截斷以確保穩定性
        delta_clamped = torch.clamp(delta, -2, 2)
    
    cx = bbox[:, 0] + delta_clamped[:, 0] * bbox[:, 2]
    cy = bbox[:, 1] + delta_clamped[:, 1] * bbox[:, 3]
    w = bbox[:, 2] * torch.exp(delta_clamped[:, 2])
    h = bbox[:, 3] * torch.exp(delta_clamped[:, 3])
    
    if training:
        # 訓練時使用smooth clamp保持梯度連續性
        cx = smooth_clamp(cx, 0.05, 0.95)
        cy = smooth_clamp(cy, 0.05, 0.95)
        w = smooth_clamp(w, 0.01, 0.9)
        h = smooth_clamp(h, 0.01, 0.9)
    else:
        # 推論時使用硬截斷
        cx = torch.clamp(cx, 0.05, 0.95)
        cy = torch.clamp(cy, 0.05, 0.95)
        w = torch.clamp(w, 0.01, 0.9)
        h = torch.clamp(h, 0.01, 0.9)
    
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