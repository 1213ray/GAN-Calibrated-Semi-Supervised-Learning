#!/usr/bin/env python3
"""
Enhanced Loss Functions for CGAN Calibration
包含IoU, GIoU, 和混合損失函數
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    """
    IoU Loss for bounding box regression
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
        iou = inter_area / (union_area + self.eps)
        
        # Return 1 - IoU as loss (lower is better)
        return 1 - iou.mean()

class GIoULoss(nn.Module):
    """
    Generalized IoU Loss for bounding box regression
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
        iou = inter_area / (union_area + self.eps)
        
        # Calculate enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # Calculate GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + self.eps)
        
        # Return 1 - GIoU as loss (lower is better)
        return 1 - giou.mean()

class FocalLoss(nn.Module):
    """Focal Loss for hard examples"""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class HybridLoss(nn.Module):
    """
    Enhanced hybrid loss with focal loss for hard examples
    """
    def __init__(self, 
                 lambda_iou=2.0, 
                 lambda_l1=0.05, 
                 lambda_focal=0.5,
                 use_giou=True, 
                 eps=1e-6):
        super().__init__()
        self.lambda_iou = lambda_iou
        self.lambda_l1 = lambda_l1
        self.lambda_focal = lambda_focal
        self.use_giou = use_giou
        
        if use_giou:
            self.iou_loss = GIoULoss(eps=eps)
        else:
            self.iou_loss = IoULoss(eps=eps)
        
        self.l1_loss = nn.SmoothL1Loss()
        self.focal_loss = FocalLoss()
    
    def forward(self, pred_deltas, target_deltas, pred_boxes, target_boxes):
        """
        Args:
            pred_deltas: (N, 4) predicted delta values
            target_deltas: (N, 4) target delta values
            pred_boxes: (N, 4) predicted boxes after applying deltas
            target_boxes: (N, 4) target boxes
        """
        # IoU/GIoU loss on boxes
        iou_loss = self.iou_loss(pred_boxes, target_boxes)
        
        # L1 loss on deltas
        l1_loss = self.l1_loss(pred_deltas, target_deltas)
        
        # Focal loss for hard examples
        delta_diff = torch.abs(pred_deltas - target_deltas)
        focal_loss = self.focal_loss(delta_diff, torch.zeros_like(delta_diff))
        
        # Combine losses
        total_loss = (self.lambda_iou * iou_loss + 
                     self.lambda_l1 * l1_loss + 
                     self.lambda_focal * focal_loss)
        
        return total_loss, iou_loss, l1_loss, focal_loss

def smooth_clamp(x, min_val, max_val, temperature=0.1):
    """Smooth clamp function that maintains gradients"""
    return min_val + (max_val - min_val) * torch.sigmoid((x - (min_val + max_val) / 2) / temperature)

def apply_delta_to_bbox(bbox, delta):
    """
    Apply predicted deltas to bounding boxes with smooth clamping for better gradients
    Args:
        bbox: (N, 4) [cx, cy, w, h]
        delta: (N, 4) [dx_rel, dy_rel, log_dw, log_dh]
    Returns:
        calibrated_bbox: (N, 4) [cx, cy, w, h]
    """
    # 使用smooth clamp保持梯度連續性
    delta_smooth = smooth_clamp(delta, -2, 2)
    
    cx = bbox[:, 0] + delta_smooth[:, 0] * bbox[:, 2]
    cy = bbox[:, 1] + delta_smooth[:, 1] * bbox[:, 3]
    w = bbox[:, 2] * torch.exp(delta_smooth[:, 2])
    h = bbox[:, 3] * torch.exp(delta_smooth[:, 3])
    
    # 使用smooth clamp保持梯度連續性
    cx = smooth_clamp(cx, 0.05, 0.95)
    cy = smooth_clamp(cy, 0.05, 0.95)
    w = smooth_clamp(w, 0.01, 0.9)
    h = smooth_clamp(h, 0.01, 0.9)
    
    return torch.stack([cx, cy, w, h], dim=-1)

def apply_delta_to_bbox_eval(bbox, delta):
    """
    Apply deltas for evaluation with hard clamp for safety
    """
    delta_clamped = torch.clamp(delta, -2, 2)
    
    cx = bbox[:, 0] + delta_clamped[:, 0] * bbox[:, 2]
    cy = bbox[:, 1] + delta_clamped[:, 1] * bbox[:, 3]
    w = bbox[:, 2] * torch.exp(delta_clamped[:, 2])
    h = bbox[:, 3] * torch.exp(delta_clamped[:, 3])
    
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