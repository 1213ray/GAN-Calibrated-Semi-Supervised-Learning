#!/usr/bin/env python3
"""
Test script to verify CGAN fixes
"""

import torch
import yaml
from pathlib import Path
from models import GeneratorUNet, GeneratorSimpleRegressor, Discriminator
from losses import HybridLoss, apply_delta_to_bbox, iou_metric
from dataset import CalibratorDataset
from torch.utils.data import DataLoader
import numpy as np

def test_loss_functions():
    """測試損失函數是否正常工作"""
    print("Testing loss functions...")
    
    # 創建測試數據
    batch_size = 4
    pred_deltas = torch.randn(batch_size, 4) * 0.1
    target_deltas = torch.randn(batch_size, 4) * 0.1
    pred_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]] * batch_size)
    
    # 應用 delta 計算目標框
    pred_calibrated = apply_delta_to_bbox(pred_boxes, pred_deltas)
    target_boxes = apply_delta_to_bbox(pred_boxes, target_deltas)
    
    # 測試混合損失
    hybrid_loss = HybridLoss(lambda_iou=1.0, lambda_l1=0.1, use_giou=True)
    total_loss, iou_loss, l1_loss = hybrid_loss(pred_deltas, target_deltas, pred_calibrated, target_boxes)
    
    print(f"  Hybrid loss: {total_loss:.4f}")
    print(f"  IoU loss: {iou_loss:.4f}")
    print(f"  L1 loss: {l1_loss:.4f}")
    
    # 測試 IoU 度量
    iou_scores = iou_metric(pred_calibrated, target_boxes)
    print(f"  IoU scores: {iou_scores.mean():.4f}")
    
    return total_loss.item() > 0

def test_models():
    """測試模型是否正常工作"""
    print("\nTesting models...")
    
    # 測試參數
    batch_size = 2
    img_size = 256
    delta_scale = 0.7
    
    # 創建測試輸入
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    # 測試 U-Net 生成器
    print("  Testing U-Net generator...")
    gen_unet = GeneratorUNet(delta_scale=delta_scale)
    delta_unet = gen_unet(x)
    print(f"    Output shape: {delta_unet.shape}")
    print(f"    Output range: [{delta_unet.min():.3f}, {delta_unet.max():.3f}]")
    
    # 測試簡化生成器
    print("  Testing Simple generator...")
    gen_simple = GeneratorSimpleRegressor(delta_scale=delta_scale)
    delta_simple = gen_simple(x)
    print(f"    Output shape: {delta_simple.shape}")
    print(f"    Output range: [{delta_simple.min():.3f}, {delta_simple.max():.3f}]")
    
    # 測試判別器
    print("  Testing Discriminator...")
    disc = Discriminator(spectral_norm=True)
    disc_output = disc(x, x)
    print(f"    Output shape: {disc_output.shape}")
    
    return delta_unet.shape[1] == 4 and delta_simple.shape[1] == 4

def test_dataset():
    """測試數據集是否正常工作"""
    print("\nTesting dataset...")
    
    # 載入配置
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    try:
        # 創建數據集
        dataset = CalibratorDataset(config['data_dir'], img_size=config['img_size'])
        print(f"  Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # 測試數據加載
            pred_patch, gt_patch, delta_true, pred_box, img_path = dataset[0]
            print(f"  Patch shape: {pred_patch.shape}")
            print(f"  Delta shape: {delta_true.shape}")
            print(f"  Pred box shape: {pred_box.shape}")
            
            # 測試數據一致性
            gt_box_reconstructed = dataset._apply_delta_to_bbox(pred_box, delta_true)
            delta_reconstructed = dataset._bbox2delta(gt_box_reconstructed, pred_box)
            
            consistency_error = torch.abs(delta_true - delta_reconstructed).max()
            print(f"  Consistency error: {consistency_error:.6f}")
            
            return consistency_error < 1e-5
        else:
            print("  Warning: Dataset is empty!")
            return False
            
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_training_step():
    """測試訓練步驟"""
    print("\nTesting training step...")
    
    # 載入配置
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    try:
        # 創建小批量測試
        dataset = CalibratorDataset(config['data_dir'], img_size=config['img_size'])
        if len(dataset) == 0:
            print("  Skipping: No data available")
            return False
            
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 創建模型
        generator = GeneratorSimpleRegressor(delta_scale=config['delta_scale'])
        discriminator = Discriminator(spectral_norm=config['spectral_norm'])
        
        # 創建損失函數
        hybrid_loss = HybridLoss(
            lambda_iou=config['lambda_iou'],
            lambda_l1=config['lambda_l1'],
            use_giou=config['use_giou']
        )
        
        # 測試一個批次
        for pred_patch, gt_patch, delta_true, pred_box, img_path in loader:
            # 前向傳播
            delta_pred = generator(pred_patch)
            
            # 計算校正框
            calibrated_boxes = apply_delta_to_bbox(pred_box, delta_pred)
            gt_boxes = apply_delta_to_bbox(pred_box, delta_true)
            
            # 計算損失
            total_loss, iou_loss, l1_loss = hybrid_loss(
                delta_pred, delta_true, calibrated_boxes, gt_boxes
            )
            
            # 計算 IoU 改善
            iou_before = iou_metric(pred_box, gt_boxes)
            iou_after = iou_metric(calibrated_boxes, gt_boxes)
            delta_iou = iou_after.mean() - iou_before.mean()
            
            print(f"  Batch tested successfully")
            print(f"  Total loss: {total_loss:.4f}")
            print(f"  IoU loss: {iou_loss:.4f}")
            print(f"  L1 loss: {l1_loss:.4f}")
            print(f"  Delta IoU: {delta_iou:.4f}")
            
            return True
            
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    """主測試函數"""
    print("=== CGAN Fixes Verification ===\n")
    
    tests = [
        ("Loss Functions", test_loss_functions),
        ("Models", test_models),
        ("Dataset", test_dataset),
        ("Training Step", test_training_step)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASS" if result else "FAIL"
            print(f"{status}\n")
        except Exception as e:
            results.append((test_name, False))
            print(f"FAIL - Exception: {e}\n")
    
    # 總結
    print("=== Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! The fixes should work correctly.")
    else:
        print("Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()