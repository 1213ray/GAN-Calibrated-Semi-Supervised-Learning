#!/usr/bin/env python3
"""
Debug dataset loading
"""

import sys
from pathlib import Path
import yaml

# Add cgan to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import CalibratorDataset

def main():
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['data_dir']
    print(f"Data directory: {data_dir}")
    
    # Check if directory exists
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Data directory does not exist: {data_path}")
        return
    
    # Check subdirectories
    for subdir in ['images', 'labels_gt', 'labels_pred']:
        subpath = data_path / subdir
        if subpath.exists():
            files = list(subpath.glob('*'))
            print(f"OK {subdir}: {len(files)} files")
        else:
            print(f"ERROR {subdir}: directory not found")
    
    # Try to create dataset
    try:
        print("\nCreating CalibratorDataset...")
        dataset = CalibratorDataset(data_dir, img_size=config['img_size'])
        print(f"OK Dataset created successfully")
        print(f"  Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            print(f"  Samples overview:")
            for i, sample in enumerate(dataset.samples[:5]):  # Show first 5
                img_path, cls, pred_box, delta, gt_box = sample
                print(f"    Sample {i}: {img_path.name}")
                print(f"      pred_box: {pred_box}")
                print(f"      gt_box: {gt_box}")
                print(f"      delta: {delta}")
                print()
        
    except Exception as e:
        print(f"ERROR creating dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()