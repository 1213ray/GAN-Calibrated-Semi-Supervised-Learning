
"""
比較真實標籤 (GT)、校準前標籤、校準後標籤的品質。

此腳本會執行以下操作：
1.  分別計算「校準前」和「校準後」標籤相對於 GT 的 TP, FP, FN, Precision, Recall, F1-Score。
2.  基於 IoU 變化量 (Delta IoU) 來分析校準成效。

用法:
    python compare_pseudo_vs_gt.py `
        --gt-dir   path/to/ground_truth_labels `
        --pre-dir  path/to/pre_calibration_labels `
        --post-dir path/to/post_calibration_labels `
        --iou-threshold 0.5
"""
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# --- Utility Functions ---

def iou_yolo(b1, b2):
    """計算兩個 YOLO 格式 (cx, cy, w, h) 邊界框的 IoU。"""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa1, ya1, xa2, ya2 = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
    xb1, yb1, xb2, yb2 = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2
    iw = max(0, min(xa2, xb2) - max(xa1, xb1))
    ih = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return 0. if union == 0 else inter / union

def load_labels(path: Path):
    """將 YOLO 標籤檔案載入為 (class_id, (cx, cy, w, h)) 的列表。"""
    if not path.exists(): return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        parts = list(map(float, line.split()))
        out.append((int(parts[0]), tuple(parts[1:5])))
    return out

def calculate_metrics(pred_boxes, gt_boxes, iou_threshold):
    """執行貪婪匹配並計算 TP, FP, FN。"""
    stats = {"tp": 0, "fp": 0, "fn": 0}
    matched_preds = set()
    for gt_cls, gt_box in gt_boxes:
        best_iou, best_pred_idx = -1, -1
        for i, (pred_cls, pred_box) in enumerate(pred_boxes):
            if i in matched_preds or pred_cls != gt_cls: continue
            iou = iou_yolo(gt_box, pred_box)
            if iou > best_iou:
                best_iou, best_pred_idx = iou, i
        if best_iou >= iou_threshold:
            stats["tp"] += 1
            if best_pred_idx != -1: matched_preds.add(best_pred_idx)
        else:
            stats["fn"] += 1
    stats["fp"] = len(pred_boxes) - len(matched_preds)
    return stats

def print_report(title, stats, total_images, missing_files=0):
    """印出格式化的評估報告。"""
    p = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
    r = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"\n=== {title} ({total_images} 張圖像) ===")
    print(f"TP: {stats['tp']:<6} FP: {stats['fp']:<6} FN: {stats['fn']:<6}")
    print(f"精確率: {p:.3f}  召回率: {r:.3f}  F1-Score: {f1:.3f}")
    if missing_files > 0:
        print(f"警告：因檔案不齊全，跳過了 {missing_files} 組標籤。")

# --- Main Logic ---

def main(args):
    gt_dir, pre_dir, post_dir = Path(args.gt_dir), Path(args.pre_dir), Path(args.post_dir)

    stats_pre, stats_post = defaultdict(int), defaultdict(int)
    delta_ious = []
    total_gt_boxes = 0
    total_files, missing_files = 0, 0

    label_files = list(post_dir.glob("*.txt"))
    if not label_files:
        print(f"錯誤：在校準後標籤目錄中找不到任何 .txt 檔案: {post_dir}")
        return

    for post_file in label_files:
        basename = post_file.name
        gt_file, pre_file = gt_dir / basename, pre_dir / basename

        if not (gt_file.exists() and pre_file.exists()):
            missing_files += 1
            continue
        
        total_files += 1
        gt_boxes = load_labels(gt_file)
        pre_boxes = load_labels(pre_file)
        post_boxes = load_labels(post_file)
        total_gt_boxes += len(gt_boxes)

        # 1. 計算校準前後的整體指標
        pre_metrics = calculate_metrics(pre_boxes, gt_boxes, args.iou_threshold)
        post_metrics = calculate_metrics(post_boxes, gt_boxes, args.iou_threshold)
        for key in pre_metrics: stats_pre[key] += pre_metrics[key]
        for key in post_metrics: stats_post[key] += post_metrics[key]

        # 2. 根據每個 GT 框計算 Delta IoU
        for gt_cls, gt_box in gt_boxes:
            best_iou_pre = max([iou_yolo(gt_box, pre_box) for pre_cls, pre_box in pre_boxes if pre_cls == gt_cls], default=0.0)
            best_iou_post = max([iou_yolo(gt_box, post_box) for post_cls, post_box in post_boxes if post_cls == gt_cls], default=0.0)
            delta_ious.append(best_iou_post - best_iou_pre)

    # --- 最終報告 ---
    print(f"使用 IoU 閾值進行比較: {args.iou_threshold}")
    print_report("校準前 (Before)", stats_pre, total_files, missing_files)
    print_report("校準後 (After)", stats_post, total_files)
    
    # Delta IoU 分析
    if delta_ious:
        delta_ious_np = np.array(delta_ious)
        avg_iou_pre = np.mean([d - delta for d, delta in zip(delta_ious_np, delta_ious_np)]) # This is incorrect, need to recalculate
        avg_iou_post = np.mean([d for d in delta_ious_np]) # This is incorrect, need to recalculate
        
        all_ious_pre = []
        all_ious_post = []
        for post_file in label_files:
            basename = post_file.name
            gt_file, pre_file = gt_dir / basename, pre_dir / basename
            if not (gt_file.exists() and pre_file.exists()): continue
            gt_boxes, pre_boxes, post_boxes = load_labels(gt_file), load_labels(pre_file), load_labels(post_file)
            for gt_cls, gt_box in gt_boxes:
                all_ious_pre.append(max([iou_yolo(gt_box, b) for c, b in pre_boxes if c == gt_cls], default=0.0))
                all_ious_post.append(max([iou_yolo(gt_box, b) for c, b in post_boxes if c == gt_cls], default=0.0))

        print("\n--- Delta IoU 校準分析 ---")
        print(f"分析的 GT 邊界框總數: {total_gt_boxes}")
        print(f"平均 IoU (校準前): {np.mean(all_ious_pre):.4f}")
        print(f"平均 IoU (校準後): {np.mean(all_ious_post):.4f}")
        print(f"平均 Delta IoU (IoU 變化量): {np.mean(delta_ious_np):+.4f}")
        print(f"IoU 提升的框數: {np.sum(delta_ious_np > 1e-6):>6}")
        print(f"IoU 下降的框數: {np.sum(delta_ious_np < -1e-6):>6}")
        print(f"IoU 不變的框數: {np.sum(np.abs(delta_ious_np) <= 1e-6):>6}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比較 GT、校準前和校準後標籤的品質。", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--gt-dir', type=str, required=True, help="真實標籤 (Ground Truth) 的資料夾路徑。")
    parser.add_argument('--pre-dir', type=str, required=True, help="校準前 (偽) 標籤的資料夾路徑。")
    parser.add_argument('--post-dir', type=str, required=True, help="校準後標籤的資料夾路徑。")
    parser.add_argument('--iou-threshold', type=float, default=0.5, help="用於判斷是否匹配 (TP) 的 IoU 閾值。預設值: 0.5")
    args = parser.parse_args()
    main(args)
