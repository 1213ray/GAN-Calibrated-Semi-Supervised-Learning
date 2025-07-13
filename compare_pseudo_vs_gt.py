#!/usr/bin/env python3
"""
以偽標籤檔為基準，比對對應的手動標記：
  - 只統計「偽標籤存在」的影像品質
  - 額外報告：有偽標籤但找不到 GT 的檔案數
"""

from pathlib import Path
from collections import defaultdict

GT_LABELS_DIR     = Path(r"C:\Users\20039\PycharmProjects\yoloV8\pythonProject\datasets\original_data\labels")
PSEUDO_LABELS_DIR = Path(r"C:\Users\20039\PycharmProjects\yoloV8\pythonProject\datasets\pseudo_dataset_person_two_pass\v3\stage2\high\labels")
IOU_THRESHOLD     = 0.40
CLASS_NAMES       = None      # 自行填 ["person"] 之類即可

# ---------- IoU ----------
def iou_yolo(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xa1,ya1,xa2,ya2 = x1-w1/2, y1-h1/2, x1+w1/2, y1+h1/2
    xb1,yb1,xb2,yb2 = x2-w2/2, y2-h2/2, x2+w2/2, y2+h2/2
    iw = max(0, min(xa2, xb2) - max(xa1, xb1))
    ih = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter = iw * ih
    union = w1*h1 + w2*h2 - inter
    return 0. if union == 0 else inter / union

def load(path: Path):
    if not path.exists(): return []
    out=[]
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        p=list(map(float,line.split()))
        out.append((int(p[0]), *p[1:5]))
    return out

# ---------- 統計 ----------
per_cls = defaultdict(lambda: {"tp":0,"fp":0,"fn":0})
overall = {"tp":0,"fp":0,"fn":0}
missing_gt = 0
total_imgs = 0

for pseudo_file in PSEUDO_LABELS_DIR.glob("*.txt"):
    total_imgs += 1
    gt_file = GT_LABELS_DIR / pseudo_file.name
    if not gt_file.exists():
        missing_gt += 1
        continue            # 找不到 GT，跳過但記錄

    pred_boxes = load(pseudo_file)
    gt_boxes   = load(gt_file)

    # greedy matching
    matched=set()
    for g in gt_boxes:
        best_iou, best_j = 0., None
        for j,p in enumerate(pred_boxes):
            if j in matched or p[0]!=g[0]: continue
            iou = iou_yolo(g[1:], p[1:])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= IOU_THRESHOLD and best_j is not None:
            overall["tp"]+=1; per_cls[g[0]]["tp"]+=1; matched.add(best_j)
        else:
            overall["fn"]+=1; per_cls[g[0]]["fn"]+=1
    for j,p in enumerate(pred_boxes):
        if j not in matched:
            overall["fp"]+=1; per_cls[p[0]]["fp"]+=1

# ---------- 報表 ----------
def prf(d):
    p = d["tp"]/(d["tp"]+d["fp"]) if d["tp"]+d["fp"] else 0
    r = d["tp"]/(d["tp"]+d["fn"]) if d["tp"]+d["fn"] else 0
    f = 2*p*r/(p+r) if p+r else 0
    return p,r,f

P,R,F = prf(overall)
print(f"\n=== Quality on {total_imgs - missing_gt}/{total_imgs} images (IoU ≥ {IOU_THRESHOLD}) ===")
print(f"TP {overall['tp']}  FP {overall['fp']}  FN {overall['fn']}")
print(f"Precision {P:.3f}  Recall {R:.3f}  F1 {F:.3f}")

if missing_gt:
    print(f"\nWARNING: {missing_gt} pseudo-label files had **no matching GT**.")

if per_cls:
    print(f"\n--- Per-class ---")
    print(f"{'class':<15}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>9}{'R':>9}{'F1':>9}")
    for cid, s in sorted(per_cls.items()):
        p,r,f = prf(s)
        name = str(cid) if CLASS_NAMES is None or cid>=len(CLASS_NAMES) else CLASS_NAMES[cid]
        print(f"{name:<15}{s['tp']:>6}{s['fp']:>6}{s['fn']:>6}{p:>9.3f}{r:>9.3f}{f:>9.3f}")
