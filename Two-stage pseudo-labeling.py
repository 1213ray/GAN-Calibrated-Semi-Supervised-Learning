
# ─────────────────── 參數區 ───────────────────
WEIGHTS_PATH = r"C:\Users\alian\PycharmProjects\yolov8\models\person_v1.pt"        # models
SOURCE_DIR   = r"C:\Users\alian\PycharmProjects\yolov8\datasets\500_100_100\unlabels\images"     # 原始影像資料夾
DEST_DIR     = r"C:\Users\alian\PycharmProjects\yolov8\datasets\500_100_100\two_stage"     # 輸出根目錄

CONF_HIGH    = 0.85      # 高置信度門檻
CONF_LOW     = 0.40      # 低置信度門檻
OVL_THR      = 0.35      # 高低框 IoU 上限
IOU_NMS      = 0.4       # NMS IoU
CLASS_FILTER = None      # 只偵測 person→[0]；全類別=None
NEED_ONLY_LOW = False    # True→將「僅低框」影像另存 only_low/
AUGMENT_LOW   = True     # 第二階段是否開 TTA
# ─────────────────────────────────────────────

import shutil
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

# ---------- 工具函式 ----------
def iou_max(a_xyxy, b_xyxy):
    return box_iou(a_xyxy, b_xyxy).max(dim=1).values if len(a_xyxy) and len(b_xyxy) else torch.zeros(len(a_xyxy))

def save_txt(path, xywhn, cls_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for (cx, cy, w, h), c in zip(xywhn, cls_ids):
            f.write(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# ---------- 階段 1：僅高框 ----------
stage1 = Path(DEST_DIR, "stage1")
stage2 = Path(DEST_DIR, "stage2")

# 建立 Stage1 的目錄
high_img1 = stage1/"high"/"images";  high_img1.mkdir(parents=True, exist_ok=True)
high_lab1 = stage1/"high"/"labels";  high_lab1.mkdir(parents=True, exist_ok=True)
high_list = stage1/"high_list.txt"

low_img1 = stage1/"low"/"images";   low_img1.mkdir(parents=True, exist_ok=True)


model = YOLO(WEIGHTS_PATH)
img_paths = [p for p in Path(SOURCE_DIR).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}]

with high_list.open("w") as hlst:
    for p in tqdm(img_paths, desc="Stage-1 High"):
        res = model.predict(p, conf=CONF_HIGH, iou=IOU_NMS, classes=CLASS_FILTER, verbose=False, augment=True)[0]
        if not len(res.boxes):
            shutil.copy2(p, low_img1 / p.name)
            continue  # 無高框→不納入
        # 複製影像 & 寫高框 .txt
        shutil.copy2(p, high_img1/p.name)
        save_txt(high_lab1/f"{p.stem}.txt", res.boxes.xywhn.cpu(), res.boxes.cls.cpu())
        hlst.write(p.name + "\n")   # 記錄檔名供第二階段

print(f"Stage-1 完成，高框影像數：{len(list(high_img1.iterdir()))}")
print(f"Stage-1 完成，低框影像數：{len(list(low_img1.iterdir()))}")

# ---------- 階段 2：補漏低框 ----------
stage2 = Path(DEST_DIR, "stage2")
high_img2 = stage2/"high"/"images";  high_img2.mkdir(parents=True, exist_ok=True)
high_lab2 = stage2/"high"/"labels";  high_lab2.mkdir(parents=True, exist_ok=True)
only_low_img = stage2/"only_low"/"images"; only_low_lab = stage2/"only_low"/"labels"

with high_list.open() as hlst:
    names = [n.strip() for n in hlst if n.strip()]

for name in tqdm(names, desc="Stage-2 Low pass"):
    src_img = Path(SOURCE_DIR)/name
    # 讀取第一階段高框
    txt1 = high_lab1/f"{Path(name).stem}.txt"
    boxes_hi_xywhn = torch.tensor([[float(x) for x in line.split()[1:]] for line in txt1.read_text().splitlines()])
    boxes_hi_cls   = torch.zeros(len(boxes_hi_xywhn)) if len(boxes_hi_xywhn) else torch.tensor([])

    # 第二次推論：低門檻
    res2 = model.predict(src_img, conf=CONF_LOW, iou=IOU_NMS,
                         classes=CLASS_FILTER, verbose=False, augment=AUGMENT_LOW)[0]
    # 分割高 / 低（重算一次避免精度差異）
    confs2 = res2.boxes.conf
    mask_hi2 = confs2 >= CONF_HIGH
    boxes_lo = res2.boxes[~mask_hi2]

    # IoU 篩選補漏框
    keep_lo = []
    if len(boxes_lo):
        if len(boxes_hi_xywhn):
            iou_max_vals = iou_max(boxes_lo.xyxy, res2.boxes[mask_hi2|mask_hi2].xyxy if len(res2.boxes) else torch.empty((0,4)))
            keep_lo = boxes_lo[iou_max_vals < OVL_THR]
        else:
            keep_lo = boxes_lo

    # 組最終框
    xywhn_all = torch.cat([boxes_hi_xywhn, keep_lo.xywhn.cpu()], dim=0) if len(keep_lo) else boxes_hi_xywhn
    cls_all   = torch.cat([boxes_hi_cls,   keep_lo.cls.cpu()],   dim=0) if len(keep_lo) else boxes_hi_cls

    if len(xywhn_all):
        # 有框 ⇒ 視為 high
        shutil.copy2(src_img, high_img2/src_img.name)
        save_txt(high_lab2/f"{src_img.stem}.txt", xywhn_all, cls_all)
        # 若只含低框且需要分流
        if NEED_ONLY_LOW and len(boxes_hi_xywhn)==0:
            only_low_img.mkdir(parents=True, exist_ok=True)
            only_low_lab.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, only_low_img/src_img.name)
            save_txt(only_low_lab/f"{src_img.stem}.txt", xywhn_all, cls_all)
    else:
        # 極端情況：第二次仍無框，可忽略或另存
        pass

print("Stage-2 完成！")
print(f"最終 high 影像：{len(list(high_img2.iterdir()))}")
if NEED_ONLY_LOW:
    print(f"  其中『僅低框』影像：{len(list(only_low_img.iterdir()))}")
