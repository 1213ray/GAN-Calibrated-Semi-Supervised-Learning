#!/usr/bin/env python3
"""
KFOLD 交叉偽標籤產生腳本
────────────────────────────────────────────────────
1. 依據 YAML 中的 `train` 影像集合切成 K 折
2. 針對每一折：
   ‣ 使用其餘 K‑1 折資料訓練 YOLOv8 模型
   ‣ 對被劃為「推論折」的影像執行推論，輸出偽標籤（YOLO txt）
3. 每折輸出結構：
   outputs/kfold_pseudo/fold_01/
       ├─ data_fold.yaml           # 本折用的 data 設定檔
       ├─ train/                  # Ultralytics 預設訓練輸出
       └─ pseudo_labels/          # 推論得到的偽標籤

注意：驗證集（val）仍沿用原 YAML 中設定，不參與 KFOLD 切分。
"""

# ────────────── 參數區 ──────────────
YAML_PATH          = r"C:\Users\alian\PycharmProjects\yolov8\yaml\person.yaml"  # 原始資料 yaml
PRETRAINED_WEIGHTS = r"C:\Users\alian\PycharmProjects\yolov8\models\yolov8m.pt" # 起始權重
OUTPUT_ROOT        = r"C:\Users\alian\PycharmProjects\yolov8\datasets\500_100_100\kfold" # 輸出根目錄

KFOLDS             = 5      # 折數
EPOCHS             = 400    # 訓練輪數
BATCH_SIZE         = 16     # 批次大小
PATIENCE           = 20
IMG_SIZE           = 640    # 輸入解析度
WORKERS            = 4      # Dataloader 執行緒
DEVICE             = 0      # GPU id，CPU 請設為 "cpu"
CONF_THR           = 0.1   # 偽標籤推論置信度下限
IOU_NMS            = 0.9    # NMS IoU 上限

from pathlib import Path           # ★新增，後續用到

MERGED_DIR  = Path(OUTPUT_ROOT) / "merged_pseudo"  # ★所有折偽標籤最終集中處
COPY_IMAGES = True                                # ★若也想搬 pred 圖片就設 True

# ────────────── 程式區 ──────────────
import os, yaml, shutil, tempfile, random, math
from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO

if __name__ == '__main__':
    # ---------- 讀取 YAML ----------
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    train_dir = Path(data_cfg['train']).resolve()
    val_dir   = Path(data_cfg['val']).resolve()

    # ---------- 收集影像 ----------
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_paths = sorted([p for p in train_dir.rglob('*')
                        if p.suffix.lower() in IMG_EXTS])

    print(f"共收集到 {len(img_paths)} 張影像作 KFOLD")

    # ---------- KFOLD 分割 ----------
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
    for fold, (train_idx, infer_idx) in enumerate(kf.split(img_paths), start=1):
        fold_name = f"fold_{fold:02d}"
        fold_root = Path(OUTPUT_ROOT) / fold_name
        train_txt = fold_root / 'train_list.txt'
        infer_txt = fold_root / 'infer_list.txt'
        data_yaml = fold_root / 'data_fold.yaml'
        pseudo_dir = fold_root / 'pseudo_labels'

        # 建立資料夾
        pseudo_dir.mkdir(parents=True, exist_ok=True)

        # ---------- 產生 list 檔 ----------
        with open(train_txt, 'w', encoding='utf-8') as f_train:
            for idx in train_idx:
                f_train.write(str(img_paths[idx]) + '\n')

        with open(infer_txt, 'w', encoding='utf-8') as f_inf:
            for idx in infer_idx:
                f_inf.write(str(img_paths[idx]) + '\n')

        # ---------- 產生 data YAML ----------
        data_fold = {
            'path': '',  # 不使用根資料夾
            'train': str(train_txt),
            'val': str(val_dir),
            'names': data_cfg.get('names', {0: 'object'})
        }
        with open(data_yaml, 'w', encoding='utf-8') as f_yaml:
            yaml.dump(data_fold, f_yaml, allow_unicode=True)

        print(f"[Fold {fold}] 訓練樣本: {len(train_idx)}, 推論樣本: {len(infer_idx)}")

        # ---------- 訓練 ----------
        model = YOLO(PRETRAINED_WEIGHTS)
        model.train(
            data=str(data_yaml),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            patience=PATIENCE,
            workers=WORKERS,
            device=DEVICE,
            project=str(fold_root),
            half=True,  # 啟用混合精度訓練
            name='train',
            exist_ok=True,
        )

        # 取得最佳權重
        best_ckpt = Path(fold_root / 'train' / 'weights' / 'best.pt')
        if not best_ckpt.exists():
            raise FileNotFoundError(f"找不到最佳權重: {best_ckpt}")

        # ---------- 推論 (產生偽標籤) ----------
        infer_model = YOLO(str(best_ckpt))
        infer_model.predict(
            source=str(infer_txt),   # 直接用 txt list
            imgsz=IMG_SIZE,
            conf=CONF_THR,
            iou=IOU_NMS,
            device=DEVICE,
            save_txt=True,
            save_conf=True,
            project=str(pseudo_dir),
            name='pred',
            exist_ok=True,
            verbose=False,
        )

        print(f"[Fold {fold}] 偽標籤輸出完成 → {pseudo_dir / 'pred'}")

    MERGED_LABEL_DIR = MERGED_DIR / "labels"
    MERGED_IMG_DIR = MERGED_DIR / "images"
    MERGED_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    if COPY_IMAGES:
        MERGED_IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("\\n[Merge] 搬移各折偽標籤到:", MERGED_DIR)
    root_path = Path(OUTPUT_ROOT)
    for pred_labels_dir in root_path.glob('fold_*/pseudo_labels/pred/labels'):
        # 搬 .txt
        for txt_file in pred_labels_dir.glob('*.txt'):
            shutil.move(str(txt_file), MERGED_LABEL_DIR / txt_file.name)

        # (可選) 搬影像
        if COPY_IMAGES:
            pred_images_dir = pred_labels_dir.parent / 'images'
            if pred_images_dir.exists():
                for img_file in pred_images_dir.glob('*'):
                    shutil.move(str(img_file), MERGED_IMG_DIR / img_file.name)

    print("[Merge] 完成，總偽標籤數:", len(list(MERGED_LABEL_DIR.glob('*.txt'))))

    print("\nKFOLD 偽標籤流程完成 ✔")
