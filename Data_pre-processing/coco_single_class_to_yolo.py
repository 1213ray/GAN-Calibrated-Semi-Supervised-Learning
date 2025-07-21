#!/usr/bin/env python3
"""Filter a single category from COCO annotations and convert to YOLO txt.

Usage:
    python coco_single_class_to_yolo.py \
        --json /path/to/instances_train2017.json \
        --images-dir /path/to/train2017 \
        --category person \
        --out-dir ./coco_person_yolo/train \
        [--copy-images] [--no-copy]

Requirements:
    pip install pycocotools tqdm pillow
"""

import json
import argparse
import shutil
import os
from pathlib import Path
from typing import List

from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image


def make_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def coco_to_yolo_bbox(bbox, img_w, img_h):
    # COCO format: [x_min, y_min, width, height]
    x_min, y_min, w, h = bbox
    x_center = x_min + w / 2
    y_center = y_min + h / 2
    return [
        x_center / img_w,
        y_center / img_h,
        w / img_w,
        h / img_h,
    ]


def filter_and_convert(
    src_json: Path,
    images_dir: Path,
    out_dir: Path,
    category: str,
    copy_images: bool = True,
):
    coco = COCO(str(src_json))

    # Map category name/ID
    if category.isdigit():
        cat_ids = [int(category)]
    else:
        cat_ids = coco.getCatIds(catNms=[category])
        if not cat_ids:
            raise ValueError(f"Category '{category}' not found in COCO annotations.")
    cat_id = cat_ids[0]

    img_ids = coco.getImgIds(catIds=[cat_id])
    images = coco.loadImgs(img_ids)

    make_dirs(out_dir / "labels")
    if copy_images:
        make_dirs(out_dir / "images")

    # index annotations by image
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=[cat_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    anns_by_img = {}
    for ann in anns:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for img in tqdm(images, desc="Converting"):
        file_name = img["file_name"]
        img_w, img_h = img["width"], img["height"]
        label_lines: List[str] = []

        for ann in anns_by_img.get(img["id"], []):
            yolo_box = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
            # single class -> id 0
            line = f"0 {' '.join(f'{v:.6f}' for v in yolo_box)}"
            label_lines.append(line)

        if not label_lines:
            # if no annotations for this image (shouldn't happen), skip copying
            continue

        # write label file
        txt_name = Path(file_name).with_suffix(".txt").name
        with open(out_dir / "labels" / txt_name, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        # copy image
        if copy_images:
            src_img_path = images_dir / file_name
            dst_img_path = out_dir / "images" / file_name
            dst_img_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_img_path, dst_img_path)

    print(
        f"Done. Converted {len(images)} images with category '{category}' "
        f"to YOLO format at {out_dir}."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="COCO single-class to YOLO converter")
    parser.add_argument("--json", type=Path, required=True, help="COCO annotation json")
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing COCO images (train2017/ etc.)",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Target category name (e.g., person) or id (e.g., 1)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output root dir")
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy images, only write labels",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    filter_and_convert(
        src_json=args.json,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        category=args.category,
        copy_images=not args.no_copy,
    )


if __name__ == "__main__":
    main()
