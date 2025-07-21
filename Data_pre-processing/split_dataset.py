import random
import shutil
from pathlib import Path
from typing import List, Sequence

# ───✨ 修改這三行即可 ✨─────────────────────────────
SRC_DIR = r"C:\Users\alian\PycharmProjects\yolov8\datasets\original_data"
DST_DIR = r"C:\Users\alian\PycharmProjects\yolov8\datasets\500_100_100"
COUNTS  = [500, 100, 100]          # 例： [300, 100] 代表 300/100/其餘
SEED    = 42             # 想每次結果不同就改成 None
MOVE    = False          # True=移動檔案；False=複製


def split_list(items: Sequence[Path], counts: List[int]) -> List[List[Path]]:
    random.shuffle(items)
    splits, idx = [], 0
    for c in counts:
        splits.append(items[idx : idx + c])
        idx += c
    splits.append(items[idx:])  # 剩下的
    return splits


def copy_pair(img_path: Path, src_root: Path, dst_root: Path, move: bool):
    rel_name = img_path.name
    lbl_path = src_root / "labels" / rel_name.replace(img_path.suffix, ".txt")

    dst_img = dst_root / "images" / rel_name
    dst_lbl = dst_root / "labels" / lbl_path.name
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)

    if move:
        shutil.move(str(img_path), dst_img)
        shutil.move(str(lbl_path), dst_lbl)
    else:
        shutil.copy2(img_path, dst_img)
        shutil.copy2(lbl_path, dst_lbl)


def main():
    if SEED is not None:
        random.seed(SEED)

    src = Path(SRC_DIR)
    imgs = sorted((src / "images").glob("*.*"))
    if not imgs:
        raise FileNotFoundError("找不到 images")

    splits = split_list(imgs, COUNTS)
    total = len(imgs)

    print(f"總影像數: {total} 份數設定: {COUNTS} (+1 份收尾)")
    for i, part in enumerate(splits, 1):
        part_dir = Path(DST_DIR) / f"part{i}"
        for img in part:
            copy_pair(img, src_root=src, dst_root=part_dir, move=MOVE)
        print(f"part{i}: {len(part):>5} 張 -> {part_dir}")

    print("拆分完成！")


if __name__ == "__main__":
    main()
