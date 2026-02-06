# CampusGuard_Models/test_scripts/dataset_sanity.py
# (Includes your dataset paths hardcoded + CLI overrides)

import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# =========================
# HARD-CODED DATASET PATHS
# =========================
DATASET_PATHS = {
    "mot17": Path(r"C:\Users\qc_de\Downloads\MOT17\MOT17"),
    "custom_images": Path(r"C:\Users\qc_de\Downloads\dataset\dataset\images"),
    "avenue": Path(r"C:\Users\qc_de\Downloads\Avenue_Dataset\Avenue Dataset"),
    "coco2017": Path(r"C:\Users\qc_de\Downloads\coco-2017-DatasetNinja"),
}

# Where we prepare samples + reports (relative to repo layout)
DEFAULT_PREP_ROOT = Path("CampusGuard_Models") / "datasets_prepared"


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
ANN_EXTS = {".txt", ".json", ".xml"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def count_extensions(root: Path) -> Dict[str, int]:
    counts = {"images": 0, "videos": 0, "txt": 0, "json": 0, "xml": 0, "other": 0}
    if not root.exists():
        return counts

    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        ext = fp.suffix.lower()
        if ext in IMG_EXTS:
            counts["images"] += 1
        elif ext in VID_EXTS:
            counts["videos"] += 1
        elif ext == ".txt":
            counts["txt"] += 1
        elif ext == ".json":
            counts["json"] += 1
        elif ext == ".xml":
            counts["xml"] += 1
        else:
            counts["other"] += 1
    return counts


def detect_structure_notes(name: str, root: Path) -> List[str]:
    notes: List[str] = []

    # Common warnings
    if " " in str(root):
        notes.append("Path contains spaces (should still work, but be careful in shells).")

    if name == "custom_images":
        # nested duplicate folder warning
        parts = [p.lower() for p in root.parts]
        for i in range(len(parts) - 1):
            if parts[i] == parts[i + 1]:
                notes.append("Nested duplicate folder name detected (e.g., dataset\\dataset).")
                break

    if name == "coco2017":
        # search for common COCO annotation json names
        ann = list(root.rglob("instances_*.json")) + list(root.rglob("*annotations*.json"))
        if not ann:
            notes.append("Could not find COCO annotations JSON (instances_*.json).")
        else:
            notes.append(f"Found COCO annotation JSON candidates: {min(len(ann), 3)} (showing up to 3): " +
                         ", ".join([str(a) for a in ann[:3]]))

    if name == "mot17":
        # check for MOT17 train/test directories
        has_train = (root / "train").exists()
        has_test = (root / "test").exists()
        if not (has_train or has_test):
            notes.append("MOT17 train/test folders not found at root; may be nested differently.")
        # check for common frames folder 'img1'
        img1 = list(root.rglob("img1"))
        if not img1:
            notes.append("No img1 folders found (MOT format typically has seq/img1/*.jpg).")
        else:
            notes.append(f"Found {len(img1)} 'img1' folders (MOT frames).")

    if name == "avenue":
        # Avenue is often videos or frames; check for common train/test dirs
        has_train = any((root / d).exists() for d in ["training_videos", "training", "train", "Train", "train_videos"])
        has_test = any((root / d).exists() for d in ["testing_videos", "testing", "test", "Test", "test_videos"])
        if not has_train and not has_test:
            notes.append("No obvious train/test video folders found (Avenue structure varies).")

    return notes


def list_image_files(root: Path, max_files: int = 50000) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files
    for ext in IMG_EXTS:
        files.extend(root.rglob(f"*{ext}"))
        if len(files) >= max_files:
            break
    return files[:max_files]


def list_video_files(root: Path, max_files: int = 5000) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files
    for ext in VID_EXTS:
        files.extend(root.rglob(f"*{ext}"))
        if len(files) >= max_files:
            break
    return files[:max_files]


def sample_images_from_dataset(name: str, root: Path, sample_dir: Path, n: int) -> Tuple[List[Path], List[str]]:
    """
    Returns (copied_sample_paths, warnings)
    Strategy:
      - If image files exist: copy random images
      - Else if video files exist: extract frames from random videos
    """
    warnings: List[str] = []
    ensure_dir(sample_dir)

    img_files = list_image_files(root)
    if img_files:
        picks = random.sample(img_files, k=min(n, len(img_files)))
        out_paths: List[Path] = []
        for i, src in enumerate(picks):
            dst = sample_dir / f"{name}_{i:05d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            out_paths.append(dst)
        return out_paths, warnings

    vid_files = list_video_files(root)
    if not vid_files:
        warnings.append("No images or videos found to sample from.")
        return [], warnings

    # Extract frames from videos
    picks = random.sample(vid_files, k=min(3, len(vid_files)))  # try up to 3 videos
    out_paths: List[Path] = []
    frames_needed = n

    for v in picks:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            warnings.append(f"Could not open video: {v}")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        # choose evenly spaced frames
        if total > 0:
            idxs = np.linspace(0, max(0, total - 1), num=min(frames_needed, 40), dtype=int)
        else:
            idxs = np.arange(0, min(frames_needed, 40), dtype=int)

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            dst = sample_dir / f"{name}_{len(out_paths):05d}.jpg"
            cv2.imwrite(str(dst), frame)
            out_paths.append(dst)
            if len(out_paths) >= n:
                break

        cap.release()
        if len(out_paths) >= n:
            break

    if not out_paths:
        warnings.append("Videos found, but failed to extract frames.")
    elif len(out_paths) < n:
        warnings.append(f"Only extracted {len(out_paths)}/{n} frames from videos.")

    return out_paths, warnings


def make_contact_sheet(image_paths: List[Path], out_path: Path, grid: Tuple[int, int] = (4, 4), tile: int = 256) -> None:
    """
    Creates a simple contact sheet using OpenCV (no matplotlib).
    """
    rows, cols = grid
    needed = rows * cols
    picks = image_paths[:needed]
    canvas = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)

    for i, p in enumerate(picks):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        r = i // cols
        c = i % cols
        canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = img

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), canvas)


def write_csv(paths: List[Path], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "path"])
        for i, p in enumerate(paths):
            w.writerow([i, str(p)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep_root", type=str, default=str(DEFAULT_PREP_ROOT), help="Prepared dataset root")
    parser.add_argument("--samples_per_dataset", type=int, default=64, help="How many sample images to copy/extract per dataset")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    # Optional overrides
    parser.add_argument("--mot17", type=str, default=str(DATASET_PATHS["mot17"]), help="Override MOT17 path")
    parser.add_argument("--custom_images", type=str, default=str(DATASET_PATHS["custom_images"]), help="Override custom images path")
    parser.add_argument("--avenue", type=str, default=str(DATASET_PATHS["avenue"]), help="Override Avenue path")
    parser.add_argument("--coco2017", type=str, default=str(DATASET_PATHS["coco2017"]), help="Override COCO2017 path")
    args = parser.parse_args()

    random.seed(args.seed)

    prep_root = Path(args.prep_root)
    ensure_dir(prep_root)
    ensure_dir(prep_root / "reports")
    ensure_dir(prep_root / "samples")

    ds_paths = {
        "mot17": Path(args.mot17),
        "custom_images": Path(args.custom_images),
        "avenue": Path(args.avenue),
        "coco2017": Path(args.coco2017),
    }

    # Report header
    print("=== DATASET SANITY REPORT ===")
    print(f"Prepared root: {prep_root.resolve()}\n")

    summary_rows = []

    for name, root in ds_paths.items():
        print(f"\n--- {name.upper()} ---")
        print(f"Path: {root}")

        exists = root.exists()
        print(f"Exists: {exists}")
        if not exists:
            summary_rows.append([name, str(root), "MISSING", 0, 0, 0, 0, 0, 0, ""])
            continue

        counts = count_extensions(root)
        notes = detect_structure_notes(name, root)

        print(f"Counts: images={counts['images']}, videos={counts['videos']}, txt={counts['txt']}, json={counts['json']}, xml={counts['xml']}, other={counts['other']}")
        for n in notes:
            print(f"Warning: {n}")

        # Sample extraction/copy
        sample_dir = prep_root / "samples" / name
        sample_paths, sample_warnings = sample_images_from_dataset(name, root, sample_dir, n=args.samples_per_dataset)
        for w in sample_warnings:
            print(f"Warning: {w}")

        # Save paths CSV
        csv_out = prep_root / "reports" / f"{name}_samples.csv"
        write_csv(sample_paths, csv_out)
        print(f"Saved sample list: {csv_out}")

        # Contact sheet
        if sample_paths:
            sheet_out = prep_root / "reports" / f"{name}_contact_sheet.png"
            make_contact_sheet(sample_paths, sheet_out, grid=(4, 4), tile=256)
            print(f"Saved contact sheet: {sheet_out}")
        else:
            print("No samples available to create contact sheet.")

        summary_rows.append([
            name, str(root), "OK",
            counts["images"], counts["videos"], counts["txt"], counts["json"], counts["xml"], counts["other"],
            " | ".join(notes + sample_warnings)
        ])

    # Summary CSV
    summary_csv = prep_root / "reports" / "dataset_summary.csv"
    ensure_dir(summary_csv.parent)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "path", "status", "images", "videos", "txt", "json", "xml", "other", "notes"])
        w.writerows(summary_rows)

    print(f"\n[OK] Wrote summary: {summary_csv}")


if __name__ == "__main__":
    main()
