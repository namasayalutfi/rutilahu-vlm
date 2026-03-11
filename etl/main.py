import os
import json
import argparse

from image_downloader import ImageDownloader
from images_processor import ImageProcessor 
from clean_annotations import AnnotationCleaner
from generate_sft_dataset import SFTGenerator

DEFAULTS = {
    "input_urls": "data/gambar_rutilahu.txt",
    "raw_dir": "data/raw_img",
    "clean_img_dir": "data/mkn_img",
    "duplicates_dir": "data/duplicates",       # Folder untuk duplikat
    "invalid_dir": "data/invalid_files",      # Folder untuk format aneh/rusak
    "annotations_raw": "data/annotations_raw/mkn_annotations_raw.json",
    "annotations_clean_dir": "data/annotations_clean",
    "annotations_clean_file": "data/annotations_clean/mkn_annotations_clean.json",
    "sft_out_file": "data/sft_dataset/train_qwen_vl.jsonl",
    "min_size_bytes": 1024,
    "rate_limit_seconds": 0.5,
    "start_index": 111,
    "img_size": (640, 640),
    "phash_threshold": 5
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="Download images from URLs")
    p.add_argument("--process", action="store_true", help="Filter duplicates, convert formats, and resize")
    p.add_argument("--clean", action="store_true", help="Clean annotation metadata")
    p.add_argument("--generate", action="store_true", help="Generate SFT dataset for VLM")
    p.add_argument("--all", action="store_true", help="Run all stages")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--config", type=str, default=None)
    return p.parse_args()

def load_config(path=None):
    cfg = DEFAULTS.copy()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    return cfg

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 1. Tahap Download
    downloader = ImageDownloader(
        cfg["input_urls"], 
        cfg["raw_dir"], 
        min_size=cfg["min_size_bytes"], 
        rate_limit=cfg["rate_limit_seconds"], 
        start_index=cfg["start_index"]
    )

    # 2. Tahap Processor (Gabungan Filter + Resize + Convert)
    processor = ImageProcessor(
        input_dir=cfg["raw_dir"],
        output_dir=cfg["clean_img_dir"],
        duplicates_dir=cfg["duplicates_dir"],
        invalid_dir=cfg["invalid_dir"],
        size=cfg["img_size"],
        phash_threshold=cfg["phash_threshold"]
    )

    # 3. Tahap Metadata & SFT
    cleaner = AnnotationCleaner(cfg["annotations_raw"], cfg["annotations_clean_dir"], cfg["annotations_clean_file"])
    sft = SFTGenerator(cfg["annotations_clean_file"], cfg["sft_out_file"], raw_img_prefix=os.path.relpath(cfg["resized_dir"]).replace("\\","/") + "/")

    if not any([args.download, args.process, args.clean, args.generate, args.all]):
        print("No stage selected. Use --all or specific flags (e.g., --process).")
        return

    # --- EKSEKUSI PIPELINE ---
    
    if args.all or args.download:
        downloader.run(limit=args.limit)

    if args.all or args.process:
        # Menjalankan pembersihan format, filter duplikat, dan resize sekaligus
        processor.run()

    if args.all or args.clean:
        cleaner.run()

    if args.all or args.generate:
        sft.run()

if __name__ == "__main__":
    main()