# main.py
import os
import json
import argparse

from image_downloader import ImageDownloader
from resize_images import ImageResizer
from clean_annotations import AnnotationCleaner
from generate_sft_dataset import SFTGenerator

DEFAULTS = {
    "input_urls":"data/crawler_url_image.txt",
    "raw_dir":"data/raw_img2",
    "resized_dir":"data/resized_img",
    "annotations_raw":"data/annotations_raw/mkn_annotations_raw.json",
    "annotations_clean_dir":"data/annotations_clean",
    "annotations_clean_file":"data/annotations_clean/mkn_annotations_clean.json",
    "sft_out_file":"data/sft_dataset/train_qwen_vl.jsonl",
    "min_size_bytes":1024,
    "rate_limit_seconds":0.5,
    "start_index":111,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    p.add_argument("--resize", action="store_true")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--config", type=str, default=None)
    return p.parse_args()

def load_config(path=None):
    cfg = DEFAULTS.copy()
    if path and os.path.exists(path):
        with open(path,"r",encoding="utf-8") as fh:
            cfg.update(json.load(fh))
    return cfg

def main():
    args = parse_args()
    cfg = load_config(args.config)

    downloader = ImageDownloader(cfg["input_urls"], cfg["raw_dir"], min_size=cfg["min_size_bytes"], rate_limit=cfg["rate_limit_seconds"], start_index=cfg["start_index"])
    resizer = ImageResizer(cfg["raw_dir"], cfg["resized_dir"])
    cleaner = AnnotationCleaner(cfg["annotations_raw"], cfg["annotations_clean_dir"], cfg["annotations_clean_file"])
    sft = SFTGenerator(cfg["annotations_clean_file"], cfg["sft_out_file"], raw_img_prefix=os.path.relpath(cfg["resized_dir"]).replace("\\","/") + "/")

    if not any([args.download, args.resize, args.clean, args.generate, args.all]):
        print("No stage selected. Use --all or stage flags.")
        return

    if args.all:
        downloader.run(limit=args.limit)
        resizer.run()
        cleaner.run()
        sft.run()
        return

    if args.download:
        downloader.run(limit=args.limit)
    if args.resize:
        resizer.run()
    if args.clean:
        cleaner.run()
    if args.generate:
        sft.run()

if __name__ == "__main__":
    main()