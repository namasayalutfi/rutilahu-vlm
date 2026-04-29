import json
import random
from pathlib import Path
from collections import defaultdict

# =========================
# CONFIG
# =========================
ROOT_DIR = Path("data/mkn_img")

OUTPUT_DIR = Path("data/cnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "mkn_image_metadata.json"

IMAGE_EXTS = {".jpg",".jpeg",".png",".webp"}

SEED=42
random.seed(SEED)

TRAIN_RATIO=0.70
VAL_RATIO=0.15
TEST_RATIO=0.15


FOLDER_CONFIG = {
    "rlh_ext": {
        "kelayakan":"layak",
        "view":"exterior"
    },
    "rlh_int": {
        "kelayakan":"layak",
        "view":"interior"
    },
    "rtlh_ext": {
        "kelayakan":"tidak_layak",
        "view":"exterior"
    },
    "rtlh_int": {
        "kelayakan":"tidak_layak",
        "view":"interior"
    }
}


def is_image(p):
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def sorted_images(folder):
    return sorted(
        [p for p in folder.iterdir() if is_image(p)],
        key=lambda x:x.name.lower()
    )


def new_id(i):
    return f"IMG{i:06d}"


# =========================
# BUILD METADATA
# =========================
def build_metadata():

    records=[]
    idx=1

    for folder_name in sorted(FOLDER_CONFIG.keys()):

        folder=ROOT_DIR/folder_name

        if not folder.exists():
            print(f"[WARN] Missing {folder}")
            continue

        cfg=FOLDER_CONFIG[folder_name]

        for img in sorted_images(folder):

            if cfg["view"]=="exterior":
                mat_atap="belum_teridentifikasi"
                mat_dinding="belum_teridentifikasi"
                mat_lantai=None

            else:
                mat_atap=None
                mat_dinding="belum_teridentifikasi"
                mat_lantai="belum_teridentifikasi"

            records.append({
                "id":new_id(idx),
                "image_path":f"{folder_name}/{img.name}",
                "view_type":cfg["view"],
                "kelayakan_rumah":cfg["kelayakan"],
                "material_atap":mat_atap,
                "material_dinding":mat_dinding,
                "material_lantai":mat_lantai,
                "split":"unsplit"
            })

            idx+=1

    return records


# =========================
# SPLIT BY VIEW TYPE
# =========================
def split_by_view(records):

    groups=defaultdict(list)

    for r in records:
        groups[r["view_type"]].append(r)

    for view,samples in groups.items():

        random.shuffle(samples)

        n=len(samples)

        n_train=int(n*TRAIN_RATIO)
        n_val=int(n*VAL_RATIO)

        for x in samples[:n_train]:
            x["split"]="train"

        for x in samples[n_train:n_train+n_val]:
            x["split"]="val"

        for x in samples[n_train+n_val:]:
            x["split"]="test"

        print(
            f"{view}: "
            f"train={n_train} "
            f"val={n_val} "
            f"test={n-(n_train+n_val)}"
        )

    return records


def main():

    records=build_metadata()

    print(f"Total images: {len(records)}")

    split_by_view(records)

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            records,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nSaved to {OUTPUT_JSON}")


if __name__=="__main__":
    main()