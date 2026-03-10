import os
import json

INPUT_FILE = "../data/annotations_raw/mkn_annotations_raw.json"
OUT_DIR = "../data/annotations_clean"
OUT_FILE = "../data/annotations_clean/mkn_annotations_clean.json"

os.makedirs(OUT_DIR, exist_ok=True)

SEVERITY_MAP = {
    1: "layak",
    2: "perlu_perbaikan_ringan",
    3: "perlu_perbaikan_sedang",
    4: "tidak_layak"
}

def clean_record(rec):

    clean = {}

    image = rec.get("image") or rec.get("data", {}).get("image")
    if image:
        image = os.path.basename(image)
        if "-" in image:
            image = image.split("-", 1)[1]

    clean["image"] = image

    clean["roof_condition"] = rec.get("roof_condition")
    clean["wall_condition"] = rec.get("wall_condition")
    clean["floor_condition"] = rec.get("floor_condition")

    sev = rec.get("severity_score") or rec.get("severity")

    try:
        sev = int(sev)
    except:
        sev = None

    clean["severity_score"] = sev
    clean["severity_label"] = SEVERITY_MAP.get(sev)

    return clean


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []

    for rec in data:
        cleaned.append(clean_record(rec))

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"Clean dataset saved to {OUT_FILE}")
    print(f"Total samples: {len(cleaned)}")


if __name__ == "__main__":
    main()