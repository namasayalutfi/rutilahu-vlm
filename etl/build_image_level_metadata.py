import json
import re
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================

INPUT_JSONLS = [
    "data/sft/train.jsonl",
    "data/sft/val.jsonl",
    "data/sft/test.jsonl"
]

OUTPUT_METADATA = "data/cnn/mkn_image_metadata.json"


# ==========================================
# PARSE ASSISTANT LABELS
# ==========================================

def extract_materials(text):
    """
    Parse:
    Atap:
    - Material: genteng
    ...
    """
    patterns = {
        "atap": r"Atap:\s*- Material:\s*(.+?)\s*- Kondisi:",
        "dinding": r"Dinding:\s*- Material:\s*(.+?)\s*- Kondisi:",
        "lantai": r"Lantai:\s*- Material:\s*(.+?)\s*- Kondisi:"
    }

    labels = {}

    for k,p in patterns.items():
        m = re.search(p, text, re.S | re.I)
        labels[k] = m.group(1).strip() if m else None

    return labels


# ==========================================
# GET IMAGE PATHS FROM USER MESSAGE
# ==========================================

def extract_images_from_sample(sample):
    """
    Ambil semua image dari role=user
    """
    images = []

    for msg in sample["messages"]:
        if msg["role"] != "user":
            continue

        for c in msg["content"]:
            if c["type"] == "image":
                images.append(c["image"])

    return images


# ==========================================
# GET ASSISTANT OUTPUT
# ==========================================

def extract_assistant_text(sample):

    for msg in sample["messages"]:
        if msg["role"]=="assistant":
            for c in msg["content"]:
                if c["type"]=="text":
                    return c["text"]

    return None


# ==========================================
# VIEW TYPE INFERENCE
# ==========================================

def infer_view_type(path):
    p = path.lower()

    if "ext" in p or "exterior" in p:
        return "exterior"

    if "int" in p or "interior" in p:
        return "interior"

    return None


# ==========================================
# BUILD IMAGE LEVEL METADATA
# ==========================================

def build_metadata():

    records=[]
    img_counter=1

    seen=set()

    for file in INPUT_JSONLS:

        print(f"Processing {file}")

        with open(file,"r",encoding="utf-8") as f:

            for line in f:
                sample=json.loads(line)

                assistant_text=extract_assistant_text(sample)

                if not assistant_text:
                    continue

                labels=extract_materials(assistant_text)

                image_paths=extract_images_from_sample(sample)

                for img_path in image_paths:

                    if img_path in seen:
                        continue

                    seen.add(img_path)

                    view_type=infer_view_type(img_path)

                    rec={
                        "id":f"IMG{img_counter:06d}",
                        "image_path":img_path,
                        "view_type":view_type,

                        "material_atap":
                            labels["atap"]
                            if view_type=="exterior"
                            else None,

                        "material_dinding":
                            labels["dinding"],

                        "material_lantai":
                            labels["lantai"]
                            if view_type=="interior"
                            else None
                    }

                    records.append(rec)
                    img_counter+=1

    return records


# ==========================================
# MAIN
# ==========================================

if __name__=="__main__":

    metadata=build_metadata()

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nDone.")
    print(f"Total images: {len(metadata)}")
    print(f"Saved -> {OUTPUT_METADATA}")