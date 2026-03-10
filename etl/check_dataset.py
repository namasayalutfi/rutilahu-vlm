import json
import os

# karena script ada di folder etl, kita naik 1 level
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "data", "sft_dataset", "train_vlm.jsonl")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw_img")

missing_images = []
total = 0

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        total += 1
        item = json.loads(line)

        img = os.path.basename(item["image"])
        img_path = os.path.join(IMAGE_DIR, img)

        if not os.path.exists(img_path):
            missing_images.append(img)

print("Total dataset:", total)
print("Missing images:", len(missing_images))

if missing_images:
    print("\nList gambar yang tidak ada:")
    for m in missing_images[:20]:
        print(m)
else:
    print("\nSemua gambar tersedia ✅")