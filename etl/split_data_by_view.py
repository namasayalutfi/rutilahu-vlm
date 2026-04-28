import json
import random
from collections import defaultdict

INPUT_JSON = "data/cnn/mkn_image_metadata.json"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42
random.seed(SEED)


# =========================
# LOAD
# =========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)


# =========================
# GROUP BY VIEW TYPE
# =========================
groups = defaultdict(list)

for item in data:
    groups[item["view_type"]].append(item)


# =========================
# STRATIFIED SPLIT
# =========================
for view_type, samples in groups.items():

    random.shuffle(samples)

    n = len(samples)

    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    # assign split
    for x in samples[:n_train]:
        x["split"] = "train"

    for x in samples[n_train:n_train+n_val]:
        x["split"] = "val"

    for x in samples[n_train+n_val:]:
        x["split"] = "test"

    print(
        f"{view_type}: "
        f"train={n_train}, "
        f"val={n_val}, "
        f"test={n_test}"
    )


# =========================
# SAVE (overwrite file lama)
# =========================
with open(INPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nSplit selesai ditambahkan ke metadata.")