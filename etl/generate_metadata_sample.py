import json
import random
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_PATH = Path("data/mkn_img/house_metadata_split.json")
OUTPUT_PATH = Path("data/mkn_img/sample_15.json")

N_MULTI = 5
N_SINGLE_EXT = 5
N_SINGLE_INT = 5

RANDOM_SEED = 42


# =========================
# LOAD DATA
# =========================
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

random.seed(RANDOM_SEED)

# =========================
# HELPER FILTER
# =========================
def is_multi(x):
    return x["dataset_scheme"] == "multi"

def is_single_ext(x):
    return (
        x["dataset_scheme"] == "single"
        and x["images"][0]["view_type"] == "exterior"
    )

def is_single_int(x):
    return (
        x["dataset_scheme"] == "single"
        and x["images"][0]["view_type"] == "interior"
    )

# =========================
# FILTER PER GROUP
# =========================
multi_data = [x for x in data if is_multi(x)]
single_ext_data = [x for x in data if is_single_ext(x)]
single_int_data = [x for x in data if is_single_int(x)]

print(f"Multi total: {len(multi_data)}")
print(f"Single EXT total: {len(single_ext_data)}")
print(f"Single INT total: {len(single_int_data)}")

# =========================
# SAMPLING
# =========================
def stratified_sample(data_list, n, name):
    train_data = [x for x in data_list if x["split"] == "train"]
    val_data = [x for x in data_list if x["split"] == "val"]

    n_train = int(n * 0.6)
    n_val = n - n_train

    sampled = []

    if len(train_data) >= n_train:
        sampled += random.sample(train_data, n_train)
    else:
        sampled += train_data

    if len(val_data) >= n_val:
        sampled += random.sample(val_data, n_val)
    else:
        sampled += val_data

    # kalau kurang (edge case), ambil dari sisa
    if len(sampled) < n:
        remaining = [x for x in data_list if x not in sampled]
        needed = n - len(sampled)
        sampled += random.sample(remaining, min(needed, len(remaining)))

    print(f"{name}: train={len([x for x in sampled if x['split']=='train'])}, val={len([x for x in sampled if x['split']=='val'])}")

    return sampled

sample_multi = stratified_sample(multi_data, N_MULTI, "multi")
sample_ext = stratified_sample(single_ext_data, N_SINGLE_EXT, "single_ext")
sample_int = stratified_sample(single_int_data, N_SINGLE_INT, "single_int")

# =========================
# COMBINE
# =========================
final_sample = sample_multi + sample_ext + sample_int

# shuffle biar campur train/val
random.shuffle(final_sample)

print(f"Total sample: {len(final_sample)}")

# =========================
# SAVE
# =========================
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_sample, f, indent=2, ensure_ascii=False)

print(f"Saved to: {OUTPUT_PATH}")