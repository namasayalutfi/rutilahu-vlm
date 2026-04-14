import json
import random
from collections import defaultdict

INPUT_JSON = "data/mkn_img/house_metadata.json"
OUTPUT_JSON = "data/mkn_img/house_metadata_split.json"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)


# =========================
# LOAD DATA
# =========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)


# =========================
# STEP 1: BUILD GROUPS (ANTI-LEAKAGE)
# =========================
# group berdasarkan shared image_hash
groups = []
hash_to_group = {}

for record in data:
    hashes = set(img["image_hash"] for img in record["images"])

    # cari group yang overlap
    matched_groups = []
    for g in groups:
        if hashes & g:
            matched_groups.append(g)

    if not matched_groups:
        groups.append(set(hashes))
    else:
        # merge semua group yang overlap
        new_group = set(hashes)
        for g in matched_groups:
            new_group |= g
            groups.remove(g)
        groups.append(new_group)


# mapping hash → group_id
group_list = list(groups)
hash_to_gid = {}

for gid, g in enumerate(group_list):
    for h in g:
        hash_to_gid[h] = gid

print(f"Total groups (connected components): {len(group_list)}")


# =========================
# STEP 2: MAP RECORD → GROUP
# =========================
group_records = defaultdict(list)

for record in data:
    hashes = set(img["image_hash"] for img in record["images"])
    gid = hash_to_gid[next(iter(hashes))]  # ambil salah satu
    group_records[gid].append(record)


# =========================
# STEP 3: SPLIT PER CLASS (LAYAK / TIDAK)
# =========================
class_groups = {
    "layak": [],
    "tidak_layak": []
}

for gid, records in group_records.items():
    # ambil label dari record pertama
    label = records[0]["kelayakan_rumah"]
    class_groups[label].append(gid)


# =========================
# STEP 4: SPLIT FUNCTION
# =========================
def split_list(lst):
    random.shuffle(lst)
    n = len(lst)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = lst[:n_train]
    val = lst[n_train:n_train + n_val]
    test = lst[n_train + n_val:]

    return train, val, test


split_map = {}

for label, gids in class_groups.items():
    train, val, test = split_list(gids)

    for gid in train:
        split_map[gid] = "train"
    for gid in val:
        split_map[gid] = "val"
    for gid in test:
        split_map[gid] = "test"


# =========================
# STEP 5: ASSIGN SPLIT
# =========================
for record in data:
    hashes = set(img["image_hash"] for img in record["images"])
    gid = hash_to_gid[next(iter(hashes))]
    record["split"] = split_map[gid]


# =========================
# STEP 6: STATS (DEBUG)
# =========================
stats = defaultdict(lambda: defaultdict(int))

for record in data:
    split = record["split"]
    label = record["kelayakan_rumah"]

    stats[split][label] += 1

print("\n=== DISTRIBUSI HOUSE ===")
for split in ["train", "val", "test"]:
    print(split, dict(stats[split]))


# =========================
# STEP 7: SAVE
# =========================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\nSaved to: {OUTPUT_JSON}")