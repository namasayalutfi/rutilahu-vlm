"""
Generate SFT JSONL dataset (2 samples per record) from a centralized cleaned annotations file.

- Sample 1: Paragraph (Bahasa Indonesia) with factual reasoning (no recommendations).
- Sample 2: Structured point response with keys: roof, wall, floor, severity (uses severity_label) + final_classification.

Adjust INPUT_FILE and OUT_DIR as needed.
"""
import os
import json
import pathlib
from collections import Counter

# === CONFIG ===
INPUT_FILE = "../data/annotations_clean/mkn_annotations_clean.json"
OUT_DIR = "../data/sft_dataset"
OUT_FILE = os.path.join(OUT_DIR, "train_vlm.jsonl")

os.makedirs(OUT_DIR, exist_ok=True)


# === Label descriptions (update sesuai skema) ===
ROOF_DESC = {
    "good": "Atap utuh, tidak ada lubang atau bagian yang hilang.",
    "minor_damage": "Kerusakan ringan pada atap (mis. beberapa genteng bergeser atau retak kecil).",
    "major_damage": "Kerusakan berat pada atap (mis. banyak genteng hilang atau kerusakan signifikan).",
    "collapsed": "Atap runtuh atau hilang dalam skala besar.",
    "unknown": "Kondisi atap tidak dapat diidentifikasi dari gambar."
}

WALL_DESC = {
    "good": "Dinding dalam kondisi baik dan kokoh.",
    "minor_damage": "Kerusakan ringan pada dinding.",
    "major_damage": "Kerusakan besar pada dinding.",
    "collapsed": "Dinding runtuh.",
    "unknown": "Kondisi dinding tidak dapat diidentifikasi."
}

WALL_MATERIAL_DESC = {
    "concrete": "Dinding dari beton cor.",
    "brick": "Dinding dari bata merah.",
    "wood": "Dinding dari papan kayu.",
    "bamboo": "Dinding dari anyaman bambu.",
    "mixed": "Dinding menggunakan lebih dari satu material.",
    "unknown": "Material dinding tidak dapat diidentifikasi."
}

# Updated floor schema per user request
FLOOR_DESC = {
    "ceramic": "Lantai menggunakan keramik atau ubin.",
    "cement": "Lantai menggunakan semen atau plester.",
    "wood": "Lantai menggunakan papan kayu.",
    "dirt": "Lantai berupa tanah tanpa lapisan material.",
    "unknown": "Material lantai tidak dapat diidentifikasi."
}

# severity label map - use explicit labels suitable for government context
SEVERITY_LABEL = {
    1: "layak_huni",
    2: "perlu_perbaikan_ringan",
    3: "perlu_perbaikan_sedang",
    4: "tidak_layak_huni"
}


# === helpers ===
def normalize_floor_label(floor_raw):
    """Normalize common variants to the floor labels we use."""
    if not floor_raw:
        return "unknown"
    f = str(floor_raw).strip().lower()
    if f in ("ceramic", "tile", "ubin", "tiles", "keramik"):
        return "ceramic"
    if f in ("cement", "concrete", "semen", "plaster", "plester"):
        return "cement"
    if f in ("wood", "papan", "wooden", "boards"):
        return "wood"
    if f in ("dirt", "soil", "tanah"):
        return "dirt"
    return "unknown"


def severity_to_label(sev):
    try:
        return SEVERITY_LABEL.get(int(sev), "unknown")
    except Exception:
        return "unknown"


def infer_severity_from_components(roof, wall, floor):
    """Infer numeric severity from component labels when missing."""
    # collapsed => 4
    for v in (roof, wall):
        if v == "collapsed":
            return 4
    # any major_damage => 3
    if roof == "major_damage" or wall == "major_damage":
        return 3
    # any minor_damage => 2
    if roof == "minor_damage" or wall == "minor_damage":
        return 2
    # default => 1
    return 1


# === build paragraph response (no recommendations, add material-specific reasons) ===
def build_paragraph_response(rec):
    image = rec.get("image") or ""
    roof = (rec.get("roof_condition") or "unknown").lower()
    wall = (rec.get("wall_condition") or "unknown").lower()
    floor_raw = rec.get("floor_condition")
    floor = normalize_floor_label(floor_raw)
    wall_mat = (rec.get("wall_material") or "").lower()
    sev = rec.get("severity_score")

    # infer severity if missing or invalid
    try:
        sev = int(sev) if sev is not None else None
    except Exception:
        sev = None
    if sev is None:
        sev = infer_severity_from_components(roof, wall, floor)

    sev_label = severity_to_label(sev)

    # Compose paragraph in Indonesian (concise, factual)
    parts = []
    parts.append(f"Analisis singkat (gambar: {pathlib.Path(image).name}):")

    # conditions
    rdesc = ROOF_DESC.get(roof, ROOF_DESC["unknown"])
    parts.append(f"Atap: {rdesc}")
    wdesc = WALL_DESC.get(wall, WALL_DESC["unknown"])
    parts.append(f"Dinding: {wdesc}")
    fdesc = FLOOR_DESC.get(floor, FLOOR_DESC["unknown"])
    parts.append(f"Lantai: {fdesc}")

    # wall material if any
    if wall_mat:
        wmat_desc = WALL_MATERIAL_DESC.get(wall_mat, WALL_MATERIAL_DESC["unknown"])
        parts.append(wmat_desc)

    # severity & reasoning
    severity_text = f"Klasifikasi akhir: {sev_label} (kode {sev})."
    parts.append(severity_text)

    # reasons tying components to severity
    reasons = []
    if roof == "major_damage":
        reasons.append("Atap mengalami kerusakan besar, berisiko kebocoran dan menurunnya keselamatan struktural.")
    elif roof == "minor_damage":
        reasons.append("Atap mengalami kerusakan ringan yang dapat ditangani melalui perbaikan lokal.")
    elif roof == "collapsed":
        reasons.append("Atap runtuh — kondisi darurat, berbahaya untuk dihuni.")
    elif roof == "good":
        reasons.append("Atap tampak utuh tanpa kerusakan terlihat.")

    if wall == "major_damage":
        reasons.append("Dinding menunjukkan kerusakan signifikan yang dapat mempengaruhi kekuatan bangunan.")
    elif wall == "minor_damage":
        reasons.append("Dinding memiliki retakan ringan yang memerlukan perbaikan namun belum mengindikasikan kegagalan struktural.")
    elif wall == "collapsed":
        reasons.append("Dinding runtuh — kondisi darurat.")
    elif wall == "good":
        reasons.append("Dinding tampak kokoh dan dalam kondisi baik.")

    # Add material-specific reasons for wood and bamboo
    if wall_mat:
        if wall_mat == "wood":
            reasons.append(
                "Dinding berbahan kayu rentan terhadap serangan hama seperti rayap dan pelapukan akibat kelembaban; "
                "perlu pemeriksaan kondisi sambungan dan perlindungan terhadap kelembaban."
            )
        elif wall_mat == "bamboo":
            reasons.append(
                "Dinding anyaman bambu rentan terhadap kelembaban, pelapukan, dan serangan hama; "
                "ketahanan struktural umumnya lebih rendah dibanding beton/bata."
            )

    if floor and floor != "unknown":
        if floor == "dirt":
            reasons.append("Lantai berupa tanah yang dapat meningkatkan risiko kelembaban dan kebersihan.")
        else:
            reasons.append(f"Lantai: {FLOOR_DESC.get(floor, floor)}")

    if reasons:
        parts.append("Alasan: " + " ".join(reasons))

    paragraph = " ".join(parts)
    return paragraph, sev


# === build point response ===
def build_point_response(rec, severity):
    roof = (rec.get("roof_condition") or "unknown").lower()
    wall = (rec.get("wall_condition") or "unknown").lower()
    floor_raw = rec.get("floor_condition")
    floor = normalize_floor_label(floor_raw)

    severity_label = severity_to_label(severity)

    # include wall_material if present (helps model)
    wall_mat = rec.get("wall_material")
    wall_mat_part = f"; wall_material: {wall_mat}" if wall_mat else ""

    # structured points + final classification
    resp = (
        f"roof: {roof}; "
        f"wall: {wall}; "
        f"floor: {floor}{wall_mat_part}; "
        f"severity: {severity_label}; "
        f"final_classification: {severity_label}"
    )
    return resp


# === main converter ===
def main():
    # load input
    if not os.path.exists(INPUT_FILE):
        print("Input file not found:", INPUT_FILE)
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print("Gagal membaca JSON input:", e)
            return

    # expect data to be list of records
    if not isinstance(data, list):
        print("Input JSON harus berupa array/list of records.")
        return

    total = 0
    severity_counter = Counter()

    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        for rec in data:
            # canonicalize keys lower-case if necessary
            clean = {k: rec.get(k) for k in rec.keys()}

            # build paragraph (will infer severity if needed)
            paragraph, sev_inferred = build_paragraph_response(clean)

            # prefer explicit numeric severity_score if provided and valid
            try:
                sev_existing = int(rec.get("severity_score")) if rec.get("severity_score") is not None else None
            except Exception:
                sev_existing = None
            severity_final = sev_existing if sev_existing is not None else sev_inferred

            # SAMPLE 1: paragraph (Bahasa Indonesia)
            instr1 = "Analisis kondisi rumah pada gambar ini. Berikan penjelasan singkat dalam satu paragraf yang menyertakan alasan."
            sample1 = {
                "image": clean.get("image"),
                "instruction": instr1,
                "response": paragraph
            }
            out_f.write(json.dumps(sample1, ensure_ascii=False) + "\n")
            total += 1

            # SAMPLE 2: structured point response (uses severity label + final classification)
            instr2 = "Keluarkan output dalam format key:value untuk: roof, wall, floor, (opsional: wall_material), severity, final_classification."
            point_resp = build_point_response(clean, severity_final)
            sample2 = {
                "image": clean.get("image"),
                "instruction": instr2,
                "response": point_resp
            }
            out_f.write(json.dumps(sample2, ensure_ascii=False) + "\n")
            total += 1

            severity_counter[severity_final] += 1

    print("SFT dataset generated:", OUT_FILE)
    print("Total samples (lines):", total)
    print("Severity distribution (per house):")
    for sev, cnt in sorted(severity_counter.items()):
        label = SEVERITY_LABEL.get(sev, "unknown")
        print(f"  {sev} ({label}): {cnt}")


if __name__ == "__main__":
    main()