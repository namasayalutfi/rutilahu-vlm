"""
Generate SFT JSONL dataset langsung dalam format Qwen-VL chat (messages) 
dari centralized cleaned annotations file. 
2 samples per record: Paragraph (Bahasa Indonesia) + Structured JSON response.
"""
import os
import json
import pathlib
from collections import Counter

class SFTGenerator:
    ROOF_DESC = {
        "good":"Atap utuh, tidak ada lubang atau bagian yang hilang.",
        "minor_damage":"Kerusakan ringan pada atap.",
        "major_damage":"Kerusakan berat pada atap.",
        "collapsed":"Atap runtuh atau hilang dalam skala besar.",
        "unknown":"Kondisi atap tidak dapat diidentifikasi dari gambar."
    }
    WALL_DESC = {
        "good":"Dinding dalam kondisi baik dan kokoh.",
        "minor_damage":"Kerusakan ringan pada dinding.",
        "major_damage":"Kerusakan besar pada dinding.",
        "collapsed":"Dinding runtuh.",
        "unknown":"Kondisi dinding tidak dapat diidentifikasi."
    }
    WALL_MATERIAL_DESC = {
        "concrete":"Dinding dari beton cor.",
        "brick":"Dinding dari bata merah.",
        "wood":"Dinding dari papan kayu.",
        "bamboo":"Dinding dari anyaman bambu.",
        "mixed":"Dinding menggunakan lebih dari satu material.",
        "unknown":"Material dinding tidak dapat diidentifikasi."
    }
    FLOOR_DESC = {
        "ceramic":"Lantai menggunakan keramik atau ubin.",
        "cement":"Lantai menggunakan semen atau plester.",
        "wood":"Lantai menggunakan papan kayu.",
        "dirt":"Lantai berupa tanah tanpa lapisan material.",
        "unknown":"Material lantai tidak dapat diidentifikasi."
    }
    SEVERITY_LABEL = {
        1:"layak_huni",2:"perlu_perbaikan_ringan",3:"perlu_perbaikan_sedang",4:"tidak_layak_huni"
    }

    def __init__(self, input_file, out_file, raw_img_prefix="raw_img/"):
        self.input_file = input_file
        self.out_file = out_file
        self.raw_img_prefix = raw_img_prefix

    def normalize_floor_label(self, floor_raw):
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

    def severity_to_label(self, sev):
        try:
            return self.SEVERITY_LABEL.get(int(sev), "unknown")
        except Exception:
            return "unknown"

    def infer_severity_from_components(self, roof, wall, floor):
        for v in (roof, wall):
            if v == "collapsed":
                return 4
        if roof == "major_damage" or wall == "major_damage":
            return 3
        if roof == "minor_damage" or wall == "minor_damage":
            return 2
        return 1

    def build_paragraph_response(self, rec):
        image = rec.get("image") or ""
        roof = (rec.get("roof_condition") or "unknown").lower()
        wall = (rec.get("wall_condition") or "unknown").lower()
        floor_raw = rec.get("floor_condition")
        floor = self.normalize_floor_label(floor_raw)
        wall_mat = (rec.get("wall_material") or "").lower()
        sev = rec.get("severity_score")
        try:
            sev = int(sev) if sev is not None else None
        except Exception:
            sev = None
        if sev is None:
            sev = self.infer_severity_from_components(roof, wall, floor)
        sev_label = self.severity_to_label(sev)

        parts = [f"Analisis singkat (gambar: {pathlib.Path(image).name}):"]
        parts.extend([
            f"Atap: {self.ROOF_DESC.get(roof, self.ROOF_DESC['unknown'])}",
            f"Dinding: {self.WALL_DESC.get(wall, self.WALL_DESC['unknown'])}",
            f"Lantai: {self.FLOOR_DESC.get(floor, self.FLOOR_DESC['unknown'])}",
        ])
        if wall_mat:
            parts.append(self.WALL_MATERIAL_DESC.get(wall_mat, self.WALL_MATERIAL_DESC["unknown"]))
        parts.append(f"Klasifikasi akhir: {sev_label} (kode {sev}).")

        reasons = []
        if roof in self.ROOF_DESC:
            reasons.append(self.ROOF_DESC[roof])
        if wall in self.WALL_DESC:
            reasons.append(self.WALL_DESC[wall])
        if wall_mat == "wood":
            reasons.append("Dinding kayu rentan rayap dan pelapukan.")
        elif wall_mat == "bamboo":
            reasons.append("Dinding bambu rentan kelembaban dan hama.")
        if floor == "dirt":
            reasons.append("Lantai tanah berisiko kelembaban tinggi.")
        if reasons:
            parts.append("Alasan: " + " ".join(reasons[:3]))

        return " ".join(parts), sev

    def build_point_response(self, rec, severity):
        roof = (rec.get("roof_condition") or "unknown").lower()
        wall = (rec.get("wall_condition") or "unknown").lower()
        floor_raw = rec.get("floor_condition")
        floor = self.normalize_floor_label(floor_raw)
        wall_mat = rec.get("wall_material")
        severity_label = self.severity_to_label(severity)
        wall_mat_part = f"; wall_material: {wall_mat}" if wall_mat else ""
        return (
            f"roof: {roof}; "
            f"wall: {wall}; "
            f"floor: {floor}{wall_mat_part}; "
            f"severity: {severity_label}; "
            f"final_classification: {severity_label}"
        )

    def run(self):
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Clean annotations not found: {self.input_file}")
        with open(self.input_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("Input harus list of records")
        total = 0
        severity_counter = Counter()
        os.makedirs(os.path.dirname(self.out_file), exist_ok=True)
        with open(self.out_file, "w", encoding="utf-8") as out_f:
            for rec in data:
                clean = {k: rec.get(k) for k in rec.keys()}
                paragraph, sev_inferred = self.build_paragraph_response(clean)
                sev_existing = None
                try:
                    sev_existing = int(rec.get("severity_score")) if rec.get("severity_score") is not None else None
                except Exception:
                    pass
                severity_final = sev_existing if sev_existing is not None else sev_inferred

                qwen_sample1 = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"{self.raw_img_prefix}{clean.get('image','')}"},
                                {"type": "text", "text": "Analisis kondisi rumah pada gambar ini. Berikan penjelasan singkat dalam satu paragraf yang menyertakan alasan."}
                            ]
                        },
                        {"role": "assistant", "content": [{"type":"text","text":paragraph}]}
                    ]
                }
                out_f.write(json.dumps(qwen_sample1, ensure_ascii=False) + "\n")
                total += 1

                point_resp = self.build_point_response(clean, severity_final)
                qwen_sample2 = {
                    "images": [f"{self.raw_img_prefix}{clean.get('image','') }"],
                    "messages": [
                        {
                            "role": "user",
                            "content":[
                                {"type":"image","image":f"{self.raw_img_prefix}{clean.get('image','')}"},
                                {"type":"text","text":"Keluarkan output dalam format key:value untuk: roof, wall, floor, (opsional: wall_material), severity, final_classification."}
                            ]
                        },
                        {"role":"assistant","content":[{"type":"text","text":point_resp}]}
                    ]
                }
                out_f.write(json.dumps(qwen_sample2, ensure_ascii=False) + "\n")
                total += 1
                severity_counter[severity_final] += 1
        print("SFT dataset generated:", self.out_file)
        print(f"Total samples: {total} (2 per rumah)")
        print("Severity distribution:", dict(severity_counter))
        return total