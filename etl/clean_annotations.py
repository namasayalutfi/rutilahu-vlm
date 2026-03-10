import os
import json

class AnnotationCleaner:
    SEVERITY_MAP = {
        1: "layak",
        2: "perlu_perbaikan_ringan",
        3: "perlu_perbaikan_sedang",
        4: "tidak_layak",
    }

    def __init__(self, input_file, out_dir, out_file):
        self.input_file = input_file
        self.out_dir = out_dir
        self.out_file = out_file
        os.makedirs(self.out_dir, exist_ok=True)

    def clean_record(self, rec):
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
        clean["wall_material"] = rec.get("wall_material")
        sev = rec.get("severity_score") or rec.get("severity")
        try:
            sev = int(sev)
        except Exception:
            sev = None
        clean["severity_score"] = sev
        clean["severity_label"] = self.SEVERITY_MAP.get(sev)
        return clean

    def run(self):
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Raw annotations file not found: {self.input_file}")
        with open(self.input_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cleaned = [self.clean_record(r) for r in data]
        with open(self.out_file, "w", encoding="utf-8") as out:
            json.dump(cleaned, out, ensure_ascii=False, indent=2)
        print(f"Clean dataset saved to {self.out_file}. total: {len(cleaned)}")
        return len(cleaned)