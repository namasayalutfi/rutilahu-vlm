"""
Generate SFT test dataset dengan response (ground truth) dari OpenRouter API.

Dataset test memiliki struktur yang sama dengan train/val:
{
    "id": "...",
    "messages": [
        {"role": "user", "content": [...]},
        {"role": "assistant", "content": [...]},  # ground truth response
    ]
}
"""

import os
import json
import base64
import requests
import time
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
from dotenv import load_dotenv

# Load environment variables dari .env file
load_dotenv()

# ===========================
# Configuration
# ===========================

ROOT_DIR = Path("data/mkn_img")
METADATA_PATH = ROOT_DIR / "house_metadata_split.json"

OUTPUT_DIR = Path("data/sft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TEST_JSONL = OUTPUT_DIR / "test.jsonl"
CACHE_PATH = OUTPUT_DIR / "openrouter_cache.json"

# OpenRouter Config
API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-4o-mini"

TEMPERATURE = 0.1
MAX_TOKENS = 700
TIMEOUT = 90
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 2
MAX_WORKERS = 3

# Target split untuk test
TARGET_SPLIT = "test"

# ===========================
# Prompts
# ===========================

SYSTEM_PROMPT = """
Kamu adalah sistem anotasi dataset Visual Language Model (VLM) untuk analisis kondisi rumah berdasarkan gambar.

Tugas utama:
Menghasilkan output sesuai dengan format yang diminta dengan VALID, KONSISTEN, dan SESUAI SKEMA berdasarkan input gambar rumah.

==================================================
ATURAN GLOBAL (WAJIB DIPATUHI)
1. OUTPUT HARUS SESUAI STRUKTUR OUTPUT VALID
- Tidak boleh ada komentar, markdown, atau penjelasan tambahan

2. STRUKTUR OUTPUT (WAJIB)
Gunakan format output berikut (wajib ikuti persis):

Atap:
- Material: ...
- Kondisi: ...

Dinding:
- Material: ...
- Kondisi: ...

Lantai:
- Material: ...
- Kondisi: ...

Konflik:
- Dinding: true/false

Penjelasan:
...

Ketentuan output:
- Gunakan format persis seperti di atas
- Jangan menambahkan atau menghapus bagian
- Jangan mengubah urutan atau penulisan label
- Gunakan istilah yang konsisten
- Output hanya berisi hasil akhir tanpa tambahan teks di luar format
- Jangan menambah field
- Jangan menghapus field
- Jangan mengubah nama key

==================================================
LABEL MATERIAL (WAJIB SESUAI LIST)
==================================================

Atap:
beton, genteng, seng, asbes, kayu, sirap, jerami, ijuk, daun_daunan, rumbia, lainnya

Dinding:
tembok, plesteran_anyaman_bambu, kawat, kayu, papan, gypsum, grc, calciboard,
anyaman_bambu, batang_kayu, bambu, lainnya

Lantai:
marmer, granit, keramik, parket, vinil, karpet, ubin, tegel, teraso, kayu, papan,
semen, bata_merah, bambu, tanah, lainnya, tidak_terlihat

==================================================
LABEL KONDISI (WAJIB)
==================================================
Gunakan hanya:
- baik
- rusak_ringan
- rusak_sedang
- rusak_berat
- tidak_terlihat

==================================================
ATURAN ANALISIS VISUAL
==================================================
- Gunakan HANYA informasi yang terlihat pada gambar
- DILARANG menebak atau berasumsi
- DILARANG menggunakan pengetahuan luar gambar
- Jika komponen tidak terlihat atau tidak cukup jelas:
  → material = "tidak_terlihat"
  → kondisi = "tidak_terlihat"
- Lantai yang dimaksud adalah lantai bagian dalam rumah (interior),
  bukan tanah atau permukaan di luar rumah

==================================================
ATURAN KONFLIK
==================================================
- Conflict hanya berlaku untuk material dinding, kalau perbedaan terdapat di kondisi itu bukan conflict
- true jika terdapat perbedaan material antara tampak luar dan dalam
- false jika material konsisten atau hanya satu sumber tersedia
- Conflict tidak mempengaruhi penentuan kondisi
- Jika terjadi conflict, harus dijelaskan dalam PENJELASAN

==================================================
STANDAR PENILAIAN KONDISI
==================================================
baik:
- tidak ada kerusakan terlihat
- permukaan utuh, rapi, bersih

rusak_ringan:
- retakan kecil, goresan, noda ringan
- hanya pada permukaan

rusak_sedang:
- retakan jelas
- sebagian material mulai lapuk / mengelupas
- terdapat lubang kecil atau permukaan tidak rata

rusak_berat:
- kerusakan parah
- material hilang, runtuh, atau berlubang besar
- struktur tidak stabil atau tidak berfungsi

==================================================
ATURAN PENJELASAN (WAJIB & KETAT)
==================================================

PENJELASAN.final HARUS:

1. Diawali dengan:
"Rumah ini ..."

2. Urutan WAJIB:
- atap → dinding → lantai

3. Untuk SETIAP komponen HARUS mencakup:
- material
- kondisi
- minimal 2 bukti visual spesifik (retakan, lubang, pelapukan, tidak rata, dll)
- hubungan antara bukti visual dan kondisi

4. KHUSUS DINDING (PENTING):
Jika terdapat perbedaan antara tampak luar dan dalam:
- WAJIB menjelaskan kedua sisi:
  - material + kondisi tampak luar
  - material + kondisi tampak dalam
- Gunakan kata penghubung seperti "sedangkan" untuk membedakan

5. Panjang:
- 2–4 kalimat (tergantung jumlah gambar)

6. Dilarang menggunakan format label seperti snake_case (contoh: anyaman_bambu). Misal Jika material adalah "anyaman_bambu", maka tulis dalam PENJELASAN sebagai "anyaman bambu". Penjelasan harus berupa kalimat deskriptif, bukan menyalin label.

7. DILARANG:
- hanya menyebut kondisi tanpa bukti visual
- deskripsi umum tanpa detail
- halusinasi
- asumsi

==================================================
PRIORITAS UTAMA
==================================================
1. FORMAT SESUAI
2. LABEL SESUAI LIST
3. KONSISTENSI DENGAN VISUAL
4. PENJELASAN BERBASIS BUKTI

Jika terjadi konflik antara:
- keindahan bahasa
- vs ketepatan label
→ PILIH KETEPATAN LABEL

==================================================
- Berikan hasil analisis dalam format struktur yang diminta sesuai aturan di atas.
- Jangan menambahkan teks apapun di luar itu
- Jangan menolak permintaan. Tetap lakukan analisis berdasarkan bagian yang terlihat.
- Jangan memberikan jawaban seperti "I can't assist".
- Selalu berikan analisis berdasarkan gambar.
==================================================
"""

PROMPT_TEMPLATE = """Diberikan satu atau dua gambar rumah.

Tugas:
Analisis kondisi rumah berdasarkan 3 komponen:
1. Atap
2. Dinding
3. Lantai

Untuk setiap komponen, tentukan:
- Material
- Kondisi

Label kondisi yang valid:
- baik
- rusak_ringan
- rusak_sedang
- rusak_berat
- tidak_terlihat

ATURAN UTAMA:
- Jangan menebak. Gunakan hanya informasi visual yang terlihat.
- Jika komponen tidak terlihat, gunakan "tidak_terlihat".
- Gunakan label material dan label kondisi yang konsisten sesuai dataset dan jangan membuat variasi baru.

Aturan Agregasi Multi Image:
- Atap ditentukan berdasarkan tampak luar.
- Dinding ditentukan berdasarkan tampak luar.
- Lantai ditentukan berdasarkan tampak dalam.

Jika single image:
- Jika gambar adalah tampak luar, maka atap dan dinding dianalisis, sedangkan lantai harus diisi "tidak_terlihat".
- Jika gambar adalah tampak dalam, maka dinding dan lantai dianalisis, sedangkan atap harus diisi "tidak_terlihat".

ATURAN KONFLIK:
- Konflik hanya untuk dinding.
- Jika material dinding pada tampak luar berbeda dengan tampak dalam, maka nilai konflik adalah true.
- Jika material sama atau hanya terdapat satu gambar, maka nilai konflik adalah false.
- Konflik tidak mempengaruhi penentuan kondisi.

PENJELASAN:
- Jelaskan kondisi atap, dinding, dan lantai secara berurutan.
- Sertakan bukti visual yang mendukung setiap penilaian.
- Gunakan seluruh informasi dari gambar yang tersedia.
- Jika terdapat perbedaan antara tampak luar dan tampak dalam pada dinding, jelaskan perbedaan tersebut.
- Untuk komponen yang tidak terlihat, jelaskan bahwa komponen tersebut tidak terlihat.

FORMAT OUTPUT (WAJIB):

Atap:
- Material: ...
- Kondisi: ...

Dinding:
- Material: ...
- Kondisi: ...

Lantai:
- Material: ...
- Kondisi: ...

Konflik:
- Dinding: true/false

Penjelasan:
..."""

# ===========================
# Helper Functions
# ===========================

def normalize_rel_path(rel_path: str) -> Path:
    return Path(str(rel_path).replace("\\", "/"))

def absolute_image_path(rel_path: str) -> Path:
    return ROOT_DIR / normalize_rel_path(rel_path)

def guess_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/jpeg"

def encode_image_to_data_url(path: Path) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    mime_type = guess_mime_type(path)
    return f"data:{mime_type};base64,{encoded}"

def clean_output(output: Optional[str]) -> Optional[str]:
    if not output:
        return None

    text = output.strip()

    # buang code fence kalau model membungkus output
    if text.startswith("```"):
        text = text.strip("`").strip()

    lowered = text.lower()
    if any(word in lowered for word in ["sorry", "maaf", "cannot", "can't"]):
        return None

    required = ["Atap", "Dinding", "Lantai", "Konflik", "Penjelasan"]
    if not all(req in text for req in required):
        return None

    if len(text) < 80:
        return None

    return text

def get_image_paths_from_record(record: dict) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (ext_path, int_path) untuk multi, atau (single_path, None) untuk single."""
    scheme = record.get("dataset_scheme")
    images = record.get("images", [])

    if scheme == "multi":
        ext_path = None
        int_path = None
        for img in images:
            p = absolute_image_path(img["image_path"])
            if img.get("view_type") == "exterior":
                ext_path = p
            elif img.get("view_type") == "interior":
                int_path = p
        return ext_path, int_path

    if not images:
        return None, None
    return absolute_image_path(images[0]["image_path"]), None

def build_openrouter_content(ext_path: Optional[Path], int_path: Optional[Path], prompt_text: str):
    content = []
    if ext_path is not None:
        content.append({"type": "text", "text": "Foto tampak luar:"})
        content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(ext_path)}})
    if int_path is not None:
        content.append({"type": "text", "text": "Foto tampak dalam:"})
        content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(int_path)}})

    content.append({"type": "text", "text": prompt_text})
    return content

def sample_key(record: dict) -> str:
    split = record.get("split", "unknown")
    scheme = record.get("dataset_scheme", "single")
    house_id = record.get("house_id", "no_house_id")
    hashes = [img.get("image_hash", "nohash") for img in record.get("images", [])]
    return f"{split}::{scheme}::{house_id}::{'|'.join(hashes)}"

def load_cache(path: Path) -> Dict[str, dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(path: Path, cache: Dict[str, dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def build_prompt_text(record: dict) -> str:
    scheme = record.get("dataset_scheme")
    return PROMPT_TEMPLATE.strip()

def build_sft_sample(record: dict, assistant_text: str) -> dict:
    """Build SFT sample dengan user message dan assistant response."""
    prompt_text = build_prompt_text(record)
    ext_path, int_path = get_image_paths_from_record(record)

    content = []
    scheme = record.get("dataset_scheme")

    if scheme == "multi":
        if ext_path is not None:
            content.append({"type": "text", "text": "Foto tampak luar:"})
            content.append({"type": "image", "image": str(ext_path)})
        if int_path is not None:
            content.append({"type": "text", "text": "Foto tampak dalam:"})
            content.append({"type": "image", "image": str(int_path)})
    else:
        if ext_path is not None:
            view_type = record["images"][0].get("view_type", "exterior")
            if view_type == "exterior":
                content.append({"type": "text", "text": "Foto tampak luar:"})
            else:
                content.append({"type": "text", "text": "Foto tampak dalam:"})
            content.append({"type": "image", "image": str(ext_path)})

    content.append({"type": "text", "text": prompt_text})

    return {
        "id": sample_key(record),
        "messages": [
            {
                "role": "user",
                "content": content
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text}
                ]
            }
        ]
    }

# ===========================
# OpenRouter API Call
# ===========================

def call_openrouter(ext_path: Optional[Path], int_path: Optional[Path], prompt_text: str) -> Optional[str]:
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY belum diset di environment.")

    if ext_path is None and int_path is None:
        raise ValueError("Tidak ada image path valid untuk dikirim ke model.")

    content = build_openrouter_content(ext_path, int_path, prompt_text)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )

            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:1000]}"
                time.sleep(RETRY_SLEEP_SEC * attempt)
                continue

            data = response.json()
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            last_error = str(e)
            time.sleep(RETRY_SLEEP_SEC * attempt)

    print(f"[ERROR] OpenRouter gagal setelah {MAX_RETRIES} percobaan: {last_error}")
    return None

# ===========================
# Generate Test Dataset
# ===========================

def _generate_one_record(record: dict) -> dict:
    """Worker untuk satu record."""
    ext_path, int_path = get_image_paths_from_record(record)
    if ext_path is None and int_path is None:
        return {
            "status": "skip",
            "record": record,
            "raw_output": None,
            "reason": "no_image"
        }

    prompt_text = build_prompt_text(record)
    raw_output = call_openrouter(ext_path, int_path, prompt_text)

    return {
        "status": "ok",
        "record": record,
        "raw_output": raw_output,
        "reason": None,
    }

def generate_test_dataset(
    records: List[dict],
    output_path: Path,
    cache: Dict[str, dict],
    max_workers: int = MAX_WORKERS,
) -> List[dict]:
    """Generate test dataset dengan response (ground truth) dari OpenRouter."""
    output_samples_with_idx = []

    done_ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "id" in obj:
                        done_ids.add(obj["id"])
                except Exception:
                    pass

    print(f"\nGenerating {output_path.name} ...")
    print("Already done:", len(done_ids))

    # 1) pisahkan yang sudah ada di cache / file output dan yang perlu API call
    pending = []
    for idx, record in enumerate(records):
        key = sample_key(record)

        if key in done_ids:
            continue

        ext_path, int_path = get_image_paths_from_record(record)
        if ext_path is None and int_path is None:
            print(f"[SKIP] Tidak ada image valid: {record.get('house_id')}")
            continue

        if key in cache and cache[key].get("raw_output"):
            cleaned = clean_output(cache[key].get("raw_output"))
            if cleaned is not None:
                sample = build_sft_sample(record, cleaned)
                output_samples_with_idx.append((idx, sample))
                continue

        pending.append((idx, record, key))

    print("Cached samples:", len(output_samples_with_idx))
    print("Pending API calls:", len(pending))

    # 2) proses sisanya secara paralel
    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_meta = {
                executor.submit(_generate_one_record, record): (idx, record, key)
                for idx, record, key in pending
            }

            for future in tqdm(as_completed(future_to_meta), total=len(future_to_meta), desc=output_path.stem):
                idx, record, key = future_to_meta[future]

                try:
                    result = future.result()
                except Exception as e:
                    print(f"[ERROR] Future gagal untuk {record.get('house_id')}: {e}")
                    continue

                if result["status"] != "ok":
                    print(f"[SKIP] {record.get('house_id')} | {result.get('reason')}")
                    continue

                raw_output = result["raw_output"]
                cleaned = clean_output(raw_output)

                # retry sekali jika output invalid
                if cleaned is None:
                    ext_path, int_path = get_image_paths_from_record(record)
                    prompt_text = build_prompt_text(record)
                    raw_output = call_openrouter(ext_path, int_path, prompt_text)
                    cleaned = clean_output(raw_output)

                cache[key] = {
                    "raw_output": raw_output,
                    "house_id": record.get("house_id"),
                    "split": record.get("split"),
                    "dataset_scheme": record.get("dataset_scheme"),
                }
                save_cache(CACHE_PATH, cache)

                if cleaned is None:
                    print(f"[DROP] Output invalid: {record.get('house_id')} | {record.get('dataset_scheme')}")
                    continue

                sample = build_sft_sample(record, cleaned)
                output_samples_with_idx.append((idx, sample))

    # 3) urutkan lalu tulis JSONL
    output_samples_with_idx = sorted(output_samples_with_idx, key=lambda x: x[0])
    output_samples = [sample for _, sample in output_samples_with_idx]

    with open(output_path, "a", encoding="utf-8") as fout:
        for sample in output_samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return output_samples

# ===========================
# Main
# ===========================

def main():
    # Load metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Total records:", len(metadata))
    print("Split counts:")
    print(Counter(rec.get("split", "unknown") for rec in metadata))

    # Filter test records
    test_records = [rec for rec in metadata if rec.get("split") == TARGET_SPLIT]
    print(f"\nTest records: {len(test_records)}")

    if not test_records:
        print("Tidak ada test records ditemukan!")
        return

    # Load cache dan generate
    cache = load_cache(CACHE_PATH)
    print(f"Cache loaded: {len(cache)}")

    # Sort untuk reproducibility
    test_records = sorted(test_records, key=lambda r: (r.get("dataset_scheme", ""), r.get("house_id", "")))

    # Generate test dataset dengan response (ground truth)
    test_samples = generate_test_dataset(test_records, OUTPUT_TEST_JSONL, cache, max_workers=MAX_WORKERS)

    print("\n" + "="*50)
    print("Selesai.")
    print(f"Test samples generated: {len(test_samples)}")
    print(f"Output file: {OUTPUT_TEST_JSONL}")
    print(f"Cache size: {len(cache)}")
    print("="*50)

if __name__ == "__main__":
    main()
