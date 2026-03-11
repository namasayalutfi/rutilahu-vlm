import os
import shutil
import hashlib
from PIL import Image
import imagehash

class ImageProcessor:
    def __init__(self, input_dir, output_dir, duplicates_dir, invalid_dir="data/invalid_files", 
                 size=(640, 640), phash_threshold=5, quality=95):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.duplicates_dir = duplicates_dir
        self.invalid_dir = invalid_dir
        self.size = size
        self.phash_threshold = phash_threshold
        self.quality = quality

        # Buat semua direktori yang diperlukan
        for d in [self.output_dir, self.duplicates_dir, self.invalid_dir]:
            os.makedirs(d, exist_ok=True)

    def _get_md5_hash(self, filepath):
        """Hash biner untuk deteksi file yang 100% identik."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def run(self):
        print(f"🚀 Memulai Pipeline: Filter + Resize di {self.input_dir}")
        
        # Ekstensi yang diizinkan (termasuk yang akan dikonversi)
        valid_ext = ('.jpg', '.jpeg', '.png', '.webp', '.jfif', '.bmp')
        
        exact_hashes = {}   # MD5 -> Filename
        visual_hashes = {}  # Filename -> pHash
        
        stats = {"total": 0, "resized": 0, "duplicates": 0, "invalid": 0}
        all_files = [f for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f))]
        stats["total"] = len(all_files)

        for filename in all_files:
            filepath = os.path.join(self.input_dir, filename)
            ext = os.path.splitext(filename)[1].lower()

            # --- TAHAP 1: FILTER FORMAT ---
            if ext not in valid_ext:
                print(f"[INVALID] Format dilarang: {filename}")
                shutil.move(filepath, os.path.join(self.invalid_dir, filename))
                stats["invalid"] += 1
                continue

            # --- TAHAP 2: BUKA GAMBAR & CEK DUPLIKAT ---
            try:
                # Cek MD5 dulu (sangat cepat, tanpa buka gambar)
                md5_hash = self._get_md5_hash(filepath)
                if md5_hash in exact_hashes:
                    print(f"[DUPLICATE] MD5 Identik: {filename}")
                    shutil.move(filepath, os.path.join(self.duplicates_dir, filename))
                    stats["duplicates"] += 1
                    continue

                # Buka gambar untuk pHash dan Resize
                with Image.open(filepath) as img:
                    img = img.convert("RGB") # Pastikan format RGB (buang Alpha/Transparansi)
                    
                    # Cek pHash (Kemiripan Visual)
                    current_phash = imagehash.phash(img)
                    is_visual_duplicate = False
                    
                    for saved_name, saved_phash in visual_hashes.items():
                        if current_phash - saved_phash <= self.phash_threshold:
                            print(f"[DUPLICATE] Visual Mirip: {filename} ≈ {saved_name}")
                            is_visual_duplicate = True
                            break
                    
                    if is_visual_duplicate:
                        shutil.move(filepath, os.path.join(self.duplicates_dir, filename))
                        stats["duplicates"] += 1
                        continue

                    # --- TAHAP 3: RESIZE & SAVE (Jika Unik) ---
                    # Ganti ekstensi jadi .jpg agar seragam untuk VLM
                    new_filename = os.path.splitext(filename)[0] + ".jpg"
                    out_path = os.path.join(self.output_dir, new_filename)
                    
                    img_resized = img.resize(self.size, Image.Resampling.LANCZOS)
                    img_resized.save(out_path, "JPEG", quality=self.quality)
                    
                    # Simpan hash untuk referensi file berikutnya
                    exact_hashes[md5_hash] = filename
                    visual_hashes[filename] = current_phash
                    stats["resized"] += 1
                    print(f"[SUCCESS] Berhasil Resize: {new_filename}")

            except Exception as e:
                print(f"[ERROR] Gagal memproses {filename}: {e}")
                shutil.move(filepath, os.path.join(self.invalid_dir, filename))
                stats["invalid"] += 1

        self._print_report(stats)

    def _print_report(self, stats):
        print("\n" + "="*40)
        print("📊 LAPORAN AKHIR PIPELINE")
        print("="*40)
        print(f"Total File di Raw     : {stats['total']}")
        print(f"Hasil Unik (Resized)  : {stats['resized']}")
        print(f"Duplikat Dibuang      : {stats['duplicates']}")
        print(f"File Rusak/Invalid    : {stats['invalid']}")
        print("="*40)