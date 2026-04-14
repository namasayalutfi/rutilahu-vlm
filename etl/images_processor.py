import os
import re
import shutil
import hashlib
from PIL import Image
import imagehash


class ImageProcessor:
    def __init__(self, input_dir, output_dir, duplicates_dir,
                 invalid_dir="data/invalid_files",
                 size=(640, 640), phash_threshold=5, quality=95,
                 output_prefix="mkn_img_"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.duplicates_dir = duplicates_dir
        self.invalid_dir = invalid_dir
        self.size = size
        self.phash_threshold = phash_threshold
        self.quality = quality
        self.output_prefix = output_prefix

        for d in [self.output_dir, self.duplicates_dir, self.invalid_dir]:
            os.makedirs(d, exist_ok=True)

    def _extract_number(self, filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else None

    def _get_md5_hash(self, filepath):
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _is_valid_image_ext(self, filename):
        valid_ext = ('.jpg', '.jpeg', '.png', '.webp', '.jfif', '.bmp')
        return os.path.splitext(filename)[1].lower() in valid_ext

    def _next_output_index(self):
        nums = []
        for f in os.listdir(self.output_dir):
            if f.startswith(self.output_prefix):
                n = self._extract_number(f)
                if n is not None:
                    nums.append(n)
        return max(nums) + 1 if nums else 1

    def _scan_output_hashes(self):
        """
        Ambil hash dari seluruh file yang sudah ada di mkn_img
        supaya file baru bisa dibandingkan dengan file lama.
        """
        exact_hashes = set()
        visual_hashes = []

        files = [
            f for f in os.listdir(self.output_dir)
            if os.path.isfile(os.path.join(self.output_dir, f))
        ]
        files.sort(key=lambda x: self._extract_number(x) or 0)

        for filename in files:
            filepath = os.path.join(self.output_dir, filename)
            if not self._is_valid_image_ext(filename):
                continue

            try:
                md5_hash = self._get_md5_hash(filepath)
                with Image.open(filepath) as img:
                    img = img.convert("RGB")
                    ph = imagehash.phash(img)

                exact_hashes.add(md5_hash)
                visual_hashes.append((filename, ph))
            except Exception:
                continue

        return exact_hashes, visual_hashes

    def _is_visual_duplicate(self, current_phash, visual_hashes):
        for saved_name, saved_phash in visual_hashes:
            if current_phash - saved_phash <= self.phash_threshold:
                return True, saved_name
        return False, None

    def run(self):
        print(f"🚀 Memulai Pipeline: Filter + Resize di {self.input_dir}")

        existing_md5s, existing_visuals = self._scan_output_hashes()
        next_index = self._next_output_index()

        current_md5s = set()
        current_visuals = []

        stats = {
            "total": 0,
            "resized": 0,
            "duplicates": 0,
            "invalid": 0
        }

        all_files = [
            f for f in os.listdir(self.input_dir)
            if os.path.isfile(os.path.join(self.input_dir, f))
        ]
        all_files.sort(key=lambda x: self._extract_number(x) or 0)
        stats["total"] = len(all_files)

        for filename in all_files:
            filepath = os.path.join(self.input_dir, filename)

            if not self._is_valid_image_ext(filename):
                print(f"[INVALID] Format dilarang: {filename}")
                shutil.move(filepath, os.path.join(self.invalid_dir, filename))
                stats["invalid"] += 1
                continue

            try:
                md5_hash = self._get_md5_hash(filepath)

                if md5_hash in existing_md5s or md5_hash in current_md5s:
                    print(f"[DUPLICATE] MD5 Identik: {filename}")
                    shutil.move(filepath, os.path.join(self.duplicates_dir, filename))
                    stats["duplicates"] += 1
                    continue

                with Image.open(filepath) as img:
                    img = img.convert("RGB")
                    current_phash = imagehash.phash(img)

                    is_dup, matched_name = self._is_visual_duplicate(
                        current_phash,
                        existing_visuals + current_visuals
                    )
                    if is_dup:
                        print(f"[DUPLICATE] Visual Mirip: {filename} ≈ {matched_name}")
                        shutil.move(filepath, os.path.join(self.duplicates_dir, filename))
                        stats["duplicates"] += 1
                        continue

                    new_filename = f"{self.output_prefix}{next_index:05d}.jpg"
                    out_path = os.path.join(self.output_dir, new_filename)

                    img_resized = img.resize(self.size, Image.Resampling.LANCZOS)
                    img_resized.save(out_path, "JPEG", quality=self.quality)

                    existing_md5s.add(md5_hash)
                    current_md5s.add(md5_hash)
                    existing_visuals.append((new_filename, current_phash))
                    current_visuals.append((new_filename, current_phash))

                    stats["resized"] += 1
                    print(f"[SUCCESS] Berhasil Resize: {new_filename}")
                    next_index += 1

            except Exception as e:
                print(f"[ERROR] Gagal memproses {filename}: {e}")
                try:
                    shutil.move(filepath, os.path.join(self.invalid_dir, filename))
                except Exception:
                    pass
                stats["invalid"] += 1

        self._print_report(stats)

    def dedupe_output(self):
        """
        Cek ulang seluruh isi mkn_img.
        File yang duplicate akan dipindahkan ke duplicates_dir.
        File pertama yang ketemu akan dipertahankan.
        """
        print(f"🧹 Mengecek ulang duplikat di folder output: {self.output_dir}")

        files = [
            f for f in os.listdir(self.output_dir)
            if os.path.isfile(os.path.join(self.output_dir, f))
        ]
        files.sort(key=lambda x: self._extract_number(x) or 0)

        seen_md5 = set()
        seen_visuals = []
        moved = 0

        for filename in files:
            filepath = os.path.join(self.output_dir, filename)
            if not self._is_valid_image_ext(filename):
                continue

            try:
                md5_hash = self._get_md5_hash(filepath)

                if md5_hash in seen_md5:
                    print(f"[OUTPUT DUPLICATE] MD5: {filename}")
                    shutil.move(filepath, os.path.join(self.duplicates_dir, filename))
                    moved += 1
                    continue

                with Image.open(filepath) as img:
                    img = img.convert("RGB")
                    ph = imagehash.phash(img)

                is_dup = False
                for saved_name, saved_ph in seen_visuals:
                    if ph - saved_ph <= self.phash_threshold:
                        print(f"[OUTPUT DUPLICATE] Visual: {filename} ≈ {saved_name}")
                        shutil.move(filepath, os.path.join(self.duplicates_dir, filename))
                        moved += 1
                        is_dup = True
                        break

                if is_dup:
                    continue

                seen_md5.add(md5_hash)
                seen_visuals.append((filename, ph))

            except Exception as e:
                print(f"[ERROR] Gagal cek output {filename}: {e}")

        print(f"✅ Output dedupe selesai. Dipindahkan ke duplicates: {moved}")
        return moved

    def _print_report(self, stats):
        print("\n" + "=" * 40)
        print("📊 LAPORAN AKHIR PIPELINE")
        print("=" * 40)
        print(f"Total File di Raw     : {stats['total']}")
        print(f"Hasil Unik (Resized)  : {stats['resized']}")
        print(f"Duplikat Dibuang      : {stats['duplicates']}")
        print(f"File Rusak/Invalid    : {stats['invalid']}")
        print("=" * 40)