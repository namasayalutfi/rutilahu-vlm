import os, requests
import urllib3
from urllib.parse import urlparse
from time import sleep

# Menonaktifkan peringatan SSL jika verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ImageDownloader:
    def __init__(self, input_txt, out_dir, min_size=1024, rate_limit=0.5, start_index=1, verify=False):
        self.input_txt = input_txt
        self.out_dir = out_dir
        self.min_size = min_size
        self.rate_limit = rate_limit
        self.start_index = start_index
        self.verify = verify
        self.prefix = "raw_img_"  # Konsistensi prefix
        os.makedirs(self.out_dir, exist_ok=True)

    def last_index(self):
        """Mencari index terakhir yang ada di folder output."""
        files = os.listdir(self.out_dir)
        nums = []
        for f in files:
            # Mencari file yang diawali dengan raw_img_
            if f.lower().startswith(self.prefix) and "." in f:
                try:
                    # Contoh: raw_img_00005.jpg -> split("_") -> ['raw', 'img', '00005.jpg']
                    # Kita ambil index ke-2, lalu hilangkan ekstensinya
                    parts = f.split("_")
                    if len(parts) >= 3:
                        num_part = parts[2].split(".")[0]
                        nums.append(int(num_part))
                except (ValueError, IndexError):
                    continue
        
        # Jika folder kosong, return start_index - 1 (agar saat +1 jadi start_index)
        return max(nums) if nums else self.start_index - 1

    def download(self, url, outpath, timeout=15):
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                stream=True,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                verify=self.verify,
            )
            if resp.status_code != 200:
                return False, f"status {resp.status_code}"
            
            content = resp.content
            if len(content) < self.min_size:
                return False, "file terlalu kecil"
            
            with open(outpath, "wb") as fh:
                fh.write(content)
            return True, None
        except Exception as e:
            return False, str(e)

    def run(self, limit=None):
        if not os.path.exists(self.input_txt):
            raise FileNotFoundError(f"File input tidak ditemukan: {self.input_txt}")
            
        with open(self.input_txt, "r", encoding="utf-8") as fh:
            urls = [l.strip() for l in fh if l.strip()]

        # Tentukan index mulai berdasarkan file yang sudah ada
        idx = self.last_index() + 1
        saved = 0
        
        print(f"Memulai download dari index: {idx}")

        for u in urls:
            if limit is not None and saved >= limit:
                break
            
            # Menggunakan prefix yang konsisten: raw_img_
            fname = f"{self.prefix}{idx:05d}"
            
            # Mendapatkan ekstensi file
            path = urlparse(u).path
            ext = os.path.splitext(path)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]:
                ext = ".jpg"
            
            outpath = os.path.join(self.out_dir, fname + ext)

            # Cek jika file sudah ada (double check)
            if os.path.exists(outpath):
                print(f"Skip: {outpath} sudah ada.")
                idx += 1
                continue

            ok, err = self.download(u, outpath)
            if ok:
                print(f"[{idx}] Berhasil: {outpath}")
                saved += 1
                idx += 1 # Naikkan index hanya jika berhasil atau jika ingin angka terus urut
                sleep(self.rate_limit)
            else:
                print(f"[{idx}] Gagal: {u} -> {err}")
                # Tetap naikkan index agar tidak menimpa jika running berikutnya
                idx += 1 
            
        print(f"--- Selesai ---")
        print(f"Total tersimpan: {saved}")
        return saved