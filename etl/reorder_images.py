import os
import re

class ImageReorderer:
    def __init__(self, target_dir, prefix="mkn_img_"):
        self.target_dir = target_dir
        self.prefix = prefix

    def _get_number(self, filename):
        """Mengekstrak angka dari nama file menggunakan regex."""
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0

    def run(self):
        print(f"🔄 Mengurutkan ulang file di: {self.target_dir}")
        
        if not os.path.exists(self.target_dir):
            print(f"Error: Folder {self.target_dir} tidak ditemukan.")
            return

        # 1. Ambil semua file gambar (hanya yang unik hasil filter sebelumnya)
        files = [f for f in os.listdir(self.target_dir) if os.path.isfile(os.path.join(self.target_dir, f))]
        
        # 2. Sort berdasarkan angka yang ada di nama file aslinya
        files.sort(key=self._get_number)

        print(f"Ditemukan {len(files)} file. Mulai rename...")

        # 3. Rename menjadi urutan baru (Temporary rename untuk menghindari tabrakan nama)
        temp_files = []
        for i, filename in enumerate(files, start=1):
            ext = os.path.splitext(filename)[1]
            old_path = os.path.join(self.target_dir, filename)
            # Gunakan temp prefix agar tidak bentrok dengan file yang belum di-rename
            temp_name = f"TEMP_{i:05d}{ext}"
            temp_path = os.path.join(self.target_dir, temp_name)
            
            os.rename(old_path, temp_path)
            temp_files.append((temp_path, ext, i))

        # 4. Final rename ke format mkn_img_XXXXX
        for temp_path, ext, index in temp_files:
            final_name = f"{self.prefix}{index:05d}{ext}"
            final_path = os.path.join(self.target_dir, final_name)
            os.rename(temp_path, final_path)

        print(f"✅ Selesai! File sekarang berurutan dari {self.prefix}00001 sampai {self.prefix}{len(files):05d}")