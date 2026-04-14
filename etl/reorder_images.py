import os
import re


class ImageReorderer:
    def __init__(self, target_dir, prefix="mkn_img_"):
        self.target_dir = target_dir
        self.prefix = prefix

    def _get_number(self, filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0

    def run(self):
        print(f"🔄 Mengurutkan ulang file di: {self.target_dir}")

        if not os.path.exists(self.target_dir):
            print(f"Error: Folder {self.target_dir} tidak ditemukan.")
            return

        files = [
            f for f in os.listdir(self.target_dir)
            if os.path.isfile(os.path.join(self.target_dir, f))
        ]
        files.sort(key=self._get_number)

        print(f"Ditemukan {len(files)} file. Mulai rename...")

        temp_files = []
        for i, filename in enumerate(files, start=1):
            ext = os.path.splitext(filename)[1]
            old_path = os.path.join(self.target_dir, filename)
            temp_name = f"TEMP_{i:05d}{ext}"
            temp_path = os.path.join(self.target_dir, temp_name)

            os.rename(old_path, temp_path)
            temp_files.append((temp_path, ext, i))

        for temp_path, ext, index in temp_files:
            final_name = f"{self.prefix}{index:05d}{ext}"
            final_path = os.path.join(self.target_dir, final_name)
            os.rename(temp_path, final_path)

        print(f"✅ Selesai! File sekarang berurutan dari {self.prefix}00001 sampai {self.prefix}{len(files):05d}")