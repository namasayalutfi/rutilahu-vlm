from PIL import Image
import os

class ImageResizer:
    def __init__(self, input_folder, output_folder, size=(640,640), quality=95):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.size = size
        self.quality = quality
        os.makedirs(self.output_folder, exist_ok=True)

    def run(self):
        if not os.path.exists(self.input_folder):
            raise FileNotFoundError(f"Input folder not found: {self.input_folder}")
        total = 0
        for fname in os.listdir(self.input_folder):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                continue
            inpath = os.path.join(self.input_folder, fname)
            outpath = os.path.join(self.output_folder, fname)
            try:
                img = Image.open(inpath).convert("RGB")
                img = img.resize(self.size, Image.BICUBIC)
                img.save(outpath, quality=self.quality)
                print(f"Resized: {fname}")
                total += 1
            except Exception as e:
                print(f"Failed to resize {fname}: {e}")
        print(f"Selesai resize semua gambar. total resized: {total}")
        return total