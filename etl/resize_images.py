from PIL import Image
import os

input_folder = "raw_img"
output_folder = "resized_img"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):

    if file.lower().endswith((".png", ".jpg", ".jpeg")):

        path = os.path.join(input_folder, file)

        img = Image.open(path).convert("RGB")

        img = img.resize((640, 640), Image.BICUBIC)

        img.save(os.path.join(output_folder, file), quality=95)

        print(f"Resized: {file}")

print("Selesai resize semua gambar.")