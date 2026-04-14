from pathlib import Path
from PIL import Image


def get_jpg_files(folder_path):
    folder = Path(folder_path)
    return sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == ".jpg"
    ])


def resize_image_to_square(image_path, output_path, size=(640, 640), keep_aspect_ratio=True):
    """
    Resize gambar JPG ke ukuran square.

    keep_aspect_ratio=True:
        - proporsi gambar dijaga
        - hasil akhir tetap 640x640 dengan padding putih

    keep_aspect_ratio=False:
        - gambar langsung di-stretch ke 640x640
    """
    img = Image.open(image_path).convert("RGB")

    if keep_aspect_ratio:
        img.thumbnail(size, Image.Resampling.LANCZOS)

        new_img = Image.new("RGB", size, (255, 255, 255))
        x = (size[0] - img.width) // 2
        y = (size[1] - img.height) // 2
        new_img.paste(img, (x, y))
        new_img.save(output_path, format="JPEG", quality=95)
    else:
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(output_path, format="JPEG", quality=95)


def resize_folder_jpg(folder_path, output_folder=None, size=(640, 640), keep_aspect_ratio=True):
    """
    Resize semua JPG di folder.

    Jika output_folder None, file akan dioverwrite di folder yang sama.
    """
    input_folder = Path(folder_path)
    output_folder = Path(output_folder) if output_folder else input_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    files = get_jpg_files(input_folder)

    if not files:
        print("Tidak ada file .jpg ditemukan.")
        return

    print(f"Total file ditemukan: {len(files)}")

    for file_path in files:
        output_path = output_folder / file_path.name
        resize_image_to_square(
            image_path=file_path,
            output_path=output_path,
            size=size,
            keep_aspect_ratio=keep_aspect_ratio
        )
        print(f"{file_path.name} -> {output_path.name}")

    print("Resize selesai.")


if __name__ == "__main__":
    folder = "data/mkn_img/rtlh_int"

    resize_folder_jpg(
        folder_path=folder,
        output_folder=None,   # isi folder baru kalau mau simpan hasil terpisah
        size=(640, 640),
        keep_aspect_ratio=False
    )