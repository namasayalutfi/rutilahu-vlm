import os
from pathlib import Path
from PIL import Image


def get_sorted_files(folder_path, extensions=None):
    folder = Path(folder_path)
    
    if extensions:
        files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    else:
        files = [f for f in folder.iterdir() if f.is_file()]
    
    return sorted(files)


def convert_to_jpg(folder_path, extensions=None, delete_original=False):
    """
    Convert semua image ke JPG.
    
    Parameters:
    - folder_path: path folder
    - extensions: file yang ingin di-convert
    - delete_original: hapus file lama setelah convert
    """
    files = get_sorted_files(folder_path, extensions)

    print(f"Convert {len(files)} file ke JPG...\n")

    for file_path in files:
        if file_path.suffix.lower() == ".jpg":
            continue  # skip kalau sudah jpg

        try:
            img = Image.open(file_path).convert("RGB")
            new_path = file_path.with_suffix(".jpg")

            img.save(new_path, "JPEG", quality=95)

            print(f"[CONVERT] {file_path.name} -> {new_path.name}")

            if delete_original:
                os.remove(file_path)

        except Exception as e:
            print(f"[ERROR] {file_path.name}: {e}")


def generate_new_name(index, prefix, padding, extension):
    number = str(index).zfill(padding)
    return f"{prefix}{number}{extension}"


def rename_files(folder_path, prefix, start_index=1, padding=4, extensions=None):
    files = get_sorted_files(folder_path, extensions)

    if not files:
        print("Tidak ada file ditemukan.")
        return

    print(f"\nTotal file ditemukan: {len(files)}\n")

    temp_paths = []

    # STEP 1: rename ke temporary name
    for i, file_path in enumerate(files):
        temp_name = f"temp_{i}{file_path.suffix}"
        temp_path = file_path.parent / temp_name

        os.rename(file_path, temp_path)
        temp_paths.append(temp_path)

    print("Step 1 selesai (rename ke temporary)\n")

    # STEP 2: rename ke final name
    for i, file_path in enumerate(temp_paths, start=start_index):
        new_name = generate_new_name(i, prefix, padding, file_path.suffix.lower())
        new_path = file_path.parent / new_name

        print(f"{file_path.name} -> {new_name}")
        os.rename(file_path, new_path)

    print("\nRename selesai (safe mode).")


# =========================
# 🔥 MAIN PIPELINE
# =========================
if __name__ == "__main__":
    folder = "data/mkn_img/multi_images/rtlh_int"

    # 1. Convert ke JPG dulu
    convert_to_jpg(
        folder_path=folder,
        extensions=[".png", ".jpeg", ".webp"],  # file yang akan di-convert
        delete_original=True  # True = hapus file lama
    )

    # 2. Rename hasilnya
    rename_files(
        folder_path=folder,
        prefix="",
        start_index=1,
        padding=1,
        extensions=[".jpg"]
    )