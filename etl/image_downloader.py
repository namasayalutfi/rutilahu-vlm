import os, requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from urllib.parse import urlparse
from time import sleep

INPUT = r"C:\Users\Lutfi\Documents\Project\AITF\rutilahu-vlm\data\gambar_rutilahu.txt"
OUT_DIR = r"C:\Users\Lutfi\Documents\Project\AITF\rutilahu-vlm\data\raw_img"

MIN_SIZE_BYTES = 1024
RATE_LIMIT_SECONDS = 0.5
START_INDEX = 111

os.makedirs(OUT_DIR, exist_ok=True)

def last_index():
    files = os.listdir(OUT_DIR)
    nums = []

    for f in files:
        if f.lower().startswith("mkn_img_") and "." in f:
            try:
                n = int(f.split("_")[2].split(".")[0])
                nums.append(n)
            except:
                pass

    if nums:
        return max(nums)
    else:
        return START_INDEX - 1

def download(url, outpath):
    try:
        resp = requests.get(
            url,
            timeout=15,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0"},
            verify=False
        )

        if resp.status_code != 200:
            return False, f"status {resp.status_code}"

        content = resp.content

        if len(content) < MIN_SIZE_BYTES:
            return False, "too small"

        with open(outpath, "wb") as f:
            f.write(content)

        return True, None

    except Exception as e:
        return False, str(e)

def main():
    start = last_index() + 1

    with open(INPUT, "r", encoding="utf-8") as fh:
        urls = [l.strip() for l in fh if l.strip()]

    idx = start

    for u in urls:
        fname = f"mkn_img_{idx:05d}"

        path = urlparse(u).path
        ext = os.path.splitext(path)[1].lower() or ".jpg"

        outpath = os.path.join(OUT_DIR, fname + ext)

        if os.path.exists(outpath):
            print(f"Skip exists {outpath}")
            idx += 1
            continue

        ok, err = download(u, outpath)

        if ok:
            print(f"[{idx}] saved {outpath}")
            idx += 1
        else:
            print(f"[{idx}] failed {u} -> {err}")

        sleep(RATE_LIMIT_SECONDS)

if __name__ == "__main__":
    main()