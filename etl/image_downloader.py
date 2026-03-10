import os, requests
import urllib3
from urllib.parse import urlparse
from time import sleep
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ImageDownloader:
    def __init__(self, input_txt, out_dir, min_size=1024, rate_limit=0.5, start_index=111, verify=False):
        self.input_txt = input_txt
        self.out_dir = out_dir
        self.min_size = min_size
        self.rate_limit = rate_limit
        self.start_index = start_index
        self.verify = verify
        os.makedirs(self.out_dir, exist_ok=True)

    def last_index(self):
        files = os.listdir(self.out_dir)
        nums = []
        for f in files:
            if f.lower().startswith("mkn_img_") and "." in f:
                try:
                    n = int(f.split("_")[2].split(".")[0])
                    nums.append(n)
                except Exception:
                    pass
        return max(nums) if nums else self.start_index - 1

    def download(self, url, outpath, timeout=15):
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                stream=True,
                headers={"User-Agent": "Mozilla/5.0"},
                verify=self.verify,
            )
            if resp.status_code != 200:
                return False, f"status {resp.status_code}"
            content = resp.content
            if len(content) < self.min_size:
                return False, "too small"
            with open(outpath, "wb") as fh:
                fh.write(content)
            return True, None
        except Exception as e:
            return False, str(e)

    def run(self, limit=None):
        if not os.path.exists(self.input_txt):
            raise FileNotFoundError(f"Input URL file not found: {self.input_txt}")
        with open(self.input_txt, "r", encoding="utf-8") as fh:
            urls = [l.strip() for l in fh if l.strip()]

        start = self.last_index() + 1
        idx = start
        saved = 0
        for u in urls:
            if limit is not None and saved >= limit:
                break
            fname = f"mkn_img_{idx:05d}"
            path = urlparse(u).path
            ext = os.path.splitext(path)[1].lower() or ".jpg"
            if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]:
                ext = ".jpg"
            outpath = os.path.join(self.out_dir, fname + ext)
            if os.path.exists(outpath):
                print(f"Skip exists {outpath}")
                idx += 1
                continue
            ok, err = self.download(u, outpath)
            if ok:
                print(f"[{idx}] saved {outpath}")
                saved += 1
            else:
                print(f"[{idx}] failed {u} -> {err}")
            idx += 1
            sleep(self.rate_limit)
        print(f"Downloader finished. saved: {saved}")
        return saved