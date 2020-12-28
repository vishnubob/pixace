import os
import io
import requests
import fnmatch
from PIL import Image
import tempfile

def download_image_from_web(url):
    resp = requests.get(url)
    if resp.status_code != 200:
        msg = "Server reported {resp.status_code} for {url}"
        raise ValueError(msg)
    fh = io.BytesIO(resp.content)
    fh.seek(0)
    img = Image.open(fh)
    img = img.convert("RGB")
    return img

def scan_for_files(path, patterns="*"):
    matches = {}
    match_img = lambda fn: any([fnmatch(fn.lower(), pat) for pat in patterns])

    for (root, dirs, files) in os.walk(path):
        root = Path(root)
        for fn in files:
            if not match_img(fn):
                continue
            pt = root.joinpath(fn)
            yield pt

def scan_for_images(path, patterns=None):
    patterns = patterns or ["*.jpg", "*.jpeg", "*.png"]
    return scan_for_files(path, patterns)
