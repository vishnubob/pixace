import os
import io
import tempfile
from fnmatch import fnmatch
from pathlib import Path
import requests
from PIL import Image

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
    patterns = patterns or ["*.jpg", "*.jpeg", "*.png", "*.gif"]
    return scan_for_files(path, patterns)
