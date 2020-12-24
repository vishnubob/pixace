import os
import io
import requests
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
