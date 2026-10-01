import io
import math
import numpy as np
from PIL import Image

def whatsapp_pipeline(img, target_qf=70):
    w, h = img.size
    if max(w, h) > 1600:
        scale = 1600.0 / float(max(w, h))
        nw = (int(round(w * scale)) // 8) * 8
        nh = (int(round(h * scale)) // 8) * 8
        work_img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    else:
        work_img = img.copy()

    if work_img.mode != "RGB":
        work_img = work_img.convert("RGB")

    buf = io.BytesIO()
    work_img.save(buf, format="JPEG", quality=target_qf, subsampling=1)
    buf.seek(0)
    return Image.open(buf)

def instagram_pipeline(img, target_qf=75):
    w, h = img.size
    if w > 1080:
        scale = 1080.0 / float(w)
        nw = 1080
        nh = (int(round(h * scale)) // 8) * 8
        work_img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    else:
        work_img = img.copy()

    if work_img.mode != "RGB":
        work_img = work_img.convert("RGB")

    buf = io.BytesIO()
    work_img.save(buf, format="JPEG", quality=target_qf, subsampling=1)
    buf.seek(0)
    return Image.open(buf)

def lossy_cdn_recompress(img, qf=60, passes=1):
    curr = img.copy()
    for _ in range(passes):
        buf = io.BytesIO()
        curr.save(buf, format="JPEG", quality=qf, subsampling=1)
        buf.seek(0)
        curr = Image.open(buf)
    return curr

def compute_psnr(img1, img2):
    a1 = np.array(img1.convert("RGB"), dtype=np.float64)
    a2 = np.array(img2.convert("RGB"), dtype=np.float64)
    mse = np.mean((a1 - a2) ** 2)
    if mse == 0:
        return 99.99
    return 10.0 * math.log10((255.0 ** 2) / mse)

def compute_ssim(img1, img2):
    y1 = np.array(img1.convert("L"), dtype=np.float64)
    y2 = np.array(img2.convert("L"), dtype=np.float64)
    
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu1, mu2 = np.mean(y1), np.mean(y2)
    var1, var2 = np.var(y1), np.var(y2)
    cov = np.mean((y1 - mu1) * (y2 - mu2))

    num = (2.0 * mu1 * mu2 + c1) * (2.0 * cov + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)
    return float(num / den)
