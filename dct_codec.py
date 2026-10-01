import math
import numpy as np
from PIL import Image

N = 8
T = np.zeros((N, N), dtype=np.float64)
for i in range(N):
    for j in range(N):
        c = 1.0 / math.sqrt(N) if i == 0 else math.sqrt(2.0 / N)
        T[i, j] = c * math.cos((2 * j + 1) * i * math.pi / (2 * N))

COEFFS = [(1, 2), (2, 1), (2, 2), (3, 1), (1, 3)]

def embed_dct(carrier_img, bits, delta=38.0):
    if carrier_img.mode != "RGB":
        carrier_img = carrier_img.convert("RGB")
    
    img_ycbcr = carrier_img.convert("YCbCr")
    y, cb, cr = img_ycbcr.split()
    y_arr = np.array(y, dtype=np.float64)
    h, w = y_arr.shape

    blocks_h = h // 8
    blocks_w = w // 8
    max_bits = blocks_h * blocks_w * len(COEFFS)
    
    if len(bits) > max_bits:
        raise ValueError(f"Payload too large: {len(bits)} bits exceeds {max_bits} capacity")

    bit_stream = list(bits) + [0] * (max_bits - len(bits))

    bit_idx = 0
    stego_y = y_arr.copy()
    
    for by in range(0, h - (h % 8), 8):
        for bx in range(0, w - (w % 8), 8):
            blk = stego_y[by:by+8, bx:bx+8] - 128.0
            dct = T @ blk @ T.T
            for u, v in COEFFS:
                b = bit_stream[bit_idx]
                dither = -delta / 4.0 if b == 0 else delta / 4.0
                dct[u, v] = np.round((dct[u, v] - dither) / delta) * delta + dither
                bit_idx += 1
            stego_y[by:by+8, bx:bx+8] = np.clip(T.T @ dct @ T + 128.0, 0.0, 255.0)

    for _ in range(2):
        q_y = np.round(np.clip(stego_y, 0.0, 255.0))
        bit_idx = 0
        fixes = 0
        for by in range(0, h - (h % 8), 8):
            for bx in range(0, w - (w % 8), 8):
                blk = q_y[by:by+8, bx:bx+8] - 128.0
                dct = T @ blk @ T.T
                adj = False
                for u, v in COEFFS:
                    target_b = bit_stream[bit_idx]
                    val = dct[u, v]
                    d0, d1 = -delta / 4.0, delta / 4.0
                    q0 = np.round((val - d0) / delta) * delta + d0
                    q1 = np.round((val - d1) / delta) * delta + d1
                    curr_b = 0 if abs(val - q0) < abs(val - q1) else 1
                    if curr_b != target_b:
                        t_dither = d0 if target_b == 0 else d1
                        dct[u, v] = np.round((val - t_dither) / delta) * delta + t_dither
                        adj = True
                        fixes += 1
                    bit_idx += 1
                if adj:
                    stego_y[by:by+8, bx:bx+8] = np.clip(T.T @ dct @ T + 128.0, 0.0, 255.0)
        if fixes == 0:
            break

    final_y = Image.fromarray(np.round(np.clip(stego_y, 0.0, 255.0)).astype(np.uint8), mode="L")
    return Image.merge("YCbCr", (final_y, cb, cr)).convert("RGB")

def extract_dct(img, nbits, delta=38.0):
    img_ycbcr = img.convert("YCbCr")
    y_arr = np.array(img_ycbcr.split()[0], dtype=np.float64)
    h, w = y_arr.shape

    out = []
    d0, d1 = -delta / 4.0, delta / 4.0
    for by in range(0, h - (h % 8), 8):
        for bx in range(0, w - (w % 8), 8):
            blk = y_arr[by:by+8, bx:bx+8] - 128.0
            dct = T @ blk @ T.T
            for u, v in COEFFS:
                val = dct[u, v]
                q0 = np.round((val - d0) / delta) * delta + d0
                q1 = np.round((val - d1) / delta) * delta + d1
                out.append(0 if abs(val - q0) < abs(val - q1) else 1)
                if len(out) == nbits:
                    return out
    return out
