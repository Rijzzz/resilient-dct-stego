import os
import sys
import struct
import numpy as np
from PIL import Image

from reed_solomon import (
    rs_encode_block,
    rs_decode_block,
    interleave_bytes,
    deinterleave_bytes,
)
from dct_codec import (
    embed_dct,
    extract_dct,
    COEFFS,
)
from pipeline_sim import (
    whatsapp_pipeline,
    instagram_pipeline,
    lossy_cdn_recompress,
    compute_psnr,
    compute_ssim,
)

BENCHMARK_IMAGES = {
    "Lena": "lena.png",
    "Baboon": "baboon.png",
    "Fruits": "fruits.png",
}

BLOCK_DATA = 14
BLOCK_PARITY = 10
QIM_DELTA = 38.0
INTERLEAVE_COLS = 16

def build_packet(msg_bytes):
    hdr = struct.pack(">4sH", b"RS91", len(msg_bytes))
    raw = hdr + msg_bytes
    
    chunks = []
    for i in range(0, len(raw), BLOCK_DATA):
        chunk = raw[i:i + BLOCK_DATA]
        if len(chunk) < BLOCK_DATA:
            chunk = chunk + b"\x00" * (BLOCK_DATA - len(chunk))
        chunks.append(rs_encode_block(chunk, BLOCK_PARITY))

    encoded = b"".join(chunks)
    interleaved = interleave_bytes(encoded, num_cols=INTERLEAVE_COLS)
    bits = [(b >> s) & 1 for b in interleaved for s in (7, 6, 5, 4, 3, 2, 1, 0)]
    return bits, len(encoded), len(msg_bytes)

def decode_packet(bits, enc_len, orig_len):
    raw_bytes = bytearray()
    for i in range(0, len(bits), 8):
        val = 0
        for s in range(8):
            val = (val << 1) | bits[i + s]
        raw_bytes.append(val)

    deint = deinterleave_bytes(raw_bytes, enc_len, num_cols=INTERLEAVE_COLS)
    
    chunks = []
    chunk_len = BLOCK_DATA + BLOCK_PARITY
    for i in range(0, len(deint), chunk_len):
        blk = deint[i:i + chunk_len]
        dec = rs_decode_block(blk, BLOCK_PARITY)
        if dec is None:
            return None, False
        chunks.append(dec)

    full = b"".join(chunks)
    if len(full) < 6:
        return None, False
    
    magic, dec_len = struct.unpack(">4sH", full[:6])
    if magic != b"RS91":
        return None, False
        
    return full[6:6 + dec_len], True

def run_spatial_lsb_test(carrier, bits, pipeline_fn):
    arr = np.array(carrier.convert("RGB")).copy()
    flat = arr.flatten()
    nbits = min(len(bits), len(flat))
    for i in range(nbits):
        flat[i] = (flat[i] & 0xFE) | bits[i]
    stego = Image.fromarray(flat.reshape(arr.shape))
    
    proc = pipeline_fn(stego)
    rec_flat = np.array(proc.convert("RGB")).flatten()
    errs = sum((rec_flat[i] & 1) != bits[i] for i in range(nbits))
    return float(errs) / nbits

def run_full_experiment():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    msg = b"Meta covert stego verification: frequency domain QIM with Reed Solomon coding [2026]"
    
    bits, enc_len, orig_len = build_packet(msg)
    total_bits = len(bits)
    print(f"Test secret message: {msg.decode('latin1')}")
    print(f"Payload packet: {orig_len} message bytes, {enc_len} encoded bytes ({total_bits} bits)\n")

    pipelines = [
        ("WhatsApp (QF 70)", lambda img: whatsapp_pipeline(img, target_qf=70)),
        ("Instagram (QF 75)", lambda img: instagram_pipeline(img, target_qf=75)),
        ("Multi-hop 2x (QF 70)", lambda img: lossy_cdn_recompress(img, qf=70, passes=2)),
        ("Aggressive CDN (QF 60)", lambda img: lossy_cdn_recompress(img, qf=60, passes=1)),
    ]

    results = []

    for img_name, img_file in BENCHMARK_IMAGES.items():
        img_path = os.path.join(base_dir, img_file)
        if not os.path.exists(img_path):
            continue
            
        carrier = Image.open(img_path).convert("RGB")
        stego = embed_dct(carrier, bits, delta=QIM_DELTA)
        
        psnr_val = compute_psnr(carrier, stego)
        ssim_val = compute_ssim(carrier, stego)
        
        print(f"=== Benchmark: {img_name} ({carrier.size[0]}x{carrier.size[1]}) ===")
        print(f"Stego Quality: PSNR = {psnr_val:.2f} dB | SSIM = {ssim_val:.4f}")

        for pipe_name, pipe_fn in pipelines:
            lsb_ber = run_spatial_lsb_test(carrier, bits, pipe_fn)
            
            proc_img = pipe_fn(stego)
            extracted_bits = extract_dct(proc_img, total_bits, delta=QIM_DELTA)
            raw_errs = sum(a != b for a, b in zip(bits, extracted_bits))
            raw_ber = float(raw_errs) / total_bits
            
            recovered, ok = decode_packet(extracted_bits, enc_len, orig_len)
            exact = (recovered == msg) if ok else False
            
            row = {
                "image": img_name,
                "pipeline": pipe_name,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lsb_ber": lsb_ber * 100.0,
                "qim_ber": raw_ber * 100.0,
                "recovered": "YES (100%)" if exact else "NO",
            }
            results.append(row)
            
            print(f"  [{pipe_name}]")
            print(f"    Spatial LSB BER : {lsb_ber * 100.0:.2f}% (fails)")
            print(f"    Raw QIM BER     : {raw_ber * 100.0:.2f}%")
            print(f"    Post-ECC Status : {'RECOVERED 100%' if exact else 'FAILED'}")

        print()

    print("=" * 88)
    print("EXPERIMENT BENCHMARK SUMMARY")
    print("=" * 88)
    header_str = f"{'Carrier':<8} | {'Pipeline Channel':<22} | {'PSNR (dB)':<9} | {'SSIM':<6} | {'LSB BER':<8} | {'Raw BER':<8} | {'Recovery'}"
    print(header_str)
    print("-" * 88)
    for r in results:
        print(f"{r['image']:<8} | {r['pipeline']:<22} | {r['psnr']:<9.2f} | {r['ssim']:<6.4f} | {r['lsb_ber']:<7.2f}% | {r['qim_ber']:<7.2f}% | {r['recovered']}")
    print("=" * 88)

    return results

if __name__ == "__main__":
    run_full_experiment()
