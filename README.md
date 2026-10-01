# resilient dct steganography

Hiding secret payloads in JPEG discrete cosine transform coefficients so they survive WhatsApp, Instagram, and lossy social media recompression.

## Problem

Almost all steganography algorithms work in the spatial domain (like LSB replacement or LSB matching). When you upload an image to WhatsApp or Instagram, the platforms downscale, convert colors to YCbCr with 4:2:0 subsampling, and quantize DCT coefficients at quality 70 to 75. This completely wipes out spatial LSBs, causing bit error rates around 50%.

## Solution

This project implements:

* 8x8 block 2D DCT on the luminance Y channel
* Mid frequency coefficient band selection (1,2), (2,1), (2,2), (3,1), (1,3)
* Quantization Index Modulation (QIM) with step delta tuned to JPEG quantizer bins
* Spatial drift feedback loop to fix RGB integer rounding errors
* Reed Solomon RS(24, 14) error correction over GF(2^8) with Berlekamp Massey decoding
* 16 way block interleaving to spread burst errors across codewords

## Benchmark Images

* lena.png (512x512)
* baboon.png (512x512)
* fruits.png (512x480)

## How to Run

Requirements: Python 3.13, Pillow, and NumPy.

```bash
python experiment.py
```

