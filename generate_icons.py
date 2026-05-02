"""Generate PWA icons (run once)."""
import os
import math

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False
    print("Pillow not found – writing minimal fallback PNG")

os.makedirs("static", exist_ok=True)


def make_icon_pillow(size: int, path: str):
    img  = Image.new("RGBA", (size, size), (10, 15, 30, 255))
    draw = ImageDraw.Draw(img)

    # Blue rounded background
    m = size // 8
    draw.rounded_rectangle([m, m, size - m, size - m],
                            radius=size // 5, fill=(37, 99, 235, 255))

    # White "K"
    fs = int(size * 0.52)
    font = None
    for candidate in ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf",
                       "arial.ttf", "DejaVuSans.ttf"]:
        try:
            font = ImageFont.truetype(candidate, fs)
            break
        except OSError:
            pass
    if font is None:
        font = ImageFont.load_default()

    bb = draw.textbbox((0, 0), "K", font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text(((size - tw) / 2 - bb[0], (size - th) / 2 - bb[1]),
              "K", fill=(255, 255, 255, 255), font=font)

    img.save(path, "PNG")
    print(f"  {path} ({size}x{size})")


def make_minimal_png(size: int, path: str):
    """Write a tiny valid PNG without Pillow (solid #0a0f1e square)."""
    import zlib, struct

    def chunk(tag, data):
        c = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", c)

    raw_row  = b"\x00" + bytes([10, 15, 30] * size)   # filter + RGB
    raw_data = raw_row * size
    compressed = zlib.compress(raw_data)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", compressed)
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
    print(f"  {path} ({size}x{size}, minimal fallback)")


print("Generating PWA icons...")
for sz, name in [(192, "static/icon-192.png"), (512, "static/icon-512.png"),
                 (180, "static/apple-touch-icon.png")]:
    if HAS_PILLOW:
        make_icon_pillow(sz, name)
    else:
        make_minimal_png(sz, name)

print("Done.")
