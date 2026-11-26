#!/usr/bin/env python3
"""Find hateful messages anywhere on the canvas and paint over them as cheaply as possible.
Needs paint_flag.py in the same folder.

    sudo apt install tesseract-ocr          (macOS: brew install tesseract)
    pip install pytesseract scipy better-profanity numpy pillow requests

    python3 cover_hate.py --dry-run         # list what it would cover, save cover_preview.png
    python3 cover_hate.py                   # cover what it finds once
    python3 cover_hate.py --guard           # rescan and cover again every --interval seconds

How it works:
  1. Text is found by shape, not colour: connected blobs of dark pixels (dark letters on a light
     background) and of light pixels (light letters with a dark outline, meme style) that are
     letter-sized are kept; noise dots and big areas are dropped. Tesseract reads what is left.
  2. A line is hateful if it contains profanity or slurs (better-profanity's word list) or a
     hostile pattern such as "hate X", "X sucks", "death to X", "f* you" (HOSTILE below).
     Lines read with low confidence are ignored, so stray art is left alone.
  3. Each hateful line's box (shrunk to its letters) is filled with one colour already in it,
     whichever needs the fewest pixels repainted: e.g. white for white letters with black
     outlines (only the outlines and the gaps get painted), the background colour for dark
     letters on a plain background. Where the box lies on our own art (croissant flag, heart),
     our colours are used instead, so this script and the guards never fight each other.
"""
import argparse, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytesseract
from better_profanity import profanity
from PIL import Image, ImageDraw
from scipy import ndimage

import paint_flag as engine

HOSTILE = [
    r"\bhate[sd]?\b", r"\bhating\b", r"\bhaters?\b", r"\bsucks?\b", r"\bdeath\s+to\b", r"\bkill\b",
    r"\bdie\b", r"\bf+[\W_]*[u*]+[\W_]*c+[\W_]*k+", r"\bstfu\b", r"\bkys\b", r"\bgo\s+home\b",
    r"\bscum\b", r"\btrash\b", r"\bpathetic\b", r"\bloser?s?\b", r"\bdisgusting\b",
]
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "|": "l"})
profanity.load_censor_words()


def luminance(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def letter_mask(sel, min_area=6, max_area=900, min_h=4, max_h=60):
    """Keep connected blobs of `sel` that are letter-sized."""
    lab, n = ndimage.label(sel, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = int((lab[sl] == i).sum())
        if min_area <= area <= max_area and min_h <= h <= max_h and w <= 3 * max_h and area < 0.85 * h * w:
            keep[i] = True
    return keep[lab]


def ocr_lines(mask, scale=3, min_conf=60):
    """Tesseract over a binary mask (True = ink). Returns [(text, conf, (x0, y0, x1, y1))]."""
    img = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).resize(
        (mask.shape[1] * scale, mask.shape[0] * scale), Image.NEAREST)
    d = pytesseract.image_to_data(img, config="--psm 11", output_type=pytesseract.Output.DICT)
    lines = {}
    for i, word in enumerate(d["text"]):
        word, conf = word.strip(), float(d["conf"][i])
        if not word or conf < 0:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        x0, y0 = d["left"][i] // scale, d["top"][i] // scale
        x1, y1 = x0 + d["width"][i] // scale, y0 + d["height"][i] // scale
        lines.setdefault(key, []).append((word, conf, (x0, y0, x1, y1)))
    out = []
    for words in lines.values():
        text = " ".join(w for w, _, _ in words)
        conf = min(c for _, c, _ in words)
        box = (min(b[0] for _, _, b in words), min(b[1] for _, _, b in words),
               max(b[2] for _, _, b in words), max(b[3] for _, _, b in words))
        out.append((text, conf, box, words))
    return out


def hateful(text):
    t = text.lower().translate(LEET)
    t = re.sub(r"[^a-z\s*]", " ", t)
    return profanity.contains_profanity(t) or any(re.search(p, t) for p in HOSTILE)


def find(board, min_conf=60):
    """Hateful lines on the board: [(text, (x0, y0, x1, y1))]."""
    lum = luminance(board.astype(np.float32))
    found = []
    for mode, sel in (("dark", lum < 100), ("light", lum > 200)):
        ink = letter_mask(sel)
        for text, _, box, words in ocr_lines(ink):
            good = [w for w, c, _ in words if c >= min_conf and len(re.sub(r"\W", "", w)) >= 2]
            if good and hateful(text) and hateful(" ".join(good)):
                found.append((text, tight(ink, box, outline=2 if mode == "light" else 1), mode))
    return found


def tight(ink, box, outline):
    """Shrink an OCR box to the letter pixels inside it, plus room for their outline."""
    x0, y0 = max(0, box[0] - 3), max(0, box[1] - 3)
    ys, xs = np.nonzero(ink[y0:box[3] + 4, x0:box[2] + 4])
    if len(xs) == 0:
        return box
    return (int(x0 + xs.min() - outline), int(y0 + ys.min() - outline),
            int(x0 + xs.max() + outline), int(y0 + ys.max() + outline))


def our_art():
    """{(x, y): colour} of our own art: inside it, a cover restores our colours instead, so the
    flag and heart guards and this script never fight each other."""
    import paint_croissant_flag, paint_heart
    ours = {(x, y): c for x, y, c in paint_croissant_flag.croissant_flag_pixels(3, 57, 506)}
    ours.update({(x, y): c for x, y, c in paint_heart.heart_pixels(1001, 494)})
    return ours


def cheapest_cover(board, box, ours, margin=0):
    """Fill colour for the box needing the fewest repaints, our art's colours where it lies on
    our art. Returns [(x, y, colour)] to paint, the fill colour, the box."""
    h, w = board.shape[:2]
    x0, y0 = max(0, box[0] - margin), max(0, box[1] - margin)
    x1, y1 = min(w, box[2] + margin + 1), min(h, box[3] + margin + 1)
    reg = board[y0:y1, x0:x1].reshape(-1, 3)
    cols, counts = np.unique(reg, axis=0, return_counts=True)
    best = cols[np.argmax(counts)]                  # the most common colour = fewest repaints
    hexcol = "%02x%02x%02x" % tuple(int(v) for v in best)
    pixels = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            want = ours.get((x, y), hexcol)
            if board[y, x].tobytes().hex() != want:
                pixels.append((x, y, want))
    return pixels, hexcol, (x0, y0, x1, y1)


def scan(min_conf, ours):
    w, raw = engine.fetch_board()
    board = np.frombuffer(raw, dtype=np.uint8).reshape(-1, w, 3)
    covers = []
    for text, box, mode in find(board, min_conf):
        pixels, col, rect = cheapest_cover(board, box, ours)
        covers.append((text, mode, rect, col, pixels))
    return board, covers


def main():
    ap = argparse.ArgumentParser(description="Detect hateful messages on the canvas and paint over them.")
    ap.add_argument("--dry-run", action="store_true", help="only list detections and save cover_preview.png")
    ap.add_argument("--guard", action="store_true", help="rescan and cover again until Ctrl-C")
    ap.add_argument("--interval", type=float, default=45)
    ap.add_argument("--min-conf", type=float, default=60, help="minimum OCR confidence (0-100) per word")
    ap.add_argument("--workers", type=int, default=min(256, 16 * (os.cpu_count() or 4)))
    ap.add_argument("--max-bits", type=int, default=12)
    ap.add_argument("--name", default=os.environ.get("NAME", "tricolore"))
    ap.add_argument("--no-ours", action="store_true", help="cover with the fill colour even on our own art")
    args = ap.parse_args()
    ours = {} if args.no_ours else our_art()

    cfg = engine.requests.get(f"{engine.BASE}/api/config", timeout=20).json()
    engine._init(cfg["pow_kib"], args.max_bits, args.name)
    print(f"hashing with {engine.backend()}, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        while True:
            t = time.time()
            board, covers = scan(args.min_conf, ours)
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] scan {time.time() - t:.0f}s: {len(covers)} hateful line(s)", flush=True)
            for text, mode, rect, col, pixels in covers:
                print(f"  {text!r:40} {mode:5} box {rect} -> fill #{col}, {len(pixels)} px to paint", flush=True)
            if args.dry_run:
                prev = Image.fromarray(board.copy())
                for _, _, rect, col, pixels in covers:
                    for x, y, c in pixels:
                        prev.putpixel((x, y), tuple(bytes.fromhex(c)))
                    ImageDraw.Draw(prev).rectangle([rect[0] - 1, rect[1] - 1, rect[2], rect[3]], outline=(255, 0, 255))
                prev.save("cover_preview.png")
                print("preview saved to cover_preview.png (magenta = covered boxes)")
                return
            todo = [p for *_, pixels in covers for p in pixels]
            if todo:
                counts = engine.run_pass(pool, todo, f"  covering {len(covers)} line(s)")
                print(f"  painted {counts.get('ok', 0)}, deferred {counts.get('deferred', 0)}", flush=True)
            if not args.guard:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
