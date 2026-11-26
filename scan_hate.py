#!/usr/bin/env python3
"""OCR scanner: every --interval seconds, read all text on the canvas, find hater words, and add a
flag rectangle for each new one to guarded.txt. It paints nothing: guard_flags.py, running next to
it, re-reads guarded.txt every round and paints the flags. Needs paint_flag.py and guard_flags.py.

    sudo apt install tesseract-ocr          (macOS: brew install tesseract)
    pip install pytesseract scipy better-profanity numpy pillow requests

    python3 scan_hate.py                    # scan every 20 s until Ctrl-C, appending to guarded.txt
    python3 scan_hate.py --dry-run          # one scan: list new words, save scan_preview.png

How it finds them:
  1. Text is found by shape: letter-like blobs of dark pixels (dark letters on a light
     background) and of light pixels (light letters with a dark outline, meme style). Noise
     dots and big areas are dropped, and tesseract reads what is left.
  2. A line is hateful if it contains profanity or slurs (better-profanity's word list) or a
     hostile pattern ("hate X", "X sucks", "death to X", "f* you", ... see HOSTILE). Identity
     words (gay, trans, queer, ...) are never treated as hate on their own.
  3. Inside a hateful line, every hater word (HATE_KEYS, matched letter by letter with one
     misread letter allowed for words of 4+ letters, plus words on the profanity list) gets a
     rectangle over its letters and their outline, no more. Benign look-alikes ("GOOD LUCK")
     are never touched: the line must first be hateful by the exact rules of step 2.
  4. A rectangle overlapping a flag already in guarded.txt is not added again.
"""
import argparse, os, re, time

import numpy as np
import pytesseract
from better_profanity import profanity
from PIL import Image, ImageDraw
from scipy import ndimage

import paint_flag as engine
from guard_flags import GUARDED, add_guarded, flag_target, load_guarded, overlap
from hate_words import HATE_KEYS, LEET, SCALE, hateful, norm

def luminance(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def letter_mask(sel, min_area=6, max_area=5000, min_h=4, max_h=60):
    """Keep connected blobs of `sel` shaped like letters (or a few letters run together)."""
    lab, n = ndimage.label(sel, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = int((lab[sl] == i).sum())
        if min_area <= area <= max_area and min_h <= h <= max_h and w <= 6 * h and area < 0.85 * h * w:
            keep[i] = True
    return keep[lab]


def ocr_lines(mask):
    """Tesseract over a binary mask (True = ink): [(line text, [(word, conf, box)])]."""
    img = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).resize(
        (mask.shape[1] * SCALE, mask.shape[0] * SCALE), Image.NEAREST)
    d = pytesseract.image_to_data(img, config="--psm 11", output_type=pytesseract.Output.DICT)
    lines = {}
    for i, word in enumerate(d["text"]):
        word, conf = word.strip(), float(d["conf"][i])
        if word and conf >= 0:
            x0, y0 = d["left"][i] // SCALE, d["top"][i] // SCALE
            box = (x0, y0, x0 + d["width"][i] // SCALE, y0 + d["height"][i] // SCALE)
            lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append((word, conf, box))
    return [(" ".join(w for w, _, _ in ws), ws) for ws in lines.values()]


def char_boxes(mask, box):
    """Tesseract character boxes for one line region: [(char, (x0, y0, x1, y1))] in board coords."""
    x0, y0 = max(0, box[0] - 4), max(0, box[1] - 4)
    crop = mask[y0:box[3] + 5, x0:box[2] + 5]
    img = Image.fromarray(np.where(crop, 0, 255).astype(np.uint8)).resize(
        (crop.shape[1] * SCALE, crop.shape[0] * SCALE), Image.NEAREST)
    out, H = [], crop.shape[0] * SCALE
    for row in pytesseract.image_to_boxes(img, config="--psm 7").splitlines():
        c, a, b, cc, dd, _ = row.split(" ")
        out.append((c, (x0 + int(a) // SCALE, y0 + (H - int(dd)) // SCALE,
                        x0 + int(cc) // SCALE, y0 + (H - int(b)) // SCALE)))
    return out


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def find_hate_words(chars):
    """[(key, box)] of hater words in a line's characters. Best matches win: exact before one
    misread letter (allowed for words of 4+ letters), longer words before shorter ones, and
    matches never overlap, so "I HATE FRANCE" gives HATE (not "HATE F" read as HATER)."""
    s = [c.upper().translate(str.maketrans("0135", "OIES")) for c, _ in chars]
    cands = []
    for key in HATE_KEYS:
        tol = 0 if len(key) <= 3 else 1
        for n in {len(key), len(key) - 1, len(key) + 1}:
            for i in range(0, len(s) - n + 1):
                d = edit_distance("".join(s[i:i + n]), key)
                if d <= tol:
                    cands.append((d, -len(key), i, i + n, key))
    used, out = set(), []
    for d, _, i, j, key in sorted(cands):
        if used.intersection(range(i, j)):
            continue
        used.update(range(i, j))
        boxes = [b for _, b in chars[i:j]]
        out.append((key, (min(b[0] for b in boxes), min(b[1] for b in boxes),
                          max(b[2] for b in boxes), max(b[3] for b in boxes))))
    return out


def word_rect(board, ink, box, mode):
    """The word's letters plus their outline (light text) or 1 px (dark text): (x0, y0, x1, y1).
    Whole letters are taken, even where tesseract's box cuts through one (e.g. a last "s")."""
    H, W = board.shape[:2]
    pad = max(4, box[3] - box[1])
    wx0, wy0 = max(0, box[0] - pad), max(0, box[1] - pad)
    win = ink[wy0:box[3] + pad + 1, wx0:box[2] + pad + 1]
    lab, n = ndimage.label(win, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    # blobs touching tesseract's box (1 px slack)
    bx0, by0, bx1, by1 = box[0] - wx0 - 1, box[1] - wy0 - 1, box[2] - wx0 + 1, box[3] - wy0 + 1
    touch = [i for i, o in enumerate(objs, 1) if o is not None and o[1].start <= bx1 and o[1].stop > bx0
             and o[0].start <= by1 and o[0].stop > by0]
    if not touch:
        return None
    # keep only letter-height blobs: stars or specks touching the word are much shorter
    tallest = max(objs[i - 1][0].stop - objs[i - 1][0].start for i in touch)
    letters = [i for i in touch if objs[i - 1][0].stop - objs[i - 1][0].start >= 0.6 * tallest]
    ys, xs = np.nonzero(np.isin(lab, letters))
    if len(xs) < 6:
        return None
    m = (2 if ys.max() - ys.min() >= 26 else 1) + 1 if mode == "light" else 1
    return (int(max(0, wx0 + xs.min() - m)), int(max(0, wy0 + ys.min() - m)),
            int(min(W - 1, wx0 + xs.max() + m)), int(min(H - 1, wy0 + ys.max() + m)))


def find_words(board, min_conf=60):
    """[(word, rect)] of hater words currently readable on the board."""
    lum = luminance(board.astype(np.float32))
    found = []
    for mode, sel in (("dark", lum < 100), ("light", lum > 200)):
        ink = letter_mask(sel)
        for text, words in ocr_lines(ink):
            good = [w for w, c, _ in words if c >= min_conf and len(re.sub(r"\W", "", w)) >= 2]
            if not good or not hateful(text):
                continue
            words = [(w, c, b) for w, c, b in words if b[3] - b[1] <= 80 and b[2] - b[0] <= 600]  # sanity
            if not words:
                continue
            line_box = (min(b[0] for _, _, b in words), min(b[1] for _, _, b in words),
                        max(b[2] for _, _, b in words), max(b[3] for _, _, b in words))
            targets = find_hate_words(char_boxes(ink, line_box))
            for w, c, b in words:  # slurs / profanity that are not in HATE_KEYS: the whole word
                if (profanity.contains_profanity(norm(w)) or hateful(w)) and not any(overlap(b, tb) for _, tb in targets):
                    targets.append((w.upper(), b))
            for key, box in targets:
                rect = word_rect(board, ink, box, mode)
                if rect:
                    found.append((key, rect))
    return found


def main():
    ap = argparse.ArgumentParser(description="Find hater words on the canvas and add flags for them to guarded.txt.")
    ap.add_argument("--interval", type=float, default=20, help="seconds between scans")
    ap.add_argument("--once", action="store_true", help="a single scan")
    ap.add_argument("--dry-run", action="store_true", help="one scan, list new words, save scan_preview.png")
    ap.add_argument("--file", default=GUARDED, help="flag list to append to (default guarded.txt)")
    ap.add_argument("--min-conf", type=float, default=60, help="minimum OCR confidence (0-100) per word")
    args = ap.parse_args()
    print(f"scanning every {args.interval:g} s, writing to {args.file}", flush=True)
    while True:
        t = time.time()
        try:
            w, raw = engine.fetch_board()
        except engine.requests.RequestException as e:
            print(f"board fetch failed ({e}), retrying", flush=True)
            time.sleep(args.interval)
            continue
        board = np.frombuffer(raw, dtype=np.uint8).reshape(-1, w, 3)
        known = [z.rect for z in load_guarded(args.file)]
        new = []
        for word, rect in find_words(board, args.min_conf):
            if not any(overlap(rect, r) for r in known + [n for n, _ in new]):
                new.append((rect, word))
        stamp = time.strftime("%H:%M:%S")
        for rect, word in new:
            print(f"[{stamp}] hater word {word!r} at {rect}" + ("" if args.dry_run else " -> added to guarded.txt"),
                  flush=True)
            if not args.dry_run:
                add_guarded(rect, word, args.file)
        print(f"[{stamp}] scan {time.time() - t:.0f}s: {len(new)} new, {len(known) + len(new)} flag(s) listed", flush=True)
        if args.dry_run:
            prev = Image.fromarray(board.copy())
            for rect, _ in new:
                for x, y, c in flag_target(rect, {}):
                    prev.putpixel((x, y), tuple(bytes.fromhex(c)))
                ImageDraw.Draw(prev).rectangle([rect[0] - 1, rect[1] - 1, rect[2] + 1, rect[3] + 1], outline=(255, 0, 255))
            prev.save("scan_preview.png")
            print("preview saved to scan_preview.png; guarded.txt not changed")
            return
        if args.once:
            return
        time.sleep(max(0.0, args.interval - (time.time() - t)))


if __name__ == "__main__":
    main()
