#!/usr/bin/env python3
"""Find hater messages anywhere on the canvas and cover every hater word with a small French flag
exactly the size of the word, then keep those flags alive. Needs paint_flag.py alongside.

    sudo apt install tesseract-ocr          (macOS: brew install tesseract)
    pip install pytesseract scipy better-profanity numpy pillow requests

    python3 cover_hate.py --dry-run         # list new words found, save cover_preview.png
    python3 cover_hate.py --guard           # flag new words and repair all flags, until Ctrl-C

How it works:
  1. Text is found by shape: letter-like blobs of dark pixels (dark letters on a light
     background) and of light pixels (light letters with a dark outline, meme style). Noise
     dots and big areas are dropped, and tesseract reads what is left.
  2. A line is hateful if it contains profanity or slurs (better-profanity's word list) or a
     hostile pattern ("hate X", "X sucks", "death to X", "f* you", ... see HOSTILE). Identity
     words (gay, trans, queer, ...) are never treated as hate on their own.
  3. Inside a hateful line, every hater word (HATE_KEYS, matched letter by letter with one
     misread letter allowed for words of 4+ letters, plus words on the profanity list) gets a
     French flag (#002395 / #ffffff / #ed2939, vertical stripes) over its letters and their
     outline, no more. Benign look-alikes ("GOOD LUCK") are never touched: the line must
     first be hateful by the exact rules of step 2.
  4. Every flag is recorded in guarded.txt ("x0 y0 x1 y1 word", one per line). At startup the
     file is read back, and every round all recorded flags are repaired, so they stay alive
     even if the hater pastes over them again. Delete a line to stop guarding that flag.
  5. On our own art (MERCI flag, heart) a flag's pixels restore our colours instead, so this
     script never fights the guards.
"""
import argparse, os, re, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytesseract
from better_profanity import profanity
from PIL import Image, ImageDraw
from scipy import ndimage

import paint_flag as engine

GUARDED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guarded.txt")
FLAG = ["002395", "ffffff", "ed2939"]  # official French blue, white, red
# hater words to cover; when several match, exact and longer matches win
HATE_KEYS = ["DISGUSTING", "PATHETIC", "FUCKING", "GARBAGE", "FUCKED", "FUCKER", "HATERS", "LOSERS", "PHUCK", "SUCKS",
             "HATER", "HATES", "HATED", "DEATH", "LOSER", "TRASH", "FUCK", "SUCK", "HATE", "KILL", "STFU", "SCUM",
             "SHIT", "FCK", "FUK", "SUX", "DIE", "KYS"]
HOSTILE = [r"\bshit", r"\bgarbage\b", r"\bsux\b",
    r"\bhate[sd]?\b", r"\bhating\b", r"\bhaters?\b", r"\bsucks?\b", r"\bdeath\s+to\b", r"\bkill\b",
    r"\bdie\b", r"\bf+[\W_]*[u*]*[\W_]*c+[\W_]*k+", r"\bstfu\b", r"\bkys\b", r"\bgo\s+home\b",
    r"\bscum\b", r"\btrash\b", r"\bpathetic\b", r"\bloser?s?\b", r"\bdisgusting\b",
]
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "|": "l"})
SCALE = 3
# identity words are not hate on their own (and may well be pride messages): never cover them
IDENTITY = ["gay", "gays", "lesbian", "lesbians", "queer", "queers", "trans", "transgender", "bi", "bisexual",
            "lgbt", "lgbtq", "homosexual", "pride", "nonbinary"]
profanity.load_censor_words(whitelist_words=IDENTITY)


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


def norm(text):
    return re.sub(r"[^a-z\s*]", " ", text.lower().translate(LEET))


def hateful(text):
    t = norm(text)
    return profanity.contains_profanity(t) or any(re.search(p, t) for p in HOSTILE)


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


def overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def word_rect(board, ink, box, mode):
    """The word's letters plus their outline (light text) or 1 px (dark text): (x0, y0, x1, y1)."""
    H, W = board.shape[:2]
    x0, y0 = max(0, box[0] - 2), max(0, box[1] - 2)
    ys, xs = np.nonzero(ink[y0:box[3] + 3, x0:box[2] + 3])
    if len(xs) < 6:
        return None
    m = (2 if ys.max() - ys.min() >= 26 else 1) + 1 if mode == "light" else 1
    return (int(max(0, x0 + xs.min() - m)), int(max(0, y0 + ys.min() - m)),
            int(min(W - 1, x0 + xs.max() + m)), int(min(H - 1, y0 + ys.max() + m)))


def flag_target(rect, ours):
    """[(x, y, colour)]: a French flag filling `rect`, or our own art's colours where it lies on it."""
    x0, y0, x1, y1 = rect
    w = x1 - x0 + 1
    return [(x, y, ours.get((x, y)) or FLAG[min(2, 3 * (x - x0) // w)])
            for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


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


def load_guarded():
    rects = []
    if os.path.exists(GUARDED):
        for line in open(GUARDED):
            parts = line.split("#")[0].split()
            if len(parts) >= 4:
                rects.append((tuple(int(v) for v in parts[:4]), " ".join(parts[4:])))
    return rects


def add_guarded(rect, word):
    new = not os.path.exists(GUARDED)
    with open(GUARDED, "a") as f:
        if new:
            f.write("# small French flags covering hater words, kept alive by cover_hate.py\n"
                    "# x0 y0 x1 y1 word   (delete a line to stop guarding that flag)\n")
        f.write(f"{rect[0]} {rect[1]} {rect[2]} {rect[3]} {word}  # {time.strftime('%Y-%m-%d %H:%M')}\n")


def our_art():
    """{(x, y): colour} of our own art: flags there restore our colours (no fights with the guards)."""
    import paint_merci_flag, paint_heart
    ours = {(x, y): c for x, y, c in paint_merci_flag.merci_pixels()}
    ours.update({(x, y): c for x, y, c in paint_heart.heart_pixels(1001, 494)})
    return ours


def main():
    ap = argparse.ArgumentParser(description="Cover hater words with small French flags and keep them alive.")
    ap.add_argument("--dry-run", action="store_true", help="only list new words, save cover_preview.png")
    ap.add_argument("--guard", action="store_true", help="keep scanning and repairing until Ctrl-C")
    ap.add_argument("--interval", type=float, default=20)
    ap.add_argument("--min-conf", type=float, default=60, help="minimum OCR confidence (0-100) per word")
    ap.add_argument("--no-ours", action="store_true", help="paint flags even over our own art")
    ap.add_argument("--workers", type=int, default=min(256, 16 * (os.cpu_count() or 4)))
    ap.add_argument("--max-bits", type=int, default=12)
    ap.add_argument("--name", default=os.environ.get("NAME", "tricolore"))
    args = ap.parse_args()
    ours = {} if args.no_ours else our_art()
    guarded = load_guarded()

    cfg = engine.requests.get(f"{engine.BASE}/api/config", timeout=20).json()
    engine._init(cfg["pow_kib"], args.max_bits, args.name)
    print(f"hashing with {engine.backend()}, {args.workers} workers, {len(guarded)} flag(s) in {GUARDED}", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        while True:
            t = time.time()
            w, raw = engine.fetch_board()
            board = np.frombuffer(raw, dtype=np.uint8).reshape(-1, w, 3)
            new = []
            for word, rect in find_words(board, args.min_conf):
                if not any(overlap(rect, r) for r, _ in guarded + new):
                    new.append((rect, word))
            stamp = time.strftime("%H:%M:%S")
            for rect, word in new:
                print(f"[{stamp}] new hater word {word!r} at {rect} -> flag", flush=True)
                if not args.dry_run:
                    add_guarded(rect, word)
            if not args.dry_run:
                guarded += new
            flags = guarded + (new if args.dry_run else [])
            todo = [(x, y, c) for rect, _ in flags for x, y, c in flag_target(rect, ours)
                    if board[y, x].tobytes().hex() != c]
            print(f"[{stamp}] scan {time.time() - t:.0f}s: {len(new)} new, {len(flags)} flag(s), "
                  f"{len(todo)} px to paint", flush=True)
            if args.dry_run:
                prev = Image.fromarray(board.copy())
                for x, y, c in todo:
                    prev.putpixel((x, y), tuple(bytes.fromhex(c)))
                d = ImageDraw.Draw(prev)
                for rect, _ in new:
                    d.rectangle([rect[0] - 1, rect[1] - 1, rect[2] + 1, rect[3] + 1], outline=(255, 0, 255))
                prev.save("cover_preview.png")
                print("preview saved to cover_preview.png (magenta = new flags); guarded.txt not changed")
                return
            if todo:
                counts = engine.run_pass(pool, engine.prioritize(todo), "  painting flags")
                print(f"  painted {counts.get('ok', 0)}, deferred {counts.get('deferred', 0)}", flush=True)
            if not args.guard:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
