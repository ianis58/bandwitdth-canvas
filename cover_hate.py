#!/usr/bin/env python3
"""Find hater messages anywhere on the canvas and rewrite every hater word as "LOVE", drawn
with a red outline ("FUCK YOU FRANCE" -> "LOVE YOU FRANCE", "FRANCE SUCKS" -> "FRANCE LOVE").
Needs paint_flag.py in the same folder.

    sudo apt install tesseract-ocr          (macOS: brew install tesseract)
    pip install pytesseract scipy better-profanity numpy pillow requests

    python3 cover_hate.py --dry-run         # list what it would do, save cover_preview.png
    python3 cover_hate.py                   # act once
    python3 cover_hate.py --guard           # rescan and act again every --interval seconds

How it works:
  1. Text is found by shape: letter-like blobs of dark pixels (dark letters on a light
     background) and of light pixels (light letters with a dark outline, meme style). Noise
     dots and big areas are dropped, and tesseract reads what is left.
  2. A line is hateful if it contains profanity or slurs (better-profanity's word list) or a
     hostile pattern ("hate X", "X sucks", "death to X", "f* you", ... see HOSTILE).
  3. Inside a hateful line, every hater word (HATE_KEYS, matched letter by letter with one
     misread letter allowed for words of 4+ letters, plus any word on the profanity/slur list)
     is redrawn as --word (LOVE) at the same size, in the original letter colour with a
     --outline (red #ed2939) outline; where old letters disappear, the background around
     them is continued. Only pixels that change are painted.
     Benign look-alikes ("LUCK", "DUCK") are never touched: a line must first be hateful by
     the exact rules of step 2.
  4. On our own art (MERCI flag, heart) the word's box is restored to our colours instead, so
     this script never fights the guards. With --no-rewrite, hater words are covered instead.
"""
import argparse, os, re, subprocess, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytesseract
from better_profanity import profanity
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import paint_flag as engine

# hater words to rewrite, longest first so "FUCKING" wins over "FUCK"
HATE_KEYS = ["DISGUSTING", "PATHETIC", "FUCKING", "GARBAGE", "FUCKED", "FUCKER", "HATERS", "LOSERS", "PHUCK", "SUCKS",
             "HATER", "HATES", "HATED", "DEATH", "LOSER", "TRASH", "FUCK", "SUCK", "HATE", "KILL", "STFU", "SCUM",
             "SHIT", "FCK", "FUK", "SUX", "DIE", "KYS"]
HOSTILE = [r"\bshit", r"\bgarbage\b", r"\bsux\b",
    r"\bhate[sd]?\b", r"\bhating\b", r"\bhaters?\b", r"\bsucks?\b", r"\bdeath\s+to\b", r"\bkill\b",
    r"\bdie\b", r"\bf+[\W_]*[u*]*[\W_]*c+[\W_]*k+", r"\bstfu\b", r"\bkys\b", r"\bgo\s+home\b",
    r"\bscum\b", r"\btrash\b", r"\bpathetic\b", r"\bloser?s?\b", r"\bdisgusting\b",
]
FONTS = ["Impact", "Anton", "Oswald:bold", "Bebas Neue", "Liberation Sans Narrow:bold", "Liberation Sans:bold",
         "DejaVu Sans:bold"]
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "|": "l"})
SCALE = 3
# identity words are not hate on their own (and may well be pride messages): never rewrite them
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
    """Tesseract character boxes for one word/line region: [(char, (x0, y0, x1, y1))] in board coords."""
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


def font_path():
    for name in FONTS:
        try:
            path = subprocess.run(["fc-match", "-f", "%{file}", name], capture_output=True, text=True).stdout
        except OSError:
            break
        if path and name.split(":")[0].split()[0].lower() in os.path.basename(path).lower():
            return path
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/impact.ttf",
                 "/Library/Fonts/Impact.ttf", "/System/Library/Fonts/Supplemental/Impact.ttf"):
        if os.path.exists(path):
            return path
    return None


_FONT = {}


def render_word(word, w, h, stroke=7):
    """Boolean mask of `word` in bold capitals, stretched to exactly w x h pixels. `stroke`
    thickens the letters towards meme-style Impact capitals."""
    if "f" not in _FONT:
        path = font_path()
        _FONT["f"] = ImageFont.truetype(path, 160) if path else ImageFont.load_default()
    img = Image.new("L", (160 * (len(word) + 1), 260), 0)
    ImageDraw.Draw(img).text((20, 20), word, fill=255, font=_FONT["f"], stroke_width=stroke, stroke_fill=255)
    img = img.crop(img.getbbox()).resize((w, h), Image.LANCZOS)
    return np.array(img) > 110


def most_common(pixels):
    if len(pixels) == 0:
        return None
    return Counter(map(bytes, pixels)).most_common(1)[0][0].hex()


def rewrite_pixels(board, ink, box, new_word, mode, tol=60, outline_color=None, ours=None):
    """Pixels to paint to turn the word in `box` into `new_word`: same letter colour and size,
    outline `outline_color` (default: the original outline). [] if unsure. Pixels already
    within `tol` (sum of RGB differences) of their target are left alone; pixels of our own art
    (`ours`) get our colour back instead."""
    H, W = board.shape[:2]
    x0, y0 = max(0, box[0] - 2), max(0, box[1] - 2)
    ys, xs = np.nonzero(ink[y0:box[3] + 3, x0:box[2] + 3])
    if len(xs) < 10:
        return []
    lx0, ly0, lx1, ly1 = x0 + xs.min(), y0 + ys.min(), x0 + xs.max(), y0 + ys.max()
    t = (1 if ly1 - ly0 < 26 else 2) if (mode == "light" or outline_color) else 0  # outline, by letter height
    rx0, ry0 = max(0, lx0 - t - 2), max(0, ly0 - t - 2)
    rx1, ry1 = min(W, lx1 + t + 3), min(H, ly1 + t + 3)
    sub = board[ry0:ry1, rx0:rx1]
    old_fill = ink[ry0:ry1, rx0:rx1].copy()
    old_fill[:ly0 - ry0] = False
    old_fill[ly1 - ry0 + 1:] = False
    old_fill[:, :lx0 - rx0] = False
    old_fill[:, lx1 - rx0 + 1:] = False
    lum = luminance(sub.astype(np.float32))
    ring = ndimage.binary_dilation(old_fill, np.ones((3, 3)), iterations=t + 1) & ~old_fill
    old_outline = ring & (lum < 120) if mode == "light" else np.zeros_like(old_fill)
    fill = most_common(sub[old_fill])
    outline = outline_color or most_common(sub[old_outline]) or "000000"
    # old text = letters, outline and their soft shadow; the background is sampled outside it,
    # from pixels that are neither near-black nor white (the picture behind the text)
    near = ndimage.binary_dilation(old_fill, np.ones((3, 3)), iterations=t + 2)
    old_text = old_fill | old_outline | (near & ((lum < 200) if mode == "light" else (lum > 60)))
    sample = ~old_text & ~near & (lum > 70) & (lum < 215) if mode == "light" else ~old_text & ~near
    if sample.sum() < 5:
        sample = ~old_text

    # continue the background over old letter pixels that are no longer covered
    _, (iy, ix) = ndimage.distance_transform_edt(~sample, return_indices=True)
    bg = sub[iy, ix]
    rgb = lambda h: np.frombuffer(bytes.fromhex(h), np.uint8)
    mine = np.zeros(sub.shape[:2], bool)
    mine_rgb = np.zeros_like(sub)
    for (x, y), c in (ours or {}).items():
        if rx0 <= x < rx1 and ry0 <= y < ry1:
            mine[y - ry0, x - rx0] = True
            mine_rgb[y - ry0, x - rx0] = rgb(c)
    best = None
    # a few weights and 1 px shifts of the new word: keep the one that repaints fewest pixels
    for stroke in (0, 2, 4, 6, 8, 10):
        glyph = render_word(new_word, lx1 - lx0 + 1, ly1 - ly0 + 1, stroke)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                new = np.zeros_like(old_fill)
                yy, xx = ly0 - ry0 + dy, lx0 - rx0 + dx
                if yy < 0 or xx < 0 or yy + glyph.shape[0] > new.shape[0] or xx + glyph.shape[1] > new.shape[1]:
                    continue
                new[yy:yy + glyph.shape[0], xx:xx + glyph.shape[1]] = glyph
                new_outline = (ndimage.binary_dilation(new, np.ones((3, 3)), iterations=t) & ~new) if t \
                    else np.zeros_like(new)
                target = np.where(new[..., None], rgb(fill), np.where(new_outline[..., None], rgb(outline),
                                  np.where(old_text[..., None], bg, sub)))
                target = np.where(mine[..., None], mine_rgb, target)
                changed = np.abs(target.astype(np.int16) - sub.astype(np.int16)).sum(axis=2) > tol
                if best is None or changed.sum() < best[0]:
                    best = (int(changed.sum()), target, changed)
    if best is None:
        return []
    _, target, changed = best
    return [(int(rx0 + x), int(ry0 + y), target[y, x].tobytes().hex()) for y, x in zip(*np.nonzero(changed))]


def cover_pixels(board, ink, box, ours, mode):
    """Fill the word's tight box with the colour already most common in it (ours on our art)."""
    H, W = board.shape[:2]
    x0, y0 = max(0, box[0] - 2), max(0, box[1] - 2)
    ys, xs = np.nonzero(ink[y0:box[3] + 3, x0:box[2] + 3])
    if len(xs) == 0:
        return []
    m = 2 if mode == "light" else 1
    bx0, by0 = max(0, x0 + xs.min() - m), max(0, y0 + ys.min() - m)
    bx1, by1 = min(W, x0 + xs.max() + m + 1), min(H, y0 + ys.max() + m + 1)
    fill = most_common(board[by0:by1, bx0:bx1].reshape(-1, 3))
    out = []
    for y in range(by0, by1):
        for x in range(bx0, bx1):
            want = ours.get((x, y), fill)
            if board[y, x].tobytes().hex() != want:
                out.append((int(x), int(y), want))
    return out


def plan(board, ours, min_conf=60, rewrite=True, word="LOVE", outline="ed2939"):
    """[(action, text, box, pixels)] for every hater word found on the board."""
    lum = luminance(board.astype(np.float32))
    actions = []
    for mode, sel in (("dark", lum < 100), ("light", lum > 200)):
        ink = letter_mask(sel)
        for text, words in ocr_lines(ink):
            good = [w for w, c, _ in words if c >= min_conf and len(re.sub(r"\W", "", w)) >= 2]
            if not good or not hateful(text):
                continue
            words = [(w, c, b) for w, c, b in words if b[3] - b[1] <= 80 and b[2] - b[0] <= 600]  # sanity
            line_box = (min(b[0] for _, _, b in words), min(b[1] for _, _, b in words),
                        max(b[2] for _, _, b in words), max(b[3] for _, _, b in words))
            targets = []
            if rewrite:
                targets = find_hate_words(char_boxes(ink, line_box))
                for w, c, b in words:  # slurs / profanity that are not in HATE_KEYS: the whole word
                    if profanity.contains_profanity(norm(w)) and not any(overlap(b, tb) for _, tb in targets):
                        targets.append((w.upper(), b))
            for key, box in targets:
                px = rewrite_pixels(board, ink, box, word, mode, outline_color=outline, ours=ours)
                if px:
                    actions.append((f"{key} -> {word}", text, box, px))
            if not targets:  # --no-rewrite, or the hater words could not be pinned down: cover them
                for w, c, b in words:
                    if hateful(w):
                        actions.append(("cover", w, b, cover_pixels(board, ink, b, ours, mode)))
    return actions


def our_art():
    """{(x, y): colour} of our own art: covers there restore our colours (no fights with the guards)."""
    import paint_merci_flag, paint_heart
    ours = {(x, y): c for x, y, c in paint_merci_flag.merci_pixels()}
    ours.update({(x, y): c for x, y, c in paint_heart.heart_pixels(1001, 494)})
    return ours


def main():
    ap = argparse.ArgumentParser(description="Detect hateful messages on the canvas and defuse them.")
    ap.add_argument("--dry-run", action="store_true", help="only list actions and save cover_preview.png")
    ap.add_argument("--guard", action="store_true", help="rescan and act again until Ctrl-C")
    ap.add_argument("--interval", type=float, default=45)
    ap.add_argument("--min-conf", type=float, default=60, help="minimum OCR confidence (0-100) per word")
    ap.add_argument("--no-rewrite", action="store_true", help="cover hater words instead of rewriting them")
    ap.add_argument("--word", default="LOVE", help="what hater words become (default LOVE)")
    ap.add_argument("--outline", default="ed2939", help="outline colour of the new word, rrggbb (default red)")
    ap.add_argument("--no-ours", action="store_true", help="cover with the fill colour even on our own art")
    ap.add_argument("--workers", type=int, default=min(256, 16 * (os.cpu_count() or 4)))
    ap.add_argument("--max-bits", type=int, default=12)
    ap.add_argument("--name", default=os.environ.get("NAME", "tricolore"))
    args = ap.parse_args()
    ours = {} if args.no_ours else our_art()

    cfg = engine.requests.get(f"{engine.BASE}/api/config", timeout=20).json()
    engine._init(cfg["pow_kib"], args.max_bits, args.name)
    print(f"hashing with {engine.backend()}, {args.workers} workers, font {font_path()}", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        while True:
            t = time.time()
            w, raw = engine.fetch_board()
            board = np.frombuffer(raw, dtype=np.uint8).reshape(-1, w, 3)
            actions = plan(board, ours, args.min_conf, not args.no_rewrite, args.word.upper(), args.outline.lower())
            print(f"[{time.strftime('%H:%M:%S')}] scan {time.time() - t:.0f}s: {len(actions)} action(s)", flush=True)
            for kind, text, box, px in actions:
                print(f"  {kind:22} {text!r:28} box {box} -> {len(px)} px to paint", flush=True)
            if args.dry_run:
                prev = Image.fromarray(board.copy())
                for _, _, box, px in actions:
                    for x, y, c in px:
                        prev.putpixel((x, y), tuple(bytes.fromhex(c)))
                prev.save("cover_preview.png")
                print("preview saved to cover_preview.png")
                return
            todo = [p for *_, px in actions for p in px]
            if todo:
                counts = engine.run_pass(pool, todo, f"  painting {len(actions)} action(s)")
                print(f"  painted {counts.get('ok', 0)}, deferred {counts.get('deferred', 0)}", flush=True)
            if not args.guard:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
