#!/usr/bin/env python3
"""Find every copy of the hater's "FUCK YOU FRANCE" meme on the board, at any size, and redraw its
"FUCK" words as "LOVE" in the meme's own style. Needs paint_flag.py and cover_hate.py alongside.

    pip install opencv-python-headless pytesseract scipy better-profanity numpy pillow requests

    python3 meme_love.py --dry-run      # list copies found and what would change, save meme_preview.png
    python3 meme_love.py --guard        # rescan every --interval seconds (the meme keeps moving)

How it works:
  1. templates/hater_meme.png is matched against the board at sizes from --min-size to
     --max-size (normalised cross-correlation, OpenCV), so copies are found wherever they are
     pasted and however big, even partly painted over.
  2. Each copy's two FUCK words are at known places in the picture (FUCK_BOXES). A word is only
     rewritten if it still looks like the meme's FUCK there (not already LOVE, not covered).
  3. The word is redrawn as LOVE by cover_hate.rewrite_pixels: same letter height, white fill and
     dark outline as measured on the board, map background continued where old letters go.
"""
import argparse, os, time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image, ImageDraw

import cover_hate
import paint_flag as engine

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "hater_meme.png")
# FUCK words as fractions of the picture: (x0, y0, x1, y1)
FUCK_BOXES = [(0.0695, 0.0254, 0.3220, 0.1305), (0.2627, 0.8441, 0.5203, 0.9441)]


def find_copies(board, tmpl, min_size=80, max_size=700, threshold=0.5):
    """[(score, x, y, size)] of meme copies, best first, non-overlapping."""
    small = cv2.resize(board, (board.shape[1] // 2, board.shape[0] // 2), interpolation=cv2.INTER_AREA)
    found = []
    size = min_size
    while size <= min(max_size, board.shape[0]):
        t = cv2.resize(tmpl, (size // 2, size // 2), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED)
        for _ in range(6):  # a few peaks per size
            _, score, _, (x, y) = cv2.minMaxLoc(res)
            if score < threshold:
                break
            found.append((float(score), 2 * x, 2 * y, size))
            r = max(2, size // 8)
            res[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = -1
        size = int(size * 1.06) + 1
    keep = []
    for f in sorted(found, reverse=True):
        _, x, y, s = f
        if all(min(x + s, kx + ks) - max(x, kx) < 0.4 * min(s, ks) or
               min(y + s, ky + ks) - max(y, ky) < 0.4 * min(s, ks) for _, kx, ky, ks in keep):
            keep.append(f)
    return keep


def word_still_there(board, tmpl, box_frac, x, y, s):
    """Similarity (-1..1) between the board and the meme's own FUCK at that place."""
    bx0, by0, bx1, by1 = (int(round(v * s)) for v in box_frac)
    tw = cv2.resize(tmpl, (s, s), interpolation=cv2.INTER_AREA)[by0:by1 + 1, bx0:bx1 + 1]
    m = 3
    crop = board[max(0, y + by0 - m):y + by1 + 1 + m, max(0, x + bx0 - m):x + bx1 + 1 + m]
    if crop.shape[0] < tw.shape[0] or crop.shape[1] < tw.shape[1] or tw.size == 0:
        return -1.0, None
    res = cv2.matchTemplate(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.cvtColor(tw, cv2.COLOR_RGB2GRAY),
                            cv2.TM_CCOEFF_NORMED)
    return float(res.max()), (x + bx0, y + by0, x + bx1, y + by1)


def plan(board, tmpl, args):
    copies = find_copies(board, tmpl, args.min_size, args.max_size, args.threshold)
    ink = cover_hate.letter_mask(cover_hate.luminance(board.astype(np.float32)) > 200)
    actions = []
    for score, x, y, s in copies:
        for box_frac in FUCK_BOXES:
            sim, box = word_still_there(board, tmpl, box_frac, x, y, s)
            px = cover_hate.rewrite_pixels(board, ink, box, "LOVE", "light") if box and sim >= args.word_match else []
            actions.append(((x, y, s, score), box, sim, px))
    return copies, actions


def main():
    ap = argparse.ArgumentParser(description="Turn every FUCK in the hater's meme into LOVE.")
    ap.add_argument("--dry-run", action="store_true", help="only list copies and changes, save meme_preview.png")
    ap.add_argument("--guard", action="store_true", help="rescan and fix again until Ctrl-C")
    ap.add_argument("--interval", type=float, default=20)
    ap.add_argument("--min-size", type=int, default=80, help="smallest meme copy to look for, in pixels")
    ap.add_argument("--max-size", type=int, default=700)
    ap.add_argument("--threshold", type=float, default=0.5, help="match score (0-1) to accept a meme copy")
    ap.add_argument("--word-match", type=float, default=0.45, help="score (0-1) for 'FUCK is still there'")
    ap.add_argument("--workers", type=int, default=min(256, 16 * (os.cpu_count() or 4)))
    ap.add_argument("--max-bits", type=int, default=12)
    ap.add_argument("--name", default=os.environ.get("NAME", "tricolore"))
    args = ap.parse_args()
    tmpl = np.array(Image.open(TEMPLATE).convert("RGB"))

    cfg = engine.requests.get(f"{engine.BASE}/api/config", timeout=20).json()
    engine._init(cfg["pow_kib"], args.max_bits, args.name)
    print(f"hashing with {engine.backend()}, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        while True:
            t = time.time()
            w, raw = engine.fetch_board()
            board = np.frombuffer(raw, dtype=np.uint8).reshape(-1, w, 3)
            copies, actions = plan(board, tmpl, args)
            print(f"[{time.strftime('%H:%M:%S')}] scan {time.time() - t:.0f}s: {len(copies)} meme copy(ies)", flush=True)
            for (x, y, s, score), box, sim, px in actions:
                state = f"FUCK -> LOVE, {len(px)} px" if px else "nothing to do (already changed or covered)"
                print(f"  copy at ({x},{y}) size {s} match {score:.2f}: word {box} similarity {sim:.2f}: {state}",
                      flush=True)
            if args.dry_run:
                prev = Image.fromarray(board.copy())
                for *_, px in actions:
                    for x, y, c in px:
                        prev.putpixel((x, y), tuple(bytes.fromhex(c)))
                d = ImageDraw.Draw(prev)
                for _, x, y, s in copies:
                    d.rectangle([x, y, x + s, y + s], outline=(255, 0, 255))
                prev.save("meme_preview.png")
                print("preview saved to meme_preview.png (magenta = meme copies found)")
                return
            todo = [p for *_, px in actions for p in px]
            if todo:
                counts = engine.run_pass(pool, todo, f"  painting {sum(1 for a in actions if a[3])} word(s)")
                print(f"  painted {counts.get('ok', 0)}, deferred {counts.get('deferred', 0)}", flush=True)
            if not args.guard:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
