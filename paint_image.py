#!/usr/bin/env python3
"""Paint a picture on the canvas, scaled to fill a rectangle, and optionally keep it alive.
Default: art/duo.png (the two guys with sunglasses) in place of the second French flag, on the
right of the board, (743,256)-(994,396). Needs paint_flag.py in the same folder.

    pip install argon2-cffi requests pillow numpy

    python3 paint_image.py --dry-run     # only save image_preview.png (the board with the picture on it)
    python3 paint_image.py               # paint the pixels that differ, once
    python3 paint_image.py --guard       # paint, then keep repairing until Ctrl-C

    python3 paint_image.py other.png --rect 10 10 109 59 --fit contain

--rect X0 Y0 X1 Y1   board rectangle, corners included (default 743 256 994 396, 252x141 px)
--fit cover          fill the rectangle, cropping what sticks out (default; here ~10 px per side)
--fit contain        whole picture inside the rectangle; the uncovered strips are left alone
--fit stretch        squash the picture to the rectangle exactly
--no-sharpen         keep the plain downscale (sharpening keeps hair and glasses crisp)
--zones-off          switch off the guarded.txt zones under the picture, so guard_flags.py
                     does not paint them back over it
Transparent pixels (alpha < 128) are not painted. Already-right pixels are always skipped.
Same options as the other scripts: --workers, --max-bits, --batch, --shard K/N, --interval, --name.
"""
import argparse, os

import numpy as np
from PIL import Image, ImageFilter

from paint_flag import fetch_board, run

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(HERE, "art", "duo.png")
DEFAULT_RECT = (743, 256, 994, 396)  # where the second French flag was, right of the board


def fitted(path, w, h, fit="cover", sharpen=True):
    """(rgb, alpha, ox, oy): the picture resized for a w x h box, as uint8 arrays, and its offset
    inside the box (non-zero only for --fit contain)."""
    im = Image.open(path).convert("RGBA")
    iw, ih = im.size
    if fit == "stretch":
        im, ox, oy = im.resize((w, h), Image.LANCZOS), 0, 0
    else:
        scale = (max if fit == "cover" else min)(w / iw, h / ih)
        sw, sh = max(1, round(iw * scale)), max(1, round(ih * scale))
        im = im.resize((sw, sh), Image.LANCZOS)
        if fit == "cover":  # centre crop
            left, top = (sw - w) // 2, (sh - h) // 2
            im, ox, oy = im.crop((left, top, left + w, top + h)), 0, 0
        else:
            ox, oy = (w - sw) // 2, (h - sh) // 2
    rgb, alpha = im.convert("RGB"), np.asarray(im.getchannel("A"))
    if sharpen:
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
    rgb = np.asarray(rgb).astype(int)
    # half-transparent pixels: blend over white, like the picture would look on a white page
    a = alpha[..., None] / 255.0
    rgb = (rgb * a + 255 * (1 - a)).round().astype(np.uint8)
    return rgb, alpha, ox, oy


def image_pixels(path, rect, fit="cover", sharpen=True):
    """[(x, y, 'rrggbb')] of the picture placed in rect (board coordinates, inclusive)."""
    x0, y0, x1, y1 = rect
    rgb, alpha, ox, oy = fitted(path, x1 - x0 + 1, y1 - y0 + 1, fit, sharpen)
    ph, pw = alpha.shape
    return [(x0 + ox + i, y0 + oy + j, "%02x%02x%02x" % tuple(rgb[j, i]))
            for j in range(ph) for i in range(pw) if alpha[j, i] >= 128]


def save_preview(pixels, rect, out):
    """The board around rect with the picture pasted on it, zoomed x3."""
    w, data = fetch_board()
    board = np.frombuffer(data, np.uint8)[:w * w * 3].reshape(w, w, 3).copy()
    for x, y, c in pixels:
        board[y, x] = [int(c[k:k + 2], 16) for k in (0, 2, 4)]
    x0, y0, x1, y1 = rect
    m = 20
    crop = board[max(0, y0 - m):min(w, y1 + m + 1), max(0, x0 - m):min(w, x1 + m + 1)]
    Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3), Image.NEAREST).save(out)
    return out


def check_zones(rect, switch_off):
    """Warn about guarded.txt zones under the picture (guard_flags.py / chat_to_board.py would
    paint over it); with switch_off, mark them "off" instead."""
    try:
        from guard_flags import GUARDED, load_guarded, overlap, set_enabled
    except ImportError:
        return
    hits = [z for z in load_guarded() if z.enabled and overlap(z.rect, rect)]
    for z in hits:
        if switch_off:
            set_enabled(z.rect, False)
            print(f"switched off zone {z.kind} {z.rect} {z.note!r} in {GUARDED}")
        else:
            print(f"WARNING: zone {z.kind} {z.rect} {z.note!r} in {GUARDED} lies under the picture: "
                  f"guard_flags.py / chat_to_board.py will paint over it. Switch it off in flag_ui.py, "
                  f"or rerun with --zones-off.")


def main():
    ap = argparse.ArgumentParser(description="Paint a picture scaled into a board rectangle.")
    ap.add_argument("image", nargs="?", default=DEFAULT_IMAGE)
    ap.add_argument("--rect", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=DEFAULT_RECT)
    ap.add_argument("--fit", choices=("cover", "contain", "stretch"), default="cover")
    ap.add_argument("--no-sharpen", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="save image_preview.png and exit, paint nothing")
    ap.add_argument("--zones-off", action="store_true", help="switch off guarded.txt zones under the picture")
    ap.add_argument("--guard", action="store_true", help="keep repairing the picture until Ctrl-C")
    ap.add_argument("--workers", type=int, default=min(256, 16 * (os.cpu_count() or 4)))
    ap.add_argument("--max-bits", type=int, default=12)
    ap.add_argument("--batch", type=int, default=3000)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--interval", type=float, default=20, help="guard: seconds between rounds when intact")
    ap.add_argument("--name", default=os.environ.get("NAME", "tricolore"))
    args = ap.parse_args()
    args.fix = True  # never repaint a pixel that is already right

    rect = tuple(args.rect)
    if not (rect[0] <= rect[2] and rect[1] <= rect[3]) or min(rect) < 0:
        ap.error("--rect needs X0 <= X1 and Y0 <= Y1, all >= 0")
    pixels = image_pixels(args.image, rect, args.fit, not args.no_sharpen)
    what = (f"{os.path.basename(args.image)} ({args.fit}) in ({rect[0]},{rect[1]})-({rect[2]},{rect[3]}), "
            f"{len(pixels)} px")
    if args.dry_run:
        out = save_preview(pixels, rect, os.path.join(HERE, "image_preview.png"))
        print(f"{what}; preview saved to {out}")
        return
    check_zones(rect, args.zones_off)
    run(pixels, args, what)


if __name__ == "__main__":
    main()
