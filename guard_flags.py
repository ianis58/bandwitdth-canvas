#!/usr/bin/env python3
"""Paint, and keep alive, the zones listed in guarded.txt (written by scan_hate.py and flag_ui.py,
or by hand). guarded.txt is re-read at every round, so zones added while this runs are painted on
the next round. Needs paint_flag.py in the same folder (and Pillow for text zones).

    python3 guard_flags.py                  # guard until Ctrl-C
    python3 guard_flags.py --once           # a single repair pass

guarded.txt: one zone per line, board coordinates (inclusive); everything after '#' is a comment.
    x0 y0 x1 y1 note                                    French flag (#002395 / #ffffff / #ed2939)
    x0 y0 x1 y1 flag note                               the same
    x0 y0 x1 y1 white note                              plain white
    x0 y0 x1 y1 text fg=ffffff ol=000000 bg=none TEXT   TEXT as big as fits in the box, centred;
                                                        "|" starts a new line
    x0 y0 x1 y1 chat note                               where chat_to_board.py shows the live chat
                                                        (painted by it, not by this script)
    x0 y0 x1 y1 off ...                                 any zone, kept in the list but not painted
  fg = text colour, ol = outline colour or none, bg = background colour or none (letters only).
Delete a line to stop guarding that zone. Where a zone lies on our own art (MERCI flag, heart)
our colours are restored instead, so the guards never fight.
Same options as the other scripts: --workers, --max-bits, --batch, --shard K/N, --interval, --name.
"""
import argparse, functools, os, subprocess, time
from typing import NamedTuple

from paint_flag import run

GUARDED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guarded.txt")
FLAG = ["002395", "ffffff", "ed2939"]  # official French blue, white, red
KINDS = ("flag", "white", "text", "chat")
OPT_KEYS = ("fg", "ol", "bg")
TEXT_DEFAULTS = {"fg": "ffffff", "ol": "000000", "bg": "none"}


class Zone(NamedTuple):
    rect: tuple       # (x0, y0, x1, y1), inclusive
    note: str         # the text of a text zone, a free note otherwise
    kind: str = "flag"
    opts: dict = {}
    enabled: bool = True  # False: listed ("off") but not painted

    def key(self):
        return (self.rect, self.note, self.kind, tuple(sorted(self.opts.items())), self.enabled)


def parse_line(line):
    parts = line.split("#")[0].split()
    try:
        x0, y0, x1, y1 = (int(v) for v in parts[:4])
    except ValueError:
        return None
    if x0 > x1 or y0 > y1:
        return None
    rest, kind, opts, enabled = parts[4:], "flag", {}, True
    if rest and rest[0].lower() == "off":
        enabled = False
        rest.pop(0)
    if rest and rest[0].lower() in KINDS:
        kind = rest.pop(0).lower()
    while rest and "=" in rest[0] and rest[0].split("=")[0].lower() in OPT_KEYS:
        k, v = rest.pop(0).split("=", 1)
        opts[k.lower()] = v.lower()
    return Zone((x0, y0, x1, y1), " ".join(rest), kind, opts, enabled)


def load_guarded(path=GUARDED):
    """[Zone] from guarded.txt; unreadable lines are skipped."""
    try:
        with open(path) as f:
            return [z for z in map(parse_line, f) if z]
    except FileNotFoundError:
        return []


def add_guarded(rect, note, path=GUARDED, kind="flag", opts=None):
    new = not os.path.exists(path)
    head = ("# zones kept alive by guard_flags.py: x0 y0 x1 y1 [flag|white|text] [fg= ol= bg=] note\n"
            "# (delete a line to stop guarding that zone)\n") if new else ""
    words = [str(v) for v in rect] + ([] if kind == "flag" else [kind])
    words += [f"{k}={v}" for k, v in (opts or {}).items()] + [note.replace("#", " ").strip() or "manual"]
    with open(path, "a") as f:  # one write per line, so a reader never sees half a line
        f.write(f"{head}{' '.join(words)}  # {time.strftime('%Y-%m-%d %H:%M')}\n")


def _rect_of(line):
    try:
        return tuple(int(v) for v in line.split("#")[0].split()[:4])
    except ValueError:
        return None


def _rewrite(path, change):
    """Apply change(lines) to the file: read just before (keeps lines another script appended)
    and replace it atomically (no reader ever sees half a file)."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(change(lines))
    os.replace(tmp, path)


def remove_guarded(rect, path=GUARDED):
    _rewrite(path, lambda lines: [l for l in lines if _rect_of(l) != tuple(rect)])


def move_guarded(old, new, path=GUARDED):
    """Give the zone at `old` the rectangle `new` (moved or resized), keeping the rest of its line."""
    def change(lines):
        out, done = [], False
        for line in lines:
            if not done and _rect_of(line) == tuple(old):
                body, sep, comment = line.partition("#")
                line = " ".join([str(v) for v in new] + body.split()[4:]) + (f"  #{comment}" if sep else "\n")
                done = True
            out.append(line)
        return out
    _rewrite(path, change)


def set_enabled(rect, enabled, path=GUARDED):
    """Switch painting of the zone at `rect` on or off (an "off" word after its coordinates)."""
    def change(lines):
        out = []
        for line in lines:
            if _rect_of(line) == tuple(rect):
                body, sep, comment = line.partition("#")
                words = body.split()
                rest = words[5:] if words[4:5] and words[4].lower() == "off" else words[4:]  # only the marker
                line = " ".join(words[:4] + ([] if enabled else ["off"]) + rest) + (f"  #{comment}" if sep else "\n")
            out.append(line)
        return out
    _rewrite(path, change)


def overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def flag_target(rect, ours, kind="flag"):
    """[(x, y, colour)]: a French flag (or plain white for kind "white") filling `rect`, or our
    own art's colours where it lies on it."""
    x0, y0, x1, y1 = rect
    w = x1 - x0 + 1
    return [(x, y, ours.get((x, y)) or ("ffffff" if kind == "white" else FLAG[min(2, 3 * (x - x0) // w)]))
            for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


@functools.lru_cache(maxsize=1)
def _font_path():
    for name in ("DejaVu Sans:bold", "Liberation Sans:bold", "Arial:bold"):
        try:
            path = subprocess.run(["fc-match", "-f", "%{file}", name], capture_output=True, text=True,
                                  timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            break
        if path and os.path.exists(path):
            return path
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"):
        if os.path.exists(path):
            return path
    return None


@functools.lru_cache(maxsize=512)
def text_masks(text, w, h, outline):
    """(letters, ring): sets of (dx, dy) for `text` ("|" = new line) in the largest bold size that
    fits w x h with its outline, centred, in crisp pixels (no anti-aliasing)."""
    from PIL import Image, ImageDraw, ImageFont
    lines, path = "\n".join(text.split("|")), _font_path()

    def font_for(size):
        if path:
            return ImageFont.truetype(path, size)
        try:
            return ImageFont.load_default(size)
        except TypeError:  # Pillow < 10.1: fixed-size default font
            return ImageFont.load_default()

    def layout(size):
        font, ow = font_for(size), (max(1, size // 16) if outline else 0)
        box = ImageDraw.Draw(Image.new("L", (1, 1))).multiline_textbbox(
            (0, 0), lines, font=font, stroke_width=ow, align="center", spacing=max(1, size // 8))
        return font, ow, size, box

    lo, hi, best = 4, max(4, 2 * h), None
    while lo <= hi:  # binary search for the biggest size that fits
        mid = (lo + hi) // 2
        font, ow, size, box = layout(mid)
        if box[2] - box[0] <= w and box[3] - box[1] <= h:
            best, lo = (font, ow, size, box), mid + 1
        else:
            hi = mid - 1
    font, ow, size, box = best or layout(4)  # box too small: smallest size, cropped
    ox, oy = (w - (box[2] - box[0])) // 2 - box[0], (h - (box[3] - box[1])) // 2 - box[1]

    # letters (255) and their outline (128) drawn in one call, so they line up exactly
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.fontmode = "1"
    d.multiline_text((ox, oy), lines, font=font, fill=255, align="center", spacing=max(1, size // 8),
                     stroke_width=ow, stroke_fill=128)
    px = list(im.getdata())
    letters = frozenset((i % w, i // w) for i, v in enumerate(px) if v > 191)
    ring = frozenset((i % w, i // w) for i, v in enumerate(px) if 64 < v <= 191)
    return letters, ring


def zone_target(zone, ours):
    """[(x, y, colour)] the zone should show; our own art keeps its colours."""
    if zone.kind == "chat":  # chat_to_board.py paints it
        return []
    if zone.kind != "text":
        return flag_target(zone.rect, ours, zone.kind)
    o = {**TEXT_DEFAULTS, **zone.opts}
    x0, y0, x1, y1 = zone.rect
    w, h = x1 - x0 + 1, y1 - y0 + 1
    letters, ring = text_masks(zone.note or " ", w, h, o["ol"] != "none")
    out = []
    for dy in range(h):
        for dx in range(w):
            if (dx, dy) in letters:
                c = o["fg"]
            elif (dx, dy) in ring:
                c = o["ol"]
            elif o["bg"] != "none":
                c = o["bg"]
            else:
                continue
            out.append((x0 + dx, y0 + dy, ours.get((x0 + dx, y0 + dy)) or c))
    return out


def our_art():
    """{(x, y): colour} of our own art, which zones must not paint over."""
    import paint_merci_flag, paint_heart
    ours = {(x, y): c for x, y, c in paint_merci_flag.merci_pixels()}
    ours.update({(x, y): c for x, y, c in paint_heart.heart_pixels(1001, 494)})
    return ours


def main():
    ap = argparse.ArgumentParser(description="Paint and keep alive the zones listed in guarded.txt.")
    ap.add_argument("--once", action="store_true", help="a single repair pass instead of guarding")
    ap.add_argument("--file", default=GUARDED, help="zone list to read (default guarded.txt next to this script)")
    ap.add_argument("--no-ours", action="store_true", help="paint zones even over our own art")
    ap.add_argument("--workers", type=int, default=min(256, 16 * (os.cpu_count() or 4)))
    ap.add_argument("--max-bits", type=int, default=12)
    ap.add_argument("--batch", type=int, default=3000)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--interval", type=float, default=10, help="seconds between rounds when all zones are intact")
    ap.add_argument("--name", default=os.environ.get("NAME", "tricolore"))
    args = ap.parse_args()
    args.guard, args.fix = not args.once, True
    ours = {} if args.no_ours else our_art()

    def pixels():  # re-read every round: scan_hate.py and flag_ui.py may have changed the list
        want = {}
        for zone in load_guarded(args.file):
            if not zone.enabled:
                continue
            for x, y, c in zone_target(zone, ours):
                want[(x, y)] = c
        return [(x, y, c) for (x, y), c in want.items()]

    run(pixels, args, f"guarding the zones in {args.file} ({len(load_guarded(args.file))} now)")


if __name__ == "__main__":
    main()
