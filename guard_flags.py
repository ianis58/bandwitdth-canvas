#!/usr/bin/env python3
"""Paint, and keep alive, the small French flags listed in guarded.txt (written by scan_hate.py,
or by hand). guarded.txt is re-read at every round, so flags added while this runs are painted
on the next round. Needs paint_flag.py in the same folder (no OCR needed).

    python3 guard_flags.py                  # guard until Ctrl-C
    python3 guard_flags.py --once           # a single repair pass

guarded.txt: one flag per line, "x0 y0 x1 y1 word" in board coordinates (inclusive); everything
after '#' is a comment. Delete a line to stop guarding that flag. A flag is the official French
tricolour (#002395 / #ffffff / #ed2939, vertical stripes) filling the rectangle; where it lies on
our own art (MERCI flag, heart) our colours are restored instead, so the guards never fight.
Same options as the other scripts: --workers, --max-bits, --batch, --shard K/N, --interval, --name.
"""
import argparse, os, time

from paint_flag import run

GUARDED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guarded.txt")
FLAG = ["002395", "ffffff", "ed2939"]  # official French blue, white, red


def load_guarded(path=GUARDED):
    """[((x0, y0, x1, y1), word)] from guarded.txt; unreadable lines are skipped."""
    rects = []
    try:
        with open(path) as f:
            for line in f:
                parts = line.split("#")[0].split()
                try:
                    x0, y0, x1, y1 = (int(v) for v in parts[:4])
                except ValueError:
                    continue
                if x0 <= x1 and y0 <= y1:
                    rects.append(((x0, y0, x1, y1), " ".join(parts[4:])))
    except FileNotFoundError:
        pass
    return rects


def add_guarded(rect, word, path=GUARDED):
    new = not os.path.exists(path)
    with open(path, "a") as f:  # one write per line, so a reader never sees half a line
        head = ("# small French flags covering hater words, kept alive by guard_flags.py\n"
                "# x0 y0 x1 y1 word   (delete a line to stop guarding that flag)\n") if new else ""
        f.write(f"{head}{rect[0]} {rect[1]} {rect[2]} {rect[3]} {word}  # {time.strftime('%Y-%m-%d %H:%M')}\n")


def overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def flag_target(rect, ours):
    """[(x, y, colour)]: a French flag filling `rect`, or our own art's colours where it lies on it."""
    x0, y0, x1, y1 = rect
    w = x1 - x0 + 1
    return [(x, y, ours.get((x, y)) or FLAG[min(2, 3 * (x - x0) // w)])
            for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def our_art():
    """{(x, y): colour} of our own art, which flags must not paint over."""
    import paint_merci_flag, paint_heart
    ours = {(x, y): c for x, y, c in paint_merci_flag.merci_pixels()}
    ours.update({(x, y): c for x, y, c in paint_heart.heart_pixels(1001, 494)})
    return ours


def main():
    ap = argparse.ArgumentParser(description="Paint and keep alive the French flags listed in guarded.txt.")
    ap.add_argument("--once", action="store_true", help="a single repair pass instead of guarding")
    ap.add_argument("--file", default=GUARDED, help="flag list to read (default guarded.txt next to this script)")
    ap.add_argument("--no-ours", action="store_true", help="paint flags even over our own art")
    ap.add_argument("--workers", type=int, default=min(256, 16 * (os.cpu_count() or 4)))
    ap.add_argument("--max-bits", type=int, default=12)
    ap.add_argument("--batch", type=int, default=3000)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--interval", type=float, default=10, help="seconds between rounds when all flags are intact")
    ap.add_argument("--name", default=os.environ.get("NAME", "tricolore"))
    args = ap.parse_args()
    args.guard, args.fix = not args.once, True
    ours = {} if args.no_ours else our_art()

    def pixels():  # re-read every round: scan_hate.py may have added flags
        want = {}
        for rect, _ in load_guarded(args.file):
            for x, y, c in flag_target(rect, ours):
                want[(x, y)] = c
        return [(x, y, c) for (x, y), c in want.items()]

    run(pixels, args, f"guarding the flags in {args.file} ({len(load_guarded(args.file))} now)")


if __name__ == "__main__":
    main()
