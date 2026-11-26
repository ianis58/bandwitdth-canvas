#!/usr/bin/env python3
"""Paint, and keep alive, the small French flags listed in guarded.txt (written by scan_hate.py,
or by hand). guarded.txt is re-read at every round, so flags added while this runs are painted
on the next round. Needs paint_flag.py in the same folder (no OCR needed).

    python3 guard_flags.py                  # guard until Ctrl-C
    python3 guard_flags.py --once           # a single repair pass

guarded.txt: one zone per line, "x0 y0 x1 y1 [flag|white] note" in board coordinates (inclusive);
everything after '#' is a comment, and a line without "flag" or "white" is a flag. Delete a line
to stop guarding that zone. A flag zone is the official French tricolour (#002395 / #ffffff /
#ed2939, vertical stripes) filling the rectangle, a white zone is plain #ffffff; where a zone lies
on our own art (MERCI flag, heart) our colours are restored instead, so the guards never fight.
Same options as the other scripts: --workers, --max-bits, --batch, --shard K/N, --interval, --name.
"""
import argparse, os, time

from paint_flag import run

GUARDED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guarded.txt")
FLAG = ["002395", "ffffff", "ed2939"]  # official French blue, white, red
KINDS = ("flag", "white")


def load_guarded(path=GUARDED):
    """[((x0, y0, x1, y1), note, kind)] from guarded.txt, kind "flag" or "white" (default flag);
    unreadable lines are skipped."""
    rects = []
    try:
        with open(path) as f:
            for line in f:
                parts = line.split("#")[0].split()
                try:
                    x0, y0, x1, y1 = (int(v) for v in parts[:4])
                except ValueError:
                    continue
                kind = parts[4].lower() if len(parts) > 4 and parts[4].lower() in KINDS else "flag"
                note = parts[5:] if len(parts) > 4 and parts[4].lower() in KINDS else parts[4:]
                if x0 <= x1 and y0 <= y1:
                    rects.append(((x0, y0, x1, y1), " ".join(note), kind))
    except FileNotFoundError:
        pass
    return rects


def add_guarded(rect, word, path=GUARDED, kind="flag"):
    new = not os.path.exists(path)
    with open(path, "a") as f:  # one write per line, so a reader never sees half a line
        head = ("# small French flags covering hater words, kept alive by guard_flags.py\n"
                "# x0 y0 x1 y1 [flag|white] note   (delete a line to stop guarding that zone)\n") if new else ""
        kind_word = "white " if kind == "white" else ""
        f.write(f"{head}{rect[0]} {rect[1]} {rect[2]} {rect[3]} {kind_word}{word}  # {time.strftime('%Y-%m-%d %H:%M')}\n")


def overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def flag_target(rect, ours, kind="flag"):
    """[(x, y, colour)]: a French flag (or plain white for kind "white") filling `rect`, or our
    own art's colours where it lies on it."""
    x0, y0, x1, y1 = rect
    w = x1 - x0 + 1
    return [(x, y, ours.get((x, y)) or ("ffffff" if kind == "white" else FLAG[min(2, 3 * (x - x0) // w)]))
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
        for rect, _, kind in load_guarded(args.file):
            for x, y, c in flag_target(rect, ours, kind):
                want[(x, y)] = c
        return [(x, y, c) for (x, y), c in want.items()]

    run(pixels, args, f"guarding the flags in {args.file} ({len(load_guarded(args.file))} now)")


if __name__ == "__main__":
    main()
