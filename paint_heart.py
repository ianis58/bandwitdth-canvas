#!/usr/bin/env python3
"""Paint a small blue/white/red heart (20x20) in the top-right of the Polish flag painted over
the original French flag, and guard it. Needs paint_flag.py in the same folder.

    pip install argon2-cffi requests

    python3 paint_heart.py            # paint it, then keep repairing it until Ctrl-C
    python3 paint_heart.py --once     # paint it once and exit

Options: --x 1001 --y 494 (top-left corner), --max-bits 12, --interval 20, --workers N, --name.
Don't run it together with paint_flag.py --guard: the heart sits inside that flag's area.
"""
import argparse, os

from paint_flag import run

COLORS = {
    "o": "14213d",  # outline, dark navy (the heart sits on white)
    "b": "0055a4",  # blue
    "w": "ffffff",  # white
    "r": "ef4135",  # red
}
HEART = [
    "...ooooo....ooooo...",
    "..obbbbwoooowrrrro..",
    ".obbbbbwwwwwwrrrrro.",
    ".obbbbbwwwwwwrrrrro.",
    "obbbbbbwwwwwwrrrrrro",
    "obbbbbbwwwwwwrrrrrro",
    "obbbbbbwwwwwwrrrrrro",
    "obbbbbbwwwwwwrrrrrro",
    ".obbbbbwwwwwwrrrrro.",
    ".obbbbbwwwwwwrrrrro.",
    ".obbbbbwwwwwwrrrrro.",
    "..obbbbwwwwwwrrrro..",
    "...obbbwwwwwwrrro...",
    "...obbbwwwwwwrrro...",
    "....obbwwwwwwrro....",
    ".....obwwwwwwro.....",
    "......oowwwwoo......",
    "........owwo........",
    ".........oo.........",
    "....................",
]


def heart_pixels(ox, oy):
    return [(ox + dx, oy + dy, COLORS[ch])
            for dy, row in enumerate(HEART) for dx, ch in enumerate(row) if ch != "."]


def main():
    ap = argparse.ArgumentParser(description="Paint and guard a small tricolour heart.")
    ap.add_argument("--x", type=int, default=1001)
    ap.add_argument("--y", type=int, default=494)
    ap.add_argument("--once", action="store_true", help="paint once and exit instead of guarding")
    ap.add_argument("--workers", type=int, default=min(32, 4 * (os.cpu_count() or 4)))
    ap.add_argument("--max-bits", type=int, default=12)
    ap.add_argument("--interval", type=float, default=20)
    ap.add_argument("--name", default=os.environ.get("NAME", "tricolore"))
    args = ap.parse_args()
    args.guard, args.fix = not args.once, True
    run(heart_pixels(args.x, args.y), args, f"heart 20x20 at ({args.x},{args.y})")


if __name__ == "__main__":
    main()
