#!/usr/bin/env python3
"""Paint (and guard) a second French flag with a croissant, over the old Polish flag
below the original one. Same options as paint_flag.py, which must sit in the same folder.

    pip install argon2-cffi requests

    python3 paint_croissant_flag.py           # paint every pixel once
    python3 paint_croissant_flag.py --fix     # repaint only pixels that differ on the board
    python3 paint_croissant_flag.py --guard   # keep repairing damaged pixels until Ctrl-C

Default position: x 877, y 672 (147x98). --no-croissant paints the plain flag.
"""
import argparse

from paint_flag import add_common_args, flag_pixels, run

# Croissant drawn over the flag. Top-left character sits at CROISSANT_AT, relative to the
# flag's top-left corner. '.' = flag shows through.
CROISSANT_COLORS = {
    "o": "4a2a0e",  # outline
    "d": "9c5518",  # shadow
    "m": "cf8429",  # base
    "l": "eaae4c",  # light
    "h": "fbe0a0",  # highlight
}
CROISSANT_AT = (32, 19)
CROISSANT = [
    "...............................ooooooooooo........oooooo",
    "............................ooooohhhhhhhllooooooooooooooooo",
    ".........................oooohhhhhhhhhhhhllllmooollhhhhhhoooo",
    "........................ooohhhhhhhhhhhhhhlllllmoolllhhhhhhhhoo",
    "......................ooohhhhhhhhhhhhhhhhlllllmmollllhhhhhhhhhoo",
    ".....................oohhhhhhhhhhhhhhhhhhlllllmmoollllhhhhhhhhhoo",
    "....................oohhhhhhhhhhhhhhhhhhhlllllmmmomllllhhhhhhhhloo",
    "...................ohhhhhhhhhhhhhhhhhhhhhlllllmmmoomllllhhhhhhhlloo",
    "..................ohhhhhhhhhhhhhhhhhhhhhhlllllmmmoomlllllhhhhhhllloo",
    ".................ohhhhhhhhhhhhhhhhhllhhhllllllmmmdommlllllhhhhlllllooo",
    "................oohhhhhhhhhhhhhhhhhllhhhllllllmmmdodmllllllhhllllllmooooo",
    "...............oohhhhhhhhhhhhhhhhhllllllllllllmmmdommmllllllllllllmmoloooo",
    "...............ohhhhhhhhhhhhhhhhhllllllllllllmmmmdommmllllllllllllmmmollhoo",
    "..............ohhhhhhhhhhhhhhhhhlllllllllllllmmmddommmlllllllllllmmmmollhhoo",
    "..............ohhhhhhhhhhhhhhhhllllllllllllllmmmddommmmllllllllllmmmoolllhhoo",
    "............oollhhhhhhhhhhhlllllllllllllllllmmmmddommmmlllllllllmmmmoomllhhhoo",
    "..........oooolllhhhhhhhhhllllllllllllllllllmmmddoodmmmlllllllmmmmmdommlllhhho",
    ".........ooolollllhhhhhhhhlllllllllllllllllmmmmddoddmmmmlllllmmmmmddommlllhhhoo",
    "........oollloollllhhhhhhlllllllllllllmllllmmmdddodmmmmmlllmmmmmmddodmmllllhhho",
    ".......oohllloolllllllllllllllllllllllmlllmmmmdddodmmmmmmmmmmmmdddoommmlllllhhlo",
    "......oohhlllmolllllllllllllllllllllllmmlmmmmdddoddmmmmmmmmmmmdddoodmmmmlllllllo",
    ".....oohhhlllmoollllllllllllllllllllllmmlmmmmdddoddmmmmmmmmmddddooddmmmmllllllloo",
    ".....ohhhhhllmmodllllllllllllllllllllmmmmmmmdddooddmmmmmmmmddddooddmmmmmllllllllo",
    ".....ohhhhhllmmoommlllllllllllllllllmmmmmmmmdddooddmmmmmmddddooddddmmmmmllllllllo",
    "....oohhhhhlllmmoommmllllllllllllllmmmmmmmmddooooommmmmmooooooooddmmmmmmmmlllllmmo",
    "....ohhhhhhlllmmmoommmmmllllllllmmmmmmmmmmmdoo...ooooooo......ooommmmmmmmmmmmmmmmoo",
    "....ohhhhhhlllmmmdoommmmmmmmllllmmmmmmmmmmmo....................oooommmmmmmmmmmmmoo",
    "...oohhhhhhhlllmmmdoommmmmmmmmmmmmmmmmmmmoo........................oooooddddddddoooo",
    "...ohhhhhhhhlllmmmmddoodmmmmmmmmmmmmmmmooo..............................ooodddddoooo",
    "...ohhhhhhhhllllmmmmddoooddmmmmmmmmmoooo...................................oooooomlo",
    "...ohhhhhhhhhllllmmmmdddooodddmmoooooo......................................oooddmloo",
    "...ohhhhhhhhhlllllmmmmdddddooooo.............................................oodmmmlo",
    "...ohhhhhhhhhllllllmmmmmdddoo.................................................oommmlo",
    "...ohhhhhhhhhlllllllmmmmmddo...................................................oommlo",
    "...ohhhhhhhhhllllllllmmmmmoo.....................................................omlo",
    "...ohhhhhhhhllllllllllmmmmoo......................................................oooo",
    "..oollhhhhhlllllllllllmmmoo........................................................ooo",
    "..ooolllllllllllllllmmmoo...........................................................oo",
    ".oolodlllllllllllmmmmmoo.............................................................o",
    ".oollodlllllllmmmmmmoo",
    ".ohllmoommmmmmmmmmmo",
    "oohhllmoooommmmddoo",
    "oohhllmmmooooooodoo",
    "oohhhllmmmmddddddoo",
    "oohhhlllmmmmmmmmmo",
    "oohhhhllllmmmmmmoo",
    ".ohhhhhlllllmmmmo",
    ".ohhhhhhllllllloo",
    ".ohhhhhhlllllllo",
    ".oohhhhhhlllllo",
    "..ohhhhhhllllmo",
    "..ohhhhhhllllo",
    "...ohhhhhllloo",
    "...ohhhhlllmo",
    "...ohhhlllmoo",
    "....ollllmmoo",
    "....oolllmdoo",
    "....oooooooo",
    "....oooooodo",
    ".....olmmmmo",
    ".....oolmmmo",
    ".....oohlllo",
    "......oolloo",
    ".......ohloo",
    ".......oohoo",
    "........oooo",
    ".........ooo",
    "..........oo",
    "..........oo",
]

# Pixels left to the fox art next to this flag (two sparkles and the tip of its tail), with a
# 1 px margin. '#' = theirs. Top-left character sits at FOX_AT, relative to the flag.
FOX_AT = (0, 3)
FOX_MASK = [
    "......###........",
    "......######.....",
    "......######.....",
    "......######.....",
    "....#######......",
    "...#########.....",
    "...#########.....",
    "..###########....",
    "..###########....",
    "..###########....",
    "...##########....",
    "....############.",
    "....######..####.",
    "....#####..######",
    "..........#######",
    "..........#######",
    "..........#######",
    "..........######.",
    "...........####..",
    "...........####..",
    "...........####..",
    ".................",
    ".................",
    ".................",
    ".................",
    ".................",
    ".................",
    ".................",
    ".................",
    ".................",
    ".................",
    ".................",
    ".................",
    "##...............",
    "###..............",
    "######...........",
    "######...........",
    "######...........",
    "######...........",
    "#####............",
    "####.............",
    "###..............",
    "##...............",
]

# Left to Exil_S, who paints a red heart straddling the flag's bottom edge (white band), with a
# 1 px margin. '#' = theirs. Top-left character sits at EXIL_AT, relative to the flag.
EXIL_AT = (67, 90)
EXIL_MASK = [
    "...####..####...",
    "...####..####...",
    ".##############.",
    "################",
    "################",
    "################",
    "################",
    "################",
]


def art(at, rows, keep=lambda ch: ch != "."):
    return {(at[0] + dx, at[1] + dy): ch
            for dy, row in enumerate(rows) for dx, ch in enumerate(row) if keep(ch)}


FOX_PIXELS = set(art(FOX_AT, FOX_MASK, lambda ch: ch == "#"))
EXIL_PIXELS = set(art(EXIL_AT, EXIL_MASK, lambda ch: ch == "#"))
GIVEN_AWAY = FOX_PIXELS | EXIL_PIXELS
CROISSANT_PIXELS = art(CROISSANT_AT, CROISSANT)


def croissant_flag_pixels(ox, oy, width, skip=GIVEN_AWAY, croissant=True):
    for x, y, color in flag_pixels(ox, oy, width, skip):
        ch = CROISSANT_PIXELS.get((x - ox, y - oy)) if croissant else None
        yield x, y, CROISSANT_COLORS[ch] if ch else color


def main():
    ap = argparse.ArgumentParser(description="Paint and guard a French flag with a croissant.")
    add_common_args(ap, x=877, y=672)
    ap.add_argument("--no-croissant", action="store_true", help="paint the plain flag")
    args = ap.parse_args()
    skip = set() if args.no_fox else GIVEN_AWAY
    pixels = list(croissant_flag_pixels(args.x, args.y, args.width, skip, not args.no_croissant))
    run(pixels, args, f"croissant flag {args.width}x{round(args.width * 2 / 3)} at ({args.x},{args.y}) "
                      f"minus {len(skip)} px left to the fox and Exil_S's heart")


if __name__ == "__main__":
    main()
