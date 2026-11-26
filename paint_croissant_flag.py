#!/usr/bin/env python3
"""Paint (and guard) a big French flag with a croissant, filling the free area at the top left of
the canvas. Same options as paint_flag.py, which must sit in the same folder.

    pip install requests pynacl        (and python3 fastpow/build.py for ~3x faster hashing)

    python3 paint_croissant_flag.py --fix     # paint the pixels that differ (skips what's right)
    python3 paint_croissant_flag.py           # paint every pixel once
    python3 paint_croissant_flag.py --guard   # keep repairing damaged pixels until Ctrl-C

Default: x 3, y 57, 506x337 - between the board's left edge, the rainbow bar and the art on the
right. The croissant is drawn for any --width. Previous place: --x 877 --y 672 --width 147.
--no-croissant paints the plain flag.
"""
import argparse, math

from paint_flag import add_common_args, flag_pixels, run

# 1 outline, 2 shadow, 3 base, 4 light, 5 highlight
CROISSANT_COLORS = {1: "4a2a0e", 2: "9c5518", 3: "cf8429", 4: "eaae4c", 5: "fbe0a0"}

# Croissant design "B" (tilted, rising to the right), in coordinates of a 147x98 flag. The body is
# a crescent along an arc of radius R around (cx, cy), tapering to a point at both tips, cut into
# n rolled bands that bulge a little and are shaded like cylinders, lit from the top-left.
CROISSANT = dict(cx=80, cy=72, R=40, phi=92, h=15, n=7, rot=-22, band_warp=1.18, taper=1.8, curve=0.28)


def _sample(X, Y, c, k, lw):
    """Palette index at a sub-pixel position (0 = outside). k: flag scale, lw: line width."""
    dx, dy = X - c["cx"] * k, Y - c["cy"] * k
    phi = math.atan2(dx, -dy) - math.radians(c["rot"])
    phi = (phi + math.pi) % (2 * math.pi) - math.pi
    PHI = math.radians(c["phi"])
    if abs(phi) >= PHI:
        return 0
    R = c["R"] * k
    rho = math.hypot(dx, dy)
    t = phi / PHI                                            # -1 .. 1 along the body
    h = c["h"] * k * (1 - abs(t) ** c["taper"]) ** 1.1
    g = math.copysign(abs(t) ** c["band_warp"], t)           # middle bands wider
    nrm0 = max(-1.0, min(1.0, (rho - R) / max(h, 0.5 * k)))
    # band boundaries bow towards the tips at the edges: rolled-dough look
    pos = (g + 1) / 2 * c["n"] + c["curve"] * (1 - nrm0 ** 2) * -math.copysign(1, t) * (abs(t) > 0.08)
    f = pos - math.floor(pos)                                # 0..1 inside a band
    h_eff = h * (0.80 + 0.20 * math.sin(math.pi * f)) + 0.3 * k
    off = rho - R
    if abs(off) > h_eff:
        return 0
    if abs(off) > h_eff - 1.15 * lw:                         # outline
        return 1
    dpos = min(f, 1 - f)                                     # separator between bands
    band_px = 2 * PHI * R / c["n"] * (0.6 if abs(t) > 0.6 else 1.0)
    if dpos * band_px < 0.75 * lw and abs(off) < h_eff - 1.5 * lw:
        return 1 if dpos * band_px < 0.45 * lw else 2
    nrm = off / h_eff                                        # -1 inner .. +1 outer
    sc = 0.75 * math.sin(math.pi * f) ** 0.8 + 0.45 * nrm - 0.25 * (f - 0.5) - 0.2 * t - 0.2
    return 5 if sc > 0.62 else 4 if sc > 0.3 else 3 if sc > -0.05 else 2


def croissant_pixels(width, c=CROISSANT, ss=4):
    """{(dx, dy): colour} relative to the flag's top-left corner, for a flag `width` wide."""
    k = width / 147
    lw = max(1.0, 0.8 * k)
    height = round(width * 2 / 3)
    R, hmax, PHI = c["R"] * k, c["h"] * k + 0.3 * k, math.radians(c["phi"])
    subs = [(i + 0.5) / ss for i in range(ss)]
    out = {}
    for y in range(height):
        for x in range(width):
            dx, dy = x + 0.5 - c["cx"] * k, y + 0.5 - c["cy"] * k
            if abs(math.hypot(dx, dy) - R) > hmax + 2:      # quick reject: far from the body
                continue
            phi = (math.atan2(dx, -dy) - math.radians(c["rot"]) + math.pi) % (2 * math.pi) - math.pi
            if abs(phi) > PHI + 2 / R:
                continue
            votes = [_sample(x + ox, y + oy, c, k, lw) for oy in subs for ox in subs]
            nz = [v for v in votes if v]
            if len(nz) < len(votes) / 2:
                continue
            v = 1 if votes.count(1) / len(votes) >= 0.3 else max(set(nz), key=nz.count)
            out[(x, y)] = CROISSANT_COLORS[v]
    return out


def mask(at, rows):
    return {(at[0] + dx, at[1] + dy) for dy, row in enumerate(rows) for dx, ch in enumerate(row) if ch == "#"}


# Areas given away at the previous place (x 877, y 672), in board coordinates: they only matter
# if the flag is painted there again. Fox art (two sparkles, tail tip) and Exil_S's red heart,
# each with a 1 px margin. '#' = theirs.
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
] + ["................."] * 12 + [
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
GIVEN_AWAY = mask((877, 675), FOX_MASK) | mask((944, 762), EXIL_MASK)


def croissant_flag_pixels(ox, oy, width, skip=GIVEN_AWAY, croissant=True):
    art = croissant_pixels(width) if croissant else {}
    for x, y, color in flag_pixels(ox, oy, width, set()):
        if (x, y) not in skip:
            yield x, y, art.get((x - ox, y - oy), color)


def main():
    ap = argparse.ArgumentParser(description="Paint and guard a French flag with a croissant.")
    add_common_args(ap, x=3, y=57)
    ap.set_defaults(width=506)
    ap.add_argument("--no-croissant", action="store_true", help="paint the plain flag")
    args = ap.parse_args()
    skip = set() if args.no_fox else GIVEN_AWAY
    pixels = list(croissant_flag_pixels(args.x, args.y, args.width, skip, not args.no_croissant))
    run(pixels, args, f"croissant flag {args.width}x{round(args.width * 2 / 3)} at ({args.x},{args.y}), "
                      f"{len(pixels)} px")


if __name__ == "__main__":
    main()
