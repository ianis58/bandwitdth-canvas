# bandwitdth-canvas

Tools for https://canvas.bandwidthyt.com. Everything runs from one window:

```bash
pip install requests pillow numpy websockets scipy pynacl
```
```bash
python3 canvas.py
```

- **Board**: live view, zoom with the wheel, move with middle-drag, Shift+drag or the arrow keys.
- **Zones** (`guarded.txt`): drag on the board to add a French flag, croissant flag, white, text or
  chat zone; move, resize, switch off or delete them.
- **Drawing**: "Enable drawing" / "Pause drawing" paints and keeps every enabled zone. Name,
  workers, hash threads, max bits, batch and interval are set in the second toolbar.
- **Artworks**: "Board" lists other painters' rectangular artworks, recorded from the live feed;
  "My library" has our own (French flag, croissant flag, VIVE LA FRANCE MERCI flag, Justice with
  its QR code, the cat, the tricolour heart, and any picture dropped into `art/library/`). Place a
  copy anywhere with a click.
- **Hate scanner**: covers hater words with small flags (needs tesseract).
- **Logs**: side panel with what the drawing, the watcher and the scanner do.

Optional, ~3x faster hashing (needs clang or gcc): `python3 fastpow/build.py`.

The YouTube chat panel is a separate program: `python3 chat_to_board.py` (its zone is placed in
`canvas.py`'s window).
