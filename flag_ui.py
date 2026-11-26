#!/usr/bin/env python3
"""Live view of the canvas with the zones of guarded.txt drawn on it. Add French flag, white or
text zones by dragging, move or resize them by drag and drop, delete them. It reads and writes the
same guarded.txt as scan_hate.py and guard_flags.py, so guard_flags.py paints your changes on its
next round. Needs paint_flag.py and guard_flags.py in the same folder.

    sudo apt install python3-tk        (Tk for Python; macOS/Windows Pythons already include it)
    python3 flag_ui.py                 # --file to use another list, --refresh 10 (seconds)

Mouse: drag on empty board = new zone of the type chosen in the toolbar (Flag / White / Text);
       drag a zone = move it; drag the square handle of the selected zone = resize it (a text zone
       re-fits its text: the box size sets the text size); right-click a zone = delete it;
       wheel = zoom where the mouse is; middle-button drag = pan.
Keys:  Delete / Backspace = delete the selected zone, + / - = zoom, r = refresh now.
Outlines: green = intact on the board, red = damaged (guard_flags.py has work there);
solid = flag, dashed = white, dotted = text. "Preview" shows what the zones will look like.
"""
import argparse, queue, threading, time
import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image, ImageTk

import paint_flag as engine
from guard_flags import (GUARDED, TEXT_DEFAULTS, add_guarded, load_guarded, move_guarded, our_art,
                         remove_guarded, zone_target)

COLOURS = {"white": "ffffff", "black": "000000", "French blue": "002395", "French red": "ed2939",
           "gold": "ffd700", "none": "none"}
DASH = {"flag": "", "white": (6, 3), "text": (2, 2)}


class TextDialog(simpledialog.Dialog):
    """Asks for a text zone's text and colours; result = (text, opts) or None."""

    def body(self, master):
        tk.Label(master, text="Text  ( | = new line ):").grid(row=0, column=0, sticky="w")
        self.text = tk.Entry(master, width=36)
        self.text.grid(row=0, column=1, columnspan=2, pady=2)
        self.vars = {}
        for row, (key, label, choices, default) in enumerate((
                ("fg", "Text colour", ["white", "black", "French blue", "French red", "gold"], "white"),
                ("ol", "Outline", ["black", "white", "none"], "black"),
                ("bg", "Background", ["none", "white", "black", "French blue", "French red"], "none")), start=1):
            tk.Label(master, text=label).grid(row=row, column=0, sticky="w")
            self.vars[key] = tk.StringVar(value=default)
            tk.OptionMenu(master, self.vars[key], *choices).grid(row=row, column=1, sticky="w")
        tk.Label(master, text="The text takes the biggest size that fits the box.", fg="#666").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        return self.text

    def validate(self):
        return bool(self.text.get().strip())

    def apply(self):
        self.result = (self.text.get().strip(), {k: COLOURS[v.get()] for k, v in self.vars.items()})


class FlagUI:
    def __init__(self, root, path, refresh):
        self.root, self.path, self.refresh = root, path, refresh
        self.zoom, self.board, self.zones, self.state, self.targets = 1, None, [], {}, {}
        self.sel, self.drag, self.last = None, None, "-"
        self.ours = our_art()
        self.queue, self.wake = queue.Queue(), threading.Event()

        root.title("Canvas zones")
        root.geometry("1150x950")
        bar = tk.Frame(root)
        bar.pack(fill="x")
        tk.Button(bar, text="Refresh (r)", command=self.wake.set).pack(side="left")
        tk.Button(bar, text="Zoom -", command=lambda: self.set_zoom(self.zoom - 1)).pack(side="left")
        tk.Button(bar, text="Zoom +", command=lambda: self.set_zoom(self.zoom + 1)).pack(side="left")
        tk.Button(bar, text="Delete selected", command=self.delete_selected).pack(side="left")
        self.kind = tk.StringVar(value="flag")
        tk.Label(bar, text="  New zone:").pack(side="left")
        for kind in ("flag", "white", "text"):
            tk.Radiobutton(bar, text=kind.capitalize(), variable=self.kind, value=kind).pack(side="left")
        self.preview = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="Preview", variable=self.preview, command=self.redraw).pack(side="left", padx=6)
        self.status = tk.Label(bar, anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=6)
        tk.Label(root, anchor="w", text="drag empty board = new zone   drag a zone = move   drag its corner handle = "
                                        "resize   right-click / Delete = remove   wheel = zoom at mouse   "
                                        "middle-drag = pan").pack(fill="x")
        frame = tk.Frame(root)
        frame.pack(fill="both", expand=True)
        self.canvas = c = tk.Canvas(frame, bg="#303030", highlightthickness=0)
        xs = tk.Scrollbar(frame, orient="horizontal", command=c.xview)
        ys = tk.Scrollbar(frame, orient="vertical", command=c.yview)
        c.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
        c.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Button-3>", self.on_right)
        c.bind("<MouseWheel>", lambda e: self.set_zoom(self.zoom + (1 if e.delta > 0 else -1), (e.x, e.y)))
        c.bind("<Button-4>", lambda e: self.set_zoom(self.zoom + 1, (e.x, e.y)))  # wheel on Linux
        c.bind("<Button-5>", lambda e: self.set_zoom(self.zoom - 1, (e.x, e.y)))
        c.bind("<ButtonPress-2>", lambda e: c.scan_mark(e.x, e.y))
        c.bind("<B2-Motion>", lambda e: c.scan_dragto(e.x, e.y, gain=1))
        for key in ("<Delete>", "<BackSpace>"):
            root.bind(key, lambda e: self.delete_selected())
        for key in ("<plus>", "<equal>", "<KP_Add>"):
            root.bind(key, lambda e: self.set_zoom(self.zoom + 1))
        for key in ("<minus>", "<KP_Subtract>"):
            root.bind(key, lambda e: self.set_zoom(self.zoom - 1))
        root.bind("r", lambda e: self.wake.set())

        threading.Thread(target=self.fetch_loop, daemon=True).start()
        root.after(200, self.poll)

    # board refresh, in the background so the window never freezes
    def fetch_loop(self):
        while True:
            try:
                w, raw = engine.fetch_board()
                self.queue.put(np.frombuffer(raw, dtype=np.uint8).reshape(-1, w, 3).copy())
            except Exception as e:  # network trouble: show it, try again next time
                self.queue.put(e)
            self.wake.wait(self.refresh)
            self.wake.clear()

    def poll(self):
        try:
            item = self.queue.get_nowait()
            if isinstance(item, Exception):
                self.status.config(text=f"board fetch failed: {item}")
            else:
                self.board, self.last = item, time.strftime("%H:%M:%S")
                self.reload()
        except queue.Empty:
            pass
        self.root.after(200, self.poll)

    def reload(self):
        """Re-read guarded.txt (scan_hate.py may have added zones) and check each zone on the board."""
        self.zones = load_guarded(self.path)
        self.targets = {z.rect: zone_target(z, self.ours) for z in self.zones}
        self.state = {}
        if self.board is not None:
            for rect, target in self.targets.items():
                self.state[rect] = all(self.board[y, x].tobytes().hex() == c for x, y, c in target)
        if self.sel not in self.state:
            self.sel = None
        self.redraw()

    def redraw(self):
        if self.board is None:
            self.status.config(text="loading the board...")
            return
        z, c = self.zoom, self.canvas
        view = self.board
        if self.preview.get() and self.targets:
            view = self.board.copy()
            for target in self.targets.values():
                for x, y, col in target:
                    view[y, x] = (int(col[0:2], 16), int(col[2:4], 16), int(col[4:6], 16))
        img = Image.fromarray(view)
        if z > 1:
            img = img.resize((img.width * z, img.height * z), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)
        c.delete("all")
        c.create_image(0, 0, image=self.photo, anchor="nw")
        c.configure(scrollregion=(0, 0, img.width, img.height))
        for zone in self.zones:
            x0, y0, x1, y1 = zone.rect
            col = {True: "#00e000", False: "#ff3030"}.get(self.state.get(zone.rect), "white")
            chosen = zone.rect == self.sel
            c.create_rectangle(x0 * z - 1, y0 * z - 1, (x1 + 1) * z, (y1 + 1) * z, outline=col,
                               width=3 if chosen else 2, dash=DASH.get(zone.kind, ""))
            label = zone.note if zone.kind == "flag" else f"{zone.kind}: {zone.note}"
            if label and (z >= 2 or chosen):
                c.create_text(x0 * z, y0 * z - 3, text=label, anchor="sw", fill=col)
            if chosen:  # resize handle
                hx, hy = (x1 + 1) * z, (y1 + 1) * z
                c.create_rectangle(hx - 5, hy - 5, hx + 5, hy + 5, fill="yellow", outline="black")
        bad = sum(1 for v in self.state.values() if v is False)
        kinds = {k: sum(1 for zn in self.zones if zn.kind == k) for k in ("flag", "white", "text")}
        self.status.config(text=f"{kinds['flag']} flag, {kinds['white']} white, {kinds['text']} text zone(s), "
                                f"{bad} damaged   board {self.last}   zoom {z}x")

    # coordinates
    def board_xy(self, e):
        return int(self.canvas.canvasx(e.x) // self.zoom), int(self.canvas.canvasy(e.y) // self.zoom)

    def zone_at(self, x, y):
        for zone in reversed(self.zones):
            r = zone.rect
            if r[0] <= x <= r[2] and r[1] <= y <= r[3]:
                return zone
        return None

    def on_handle(self, e):
        if not self.sel:
            return False
        hx, hy = (self.sel[2] + 1) * self.zoom, (self.sel[3] + 1) * self.zoom
        return abs(self.canvas.canvasx(e.x) - hx) <= 7 and abs(self.canvas.canvasy(e.y) - hy) <= 7

    def clamp(self, r):
        H, W = self.board.shape[:2]
        return (max(0, r[0]), max(0, r[1]), min(W - 1, r[2]), min(H - 1, r[3]))

    # mouse: create, move, resize
    def on_press(self, e):
        x, y = self.board_xy(e)
        if self.on_handle(e):
            self.drag = ("resize", self.sel, (x, y))
        else:
            zone = self.zone_at(x, y)
            self.sel = zone.rect if zone else None
            self.drag = ("move", zone.rect, (x, y)) if zone else ("create", None, (x, y))
        self.redraw()

    def drag_rect(self, e):
        mode, rect, (sx, sy) = self.drag
        x, y = self.board_xy(e)
        if mode == "create":
            return (min(sx, x), min(sy, y), max(sx, x), max(sy, y))
        if mode == "move":
            H, W = self.board.shape[:2]
            dx = max(-rect[0], min(W - 1 - rect[2], x - sx))
            dy = max(-rect[1], min(H - 1 - rect[3], y - sy))
            return (rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy)
        return (rect[0], rect[1], max(rect[0] + 2, x), max(rect[1] + 2, y))  # resize from the corner

    def on_drag(self, e):
        if not self.drag or self.board is None:
            return
        r, z = self.clamp(self.drag_rect(e)), self.zoom
        self.canvas.delete("rubber")
        self.canvas.create_rectangle(r[0] * z, r[1] * z, (r[2] + 1) * z, (r[3] + 1) * z, outline="yellow",
                                     dash=(3, 3), width=2, tags="rubber")

    def on_release(self, e):
        if not self.drag or self.board is None:
            return
        mode, old, _ = self.drag
        rect = self.clamp(self.drag_rect(e))
        self.drag = None
        self.canvas.delete("rubber")
        if mode in ("move", "resize"):
            if rect != old:
                move_guarded(old, rect, self.path)
                self.sel = rect
            self.reload()
            return
        if rect[2] - rect[0] < 2 or rect[3] - rect[1] < 2:
            return
        kind = self.kind.get()
        if kind == "text":
            got = TextDialog(self.root, f"New text zone {rect}").result
            if not got:
                return
            add_guarded(rect, got[0], self.path, "text", got[1])
        else:
            note = simpledialog.askstring(f"New {kind} zone", f"{kind.capitalize()} zone {rect}\nNote (optional):",
                                          parent=self.root)
            if note is None:  # cancelled
                return
            add_guarded(rect, note, self.path, kind)
        self.sel = rect
        self.reload()

    def on_right(self, e):
        zone = self.zone_at(*self.board_xy(e))
        if zone:
            self.sel = zone.rect
            self.delete_selected()

    def delete_selected(self):
        if not self.sel:
            return
        if messagebox.askyesno("Delete zone", f"Stop guarding the zone at {self.sel}?", parent=self.root):
            remove_guarded(self.sel, self.path)
            self.sel = None
            self.reload()

    def set_zoom(self, z, anchor=None):
        """Zoom keeping the board point under `anchor` (window x, y; default: view centre) in place."""
        z = max(1, min(8, z))
        if z == self.zoom or self.board is None:
            return
        c = self.canvas
        ax, ay = anchor if anchor else (c.winfo_width() // 2, c.winfo_height() // 2)
        bx, by = c.canvasx(ax) / self.zoom, c.canvasy(ay) / self.zoom
        self.zoom = z
        self.redraw()
        H, W = self.board.shape[:2]
        c.xview_moveto(max(0.0, bx * z - ax) / (W * z))
        c.yview_moveto(max(0.0, by * z - ay) / (H * z))


def main():
    ap = argparse.ArgumentParser(description="View the board and edit the zones of guarded.txt.")
    ap.add_argument("--file", default=GUARDED)
    ap.add_argument("--refresh", type=float, default=10, help="seconds between board refreshes")
    args = ap.parse_args()
    root = tk.Tk()
    FlagUI(root, args.file, args.refresh)
    root.mainloop()


if __name__ == "__main__":
    main()
