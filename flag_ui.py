#!/usr/bin/env python3
"""Live view of the canvas with the flag zones of guarded.txt drawn on it: add a zone by dragging,
delete one by clicking it and pressing Delete (or right-clicking it). It reads and writes the same
guarded.txt as scan_hate.py and guard_flags.py, so guard_flags.py paints your changes on its next
round. Needs paint_flag.py and guard_flags.py in the same folder.

    sudo apt install python3-tk        (Tk for Python; macOS/Windows Pythons already include it)
    python3 flag_ui.py                 # --file to use another list, --refresh 10 (seconds)

Mouse: drag on the board = new zone (a flag or a white zone, chosen in the toolbar), click a zone
       = select it, right-click a zone = delete it.
Keys:  Delete / Backspace = delete the selected zone, + / - or mouse wheel = zoom, r = refresh now.
Zone outlines: green = intact on the board, red = damaged (guard_flags.py has work there);
solid = flag zone, dashed = white zone.
"""
import argparse, os, queue, threading, time
import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image, ImageTk

import paint_flag as engine
from guard_flags import GUARDED, add_guarded, flag_target, load_guarded, our_art


def remove_guarded(rect, path=GUARDED):
    """Remove the lines for `rect`. The file is read again just before (so lines appended
    meanwhile by scan_hate.py are kept) and replaced atomically (guard_flags.py never reads half
    a file)."""
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return
    keep = []
    for line in lines:
        try:
            r = tuple(int(v) for v in line.split("#")[0].split()[:4])
        except ValueError:
            r = None
        if r != tuple(rect):
            keep.append(line)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(keep)
    os.replace(tmp, path)


class FlagUI:
    def __init__(self, root, path, refresh):
        self.root, self.path, self.refresh = root, path, refresh
        self.zoom, self.board, self.flags, self.state = 1, None, [], {}
        self.sel, self.drag, self.last = None, None, "-"
        self.ours = our_art()
        self.queue, self.wake = queue.Queue(), threading.Event()

        root.title("Canvas flag zones")
        root.geometry("1100x950")
        bar = tk.Frame(root)
        bar.pack(fill="x")
        tk.Button(bar, text="Refresh now (r)", command=self.wake.set).pack(side="left")
        tk.Button(bar, text="Zoom -", command=lambda: self.set_zoom(self.zoom - 1)).pack(side="left")
        tk.Button(bar, text="Zoom +", command=lambda: self.set_zoom(self.zoom + 1)).pack(side="left")
        tk.Button(bar, text="Delete selected", command=self.delete_selected).pack(side="left")
        self.kind = tk.StringVar(value="flag")
        tk.Label(bar, text="  New zone:").pack(side="left")
        tk.Radiobutton(bar, text="Flag", variable=self.kind, value="flag").pack(side="left")
        tk.Radiobutton(bar, text="White", variable=self.kind, value="white").pack(side="left")
        self.status = tk.Label(bar, anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=8)
        tk.Label(root, anchor="w", text="drag = add a zone (type chosen above)   click = select   "
                                        "Delete / right-click = remove   + / - / wheel = zoom   "
                                        "green = intact, red = damaged, dashed = white zone").pack(fill="x")
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
        c.bind("<MouseWheel>", lambda e: self.set_zoom(self.zoom + (1 if e.delta > 0 else -1)))
        c.bind("<Button-4>", lambda e: self.set_zoom(self.zoom + 1))  # wheel on Linux
        c.bind("<Button-5>", lambda e: self.set_zoom(self.zoom - 1))
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
        """Re-read guarded.txt (scan_hate.py may have added zones) and check each flag on the board."""
        self.flags = load_guarded(self.path)
        self.state = {}
        if self.board is not None:
            for rect, _, kind in self.flags:
                self.state[rect] = all(self.board[y, x].tobytes().hex() == c
                                       for x, y, c in flag_target(rect, self.ours, kind))
        if self.sel not in self.state:
            self.sel = None
        self.redraw()

    def redraw(self):
        if self.board is None:
            self.status.config(text="loading the board...")
            return
        z, c = self.zoom, self.canvas
        img = Image.fromarray(self.board)
        if z > 1:
            img = img.resize((img.width * z, img.height * z), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)
        c.delete("all")
        c.create_image(0, 0, image=self.photo, anchor="nw")
        c.configure(scrollregion=(0, 0, img.width, img.height))
        for rect, word, kind in self.flags:
            x0, y0, x1, y1 = rect
            col = {True: "#00e000", False: "#ff3030"}.get(self.state.get(rect), "white")
            chosen = rect == self.sel
            c.create_rectangle(x0 * z - 1, y0 * z - 1, (x1 + 1) * z, (y1 + 1) * z, outline=col,
                               width=3 if chosen else 2, dash=(6, 3) if kind == "white" else "")
            label = f"{kind}: {word}" if kind == "white" else word
            if label and (z >= 2 or chosen):
                c.create_text(x0 * z, y0 * z - 3, text=label, anchor="sw", fill=col)
        bad = sum(1 for v in self.state.values() if v is False)
        whites = sum(1 for _, _, kind in self.flags if kind == "white")
        self.status.config(text=f"{len(self.flags) - whites} flag + {whites} white zone(s), {bad} damaged   "
                                f"board at {self.last}   "
                                f"zoom {z}x   {self.path}")

    # mouse
    def board_xy(self, e):
        return int(self.canvas.canvasx(e.x) // self.zoom), int(self.canvas.canvasy(e.y) // self.zoom)

    def zone_at(self, x, y):
        for rect, *_ in reversed(self.flags):
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                return rect
        return None

    def on_press(self, e):
        x, y = self.board_xy(e)
        self.sel = self.zone_at(x, y)
        self.drag = None if self.sel else (x, y)
        self.redraw()

    def on_drag(self, e):
        if not self.drag:
            return
        (x0, y0), (x, y), z = self.drag, self.board_xy(e), self.zoom
        self.canvas.delete("rubber")
        self.canvas.create_rectangle(min(x0, x) * z, min(y0, y) * z, (max(x0, x) + 1) * z, (max(y0, y) + 1) * z,
                                     outline="yellow", dash=(3, 3), tags="rubber")

    def on_release(self, e):
        if not self.drag or self.board is None:
            return
        (x0, y0), (x, y) = self.drag, self.board_xy(e)
        self.drag = None
        self.canvas.delete("rubber")
        H, W = self.board.shape[:2]
        rect = (max(0, min(x0, x)), max(0, min(y0, y)), min(W - 1, max(x0, x)), min(H - 1, max(y0, y)))
        if rect[2] - rect[0] < 2 or rect[3] - rect[1] < 2:
            return
        kind = self.kind.get()
        note = simpledialog.askstring(f"New {kind} zone", f"{kind.capitalize()} zone {rect}\nNote (optional):",
                                      parent=self.root)
        if note is None:  # cancelled
            return
        add_guarded(rect, note.replace("#", " ").strip() or "manual", self.path, kind)
        self.sel = rect
        self.reload()

    def on_right(self, e):
        rect = self.zone_at(*self.board_xy(e))
        if rect:
            self.sel = rect
            self.delete_selected()

    def delete_selected(self):
        if not self.sel:
            return
        if messagebox.askyesno("Delete zone", f"Stop guarding the zone at {self.sel}?", parent=self.root):
            remove_guarded(self.sel, self.path)
            self.sel = None
            self.reload()

    def set_zoom(self, z):
        z = max(1, min(6, z))
        if z != self.zoom:
            self.zoom = z
            self.redraw()


def main():
    ap = argparse.ArgumentParser(description="View the board and edit the flag zones of guarded.txt.")
    ap.add_argument("--file", default=GUARDED)
    ap.add_argument("--refresh", type=float, default=10, help="seconds between board refreshes")
    args = ap.parse_args()
    root = tk.Tk()
    FlagUI(root, args.file, args.refresh)
    root.mainloop()


if __name__ == "__main__":
    main()
