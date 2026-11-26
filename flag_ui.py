#!/usr/bin/env python3
"""Live view of the canvas with the zones of guarded.txt drawn on it. Add French flag, white or
text zones by dragging, move or resize them by drag and drop, switch their painting on or off,
delete them. It reads and writes the same guarded.txt as scan_hate.py and guard_flags.py, so
guard_flags.py paints your changes on its next round. Needs paint_flag.py and guard_flags.py.

    sudo apt install python3-tk        (Tk for Python; macOS/Windows Pythons already include it)
    pip install websockets             (optional: live view; without it the board reloads every 10 s)
    python3 flag_ui.py                 # --file to use another list, --refresh 10 (seconds)

Mouse: drag on empty board = new zone of the type chosen in the toolbar (Flag / White / Text /
       Chat: where chat_to_board.py shows the live chat, one chat zone, 3 lines high);
       drag a zone = move it; drag the square handle of the selected zone = resize it (a text zone
       re-fits its text); the buttons above the selected zone switch its painting on/off or
       delete it; right-click a zone = delete it; wheel = zoom where the mouse is; middle-drag = pan.
Keys:  Delete / Backspace = delete the selected zone, + / - = zoom, r = full refresh now.
Outlines: green = intact on the board, red = damaged, grey = painting switched off;
solid = flag, dashed = white, dotted = text. "Preview" shows the zones as they will look.

The view is live: pixels painted by anyone appear within about half a second (the board's own
live feed), with a full reload every 60 s to stay in sync. Only the visible part of the board is
drawn, so zooming and scrolling stay fast.
"""
import argparse, asyncio, queue, threading, time
import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image, ImageTk

import paint_flag as engine
import pixfont
from guard_flags import (GUARDED, add_guarded, load_guarded, move_guarded, our_art, remove_guarded,
                         set_enabled, zone_target)

COLOURS = {"white": "ffffff", "black": "000000", "French blue": "002395", "French red": "ed2939",
           "gold": "ffd700", "none": "none"}
DASH = {"flag": "", "white": (6, 3), "text": (2, 2), "chat": (8, 2, 2, 2)}
CHAT_H = 3 * 8 + 1  # chat_to_board.py: 3 lines of the 5x7 pixel font
WS_URL = engine.BASE.replace("https://", "wss://").replace("http://", "ws://") + "/api/ws"


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
        self.zoom, self.board, self.view = 1, None, None
        self.zones, self.cache, self.state = [], {}, {}
        self.sel, self.drag, self.last, self.ant = None, None, "-", 0
        self.photo, self.img_item, self.render_pending, self.buttons = None, None, False, []
        self.live, self.live_buf, self.lock = False, [], threading.Lock()
        self.ours = our_art()
        self.queue, self.wake = queue.Queue(), threading.Event()

        root.title("Canvas zones")
        root.geometry("1150x950")
        bar = tk.Frame(root)
        bar.pack(fill="x")
        tk.Button(bar, text="Refresh (r)", command=self.wake.set).pack(side="left")
        tk.Button(bar, text="Zoom -", command=lambda: self.set_zoom(self.zoom - 1)).pack(side="left")
        tk.Button(bar, text="Zoom +", command=lambda: self.set_zoom(self.zoom + 1)).pack(side="left")
        self.kind = tk.StringVar(value="flag")
        tk.Label(bar, text="  New zone:").pack(side="left")
        for kind in ("flag", "white", "text", "chat"):
            tk.Radiobutton(bar, text=kind.capitalize(), variable=self.kind, value=kind).pack(side="left")
        self.preview = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="Preview", variable=self.preview, command=self.recompose).pack(side="left", padx=6)
        self.status = tk.Label(bar, anchor="w")
        self.status.pack(side="left", fill="x", expand=True, padx=6)
        tk.Label(root, anchor="w", text="drag empty board = new zone   drag a zone = move   drag its corner handle = "
                                        "resize   buttons above the selected zone = on/off, delete   "
                                        "wheel = zoom at mouse   middle-drag = pan").pack(fill="x")
        frame = tk.Frame(root)
        frame.pack(fill="both", expand=True)
        self.canvas = c = tk.Canvas(frame, bg="#303030", highlightthickness=0)
        xs = tk.Scrollbar(frame, orient="horizontal", command=lambda *a: (c.xview(*a), self.request_render()))
        ys = tk.Scrollbar(frame, orient="vertical", command=lambda *a: (c.yview(*a), self.request_render()))
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
        c.bind("<B2-Motion>", lambda e: (c.scan_dragto(e.x, e.y, gain=1), self.request_render()))
        c.bind("<Configure>", lambda e: self.request_render())
        for key in ("<Delete>", "<BackSpace>"):
            root.bind(key, lambda e: self.delete_selected())
        for key in ("<plus>", "<equal>", "<KP_Add>"):
            root.bind(key, lambda e: self.set_zoom(self.zoom + 1))
        for key in ("<minus>", "<KP_Subtract>"):
            root.bind(key, lambda e: self.set_zoom(self.zoom - 1))
        root.bind("r", lambda e: self.wake.set())

        threading.Thread(target=self.fetch_loop, daemon=True).start()
        threading.Thread(target=self.live_loop, daemon=True).start()
        root.after(100, self.poll)
        root.after(90, self.animate)

    # ---------------------------------------------------------------- board data (threads)
    def fetch_loop(self):
        """Full board download: at start, every --refresh s without the live feed, every 60 s with it."""
        while True:
            try:
                w, raw = engine.fetch_board()
                self.queue.put(np.frombuffer(raw, dtype=np.uint8).reshape(-1, w, 3).copy())
            except Exception as e:  # network trouble: show it, try again next time
                self.queue.put(e)
            self.wake.wait(60 if self.live else self.refresh)
            self.wake.clear()

    def live_loop(self):
        """Pixels as they are painted, from the board's live feed ("x,y,rrggbb,name" lines)."""
        try:
            import websockets
        except ImportError:
            return

        async def listen():
            async with websockets.connect(WS_URL, open_timeout=20, max_size=None, ping_interval=20) as ws:
                self.live = True
                async for msg in ws:
                    for line in str(msg).splitlines():
                        if line == "reload":
                            self.wake.set()
                            continue
                        p = line.split(",")
                        if len(p) >= 3 and len(p[2]) == 6:
                            try:
                                item = (int(p[0]), int(p[1]), bytes.fromhex(p[2]))
                            except ValueError:
                                continue
                            with self.lock:
                                self.live_buf.append(item)

        while True:
            try:
                asyncio.run(listen())
            except Exception:
                pass
            self.live = False
            time.sleep(5)

    def poll(self):
        """Apply new data on the Tk thread, a few times per second."""
        changed = False
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, Exception):
                self.status.config(text=f"board fetch failed: {item}")
            else:
                first = self.board is None
                self.board, self.last, changed = item, time.strftime("%H:%M:%S"), True
                if first:
                    self.reload_zones()
        with self.lock:
            buf, self.live_buf = self.live_buf, []
        if buf and self.board is not None:
            H, W = self.board.shape[:2]
            buf = [(x, y, c) for x, y, c in buf if 0 <= x < W and 0 <= y < H]
            if buf:
                xs, ys, cs = zip(*buf)
                self.board[np.array(ys), np.array(xs)] = np.frombuffer(b"".join(cs), np.uint8).reshape(-1, 3)
                changed = True
        if changed and self.board is not None:
            old = dict(self.state)
            self.update_state()
            self.recompose()
            if old != self.state:
                self.draw_overlays()
            else:
                self.update_status()
        self.root.after(250, self.poll)

    # ---------------------------------------------------------------- zones
    @staticmethod
    def chat_sample(zone):
        """What a chat zone will look like (chat_to_board.py paints it): 3 sample lines."""
        x0, y0, x1, _ = zone.rect
        w = x1 - x0 + 1
        img = np.full((CHAT_H, w, 3), 255, np.uint8)
        for i, (who, msg) in enumerate((("Marie", "Vive la France !"), ("Tom_42", "the croissant flag was so good"),
                                        ("Zoe", "allez les bleus"))):
            if i == 2:
                pixfont.draw(img, 2, 1 + i * 8, ">", (0xed, 0x29, 0x39))
            x = pixfont.draw(img, 14, 1 + i * 8, who + ": ", (0x00, 0x23, 0x95))
            pixfont.draw(img, x, 1 + i * 8, msg, (0, 0, 0))
        ys, xs = np.mgrid[0:CHAT_H, 0:w]
        return [(x0 + int(x), y0 + int(y), img[y, x].tobytes().hex()) for y, x in zip(ys.ravel(), xs.ravel())]

    def arrays(self, zone):
        """(xs, ys, colours) numpy arrays of the zone's target, cached per zone."""
        k = zone.key()
        if k not in self.cache:
            t = self.chat_sample(zone) if zone.kind == "chat" else zone_target(zone, self.ours)
            if t:
                xs, ys, cs = zip(*t)
                self.cache[k] = (np.array(xs), np.array(ys),
                                 np.frombuffer(bytes.fromhex("".join(cs)), np.uint8).reshape(-1, 3))
            else:
                self.cache[k] = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3), np.uint8))
        return self.cache[k]

    def reload_zones(self):
        """Re-read guarded.txt and refresh everything that depends on it."""
        self.zones = load_guarded(self.path)
        keys = {z.key() for z in self.zones}
        self.cache = {k: v for k, v in self.cache.items() if k in keys}
        if self.sel not in {z.rect for z in self.zones}:
            self.sel = None
        if self.board is not None:
            self.update_state()
            self.recompose()
        self.draw_overlays()

    def update_state(self):
        self.state = {}
        for z in self.zones:
            if z.enabled and z.kind != "chat":  # the chat panel's content is not known here
                xs, ys, cs = self.arrays(z)
                self.state[z.rect] = bool((self.board[ys, xs] == cs).all()) if len(xs) else True

    def recompose(self):
        """The image to show: the board, with the zones' targets drawn over it in preview mode."""
        if self.board is None:
            return
        if self.preview.get():
            self.view = self.board.copy()
            for z in self.zones:
                if z.enabled:
                    xs, ys, cs = self.arrays(z)
                    self.view[ys, xs] = cs
        else:
            self.view = self.board
        self.request_render()

    # ---------------------------------------------------------------- drawing (visible part only)
    def request_render(self):
        if not self.render_pending:
            self.render_pending = True
            self.root.after_idle(self.render)

    def render(self):
        self.render_pending = False
        if self.view is None:
            return
        z, c = self.zoom, self.canvas
        H, W = self.view.shape[:2]
        c.configure(scrollregion=(0, 0, W * z, H * z))
        vx, vy = c.canvasx(0), c.canvasy(0)
        vw, vh = max(1, c.winfo_width()), max(1, c.winfo_height())
        bx0, by0 = max(0, int(vx // z) - 1), max(0, int(vy // z) - 1)
        bx1, by1 = min(W, int((vx + vw) // z) + 2), min(H, int((vy + vh) // z) + 2)
        crop = self.view[by0:by1, bx0:bx1]
        if z > 1:
            crop = crop.repeat(z, axis=0).repeat(z, axis=1)
        self.photo = ImageTk.PhotoImage(Image.fromarray(np.ascontiguousarray(crop)))
        if self.img_item is None:
            self.img_item = c.create_image(0, 0, anchor="nw", tags="img")
        c.itemconfigure(self.img_item, image=self.photo)
        c.coords(self.img_item, bx0 * z, by0 * z)
        c.tag_lower("img")

    def draw_overlays(self):
        """Zone outlines, labels, and the selected zone's animated border, handle and buttons."""
        c, z = self.canvas, self.zoom
        c.delete("zone")
        for b in self.buttons:
            b.destroy()
        self.buttons = []
        for zone in self.zones:
            x0, y0, x1, y1 = zone.rect
            box = (x0 * z - 1, y0 * z - 1, (x1 + 1) * z, (y1 + 1) * z)
            col = "#9a9a9a" if not zone.enabled else {True: "#00e000", False: "#ff3030"}.get(self.state.get(zone.rect), "white")
            label = zone.note if zone.kind == "flag" else f"{zone.kind}: {zone.note}"
            if zone.rect == self.sel:
                c.create_rectangle(*box, outline="white", width=2, tags="zone")
                c.create_rectangle(*box, outline="black", width=2, dash=(4, 4), dashoffset=self.ant, tags=("zone", "ants"))
                hx, hy = box[2], box[3]
                c.create_rectangle(hx - 5, hy - 5, hx + 5, hy + 5, fill="yellow", outline="black", tags="zone")
                self.add_buttons(zone, box)
            else:
                c.create_rectangle(*box, outline=col, width=2, dash=DASH.get(zone.kind, ""), tags="zone")
            if label and (z >= 2 or zone.rect == self.sel):
                c.create_text(box[0] + 1, box[1] - (24 if zone.rect == self.sel else 2), text=label, anchor="sw",
                              fill=col, tags="zone")
        self.update_status()

    def add_buttons(self, zone, box):
        c = self.canvas
        on = zone.enabled
        toggle = tk.Button(c, text="Drawing: ON" if on else "Drawing: OFF", font=("TkDefaultFont", 8), padx=3, pady=0,
                           bg="#2e7d32" if on else "#8e2b2b", fg="white", activebackground="#555",
                           command=lambda: self.toggle(zone))
        delete = tk.Button(c, text="Delete", font=("TkDefaultFont", 8), padx=3, pady=0,
                           command=self.delete_selected)
        above = box[1] - 3 >= self.canvas.canvasy(0) + 22  # room above, else put them under the box
        y, anchor = (box[1] - 3, "sw") if above else (box[3] + 3, "nw")
        c.create_window(box[0], y, window=toggle, anchor=anchor, tags="zone")
        c.create_window(box[0] + toggle.winfo_reqwidth() + 3, y, window=delete, anchor=anchor, tags="zone")
        self.buttons = [toggle, delete]

    def animate(self):
        self.ant = (self.ant - 1) % 8
        self.canvas.itemconfigure("ants", dashoffset=self.ant)
        self.root.after(90, self.animate)

    def update_status(self):
        bad = sum(1 for v in self.state.values() if v is False)
        off = sum(1 for zn in self.zones if not zn.enabled)
        kinds = {k: sum(1 for zn in self.zones if zn.kind == k) for k in ("flag", "white", "text", "chat")}
        mode = "live" if self.live else f"reload every {self.refresh:g} s"
        self.status.config(text=f"{kinds['flag']} flag, {kinds['white']} white, {kinds['text']} text zone(s), "
                                f"{'chat panel placed, ' if kinds['chat'] else ''}{bad} damaged, {off} off   "
                                f"{mode}, full {self.last}   zoom {self.zoom}x")

    # ---------------------------------------------------------------- mouse
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

    def on_press(self, e):
        if self.board is None:
            return
        x, y = self.board_xy(e)
        if self.on_handle(e):
            self.drag = ("resize", self.sel, (x, y))
        else:
            zone = self.zone_at(x, y)
            new_sel = zone.rect if zone else None
            self.drag = ("move", zone.rect, (x, y)) if zone else ("create", None, (x, y))
            if new_sel != self.sel:
                self.sel = new_sel
                self.draw_overlays()

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
        if self.zone_kind(rect) == "chat":  # a chat zone only changes width: its height is 3 lines
            return (rect[0], rect[1], max(rect[0] + 40, x), rect[1] + CHAT_H - 1)
        return (rect[0], rect[1], max(rect[0] + 2, x), max(rect[1] + 2, y))  # resize from the corner

    def zone_kind(self, rect):
        return next((z.kind for z in self.zones if z.rect == rect), None)

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
                self.reload_zones()
            return
        if rect[2] - rect[0] < 2 or rect[3] - rect[1] < 2:
            return
        kind = self.kind.get()
        if kind == "chat":  # one chat panel: replace the previous one; height = 3 lines
            for z in self.zones:
                if z.kind == "chat":
                    remove_guarded(z.rect, self.path)
            rect = self.clamp((rect[0], rect[1], max(rect[2], rect[0] + 40), rect[1] + CHAT_H - 1))
            add_guarded(rect, "chat", self.path, "chat")
        elif kind == "text":
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
        self.reload_zones()

    def on_right(self, e):
        zone = self.zone_at(*self.board_xy(e))
        if zone:
            self.sel = zone.rect
            self.draw_overlays()
            self.delete_selected()

    def toggle(self, zone):
        set_enabled(zone.rect, not zone.enabled, self.path)
        self.reload_zones()

    def delete_selected(self):
        if not self.sel:
            return
        if messagebox.askyesno("Delete zone", f"Stop guarding the zone at {self.sel}?", parent=self.root):
            remove_guarded(self.sel, self.path)
            self.sel = None
            self.reload_zones()

    def set_zoom(self, z, anchor=None):
        """Zoom keeping the board point under `anchor` (window x, y; default: view centre) in place."""
        z = max(1, min(8, z))
        if z == self.zoom or self.board is None:
            return
        c = self.canvas
        ax, ay = anchor if anchor else (c.winfo_width() // 2, c.winfo_height() // 2)
        bx, by = c.canvasx(ax) / self.zoom, c.canvasy(ay) / self.zoom
        self.zoom = z
        H, W = self.board.shape[:2]
        c.configure(scrollregion=(0, 0, W * z, H * z))
        c.xview_moveto(max(0.0, bx * z - ax) / (W * z))
        c.yview_moveto(max(0.0, by * z - ay) / (H * z))
        self.draw_overlays()
        self.request_render()


def main():
    ap = argparse.ArgumentParser(description="View the board and edit the zones of guarded.txt.")
    ap.add_argument("--file", default=GUARDED)
    ap.add_argument("--refresh", type=float, default=10, help="seconds between full reloads without the live feed")
    args = ap.parse_args()
    root = tk.Tk()
    FlagUI(root, args.file, args.refresh)
    root.mainloop()


if __name__ == "__main__":
    main()
