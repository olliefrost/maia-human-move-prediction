"""Interactive demo (tkinter): Maia vs Stockfish vs me, one position at a time.

    python -m maia_study gui                    # step through data/positions.json
    python -m maia_study gui --pgn games.pgn    # step through every position of a PGN

Click a piece then a target square (or type SAN) to enter *your* move.
Arrows: blue = Maia's top move, green = Stockfish's best, orange = your move.
"""

import argparse
import json
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, ttk

import chess
import chess.pgn

from . import config as C
from .data import board_from_entry
from .engines import Engines

RATINGS = C.RATINGS

# Colours shared with the charts (colour-blind-safe set)
MAIA = "#2a78d6"
SF = "#1baf7a"
ME = "#eb6834"
BG = "#1e1f22"
PANEL = "#2b2d31"
INK = "#f0efec"
INK2 = "#b5b4ad"
LIGHT_SQ = "#f0d9b5"
DARK_SQ = "#b58863"
SEL_SQ = "#f6f669"

SQ = 84  # board square size in px; shrunk in main() to fit small screens
SF_DEPTH = 16  # deeper than the analysis (14): the GUI analyses one position at a time
GLYPH = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}


def pick_font(root, preferred):
    fams = set(tkfont.families(root))
    return next((f for f in preferred if f in fams), "TkDefaultFont")


# ------------------------------------------------------------------ engine worker
class EngineWorker(threading.Thread):
    """Owns the engines; runs jobs off the UI thread and posts results back on a queue."""

    def __init__(self, results: queue.Queue):
        super().__init__(daemon=True)
        self.jobs = queue.Queue()
        self.results = results

    def run(self):
        with Engines() as eng:
            while True:
                kind, token, payload = self.jobs.get()
                if kind == "quit":
                    return
                try:
                    if kind == "analyse":
                        board = payload  # a copy, including move history when we have it
                        dists = {}
                        for r in RATINGS:
                            dists[r] = eng.maia_distribution(board, r)
                            # Maia is fast; post a partial result so the panel fills immediately
                            if r == RATINGS[-1] or len(dists) == 1:
                                self.results.put(("maia", token, dict(dists)))
                        sf = eng.stockfish_analysis(board, depth=SF_DEPTH, multipv=3)
                        self.results.put(("sf", token, sf))
                    elif kind == "move_cp":
                        board, move = payload
                        cp = eng.stockfish_move_cp(board, move, depth=SF_DEPTH)
                        self.results.put(("move_cp", token, (move, cp)))
                except Exception as exc:  # surface engine errors in the UI instead of dying silently
                    self.results.put(("error", token, repr(exc)))

    def submit(self, kind, token, payload=None):
        self.jobs.put((kind, token, payload))


# ------------------------------------------------------------------ app
class App:
    def __init__(self, root, positions):
        self.root = root
        self.positions = positions
        self.idx = 0
        self.rating = tk.IntVar(value=1500)
        self.token = 0
        self.maia = {}
        self.sf = None
        self.my_move = None
        self.my_cp = None
        self.selected = None
        self.session = {}  # fen -> {"maia": bool, "sf": bool, "p": float, "cp": int}

        self.results = queue.Queue()
        self.worker = EngineWorker(self.results)
        self.worker.start()

        ui = pick_font(root, ["Segoe UI", "Noto Sans", "DejaVu Sans", "Helvetica"])
        self.f_piece = (pick_font(root, ["Segoe UI Symbol", "DejaVu Sans", "Noto Sans Symbols2"]), int(SQ * 0.62))
        self.f_title = (ui, 20, "bold")
        self.f_h = (ui, 17, "bold")
        self.f = (ui, 15)
        self.f_small = (ui, 13)
        self.f_mono = (pick_font(root, ["Cascadia Mono", "Consolas", "DejaVu Sans Mono"]), 15)
        self.f_verdict = (ui, 17, "bold")

        root.title("Maia vs Stockfish vs Me")
        root.configure(bg=BG)
        self._build()
        self.load(0)
        root.after(50, self._poll)
        root.protocol("WM_DELETE_WINDOW", self.quit)

    # ---------------- layout
    def _build(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TButton", font=self.f, padding=(12, 6), width=-1, background=PANEL, foreground=INK, bordercolor="#44464c"
        )
        style.map("TButton", background=[("active", "#3a3c42")])
        # rating buttons as flat toggles (no radio dot) so all nine fit on one row
        style.layout(
            "Rating.TRadiobutton",
            [("Radiobutton.padding", {"sticky": "nswe", "children": [("Radiobutton.label", {"sticky": "nswe"})]})],
        )
        style.configure(
            "Rating.TRadiobutton",
            font=self.f_small,
            background="#3a3c42",
            foreground=INK,
            padding=(7, 5),
            anchor="center",
        )
        style.map(
            "Rating.TRadiobutton",
            background=[("selected", MAIA), ("active", "#3a3c42")],
            indicatorcolor=[("selected", MAIA)],
        )

        left = tk.Frame(self.root, bg=BG)
        left.grid(row=0, column=0, sticky="n", padx=(24, 12), pady=18)
        right = tk.Frame(self.root, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 24), pady=18)
        self.root.columnconfigure(1, weight=1)

        # --- left: title, board, navigation
        self.lbl_title = tk.Label(
            left, text="", font=self.f_title, fg=INK, bg=BG, anchor="w", justify="left", wraplength=8 * SQ + 28
        )
        self.lbl_title.pack(fill="x")
        self.lbl_sub = tk.Label(
            left, text="", font=self.f, fg=INK2, bg=BG, anchor="w", justify="left", wraplength=8 * SQ + 28
        )
        self.lbl_sub.pack(fill="x", pady=(0, 8))
        self.canvas = tk.Canvas(left, width=8 * SQ + 28, height=8 * SQ + 28, bg=BG, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        legend = tk.Frame(left, bg=BG)
        legend.pack(fill="x", pady=(6, 4))
        for colour, text in ((MAIA, "Maia top move"), (SF, "Stockfish best"), (ME, "My move")):
            tk.Label(legend, text="━━", font=self.f_h, fg=colour, bg=BG).pack(side="left")
            tk.Label(legend, text=text + "    ", font=self.f, fg=INK, bg=BG).pack(side="left")

        nav = tk.Frame(left, bg=BG)
        nav.pack(fill="x", pady=(6, 0))
        ttk.Button(nav, text="◀ Prev", command=lambda: self.load(self.idx - 1)).pack(side="left")
        ttk.Button(nav, text="Next ▶", command=lambda: self.load(self.idx + 1)).pack(side="left", padx=6)
        ttk.Button(nav, text="Open PGN…", command=self.open_pgn).pack(side="left", padx=6)
        self.lbl_count = tk.Label(nav, text="", font=self.f, fg=INK2, bg=BG)
        self.lbl_count.pack(side="right")

        fenrow = tk.Frame(left, bg=BG)
        fenrow.pack(fill="x", pady=(8, 0))
        self.fen_var = tk.StringVar()
        e = tk.Entry(
            fenrow, textvariable=self.fen_var, font=self.f_small, bg=PANEL, fg=INK, insertbackground=INK, relief="flat"
        )
        e.pack(side="left", fill="x", expand=True, ipady=6)
        e.bind("<Return>", lambda _: self.load_fen())
        ttk.Button(fenrow, text="Load FEN", command=self.load_fen).pack(side="left", padx=(6, 0))

        # --- right: rating selector
        box = self._card(right, "Maia rating")
        rrow = tk.Frame(box, bg=PANEL)
        rrow.pack(fill="x")
        for r in RATINGS:
            ttk.Radiobutton(
                rrow, text=str(r), value=r, variable=self.rating, style="Rating.TRadiobutton", command=self.redraw
            ).pack(side="left", padx=2)

        # Maia distribution
        box = self._card(right, None)
        self.lbl_maia = tk.Label(box, text="Maia", font=self.f_h, fg=MAIA, bg=PANEL, anchor="w")
        self.lbl_maia.pack(fill="x")
        self.maia_canvas = tk.Canvas(box, width=520, height=5 * 30 + 4, bg=PANEL, highlightthickness=0)
        self.maia_canvas.pack(fill="x", pady=(4, 0))

        # Stockfish
        box = self._card(right, None)
        tk.Label(box, text=f"Stockfish 17.1 · depth {SF_DEPTH}", font=self.f_h, fg=SF, bg=PANEL, anchor="w").pack(
            fill="x"
        )
        self.sf_text = tk.Label(box, text="", font=self.f_mono, fg=INK, bg=PANEL, anchor="w", justify="left")
        self.sf_text.pack(fill="x", pady=(4, 0))

        # My move + scoreboard side by side
        bottom = tk.Frame(right, bg=BG)
        bottom.pack(fill="x")
        box = self._card(bottom, None, side="left")
        tk.Label(box, text="My move", font=self.f_h, fg=ME, bg=PANEL, anchor="w").pack(fill="x")
        mrow = tk.Frame(box, bg=PANEL)
        mrow.pack(fill="x", pady=(4, 6))
        self.move_var = tk.StringVar()
        me = tk.Entry(
            mrow, textvariable=self.move_var, font=self.f, width=10, bg=BG, fg=INK, insertbackground=INK, relief="flat"
        )
        me.pack(side="left", ipady=5)
        me.bind("<Return>", lambda _: self.submit_san())
        ttk.Button(mrow, text="Play", command=self.submit_san).pack(side="left", padx=6)
        ttk.Button(mrow, text="Clear", command=self.clear_move).pack(side="left")
        tk.Label(
            box, text="Type SAN, or click a piece then a square", font=self.f_small, fg=INK2, bg=PANEL, anchor="w"
        ).pack(fill="x", pady=(0, 6))
        self.cmp_text = tk.Label(
            box, text="", font=self.f_small, fg=INK, bg=PANEL, anchor="w", justify="left", wraplength=370
        )
        self.cmp_text.pack(fill="x")
        self.verdict = tk.Label(
            box, text="", font=self.f_verdict, fg=INK, bg=PANEL, anchor="w", justify="left", wraplength=370
        )
        self.verdict.pack(fill="x", pady=(6, 0))

        # Scoreboard
        box = self._card(bottom, "Session scoreboard", side="left")
        self.score_text = tk.Label(box, text="", font=self.f, fg=INK, bg=PANEL, anchor="w", justify="left")
        self.score_text.pack(fill="x")

        self.status = tk.Label(right, text="", font=self.f_small, fg=INK2, bg=BG, anchor="w")
        self.status.pack(fill="x", pady=(6, 0))

    def _card(self, parent, title, side=None):
        frame = tk.Frame(parent, bg=PANEL, padx=16, pady=10)
        if side:
            frame.pack(side=side, fill="both", expand=True, pady=(0, 10), padx=(0, 10))
        else:
            frame.pack(fill="x", pady=(0, 10))
        if title:
            tk.Label(frame, text=title, font=self.f_h, fg=INK, bg=PANEL, anchor="w").pack(fill="x", pady=(0, 4))
        return frame

    # ---------------- loading positions
    def load(self, idx):
        if not self.positions:
            return
        self.idx = idx % len(self.positions)
        p = self.positions[self.idx]
        self.board = board_from_entry(p)
        self.fen_var.set(p["fen"])
        self.lbl_title.config(text=p.get("label", "Position"))
        side = "White" if self.board.turn == chess.WHITE else "Black"
        self.lbl_sub.config(text=f"{side} to move" + (f"  ·  {p['source']}" if p.get("source") else ""))
        self.lbl_count.config(text=f"{self.idx + 1} / {len(self.positions)}")
        self.move_var.set(p.get("my_move", "") or "")
        self.maia, self.sf, self.my_move, self.my_cp, self.selected = {}, None, None, None, None
        self.token += 1
        self.worker.submit("analyse", self.token, self.board.copy())
        self.status.config(text="Analysing…")
        self.redraw()

    def load_fen(self):
        fen = self.fen_var.get().strip()
        try:
            chess.Board(fen)
        except ValueError as exc:
            self.status.config(text=f"Invalid FEN: {exc}")
            return
        self.positions.insert(self.idx + 1, {"fen": fen, "label": "Custom position"})
        self.load(self.idx + 1)

    def open_pgn(self, path=None):
        path = path or filedialog.askopenfilename(filetypes=[("PGN", "*.pgn"), ("All files", "*")])
        if not path:
            return
        self.positions = positions_from_pgn(Path(path))
        self.load(0)

    # ---------------- my move
    def on_click(self, event):
        f, r = (event.x - 28) // SQ, 7 - event.y // SQ
        if not (0 <= f < 8 and 0 <= r < 8):
            return
        sq = chess.square(f, r)
        piece = self.board.piece_at(sq)
        if self.selected is None:
            if piece and piece.color == self.board.turn:
                self.selected = sq
        else:
            move = chess.Move(self.selected, sq)
            if self.board.piece_type_at(self.selected) == chess.PAWN and r in (0, 7):
                move.promotion = chess.QUEEN
            if move in self.board.legal_moves:
                self.set_my_move(move)
                self.selected = None
            elif piece and piece.color == self.board.turn:
                self.selected = sq
            else:
                self.selected = None
        self.redraw()

    def submit_san(self):
        text = self.move_var.get().strip()
        if not text:
            return
        try:
            self.set_my_move(self.board.parse_san(text))
        except ValueError:
            self.status.config(text=f"'{text}' is not a legal move here")

    def clear_move(self):
        self.my_move, self.my_cp = None, None
        self.move_var.set("")
        self.session.pop(self.board.fen(), None)
        self.redraw()

    def set_my_move(self, move):
        self.my_move, self.my_cp = move, None
        self.move_var.set(self.board.san(move))
        self.resolve_cp()
        self.redraw()

    def resolve_cp(self):
        """Centipawn loss of my move: free if it's in Stockfish's top 3, otherwise ask the worker."""
        if self.my_move is None or self.sf is None:
            return
        for line in self.sf:
            if line.move == self.my_move:
                self.my_cp = line.cp
                return
        self.worker.submit("move_cp", self.token, (self.board.copy(), self.my_move))
        self.status.config(text="Evaluating your move…")

    # ---------------- worker results
    def _poll(self):
        try:
            while True:
                kind, token, payload = self.results.get_nowait()
                if token != self.token:
                    continue  # stale result for a position we've left
                if kind == "maia":
                    self.maia = payload
                elif kind == "sf":
                    self.sf = payload
                    self.status.config(text="")
                    self.resolve_cp()
                elif kind == "move_cp":
                    move, cp = payload
                    if move == self.my_move:
                        self.my_cp = cp
                        self.status.config(text="")
                elif kind == "error":
                    self.status.config(text=f"Engine error: {payload}")
                self.redraw()
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    # ---------------- drawing
    def redraw(self):
        self.draw_board()
        self.draw_maia()
        self.draw_sf()
        self.draw_comparison()

    def sq_xy(self, sq):
        """Centre of a square in canvas coordinates."""
        f, r = chess.square_file(sq), chess.square_rank(sq)
        return 28 + f * SQ + SQ / 2, (7 - r) * SQ + SQ / 2

    def draw_board(self):
        c = self.canvas
        c.delete("all")
        for sq in chess.SQUARES:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            x0, y0 = 28 + f * SQ, (7 - r) * SQ
            colour = SEL_SQ if sq == self.selected else (LIGHT_SQ if (f + r) % 2 else DARK_SQ)
            if self.my_move and sq in (self.my_move.from_square, self.my_move.to_square):
                colour = "#f2b48a" if (f + r) % 2 else "#d98f5e"
            c.create_rectangle(x0, y0, x0 + SQ, y0 + SQ, fill=colour, outline="")
            piece = self.board.piece_at(sq)
            if piece:
                g = GLYPH[piece.symbol().lower()]
                cx, cy = x0 + SQ / 2, y0 + SQ / 2 + 2
                if piece.color == chess.WHITE:
                    # white pieces: white glyph with a dark outline so they read on light squares
                    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)):
                        c.create_text(cx + dx, cy + dy, text=g, font=self.f_piece, fill="#222")
                    c.create_text(cx, cy, text=g, font=self.f_piece, fill="#ffffff")
                else:
                    c.create_text(cx, cy, text=g, font=self.f_piece, fill="#111111")
        for i in range(8):
            c.create_text(14, i * SQ + SQ / 2, text=str(8 - i), font=self.f_small, fill=INK2)
            c.create_text(28 + i * SQ + SQ / 2, 8 * SQ + 14, text="abcdefgh"[i], font=self.f_small, fill=INK2)

        arrows = []
        if self.maia.get(self.rating.get()):
            arrows.append((next(iter(self.maia[self.rating.get()])), MAIA))
        if self.sf:
            arrows.append((self.sf[0].move, SF))
        if self.my_move:
            arrows.append((self.my_move, ME))
        # arrows that share a move are spread side by side so every colour stays visible
        for move, colour in arrows:
            # group by squares, not Move, so c8=Q and c8=R arrows sit side by side too
            same = [c for m, c in arrows if (m.from_square, m.to_square) == (move.from_square, move.to_square)]
            i = same.index(colour)
            self.draw_arrow(move, colour, offset=(i - (len(same) - 1) / 2) * 12)

    def draw_arrow(self, move, colour, offset=0):
        (x0, y0), (x1, y1) = self.sq_xy(move.from_square), self.sq_xy(move.to_square)
        dx, dy = x1 - x0, y1 - y0
        length = max(1.0, (dx * dx + dy * dy) ** 0.5)
        nx, ny = -dy / length * offset, dx / length * offset
        # start a little off the piece centre, end just short of the target centre
        sx, sy = x0 + dx / length * SQ * 0.25 + nx, y0 + dy / length * SQ * 0.25 + ny
        ex, ey = x1 - dx / length * SQ * 0.18 + nx, y1 - dy / length * SQ * 0.18 + ny
        self.canvas.create_line(
            sx,
            sy,
            ex,
            ey,
            fill="#000000",
            width=15,
            arrow="last",
            arrowshape=(30, 34, 15),
            capstyle="round",
            stipple="gray50",
        )
        self.canvas.create_line(
            sx, sy, ex, ey, fill=colour, width=11, arrow="last", arrowshape=(26, 30, 12), capstyle="round"
        )

    def draw_maia(self):
        c = self.maia_canvas
        c.delete("all")
        r = self.rating.get()
        self.lbl_maia.config(text=f"Maia {r} · raw policy, no search")
        dist = self.maia.get(r)
        if not dist:
            c.create_text(4, 20, text="thinking…", anchor="w", font=self.f, fill=INK2)
            return
        width = max(c.winfo_width(), 520)
        bar_x, bar_w = 110, width - 110 - 90
        sf_best = self.sf[0].move if self.sf else None
        for i, (move, p) in enumerate(list(dist.items())[:5]):
            y = 2 + i * 30
            colour = ME if move == self.my_move else MAIA
            name = self.board.san(move)
            c.create_text(4, y + 14, text=name, anchor="w", font=self.f_h, fill=SF if move == sf_best else INK)
            c.create_rectangle(bar_x, y + 4, bar_x + bar_w, y + 26, fill="#3a3c42", outline="")
            c.create_rectangle(bar_x, y + 4, bar_x + max(3, bar_w * p), y + 26, fill=colour, outline="")
            c.create_text(bar_x + bar_w + 10, y + 14, text=f"{p:.1%}", anchor="w", font=self.f_h, fill=INK)

    def draw_sf(self):
        if not self.sf:
            self.sf_text.config(text="thinking…", fg=INK2)
            return
        lines = []
        for i, line in enumerate(self.sf, 1):
            pv = self.board.variation_san(line.pv[:4])
            lines.append(f"{i}. {self.board.san(line.move):<7} {line.eval_str:>7}   {pv}")
        self.sf_text.config(text="\n".join(lines), fg=INK)

    def draw_comparison(self):
        r = self.rating.get()
        dist = self.maia.get(r)
        if self.my_move is None or not dist or not self.sf:
            self.cmp_text.config(text="")
            self.verdict.config(text="")
            self.update_scoreboard()
            return
        maia_top = next(iter(dist))
        sf_best = self.sf[0].move
        p = dist.get(self.my_move, 0.0)
        rank = list(dist).index(self.my_move) + 1
        match_maia, match_sf = self.my_move == maia_top, self.my_move == sf_best
        cp_loss = None if self.my_cp is None else (0 if match_sf else max(0, self.sf[0].cp - self.my_cp))
        sf_mates = self.sf[0].score.pov(self.board.turn).is_mate() and self.sf[0].score.pov(self.board.turn).mate() > 0
        missed_mate = sf_mates and not match_sf and self.my_cp is not None and self.my_cp < self.sf[0].cp
        if cp_loss is None:
            cp_txt = "…"
        elif missed_mate:
            cp_txt = "missed a forced mate"
        else:
            cp_txt = f"{cp_loss} cp" + ("  (a mistake)" if cp_loss >= 100 else "")
        self.cmp_text.config(
            text=(
                f"Maia {r} gave {self.board.san(self.my_move)}: {p:.1%} (rank {rank}/{len(dist)})\n"
                f"Maia top: {self.board.san(maia_top)}   ·   Stockfish best: {self.board.san(sf_best)}\n"
                f"My cp loss: {cp_txt}"
            )
        )
        if match_maia and match_sf:
            v, colour = "✔ Matches both Maia and Stockfish", INK
        elif match_maia:
            v, colour = f"✔ Human-like: matches Maia {r}, not Stockfish", MAIA
        elif match_sf:
            v, colour = f"✔ Engine-like: matches Stockfish, not Maia {r}", SF
        else:
            v, colour = f"✘ Matches neither. Maia still gave it {p:.0%}", ME
        self.verdict.config(text=v, fg=colour)
        self.session[self.board.fen()] = {"maia": match_maia, "sf": match_sf, "p": p, "cp": cp_loss}
        self.update_scoreboard()

    def update_scoreboard(self):
        n = len(self.session)
        if n == 0:
            self.score_text.config(text="Play a move to start scoring.")
            return
        m = sum(s["maia"] for s in self.session.values())
        s_ = sum(s["sf"] for s in self.session.values())
        avg_p = sum(s["p"] for s in self.session.values()) / n
        cps = [s["cp"] for s in self.session.values() if s["cp"] is not None]
        avg_cp = f"{sum(cps) / len(cps):.0f} cp" if cps else "…"
        self.score_text.config(
            text=(
                f"Positions played: {n}\n"
                f"Matched Maia: {m}/{n} ({m / n:.0%})\n"
                f"Matched Stockfish: {s_}/{n} ({s_ / n:.0%})\n"
                f"Avg Maia p(my move): {avg_p:.0%}\n"
                f"Avg cp loss: {avg_cp}"
            )
        )

    def quit(self):
        self.worker.submit("quit", 0)
        self.root.after(200, self.root.destroy)


# ------------------------------------------------------------------ position sources
def load_positions_json():
    path = C.DEMO_POSITIONS
    if not path.exists():
        return [{"fen": chess.STARTING_FEN, "label": "Starting position"}]
    return json.loads(path.read_text())


def positions_from_pgn(path: Path, max_positions=400):
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        while len(out) < max_positions and (game := chess.pgn.read_game(f)) is not None:
            h = game.headers
            board = game.board()
            for move in game.mainline_moves():
                out.append(
                    {
                        "fen": board.fen(),
                        "my_move": board.san(move),
                        "history": [m.uci() for m in board.move_stack] if "FEN" not in h else [],
                        "label": f"{h.get('White', '?')} vs {h.get('Black', '?')}",
                        "source": f"move {board.fullmove_number} · game move pre-filled, press Play",
                    }
                )
                board.push(move)
    return out or [{"fen": chess.STARTING_FEN, "label": "Starting position"}]


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m maia_study gui", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pgn", type=Path, help="step through the positions of this PGN instead of positions.json")
    args = ap.parse_args(argv)
    root = tk.Tk()
    fit_to_screen(root)
    positions = positions_from_pgn(args.pgn) if args.pgn else load_positions_json()
    App(root, positions)
    root.mainloop()


def fit_to_screen(root):
    """Shrink the board on small/laptop screens so the whole window fits."""
    global SQ
    SQ = max(56, min(84, (root.winfo_screenheight() - 380) // 8, (root.winfo_screenwidth() - 800) // 8))
