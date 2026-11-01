"""The video assets (clips and stills) for the slides, built without running any engine.

    python -m maia_study video                    # running example = game 4hGkJ3rK, move 20
    python -m maia_study video --running-move 21  # the position where Bxb4 was played
    python -m maia_study video --qa               # also write frame contact sheets to figures/video/qa/

Everything is read from the saved analysis (via stats.py), results/headline_numbers.json,
results/showcase/showcase.json and results/cache/showcase_full_dists.json.

Outputs (figures/video/):
  rating_sweep_<name>.mp4 / .gif   board + arrows, Maia top-5 bars, rating 1100 -> 1900 -> 1100 (loops)
  reveal/1_board.png .. 4_mine.png running example, one layer per slide click (1920x1080)
  top1_vs_rating.mp4 / .gif / .png top-1 match by Maia rating, drawn in left to right
  manifest.json                    what each asset shows and where every number came from

Video spec: 1280x720, 30 fps, H.264 yuv420p, +faststart, no audio. The bottom-right face-cam box
(x >= 75% and y >= 75% of the frame, a margin around the requested 20%) is kept empty; every frame
that is checked has its text, bars and arrows tested against that box and the frame edges.
"""

import argparse
import json
import math
import subprocess
import sys
from io import BytesIO

import cairosvg
import chess
import chess.svg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, Rectangle
from matplotlib.text import Text
from PIL import Image
from rich.console import Console
from rich.table import Table

from . import config as C
from . import stats

ROOT = C.ROOT
OUT = C.VIDEO_DIR
QA = OUT / "qa"  # optional contact sheets (--qa), not committed
console = Console(width=120)

W, H, FPS = 1280, 720, 30
DPI = 100  # 12.8 x 7.2 in -> 1280 x 720 px
PNG_DPI = 150  # stills: same framing at 1920 x 1080
SAFE_X, SAFE_Y = 0.75 * W, 0.75 * H  # face-cam box: x >= SAFE_X and y >= SAFE_Y (from the top)
T = C.THEMES["dark"]
BEST = 1100  # best-fitting Maia (headline_numbers.json best_rating.by_top1)
RATINGS = C.RATINGS
RUNNING_GAME = "4hGkJ3rK"

HOLD, TRANS = 13, 8  # frames per rating level: 16 levels x 21 frames = 11.2 s
CHART_DRAW_S, CHART_HOLD_S = 6.0, 1.5


def pt(px):
    """Font size in points for a size given in pixels at 1280x720."""
    return px * 72 / DPI


FS = {"title": pt(40), "sub": pt(30), "body": pt(30), "rating": pt(54), "small": pt(30)}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": T["bg"],
        "axes.facecolor": T["bg"],
        "savefig.facecolor": T["bg"],
        "text.color": T["ink"],
        "axes.edgecolor": T["grid"],
    }
)


def frac(x, y_top, w=0, h=0):
    """Pixel box (left, top, width, height) at 1280x720 -> figure fraction [left, bottom, width, height]."""
    return [x / W, 1 - (y_top + h) / H, w / W, h / H]


def ease(s):
    return s * s * (3 - 2 * s)


def san(board, uci):
    return board.san(chess.Move.from_uci(uci))


def pct(p):
    return f"{p * 100:.0f}%" if p >= 0.095 else f"{p * 100:.1f}%"


# ------------------------------------------------------------------ data (read only)
RECORDS = stats.load_records()
DF, _, _ = stats.clean()
HL = json.loads((C.HEADLINE).read_text())
SHOW = json.loads((C.SHOWCASE_JSON).read_text())
DISTS = json.loads(C.SHOWCASE_DISTS.read_text())
assert HL["best_rating"]["by_top1"] == BEST


def record_for(fen=None, game_id=None, move_number=None):
    for r in RECORDS:
        if fen is not None and r["fen"] == fen:
            return r
        if game_id is not None and r["game_id"] == game_id and int(r["fen"].split()[5]) == move_number:
            return r
    raise SystemExit(f"position not in the analysis cache: {fen or (game_id, move_number)}")


def position(r, name, title, why):
    """Everything the renderers need for one position. Maia values: the full distribution where one is
    cached (showcase/fig6 positions), otherwise the per-rating top 5 stored in the analysis cache."""
    board = chess.Board(r["fen"])
    full = DISTS.get(r["fen"])
    levels = {}
    for rt in RATINGS:
        if full:
            dist = full[str(rt)]
            top5 = sorted(dist.items(), key=lambda kv: -kv[1])[:5]
        else:
            dist = None
            top5 = [tuple(x) for x in r["maia"][str(rt)]["top5"]]
        levels[rt] = {"top5": top5, "dist": dist, "top1": top5[0][0], "p_me": r["maia"][str(rt)]["p_human"]}
    in_clean = bool(((DF.game_id == r["game_id"]) & (DF.fen == r["fen"])).any())
    return {
        "name": name,
        "title": title,
        "why": why,
        "rec": r,
        "board": board,
        "levels": levels,
        "me": r["move_uci"],
        "sf": r["sf_best"],
        "source": "full distribution" if full else "cached top 5",
        "in_clean_set": in_clean,
    }


def pick_positions(running_move):
    r = record_for(game_id=RUNNING_GAME, move_number=running_move)
    out = [
        position(
            r,
            "running_example",
            "What would each Maia play here?",
            f"Running example: game {RUNNING_GAME}, move {running_move}.",
        )
    ]

    # Maia and I agree, Stockfish differs, and my move is not a mistake (< 50 cp), so this does not
    # double as the mistake example. Rule: top-1 of EVERY Maia rating = my move; pick the position with
    # the highest minimum probability across ratings (the most unanimous agreement).
    d = DF[(DF.move_uci != DF.sf_best) & (DF.cp_loss < 50)].copy()
    d = d[d[[f"top1_{rt}" for rt in RATINGS]].eq(d.move_uci, axis=0).all(axis=1)]
    d["minp"] = d[[f"p_{rt}" for rt in RATINGS]].min(axis=1)
    a = d.sort_values("minp", ascending=False).iloc[0]
    out.append(
        position(
            record_for(fen=a.fen),
            "agree",
            "Maia and I agree, Stockfish doesn't",
            f"Rule: every Maia rating's top move = my move, my move != Stockfish's best, cp loss < 50; "
            f"highest minimum probability across ratings ({len(d)} candidates in the clean set).",
        )
    )

    m = next(p for p in SHOW["positions"] if p["slug"] == "mistake")
    out.append(
        position(
            record_for(fen=m["fen"]),
            "mistake",
            "Maia saw my mistake coming",
            "Showcase position 2 (final/showcase.py): cp loss >= 150 and Maia 1100's top choice, highest probability.",
        )
    )

    s = SHOW["sweep"]
    out.append(
        position(
            record_for(fen=s["fen"]),
            "1100_vs_1900",
            "Maia 1100 and Maia 1900 disagree",
            f"The fig6 position (final/showcase.py): largest total-variation distance between the "
            f"Maia 1100 and 1900 distributions among positions whose top move changes "
            f"(TV = {s['tv_1100_vs_1900']:.2f}).",
        )
    )
    return out


def subtitle(p):
    r, b = p["rec"], p["board"]
    side = "White" if b.turn else "Black"
    return f"Game {r['game_id']} · move {b.fullmove_number} · {side} to move · I played {san(b, p['me'])}"


# ------------------------------------------------------------------ board + arrows
BOARD_BOX = (20, 108, 592)  # left, top, size in px at 1280x720
_board_cache = {}


def board_image(board, tint_me, me, size_px):
    key = (board.fen(), tint_me, size_px)
    if key not in _board_cache:
        fill = {}
        if tint_me:
            mv = chess.Move.from_uci(me)
            fill = {mv.from_square: T["me"] + "66", mv.to_square: T["me"] + "99"}
        svg = chess.svg.board(
            board,
            fill=fill,
            size=size_px,
            orientation=board.turn,
            coordinates=True,
            colors={"square light": "#f0d9b5", "square dark": "#b58863", "margin": T["bg"], "coord": T["ink2"]},
        )
        png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size_px, output_height=size_px)
        _board_cache[key] = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
    return _board_cache[key]


def square_xy(sq, white_bottom):
    """Centre of a square in board units: margin 15 + 8 x 45 squares + margin 15 = 390 (chess.svg layout)."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    if not white_bottom:
        f, r = 7 - f, 7 - r
    return 15 + 45 * f + 22.5, 15 + 45 * (7 - r) + 22.5


def arrow_specs(p, maia_uci, show):
    """[(uci, colour_key)] in draw order; coinciding arrows get side-by-side offsets so none is hidden."""
    specs = []
    if "sf" in show:
        specs.append((p["sf"], "sf"))
    if "maia" in show:
        specs.append((maia_uci, "maia"))
    if "me" in show:
        specs.append((p["me"], "me"))
    by_move = {}
    for u, c in specs:
        by_move.setdefault(u, []).append(c)
    out = []
    for u, cols in by_move.items():
        n = len(cols)
        for i, c in enumerate(cols):
            out.append((u, c, (i - (n - 1) / 2) * 11.0, n))
    return out


def draw_arrows(ax, p, arrows, alpha=1.0):
    white_bottom = p["board"].turn
    for u, c, off, n in arrows:
        mv = chess.Move.from_uci(u)
        x0, y0 = square_xy(mv.from_square, white_bottom)
        x1, y1 = square_xy(mv.to_square, white_bottom)
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        nx, ny = -dy / L, dx / L
        width = 9.0 if n == 1 else 7.0
        ax.add_patch(
            FancyArrow(
                x0 + nx * off,
                y0 + ny * off,
                dx,
                dy,
                width=width,
                head_width=width * 2.4,
                head_length=14,
                length_includes_head=True,
                color=T[c],
                alpha=0.92 * alpha,
                lw=0,
                zorder=3,
            )
        )


def board_axes(fig, p, tint_me, dpi):
    left, top, size = BOARD_BOX
    ax = fig.add_axes(frac(left, top, size, size))
    img = board_image(p["board"], tint_me, p["me"], int(size * dpi / DPI))
    ax.imshow(img, extent=(0, 390, 390, 0), interpolation="antialiased")
    ax.set_xlim(0, 390)
    ax.set_ylim(390, 0)
    ax.axis("off")
    return ax


def header(fig, title, sub):
    fig.text(20 / W, 1 - 12 / H, title, fontsize=FS["title"], weight="bold", color=T["ink"], va="top")
    fig.text(20 / W, 1 - 62 / H, sub, fontsize=FS["sub"], color=T["ink2"], va="top")


# ------------------------------------------------------------------ bars
BARS_BOX = (650, 270, 600, 260)  # left, top, width, height
LABEL_X, BAR_X0, BAR_X1 = 140, 150, 510  # inside BARS_BOX, px


def bar_state(p, rt):
    lv = p["levels"][rt]
    return {u: (i, pr) for i, (u, pr) in enumerate(lv["top5"])}


def value_at(p, rt, u):
    lv = p["levels"][rt]
    if lv["dist"] is not None:
        return lv["dist"].get(u, 0.0)
    for v, pr in lv["top5"]:
        if v == u:
            return pr
    if u == p["me"]:
        return lv["p_me"]
    return None  # outside the cached top 5: unknown


def draw_bars(fig, p, ra, rb, s, tags=("sf", "me")):
    """Bars between rating ra (s=0) and rb (s=1). Moves entering/leaving the top 5 slide in/out of view."""
    left, top, w, h = BARS_BOX
    ax = fig.add_axes(frac(left, top, w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(4.5, -0.5)
    ax.axis("off")
    A, B = bar_state(p, ra), bar_state(p, rb)
    b = p["board"]
    for u in list(dict.fromkeys(list(A) + list(B))):
        ya = A[u][0] if u in A else 5.6
        yb = B[u][0] if u in B else 5.6
        va = A[u][1] if u in A else value_at(p, ra, u)
        vb = B[u][1] if u in B else value_at(p, rb, u)
        va = vb if va is None else va
        vb = va if vb is None else vb
        alpha = 1.0 if (u in A and u in B) else (1 - s if u in A else s)
        y = ya + (yb - ya) * s
        v = va + (vb - va) * s
        ax.add_patch(
            Rectangle((BAR_X0, y - 0.32), (BAR_X1 - BAR_X0) * v, 0.64, color=T["maia"], alpha=alpha, lw=0, clip_on=True)
        )
        ax.text(
            LABEL_X,
            y,
            san(b, u),
            ha="right",
            va="center",
            fontsize=FS["body"],
            color=T["ink"],
            alpha=alpha,
            weight="bold",
            clip_on=True,
        )
        ax.text(
            BAR_X0 + (BAR_X1 - BAR_X0) * v + 8,
            y,
            pct(v),
            ha="left",
            va="center",
            fontsize=FS["body"],
            color=T["ink"],
            alpha=alpha,
            clip_on=True,
        )
        dots = [c for c in tags if p[c] == u]
        for i, c in enumerate(dots):
            ax.plot([10 + 22 * i], [y], "o", ms=13, color=T[c], alpha=alpha, clip_on=True)
    return ax


def legend(fig):
    for i, (c, label) in enumerate((("maia", "Maia's top move"), ("sf", "Stockfish's best"), ("me", "My move"))):
        y = 1 - (562 + 44 * i) / H
        fig.text(652 / W, y, "━━", fontsize=FS["small"], color=T[c], va="center", weight="bold")
        fig.text(712 / W, y, label, fontsize=FS["small"], color=T["ink2"], va="center")


def rating_header(fig, rt_label, pos):
    """'Maia 1500' plus a track with a dot at fractional level index pos (0 = 1100 ... 8 = 1900)."""
    fig.text(650 / W, 1 - 106 / H, f"Maia {rt_label}", fontsize=FS["rating"], weight="bold", color=T["maia"], va="top")
    ax = fig.add_axes(frac(650, 186, 600, 70))
    ax.set_xlim(-0.3, 8.3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0, 8], [0.75, 0.75], color=T["grid"], lw=5, solid_capstyle="round")
    ax.plot(range(9), [0.75] * 9, "o", ms=8, color=T["neutral"])
    ax.plot([pos], [0.75], "o", ms=22, color=T["maia"], mec=T["ink"], mew=2)
    for i, lab in ((0, "1100"), (4, "1500"), (8, "1900")):
        ax.text(i, 0.05, lab, ha="center", va="bottom", fontsize=FS["small"], color=T["ink2"])


# ------------------------------------------------------------------ layout checks
class LayoutError(Exception):
    pass


def check_layout(fig, where):
    """Every visible text, patch, line and image must sit inside the frame and outside the face-cam box."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fw, fh = fig.bbox.width, fig.bbox.height
    sx, sy = SAFE_X / W * fw, (1 - SAFE_Y / H) * fh  # display coords: origin bottom-left
    artists = list(fig.texts)
    for ax in fig.axes:
        artists += ax.texts + ax.patches + ax.lines + ax.images + ax.collections
        if ax.axison:
            artists += ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label]
    problems, texts = [], []
    for a in artists:
        if not a.get_visible() or (a.get_alpha() is not None and a.get_alpha() < 0.05):
            continue
        if isinstance(a, Text) and not a.get_text().strip():
            continue
        bb = a.get_window_extent(rend)
        if a.axes is not None and a.get_clip_on():
            bb = bb.intersection(bb, a.axes.bbox)
        if bb is None or bb.width <= 0 or bb.height <= 0:
            continue
        name = a.get_text() if isinstance(a, Text) else a.__class__.__name__
        box = tuple(round(v) for v in (bb.x0, bb.y0, bb.x1, bb.y1))
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > fw + 0.5 or bb.y1 > fh + 0.5:
            problems.append(f"{name!r} leaves the frame {box}")
        if bb.x1 > sx and bb.y0 < sy:
            problems.append(f"{name!r} enters the face-cam box {box}")
        if isinstance(a, Text):
            for other, obb in texts:
                if bb.overlaps(obb):
                    problems.append(f"{name!r} overlaps {other!r}")
            texts.append((name, bb))
    if problems:
        raise LayoutError(f"{where}:\n  " + "\n  ".join(problems))


# ------------------------------------------------------------------ encoding
class Encoder:
    def __init__(self, path):
        self.path = path
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{W}x{H}",
                "-r",
                str(FPS),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-tune",
                "animation",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(FPS),
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self.n = 0

    def write(self, fig):
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        assert buf.shape == (H, W, 3), buf.shape
        self.proc.stdin.write(np.ascontiguousarray(buf).tobytes())
        self.n += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise SystemExit(f"ffmpeg failed for {self.path}")


def make_gif(mp4):
    gif = mp4.with_suffix(".gif")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-vf",
            "fps=15,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
            "-loop",
            "0",
            str(gif),
        ],
        check=True,
    )
    return gif


def new_fig(dpi=DPI):
    return plt.figure(figsize=(W / DPI, H / DPI), dpi=dpi)


# ------------------------------------------------------------------ asset 1: rating sweeps
def sweep_frame(fig, p, ia, ib, s):
    fig.clf()
    ra, rb = RATINGS[ia], RATINGS[ib]
    e = ease(s)
    header(fig, p["title"], subtitle(p))
    ax = board_axes(fig, p, True, DPI)
    ta, tb = p["levels"][ra]["top1"], p["levels"][rb]["top1"]
    A, B = arrow_specs(p, ta, ("sf", "maia", "me")), arrow_specs(p, tb, ("sf", "maia", "me"))
    if A == B or e == 0:
        draw_arrows(ax, p, A)
    elif e == 1:
        draw_arrows(ax, p, B)
    else:
        draw_arrows(ax, p, [x for x in A if x in B])
        draw_arrows(ax, p, [x for x in A if x not in B], 1 - e)
        draw_arrows(ax, p, [x for x in B if x not in A], e)
    rating_header(fig, ra if e < 0.5 else rb, ia + (ib - ia) * e)
    draw_bars(fig, p, ra, rb, e)
    legend(fig)


def build_sweep(p):
    path = OUT / f"rating_sweep_{p['name']}.mp4"
    order = list(range(9)) + list(range(7, 0, -1))  # 1100..1900..1200, then wraps to 1100
    fig = new_fig()
    enc = Encoder(path)
    for k, ia in enumerate(order):
        ib = order[(k + 1) % len(order)]
        for f in range(HOLD):
            sweep_frame(fig, p, ia, ia, 0.0)
            if f == 0:
                check_layout(fig, f"{path.name} @ Maia {RATINGS[ia]}")
            enc.write(fig)
        for f in range(TRANS):
            sweep_frame(fig, p, ia, ib, (f + 1) / (TRANS + 1))
            enc.write(fig)
    enc.close()
    plt.close(fig)
    return path, enc.n


# ------------------------------------------------------------------ asset 2: reveal layers
def reveal_layers(p):
    d = OUT / "reveal"
    d.mkdir(parents=True, exist_ok=True)
    b, lv = p["board"], p["levels"][BEST]
    maia_top = lv["top1"]
    r = p["rec"]
    me_line = (
        f"My move: {san(b, p['me'])} · same as Stockfish"
        if p["me"] == p["sf"]
        else f"My move: {san(b, p['me'])} · lost {r['cp_loss'] / 100:.1f} pawns"
    )
    lines = [
        ("maia", f"Maia {BEST}'s top move: {san(b, maia_top)}"),
        ("sf", f"Stockfish's best: {san(b, p['sf'])}"),
        ("me", me_line),
    ]
    layers = [("1_board", ()), ("2_maia", ("maia",)), ("3_stockfish", ("maia", "sf")), ("4_mine", ("maia", "sf", "me"))]
    out = []
    for fname, show in layers:
        fig = new_fig(PNG_DPI)
        header(
            fig,
            f"My game {r['game_id']}, move {b.fullmove_number}",
            f"{'White' if b.turn else 'Black'} to move · Maia {BEST} is the model that fits me best",
        )
        ax = board_axes(fig, p, "me" in show, PNG_DPI)
        draw_arrows(ax, p, arrow_specs(p, maia_top, show))
        for i, (c, text) in enumerate(lines):
            if c in show:
                fig.text(
                    650 / W, 1 - (112 + 50 * i) / H, text, fontsize=FS["body"], color=T[c], weight="bold", va="top"
                )
        if "maia" in show:
            draw_bars(fig, p, BEST, BEST, 0.0, tags=tuple(c for c in ("sf", "me") if c in show))
        check_layout(fig, f"reveal {fname}")
        path = d / f"{fname}.png"
        fig.savefig(path, dpi=PNG_DPI)
        plt.close(fig)
        out.append(path)
    return out, lines


# ------------------------------------------------------------------ asset 3: top-1 chart
def chart_frame(fig, cut, final=False):
    """cut = x position (rating units) the drawing has reached."""
    fig.clf()
    m = [HL["match"][str(rt)]["top1"] for rt in RATINGS]
    y = np.array([v["rate"] * 100 for v in m])
    lo = np.array([v["ci95"][0] * 100 for v in m])
    hi = np.array([v["ci95"][1] * 100 for v in m])
    sf = HL["match"]["stockfish"]["top1"]["rate"] * 100
    rnd = HL["match"]["random"]["top1_expected"] * 100
    meta = HL["meta"]
    header(
        fig,
        "Maia predicts my moves better than Stockfish",
        f"Top-1 match with the move I played · {meta['n_positions_clean']} positions, {meta['n_games_clean']} games",
    )
    ax = fig.add_axes(frac(120, 140, 800, 465))
    x0, x1 = 1050, 1950
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, 62)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(T["grid"])
    ax.spines["bottom"].set_color(T["grid"])
    ax.grid(axis="y", color=T["grid"], lw=1)
    ax.set_axisbelow(True)
    ax.set_xticks(RATINGS, [str(r) for r in RATINGS])
    ax.set_yticks([0, 20, 40, 60])
    ax.tick_params(colors=T["ink2"], labelsize=FS["small"], length=0, pad=8)
    ax.set_ylabel("Top-1 match (%)", fontsize=FS["small"], color=T["ink2"])
    ax.set_xlabel("Maia model (training rating)", fontsize=FS["small"], color=T["ink2"])

    xs = np.array(RATINGS, float)
    if cut > x0:
        # Maia line + 95% CI band, truncated at cut with an interpolated end point
        k = xs <= cut
        xx = list(xs[k])
        if cut < xs[-1]:
            xx.append(max(cut, xs[0]))
        xx = np.array(xx)
        if len(xx) and xx[-1] >= xs[0]:
            yy, ll, hh = (np.interp(xx, xs, v) for v in (y, lo, hi))
            ax.fill_between(xx, ll, hh, color=T["maia_soft"], alpha=0.45, lw=0)
            ax.plot(xx, yy, color=T["maia"], lw=4, solid_capstyle="round", zorder=3)
        for xi, yi, hv in zip(xs, y, hi):
            a = float(np.clip((cut - xi) / 40 + 1, 0, 1))  # fades in as the line reaches the point
            if a > 0:
                ax.plot([xi], [yi], "o", ms=12, color=T["maia"], mec=T["bg"], mew=2, alpha=a, zorder=4)
                ax.text(
                    xi, hv + 1.2, f"{yi:.0f}", ha="center", va="bottom", fontsize=FS["small"], color=T["ink"], alpha=a
                )
        end = min(cut, x1)
        ax.axhspan(
            HL["match"]["stockfish"]["top1"]["ci95"][0] * 100,
            HL["match"]["stockfish"]["top1"]["ci95"][1] * 100,
            xmin=0,
            xmax=(end - x0) / (x1 - x0),
            color=T["sf"],
            alpha=0.18,
            lw=0,
        )
        ax.plot([x0, end], [sf, sf], color=T["sf"], lw=4)
        ax.plot([x0, end], [rnd, rnd], color=T["neutral"], lw=3, ls=(0, (6, 4)))
        la = float(np.clip((cut - 1500) / 120, 0, 1))
        if la > 0:
            ax.text(
                1080,
                sf + 1.6,
                f"Stockfish {sf:.1f}%",
                color=T["sf"],
                fontsize=FS["small"],
                weight="bold",
                va="bottom",
                alpha=la,
            )
            ax.text(
                1080,
                rnd + 1.6,
                f"Random legal move {rnd:.1f}%",
                color=T["neutral"],
                fontsize=FS["small"],
                va="bottom",
                alpha=la,
            )
        ma = float(np.clip((cut - 1900) / 40, 0, 1))
        if ma > 0:
            ax.text(
                1080,
                lo[0] - 1.2,
                "Maia, 95% CI",
                color=T["maia"],
                fontsize=FS["small"],
                weight="bold",
                va="top",
                alpha=ma,
            )


def build_chart():
    path = OUT / "top1_vs_rating.mp4"
    fig = new_fig()
    enc = Encoder(path)
    n_draw, n_hold = int(CHART_DRAW_S * FPS), int(CHART_HOLD_S * FPS)
    for f in range(n_draw + n_hold):
        cut = 1050 + (1990 - 1050) * min(f / (n_draw - 1), 1.0)
        chart_frame(fig, cut)
        if f in (n_draw // 2, n_draw + n_hold - 1):
            check_layout(fig, f"{path.name} frame {f}")
        enc.write(fig)
    enc.close()
    plt.close(fig)
    fig = new_fig(PNG_DPI)
    chart_frame(fig, 1990)
    png = OUT / "top1_vs_rating.png"
    fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    return path, png, enc.n


# ------------------------------------------------------------------ QA
def top_level_atoms(path):
    atoms, off, data = [], 0, path.read_bytes()
    while off + 8 <= len(data):
        size = int.from_bytes(data[off : off + 4], "big")
        kind = data[off + 4 : off + 8].decode("latin1")
        if size == 1:
            size = int.from_bytes(data[off + 8 : off + 16], "big")
        atoms.append(kind)
        if size < 8:
            break
        off += size
    return atoms


def probe(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(r.stdout)


def check_video(path, dur_range):
    info = probe(path)
    streams = info["streams"]
    v = [s for s in streams if s["codec_type"] == "video"]
    errs = []
    if len(streams) != 1 or len(v) != 1:
        errs.append(f"expected exactly one (video) stream, got {[s['codec_type'] for s in streams]}")
    v = v[0]
    dur = float(info["format"]["duration"])
    checks = [
        (v["codec_name"] == "h264", f"codec {v['codec_name']}"),
        (v["pix_fmt"] == "yuv420p", f"pix_fmt {v['pix_fmt']}"),
        ((v["width"], v["height"]) == (W, H), f"size {v['width']}x{v['height']}"),
        (v["width"] % 2 == 0 and v["height"] % 2 == 0, "odd dimension"),
        (v["r_frame_rate"] == f"{FPS}/1" and v["avg_frame_rate"] == f"{FPS}/1", f"fps {v['avg_frame_rate']}"),
        (dur_range[0] <= dur <= dur_range[1], f"duration {dur:.2f}s outside {dur_range}"),
    ]
    errs += [msg for ok, msg in checks if not ok]
    atoms = top_level_atoms(path)
    if "moov" not in atoms or "mdat" not in atoms or atoms.index("moov") > atoms.index("mdat"):
        errs.append(f"not faststart (atoms {atoms})")
    return {
        "file": str(path.relative_to(ROOT)),
        "codec": v["codec_name"],
        "profile": v.get("profile"),
        "pix_fmt": v["pix_fmt"],
        "size": f"{v['width']}x{v['height']}",
        "fps": v["avg_frame_rate"],
        "duration_s": round(dur, 2),
        "frames": int(v.get("nb_frames", 0)),
        "audio": False if len(streams) == 1 else True,
        "atoms": atoms,
        "mb": round(path.stat().st_size / 1e6, 2),
        "errors": errs,
    }


def contact_sheet(mp4, n=12):
    out = QA / f"contact_{mp4.stem}.png"
    dur = float(probe(mp4)["format"]["duration"])
    rate = n / dur
    vf = (
        f"fps={rate:.5f},drawbox=x=iw*0.75:y=ih*0.75:w=iw*0.25:h=ih*0.25:color=red@0.9:t=4,"
        f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='%{{pts\\:hms}}':"
        f"x=8:y=h-th-8:fontsize=28:fontcolor=yellow:box=1:boxcolor=black@0.6,scale=640:-2,tile=3x4:padding=6:color=gray"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, "-frames:v", "1", str(out)], check=True
    )
    return out


def contact_sheet_pngs(paths, name):
    ims = []
    for pth in paths:
        im = Image.open(pth).convert("RGB")
        from PIL import ImageDraw

        dr = ImageDraw.Draw(im)
        w, h = im.size
        dr.rectangle([int(0.75 * w), int(0.75 * h), w - 1, h - 1], outline=(255, 0, 0), width=5)
        ims.append(im.resize((960, int(960 * h / w))))
    sheet = Image.new("RGB", (960 * 2 + 6, ims[0].height * 2 + 6), (128, 128, 128))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % 2) * 966, (i // 2) * (ims[0].height + 6)))
    out = QA / f"contact_{name}.png"
    sheet.save(out)
    return out


# ------------------------------------------------------------------ main
def describe(p):
    b = p["board"]
    for c in ("me", "sf"):
        assert chess.Move.from_uci(p[c]) in b.legal_moves, (p["name"], c, p[c])
    tops = {rt: san(b, p["levels"][rt]["top1"]) for rt in RATINGS}
    for rt in RATINGS:
        assert chess.Move.from_uci(p["levels"][rt]["top1"]) in b.legal_moves
        # the cached top-1 must match what the bars show at that rating
        assert p["levels"][rt]["top1"] == (p["rec"]["maia"][str(rt)]["top1"]), (p["name"], rt)
    return tops


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m maia_study video", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--running-move",
        type=int,
        default=20,
        help="move number of the running example in game 4hGkJ3rK (default 20; 21 = where Bxb4 was played)",
    )
    ap.add_argument("--qa", action="store_true", help="write frame contact sheets to figures/video/qa/")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.qa:
        QA.mkdir(parents=True, exist_ok=True)

    positions = pick_positions(args.running_move)
    tab = Table(
        title="Positions (arrows: blue = Maia top move per rating, green = Stockfish d14 best, orange = my move)"
    )
    for col in (
        "asset",
        "game · move",
        "my move",
        "SF best",
        "cp loss",
        "Maia top-1 1100 → 1900",
        "bars from",
        "clean set",
    ):
        tab.add_column(col)
    manifest = {
        "spec": {
            "size": f"{W}x{H}",
            "fps": FPS,
            "codec": "H.264 yuv420p +faststart, no audio",
            "face_cam_box": "x >= 75% and y >= 75% of the frame kept empty",
        },
        "sources": {
            "analysis": str(C.ANALYSIS.relative_to(ROOT)),
            "headline": str(C.HEADLINE.relative_to(ROOT)),
            "showcase": str(C.SHOWCASE_JSON.relative_to(ROOT)),
            "full_dists": str(C.SHOWCASE_DISTS.relative_to(ROOT)),
        },
        "positions": [],
        "files": {},
    }
    for p in positions:
        tops = describe(p)
        b, r = p["board"], p["rec"]
        tab.add_row(
            p["name"],
            f"{r['game_id']} · {b.fullmove_number}",
            san(b, p["me"]),
            san(b, p["sf"]),
            str(r["cp_loss"]),
            " ".join(dict.fromkeys(tops.values())) + f"  ({'/'.join(tops[x] for x in (1100, 1500, 1900))})",
            p["source"],
            "yes" if p["in_clean_set"] else "NO",
        )
        manifest["positions"].append(
            {
                "asset": f"rating_sweep_{p['name']}.mp4",
                "game_id": r["game_id"],
                "move_number": b.fullmove_number,
                "fen": r["fen"],
                "my_move": san(b, p["me"]),
                "stockfish_best": san(b, p["sf"]),
                "cp_loss": r["cp_loss"],
                "maia_top1_by_rating": tops,
                "maia_top5_by_rating": {
                    rt: [[san(b, u), round(v, 4)] for u, v in p["levels"][rt]["top5"]] for rt in RATINGS
                },
                "bars_source": p["source"],
                "in_clean_set": p["in_clean_set"],
                "why_chosen": p["why"],
            }
        )
    console.print(tab)

    checks = []
    for p in positions:
        path, n = build_sweep(p)
        console.print(f"[green]wrote[/] {path.relative_to(ROOT)} ({n} frames)")
        checks.append(check_video(path, (8.0, 12.0)))
        manifest["files"][path.name] = str(make_gif(path).relative_to(ROOT))

    layers, lines = reveal_layers(positions[0])
    console.print(f"[green]wrote[/] {len(layers)} reveal layers in {layers[0].parent.relative_to(ROOT)}")
    manifest["reveal_lines"] = [t for _, t in lines]

    path, png, n = build_chart()
    console.print(f"[green]wrote[/] {path.relative_to(ROOT)} ({n} frames) and {png.name}")
    checks.append(check_video(path, (6.0, 8.0)))
    manifest["files"][path.name] = str(make_gif(path).relative_to(ROOT))

    sheets = []
    if args.qa:
        sheets = [contact_sheet(OUT / c["file"].split("/")[-1]) for c in checks]
        sheets.append(contact_sheet_pngs(layers, "reveal"))
        sheets.append(contact_sheet_pngs([png], "top1_png"))

    t = Table(title="ffprobe")
    for col in ("file", "codec", "pix_fmt", "size", "fps", "dur s", "frames", "audio", "faststart", "MB", "status"):
        t.add_column(col)
    for c in checks:
        fast = c["atoms"].index("moov") < c["atoms"].index("mdat")
        t.add_row(
            c["file"].split("/")[-1],
            f"{c['codec']} {c['profile']}",
            c["pix_fmt"],
            c["size"],
            c["fps"],
            str(c["duration_s"]),
            str(c["frames"]),
            str(c["audio"]),
            str(fast),
            str(c["mb"]),
            "[green]OK" if not c["errors"] else "[red]" + "; ".join(c["errors"]),
        )
    console.print(t)
    for g in sorted(OUT.glob("*.gif")):
        gi = probe(g)["streams"][0]
        console.print(f"gif  {g.name}: {gi['width']}x{gi['height']}, {g.stat().st_size / 1e6:.1f} MB")
    for pth in layers + [png]:
        console.print(f"png  {pth.relative_to(OUT)}: {Image.open(pth).size}")
    if sheets:
        console.print("contact sheets: " + ", ".join(str(s.relative_to(ROOT)) for s in sheets))

    manifest["ffprobe"] = checks
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if any(c["errors"] for c in checks):
        raise SystemExit("video checks failed")


def run(argv=None):
    try:
        main(argv)
    except LayoutError as e:
        console.print(f"[red]layout check failed[/] {e}")
        sys.exit(1)
