"""Pick the showcase positions and render their boards and cards.

Selection uses the clean set and the saved analysis. The only engine work is a full Maia
distribution (9 forward passes) per chosen position, because the analysis run kept only Maia's top 5
moves per rating; those distributions are cached in results/cache/showcase_full_dists.json, so a
rerun needs no engines. Stockfish numbers on the cards come from the saved depth-14 analysis.

Writes board/card images to figures/positions/ and showcase.json plus one Markdown card per position
to results/showcase/.
"""

import json
from pathlib import Path

import cairosvg
import chess
import chess.svg
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "maia-study"
import matplotlib.pyplot as plt

from . import config as C
from . import stats
from .data import history_index
from .engines import Engines

# Notes shown on the cards, written from the numbers this script prints
# (Maia 1100 is the best-matching rating; evals are Stockfish depth 14, mover's point of view).
WHY = {
    "disagreement": (
        "Stockfish sees exactly one good move here, Be6, which keeps a clear advantage; its next-best "
        "choice is already about seven pawns worse. Every Maia model from 1100 to 1900 gives Be6 under "
        "2%, while Maia 1100 put 53% on Bg3, the move I actually played, and Bg3 threw the game away."
    ),
    "mistake": (
        "Maia 1100 was 97% sure I would play Ke8, and I did, even though it gave away almost three pawns "
        "compared with Stockfish's Rg1+. Every Maia rating agreed (96 to 99%), which is the clearest case of "
        "Maia predicting a human mistake rather than the best move."
    ),
    "engine_move": (
        "Here I found Stockfish's best move, the bishop sacrifice Bxf2+, but Maia 1100 ranked it only fifth "
        "with 8%. It shows the other side of the coin: Maia predicts typical moves, so when I play better "
        "than usual, Maia is surprised too."
    ),
    "xent_example": (
        "Maia 1100 split its prediction between Qe2 (34%) and Qd2 (31%); I played Qd2, so the loss is "
        "-log 0.313 = 1.16 nats even though Maia's top choice was a different move. The two moves are only "
        "22 centipawns apart for Stockfish, so this is a fair, low-drama example of the loss at work."
    ),
}


def write_md(info, stem):
    lines = [
        f"# Position {info['id']}: {info['title']}",
        "",
        f"- Source: {info['source']}",
        f"- {info['to_move']}",
        f"- FEN: `{info['fen']}`",
        "",
        f"| Maia {info['rating']} top 3 | p |",
        "|---|---|",
    ]
    lines += [f"| {san} | {p:.1%} |" for san, p in info["maia_top3"]]
    lines += ["", f"| Stockfish top 3 (depth {C.SF_DEPTH}) | eval (White's view) |", "|---|---|"]
    lines += [f"| {san} | {ev} |" for san, ev in info["sf_top3"]]
    lines += [
        "",
        f"- My move: **{info['my_move']}**",
        f"- Maia {info['rating']} probability on my move: "
        f"{info['p_my_move']:.1%} (rank {info['rank_my_move']} of {info['n_legal']})",
        f"- My centipawn loss: {info['cp_loss']}",
        "",
        "**Why it is interesting**",
        "",
        WHY[info["slug"]],
        "",
    ]
    Path(f"{stem}_card.md").write_text("\n".join(lines))


def board_with_history(row, hist):
    moves = hist.get((row["game_id"], row["fen"]))
    board = chess.Board()
    if moves is not None:
        for u in moves:
            board.push_uci(u)
        if board.fen() == row["fen"]:
            return board
    return chess.Board(row["fen"])


def full_dists(rows, hist):
    cache = json.loads(C.SHOWCASE_DISTS.read_text()) if C.SHOWCASE_DISTS.exists() else {}
    todo = [r for r in rows if r["fen"] not in cache]
    if todo:
        with Engines() as eng:
            for r in todo:
                b = board_with_history(r, hist)
                cache[r["fen"]] = {
                    str(rt): {m.uci(): p for m, p in eng.maia_distribution(b, rt).items()} for rt in C.RATINGS
                }
        C.SHOWCASE_DISTS.write_text(json.dumps(cache))
    return cache


def select(df, best):
    """Return [(slug, title, row)] for the four showcase categories below, from distinct games where possible."""
    b = str(best)
    used = set()
    out = []

    def take(cands, slug, title):
        for _, r in cands.iterrows():
            if r["game_id"] not in used:
                used.add(r["game_id"])
                out.append((slug, title, r))
                return

    def in_top5(r, rt):
        return r["sf_best"] in [u for u, _ in stats.load_records()[r["idx"]]["maia"][str(rt)]["top5"]]

    d = df.copy()
    # 1 large Maia vs Stockfish disagreement: SF move outside Maia's top 5 at best AND 1900, big gap to SF's 2nd choice
    d["sf_out"] = d.apply(lambda r: not in_top5(r, best) and not in_top5(r, 1900), axis=1)
    take(
        d[d.sf_out & (d[f"top1_{b}"] != d.sf_best)].sort_values("sf_gap", ascending=False),
        "disagreement",
        "Maia and Stockfish want completely different moves",
    )
    # 2 a real mistake that Maia saw coming
    take(
        d[(d.cp_loss >= 150) & (d[f"rank_{b}"] == 1)].sort_values(f"p_{b}", ascending=False),
        "mistake",
        "My mistake was Maia's first choice",
    )
    # 3 I matched Stockfish but not Maia (and Stockfish's move clearly mattered)
    c3 = d[(d.move_uci == d.sf_best) & (d[f"top1_{b}"] != d.move_uci) & (d.sf_gap >= 50)]
    take(c3.sort_values(f"p_{b}"), "engine_move", "I found the engine move that Maia did not expect")
    # 4 clean worked example for cross-entropy: a sensible move, Maia's 2nd choice, mid probability
    c4 = d[(d.n_legal.between(20, 38)) & (d[f"rank_{b}"] == 2) & (d[f"p_{b}"].between(0.2, 0.45)) & (d.cp_loss <= 30)]
    c4 = c4.assign(dist=(c4[f"p_{b}"] - 0.3).abs()).sort_values("dist")
    take(c4, "xent_example", "Cross-entropy worked example")
    return out


def sweep_candidate(df):
    """Positions whose top move changes between Maia 1100 and 1900 (for fig6)."""
    return df[df.top1_1100 != df.top1_1900].sort_values("sf_gap", ascending=False).head(12)


def arrows_for(board, maia_mv, sf_mv, my_mv, theme):
    t = C.THEMES[theme]
    arrows, fill = [], {}
    arrows.append(chess.svg.Arrow(sf_mv.from_square, sf_mv.to_square, color=t["sf"] + "dd"))
    if maia_mv != sf_mv:
        arrows.append(chess.svg.Arrow(maia_mv.from_square, maia_mv.to_square, color=t["maia"] + "dd"))
    if my_mv not in (maia_mv, sf_mv):
        arrows.append(chess.svg.Arrow(my_mv.from_square, my_mv.to_square, color=t["me"] + "dd"))
    # my move's squares are always tinted orange, so it is visible even when it equals Maia or Stockfish
    fill[my_mv.from_square] = t["me"] + "66"
    fill[my_mv.to_square] = t["me"] + "99"
    return arrows, fill


def render_board(board, maia_mv, sf_mv, my_mv, stem):
    arrows, fill = arrows_for(board, maia_mv, sf_mv, my_mv, "light")
    svg = chess.svg.board(
        board,
        arrows=arrows,
        fill=fill,
        size=1000,
        orientation=board.turn,
        colors={"square light": "#f0d9b5", "square dark": "#b58863", "margin": "#2b2d31", "coord": "#e8e6e1"},
    )
    Path(f"{stem}.svg").write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=f"{stem}.png", output_width=1080, output_height=1080)


def card(info, board_png, stem, theme):
    t = C.THEMES[theme]
    fig = plt.figure(figsize=(9.6, 5.4), dpi=200, facecolor=t["bg"])
    ax_img = fig.add_axes([0.02, 0.04, 0.46, 0.92])
    ax_img.imshow(plt.imread(board_png))
    ax_img.axis("off")
    ax = fig.add_axes([0.5, 0.02, 0.49, 0.96])
    ax.axis("off")
    ax.set_facecolor(t["bg"])
    import textwrap

    title = textwrap.wrap(info["title"], 30)
    y = 0.98
    ax.text(0, y, "\n".join(title), fontsize=19, weight="bold", color=t["ink"], va="top", linespacing=1.1)
    y -= 0.075 * len(title) + 0.015
    ax.text(
        0,
        y,
        f"{info['to_move']}  ·  game {info['game_id']}, move {info['move_number']}",
        fontsize=13,
        color=t["ink2"],
        va="top",
    )
    y -= 0.075
    ax.text(0, y, f"Maia {info['rating']} top 3", fontsize=17, weight="bold", color=t["maia"], va="top")
    for san, p in info["maia_top3"]:
        y -= 0.058
        ax.text(0.03, y, f"{san:<8}{p:6.1%}", fontsize=16, family="DejaVu Sans Mono", color=t["ink"], va="top")
    y -= 0.072
    ax.text(0, y, f"Stockfish top 3 (depth {C.SF_DEPTH})", fontsize=17, weight="bold", color=t["sf"], va="top")
    for san, ev in info["sf_top3"]:
        y -= 0.058
        ax.text(0.03, y, f"{san:<8}{ev:>6}", fontsize=16, family="DejaVu Sans Mono", color=t["ink"], va="top")
    y -= 0.072
    ax.text(0, y, f"My move: {info['my_move']}", fontsize=17, weight="bold", color=t["me"], va="top")
    y -= 0.065
    ax.text(
        0.03,
        y,
        f"Maia p = {info['p_my_move']:.1%}  (rank {info['rank_my_move']})",
        fontsize=16,
        family="DejaVu Sans Mono",
        color=t["ink"],
        va="top",
    )
    y -= 0.062
    ax.text(0.03, y, f"cp loss = {info['cp_loss']}", fontsize=16, family="DejaVu Sans Mono", color=t["ink"], va="top")
    ax.text(0, 0.0, "arrows: blue Maia, green Stockfish; orange squares: me", fontsize=12, color=t["ink2"], va="bottom")
    fig.savefig(f"{stem}_{theme}.png", facecolor=t["bg"])
    plt.close(fig)


def main():
    from rich.console import Console

    console = Console(width=110)
    H = json.loads(C.HEADLINE.read_text())
    best = H["best_rating"]["by_top1"]
    df, _, _ = stats.clean()
    hist = history_index(C.PLAYER)
    picks = select(df, best)
    sweep_rows = sweep_candidate(df)
    dists = full_dists([r for _, _, r in picks] + [r for _, r in sweep_rows.iterrows()], hist)

    # fig6 position: biggest change in Maia's distribution between 1100 and 1900 (total variation distance)
    def tv(fen):
        a, z = dists[fen]["1100"], dists[fen]["1900"]
        return 0.5 * sum(abs(a.get(m, 0) - z.get(m, 0)) for m in set(a) | set(z))

    sweep_fen = max(sweep_rows.fen, key=tv)

    C.POS_DIR.mkdir(parents=True, exist_ok=True)
    C.SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)
    showcase = []
    for i, (slug, title, r) in enumerate(picks, 1):
        board = board_with_history(r, hist)
        dist = dists[r["fen"]][str(best)]
        ranked = sorted(dist.items(), key=lambda kv: -kv[1])
        maia_mv = chess.Move.from_uci(ranked[0][0])
        sf_mv = chess.Move.from_uci(r["sf_best"])
        my_mv = chess.Move.from_uci(r["move_uci"])
        rec = stats.load_records()[r["idx"]]
        sf3 = []
        for u, cp in zip(rec["sf_top3"], rec["sf_top3_cp"]):
            # cp is from the mover's point of view; show it from White's like a normal eval bar
            white_cp = cp if board.turn == chess.WHITE else -cp
            ev = "mate" if abs(cp) >= 1000 else f"{white_cp / 100:+.2f}"
            sf3.append((board.san(chess.Move.from_uci(u)), ev))
        info = {
            "id": i,
            "slug": slug,
            "title": title,
            "fen": r["fen"],
            "game_id": r["game_id"],
            "move_number": int(r["ply"] // 2 + 1),
            "to_move": "White to move" if board.turn else "Black to move",
            "source": f"{C.PLAYER} game {r['game_id']}, move {int(r['ply'] // 2 + 1)} ({r['event']})",
            "rating": best,
            "maia_top3": [(board.san(chess.Move.from_uci(u)), p) for u, p in ranked[:3]],
            "sf_top3": sf3,
            "my_move": r["move_san"],
            "my_uci": r["move_uci"],
            "p_my_move": dist.get(r["move_uci"], 0.0),
            "rank_my_move": [u for u, _ in ranked].index(r["move_uci"]) + 1,
            "cp_loss": int(r["cp_loss"]),
            "sf_best": r["sf_best_san"],
            "maia_top": board.san(maia_mv),
            "p_sf_best_by_rating": {str(rt): dists[r["fen"]][str(rt)].get(r["sf_best"], 0.0) for rt in C.RATINGS},
            "p_my_move_by_rating": {str(rt): dists[r["fen"]][str(rt)].get(r["move_uci"], 0.0) for rt in C.RATINGS},
            "n_legal": int(r["n_legal"]),
        }
        stem = C.POS_DIR / f"pos_{i}_{slug}"
        render_board(board, maia_mv, sf_mv, my_mv, f"{stem}_board")
        for theme in ("light", "dark"):
            card(info, f"{stem}_board.png", f"{stem}_card", theme)
        info["why"] = WHY[slug]
        write_md(info, C.SHOWCASE_DIR / f"pos_{i}_{slug}")
        showcase.append(info)
        console.print(f"[bold]{i}. {title}[/]  {info['source']}")
        console.print(
            f"   my move {info['my_move']} (Maia {best} p={info['p_my_move']:.1%}, rank {info['rank_my_move']}, "
            f"cp loss {info['cp_loss']})  ·  Maia top {info['maia_top']}  ·  Stockfish {info['sf_best']}"
        )
        console.print(
            "   Maia p(Stockfish move) by rating: "
            + " ".join(f"{rt}:{v:.1%}" for rt, v in info["p_sf_best_by_rating"].items())
        )
    sweep_row = df[df.fen == sweep_fen].iloc[0]
    out = {
        "best_rating": best,
        "positions": showcase,
        "sweep": {
            "fen": sweep_fen,
            "game_id": sweep_row["game_id"],
            "move_number": int(sweep_row["ply"] // 2 + 1),
            "my_move": sweep_row["move_uci"],
            "sf_best": sweep_row["sf_best"],
            "tv_1100_vs_1900": tv(sweep_fen),
        },
    }
    C.SHOWCASE_JSON.write_text(json.dumps(out, indent=2))
    console.print(
        f"[dim]fig6 sweep position: game {sweep_row['game_id']} move {int(sweep_row['ply'] // 2 + 1)}, "
        f"TV distance 1100 vs 1900 = {tv(sweep_fen):.2f}[/]"
    )
