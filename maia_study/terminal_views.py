"""Terminal views for screenshots and screen recordings (each fits about 100 x 35 characters).

    python -m maia_study terminal headline      # match rates with 95% CIs
    python -m maia_study terminal mistakes      # mistakes test + sensitivity
    python -m maia_study terminal xent          # cross-entropy per rating + worked example
    python -m maia_study terminal position 2    # one showcase position (1-4)
    python -m maia_study terminal summary       # what was run, engines, versions, filters
    python -m maia_study terminal live          # streams 30 positions through the engines

All views except `live` read the saved results (results/headline_numbers.json, results/showcase/).
"""

import argparse
import json
import math
import time

import chess
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import config as C

console = Console(width=100)
MAIA, SF, ME, DIM = "bold #5598e7", "bold #1baf7a", "bold #eb6834", "grey62"
HL = json.loads(C.HEADLINE.read_text())
BEST = HL["best_rating"]["by_top1"]


def bar(frac, width=20, style="#2a78d6"):
    full = int(round(frac * width))
    return Text("█" * full, style=style) + Text("░" * (width - full), style="grey30")


def ci(d):
    return f"{d['ci95'][0] * 100:.0f}-{d['ci95'][1] * 100:.0f}"


def header(title, sub):
    console.print(
        Panel(Text.assemble((title, "bold"), "\n", (sub, DIM)), box=box.HEAVY, border_style="#2a78d6", width=100)
    )


# ------------------------------------------------------------------ views
def v_headline(_):
    m = HL["meta"]
    header(
        "Does Maia predict my moves better than Stockfish?",
        f"{m['n_positions_clean']} positions from {m['n_games_clean']} of my Lichess games ({C.PLAYER}, "
        f"rating ~{C.MY_RATING}) · 95% confidence intervals",
    )
    t = Table(box=box.SIMPLE_HEAVY, header_style="bold", width=100)
    t.add_column("Model", width=14)
    t.add_column("Top-1 match", justify="right", width=11)
    t.add_column("", width=22)
    t.add_column("95% CI", justify="right", width=7)
    t.add_column("Top-3", justify="right", width=7)
    t.add_column("Agrees with SF", justify="right", width=14)
    for rt in C.RATINGS:
        d = HL["match"][str(rt)]
        best = rt == BEST
        t.add_row(
            Text(f"Maia {rt}" + (" ★" if best else ""), style=MAIA if best else ""),
            Text(f"{d['top1']['rate']:.1%}", style="bold" if best else ""),
            bar(d["top1"]["rate"], 22),
            ci(d["top1"]),
            f"{d['top3']['rate']:.0%}",
            f"{HL['maia_vs_stockfish_agreement'][str(rt)]['rate']:.0%}",
        )
    t.add_section()
    d = HL["match"]["stockfish"]
    t.add_row(
        Text("Stockfish 17.1", style=SF),
        Text(f"{d['top1']['rate']:.1%}", style="bold"),
        bar(d["top1"]["rate"], 22, "#1baf7a"),
        ci(d["top1"]),
        f"{d['top3']['rate']:.0%}",
        "",
    )
    t.add_row(
        Text("Random move", style=DIM),
        Text(f"{HL['match']['random']['top1_expected']:.1%}", style=DIM),
        bar(HL["match"]["random"]["top1_expected"], 22, "grey50"),
        "",
        "",
        "",
    )
    console.print(t)
    test = HL["tests"]["maia_best_vs_stockfish_top1"]
    near = HL["tests"].get(f"maia_{BEST}_vs_{HL['best_rating']['closest_maia_to_my_rating']}_top1", {})
    console.print(
        Panel(
            Text.assemble(
                ("Maia vs Stockfish: ", "bold"),
                (
                    f"Maia {BEST} matched me on {test['maia_only']} positions where Stockfish did not; "
                    f"Stockfish on {test['sf_only']} where Maia did not (McNemar p = {test['p']:.0e}). "
                    "A real difference.\n",
                    "",
                ),
                ("Between Maia ratings: ", "bold"),
                (
                    f"1100 to 1400 are within noise of each other (e.g. {BEST} vs "
                    f"{HL['best_rating']['closest_maia_to_my_rating']}: p = {near.get('p', float('nan')):.2f}); "
                    "accuracy falls by about "
                    f"{abs(HL['tests']['top1_trend_per_100_rating']['slope']):.1f} points per 100 rating.",
                    "",
                ),
            ),
            border_style="green",
            width=100,
        )
    )


def v_mistakes(_):
    e = HL["mistakes"][str(C.MISTAKE_CP)]
    mm = e[f"maia_{BEST}"]
    header(
        "Does Maia expect my mistakes?",
        f"Mistake = my move lost at least {C.MISTAKE_CP} centipawns vs Stockfish's best · {e['n_mistakes']} of "
        f"{HL['meta']['n_positions_clean']} moves · 95% CI",
    )
    t = Table(box=box.SIMPLE_HEAVY, header_style="bold", width=100)
    for c, w in (("Model", 12), ("Ranked #1", 19), ("In top 3", 12), ("", 18), ("Median p", 10)):
        t.add_column(c, justify="left" if c in ("Model", "") else "right", width=w)
    t.add_row(
        Text(f"Maia {BEST}", style=MAIA),
        f"{mm['top1']['rate']:.0%} ({ci(mm['top1'])})",
        f"{mm['top3']['rate']:.0%} ({ci(mm['top3'])})",
        bar(mm["top3"]["rate"], 18),
        f"{mm['median_p'][0]:.1%}",
    )
    s = e["stockfish"]
    t.add_row(
        Text("Stockfish", style=SF),
        "0% (by definition)",
        f"{s['top3']['rate']:.0%} ({ci(s['top3'])})",
        bar(s["top3"]["rate"], 18, "#1baf7a"),
        "n/a",
    )
    t.add_row(
        Text("Random move", style=DIM),
        Text(f"{e['random_expected_p']:.0%}", style=DIM),
        "",
        "",
        Text(f"{e['random_expected_p']:.1%}", style=DIM),
    )
    console.print(t)
    s2 = Table(
        title="Sensitivity: does the threshold change the story?", box=box.SIMPLE, header_style="bold", width=100
    )
    for c in ("Threshold", "Mistakes", f"Maia {BEST} #1", f"Maia {BEST} top 3", "Maia median p", "Stockfish top 3"):
        s2.add_column(c, justify="right")
    for thr, v in HL["mistakes"].items():
        x = v[f"maia_{BEST}"]
        s2.add_row(
            f"{thr} cp",
            str(v["n_mistakes"]),
            f"{x['top1']['rate']:.0%}",
            f"{x['top3']['rate']:.0%}",
            f"{x['median_p'][0]:.1%}",
            f"{v['stockfish']['top3']['rate']:.0%}",
        )
    console.print(s2)
    pm = HL["prob_on_my_move"]
    console.print(
        Text.assemble(
            ("  Median Maia probability on my move: ", "bold"),
            (f"good moves {pm['non_mistakes']['median'][0]:.0%}", MAIA),
            ("  vs  ", ""),
            (f"mistakes {pm['mistakes']['median'][0]:.0%}", ME),
            (f"   (random guess {e['random_expected_p']:.0%})", DIM),
        )
    )


def v_xent(_):
    header(
        "Cross-entropy: how surprised was each Maia model by my moves?",
        f"Loss = -log p(my move), averaged over {HL['meta']['n_positions_clean']} positions · lower is better · "
        "95% CI (bootstrap over games)",
    )
    t = Table(box=box.SIMPLE_HEAVY, header_style="bold", width=100)
    for c, w in (("Model", 14), ("Avg loss (nats)", 15), ("", 24), ("95% CI", 13), ("Perplexity", 11)):
        t.add_column(c, justify="left" if c in ("Model", "") else "right", width=w)
    lo = min(HL["cross_entropy"][str(r)]["mean_xent"] for r in C.RATINGS)
    hi = HL["cross_entropy"]["random"]["mean_xent"]
    for rt in C.RATINGS:
        d = HL["cross_entropy"][str(rt)]
        best = d["mean_xent"] == lo
        t.add_row(
            Text(f"Maia {rt}" + (" ★" if best else ""), style=MAIA if best else ""),
            Text(f"{d['mean_xent']:.3f}", style="bold" if best else ""),
            bar(d["mean_xent"] / hi, 24),
            f"{d['ci95'][0]:.2f}-{d['ci95'][1]:.2f}",
            f"{d['perplexity']:.2f}",
        )
    t.add_section()
    t.add_row(
        Text("Random move", style=DIM),
        Text(f"{hi:.3f}", style=DIM),
        bar(1, 24, "grey50"),
        "",
        Text(f"{math.exp(hi):.1f}", style=DIM),
    )
    console.print(t)
    show = json.loads((C.SHOWCASE_JSON).read_text())
    ex = next(p for p in show["positions"] if p["slug"] == "xent_example")
    p = ex["p_my_move"]
    console.print(
        Panel(
            Text.assemble(
                ("Worked example ", "bold"),
                (f"(game {ex['game_id']}, move {ex['move_number']}): ", DIM),
                (f"Maia {BEST} gave my move {ex['my_move']} p = {p:.3f}", ""),
                ("  →  ", DIM),
                (f"loss = -log({p:.3f}) = {-math.log(p):.2f} nats\n", ME),
                (
                    "Perplexity e^loss reads as 'Maia was as unsure as choosing between "
                    "this many equally likely moves'.",
                    DIM,
                ),
            ),
            border_style="yellow",
            width=100,
        )
    )


def render_board(fen, highlight=()):
    board = chess.Board(fen)
    text = Text()
    ranks = range(7, -1, -1) if board.turn == chess.WHITE else range(8)
    files = list(range(8)) if board.turn == chess.WHITE else list(range(7, -1, -1))
    for r in ranks:
        text.append(f" {r + 1} ", style="bold")
        for f in files:
            sq = chess.square(f, r)
            pc = board.piece_at(sq)
            bg = "#f2b48a" if sq in highlight else ("#f0d9b5" if (r + f) % 2 else "#b58863")
            g = chess.UNICODE_PIECE_SYMBOLS[pc.symbol().lower()] if pc else " "
            fg = "#ffffff" if pc and pc.color == chess.WHITE else "#000000"
            text.append(f" {g} ", style=f"bold {fg} on {bg}")
        text.append("\n")
    text.append("    " + "  ".join("abcdefgh"[f] for f in files), style="bold")
    return text


def v_position(args):
    show = json.loads((C.SHOWCASE_JSON).read_text())
    p = show["positions"][args.n - 1]
    mv = chess.Move.from_uci(p["my_uci"])
    header(f"Position {p['id']}: {p['title']}", f"{p['source']} · {p['to_move']}")
    t = Table(box=box.SIMPLE_HEAVY, header_style="bold", show_edge=False)
    t.add_column(f"Maia {p['rating']}", style="")
    t.add_column("")
    t.add_column(f"Stockfish d{C.SF_DEPTH}")
    for (msan, mp), (ssan, sev) in zip(p["maia_top3"], p["sf_top3"]):
        t.add_row(
            Text(f"{msan:<7}{mp:6.1%}", style=MAIA if msan == p["maia_top"] else ""),
            bar(mp, 10),
            Text(f"{ssan:<7}{sev:>6}", style=SF if ssan == p["sf_best"] else ""),
        )
    me = Text.assemble(
        ("\nMy move: ", "bold"),
        (p["my_move"], ME),
        (f"\nMaia p = {p['p_my_move']:.1%} (rank {p['rank_my_move']} of {p['n_legal']})", ""),
        (f"\nCentipawn loss: {p['cp_loss']}", ""),
    )
    console.print(Columns([render_board(p["fen"], {mv.from_square, mv.to_square}), Group(t, me)], padding=(0, 4)))
    console.print(Panel(p["why"], title="Why it matters", border_style="grey50", width=100))
    console.print(Text(f"FEN {p['fen']}", style=DIM))


def v_summary(_):
    m = HL["meta"]
    header("Run summary", "What was analysed, how, and with which engines")
    t = Table(box=box.SIMPLE, show_header=False, width=100)
    t.add_column(style="bold", width=12)
    t.add_column(width=84)
    t.add_row(
        "Player",
        f"{C.PLAYER} (Lichess), rating used: {C.MY_RATING} (median of in-game ratings, "
        f"range {m['player_elo_in_games']['min']}-{m['player_elo_in_games']['max']})",
    )
    t.add_row(
        "Positions",
        f"{m['n_positions_raw']} sampled from 1413 candidates (seed 7) → {m['n_positions_clean']} "
        f"after filters, {m['n_games_clean']} games",
    )
    for s in m["filter_steps"][1:]:
        t.add_row("", f"· {s['rule']}: removed {s['removed']}, {s['remaining']} left")
    t.add_row("Maia", "lc0 v0.34.0-dev (d91bd14), CPU BLAS backend, weights maia-1100 to maia-1900")
    t.add_row("", "nodes=1 (one forward pass), PolicyTemperature 1.0, real game history")
    t.add_row("Stockfish", f"Stockfish 17.1 (armv8-dotprod), 4 threads, depth {C.SF_DEPTH}, MultiPV 3")
    t.add_row("Statistics", "Wilson 95% CIs for rates, game-level bootstrap (2000) for means, exact McNemar")
    t.add_row("Time taken", "main 1000-position run: not logged (about 0.3 s per position in a 30-position test)")
    t.add_row("", "top-up: 426 extra Stockfish searches, ~33 s of search")
    t.add_row("Hardware", "ARM64 laptop, WSL2 Ubuntu, 12 cores")
    t.add_row(
        "Outputs",
        f"{len(list(C.FIG_DIR.glob('*')))} figure files, {len(list(C.FIG_DATA.glob('*.csv')))} figure-data "
        f"CSVs, {len(list(C.POS_DIR.glob('pos_*')))} position files",
    )
    console.print(t)


def v_live(args):
    """Stream positions through the engines with a live-updating table (screen-recording view)."""
    from . import stats
    from .data import history_index
    from .engines import Engines

    df, _, _ = stats.clean()
    rows = df.sample(n=args.n, random_state=11).to_dict("records")
    hist = history_index(C.PLAYER)
    tally = {"maia": 0, "maia19": 0, "sf": 0, "done": 0}
    recent = []

    def view():
        t = Table(
            box=box.SIMPLE_HEAVY,
            header_style="bold",
            width=100,
            title=f"Live: Maia {BEST} vs Maia 1900 vs Stockfish on my moves",
        )
        for c, w in (
            ("#", 3),
            ("Game / move", 18),
            ("I played", 9),
            (f"Maia {BEST}", 13),
            ("Maia 1900", 13),
            ("Stockfish", 11),
            ("p(my move)", 10),
        ):
            t.add_column(c, width=w, justify="right" if c in ("#", "p(my move)") else "left")
        for r in recent[-14:]:
            t.add_row(*r)
        n = max(1, tally["done"])
        score = Text.assemble(
            (f"  {tally['done']}/{args.n} positions   ", "bold"),
            (f"Maia {BEST} {tally['maia']}/{tally['done']} ({tally['maia'] / n:.0%})", MAIA),
            ("   ", ""),
            (f"Maia 1900 {tally['maia19']}/{tally['done']} ({tally['maia19'] / n:.0%})", "#86b6ef"),
            ("   ", ""),
            (f"Stockfish {tally['sf']}/{tally['done']} ({tally['sf'] / n:.0%})", SF),
        )
        return Group(t, score)

    t0 = time.time()
    with Engines() as eng, Live(view(), console=console, refresh_per_second=8) as live:
        for i, r in enumerate(rows, 1):
            board = chess.Board()
            for u in hist.get((r["game_id"], r["fen"]), []):
                board.push_uci(u)
            if board.fen() != r["fen"]:
                board = chess.Board(r["fen"])
            me = chess.Move.from_uci(r["move_uci"])
            d1 = eng.maia_distribution(board, BEST)
            d2 = eng.maia_distribution(board, 1900)
            sf = eng.stockfish_analysis(board, depth=C.SF_DEPTH, multipv=1)[0].move
            m1, m2 = next(iter(d1)), next(iter(d2))
            tally["maia"] += m1 == me
            tally["maia19"] += m2 == me
            tally["sf"] += sf == me
            tally["done"] += 1

            def mark(mv, st, board=board, me=me):
                return Text(f"{board.san(mv):<7}" + ("✓" if mv == me else "✗"), style=st if mv == me else DIM)

            recent.append(
                (
                    str(i),
                    f"{r['game_id']} m{int(r['ply'] // 2 + 1)}",
                    Text(board.san(me), style=ME),
                    mark(m1, MAIA),
                    mark(m2, "#86b6ef"),
                    mark(sf, SF),
                    f"{d1.get(me, 0):.0%}",
                )
            )
            live.update(view())
            time.sleep(args.delay)
    secs = round(time.time() - t0, 1)
    console.print()
    console.print(
        Text(
            f"  finished in {secs} s (including {args.delay} s display delay per position;  "
            f"live engines, Stockfish depth {C.SF_DEPTH})",
            style=DIM,
        )
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m maia_study terminal", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("headline", v_headline), ("mistakes", v_mistakes), ("xent", v_xent), ("summary", v_summary)):
        sub.add_parser(name).set_defaults(fn=fn)
    sp = sub.add_parser("position")
    sp.add_argument("n", type=int, choices=[1, 2, 3, 4])
    sp.set_defaults(fn=v_position)
    sp = sub.add_parser("live")
    sp.add_argument("--n", type=int, default=30)
    sp.add_argument("--delay", type=float, default=0.3, help="pause after each position so viewers can follow")
    sp.set_defaults(fn=v_live)
    args = ap.parse_args(argv)
    args.fn(args)
