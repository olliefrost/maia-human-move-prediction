"""Choose the 5 GUI demo positions where Maia and Stockfish disagree; writes data/positions.json.

Candidates come from the saved analysis (my real games), endgames
found in the same games, and two famous endgame studies. Each category is scored with a
cheap prefilter, the shortlist is re-analysed with full Maia distributions and a deeper
Stockfish search, and the position with the biggest Maia/Stockfish contrast wins.
"""

import argparse
import json

import chess
from rich import box
from rich.console import Console
from rich.table import Table

from . import config as C
from .data import board_from_entry, history_index, read_games, resolve_pgn
from .engines import Engines

OUT = C.DEMO_POSITIONS
SHOW = [1100, 1500, 1900]
OWN_RATING = 1400

FAMOUS = [
    {"fen": "8/2P5/8/8/3r4/8/2K5/k7 w - - 0 6", "source": "Saavedra position (1895), after 5...Rd4"},
    {"fen": "7K/8/k1P5/7p/8/8/8/8 w - - 0 1", "source": "Réti endgame study (1921)"},
]

console = Console()


def load_analysis():
    if not C.ANALYSIS.exists():
        raise SystemExit("No saved analysis found. Run `python -m maia_study analyse` first.")
    return [json.loads(line) for line in C.ANALYSIS.read_text().splitlines() if line.strip()]


def quiet(board, move):
    return not board.is_capture(move) and not board.gives_check(move) and not move.promotion


def p_top5(rec, rating, uci):
    return dict(map(tuple, rec["maia"][str(rating)]["top5"])).get(uci, 0.0)


def endgame_pool(user, limit=80):
    path, user = resolve_pgn(user)
    seen, pool = set(), []
    for game in read_games(path):
        if game.headers.get("Variant", "Standard") != "Standard" or "FEN" in game.headers:
            continue
        board = game.board()
        for move in game.mainline_moves():
            if (
                len(board.piece_map()) <= 10
                and board.ply() >= 30
                and not board.is_check()
                and board.legal_moves.count() > 3
                and board.fen() not in seen
            ):
                seen.add(board.fen())
                pool.append(
                    {
                        "fen": board.fen(),
                        "source": f"endgame from {game.headers.get('GameId', '?')}",
                        "game_id": game.headers.get("GameId"),
                    }
                )
            board.push(move)
    return pool[:: max(1, len(pool) // limit)][:limit]


def full_features(eng, entry, depth):
    board = board_from_entry(entry)
    dists = {r: eng.maia_distribution(board, r) for r in SHOW + [OWN_RATING]}
    sf = eng.stockfish_analysis(board, depth=depth, multipv=2)
    best = sf[0].move
    gap = sf[0].cp - sf[1].cp if len(sf) > 1 else 0
    return board, dists, sf, best, gap


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m maia_study pick-positions",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--user", default=None)
    ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--shortlist", type=int, default=12)
    args = ap.parse_args(argv)

    recs = load_analysis()
    user = args.user or recs[0]["player"]

    # ---- cheap prefilter on the cached analysis (Maia top-5 only, SF depth 14)
    def gap(r):
        return r["sf_top3_cp"][0] - r["sf_top3_cp"][1] if len(r["sf_top3_cp"]) > 1 else 0

    def sf_quiet(r):
        b = chess.Board(r["fen"])
        return quiet(b, chess.Move.from_uci(r["sf_best"]))

    def maxp(r):
        return max(p_top5(r, k, r["sf_best"]) for k in SHOW)

    pre = {
        "quiet": sorted(
            [
                r
                for r in recs
                if abs(r["sf_best_cp"]) <= 150
                and gap(r) <= 60
                and sf_quiet(r)
                and all(r["maia"][str(k)]["top1"] != r["sf_best"] for k in SHOW)
            ],
            key=maxp,
        ),
        "tactic": sorted(
            [
                r
                for r in recs
                if gap(r) >= 200 and -100 <= r["sf_best_cp"] <= 800 and r["maia"]["1100"]["top1"] != r["sf_best"]
            ],
            key=lambda r: p_top5(r, 1100, r["sf_best"]) - p_top5(r, 1900, r["sf_best"]),
        ),
        "engine": sorted(
            [
                r
                for r in recs
                if gap(r) >= 100 and sf_quiet(r) and all(r["maia"][str(k)]["top1"] != r["sf_best"] for k in SHOW)
            ],
            key=maxp,
        ),
        "own": sorted(
            [
                r
                for r in recs
                if r["move_uci"] == r["maia"][str(OWN_RATING)]["top1"]
                and r["move_uci"] != r["sf_best"]
                and 100 <= r["cp_loss"] <= 600
            ],
            key=lambda r: -r["maia"][str(OWN_RATING)]["p_human"],
        ),
    }
    cands = {
        k: [
            {
                "fen": r["fen"],
                "source": f"{user} game {r['game_id']}, move {r['ply'] // 2 + 1}",
                "game_id": r["game_id"],
                "human": r["move_uci"],
            }
            for r in v[: args.shortlist]
        ]
        for k, v in pre.items()
    }
    cands["endgame"] = FAMOUS + endgame_pool(user)
    # attach each game position's move history so Maia sees real previous positions
    hist = history_index(user)
    for items in cands.values():
        for c in items:
            if c.get("game_id") and (c["game_id"], c["fen"]) in hist:
                c["history"] = hist[(c["game_id"], c["fen"])]

    # ---- verify the shortlist with full distributions and deeper Stockfish
    scored = {k: [] for k in cands}
    with Engines() as eng:
        total = sum(len(v) for v in cands.values())
        with console.status(f"Verifying {total} candidate positions (Stockfish depth {args.depth})..."):
            for cat, items in cands.items():
                for c in items:
                    board, dists, sf, best, g = full_features(eng, c, args.depth)
                    pb = {r: dists[r].get(best, 0.0) for r in SHOW}
                    tops = {r: next(iter(dists[r])) for r in SHOW}
                    if cat == "quiet":
                        ok = abs(sf[0].cp) <= 200 and all(tops[r] != best for r in SHOW)
                        score = 1 - max(pb.values())
                    elif cat == "tactic":
                        ok = g >= 150 and tops[1100] != best
                        score = (1 - pb[1100]) + (pb[1900] - pb[1100])
                    elif cat == "engine":
                        ok = g >= 80 and quiet(board, best) and all(tops[r] != best for r in SHOW)
                        score = 1 - max(pb.values())
                    elif cat == "endgame":
                        ok = g >= 80 and tops[1500] != best
                        score = 1 - pb[1500] + (0.3 if "study" in c["source"] or "position" in c["source"] else 0)
                    else:  # own
                        human = chess.Move.from_uci(c["human"])
                        ok = human != best and next(iter(dists[OWN_RATING])) == human
                        score = dists[OWN_RATING].get(human, 0) - dists[OWN_RATING].get(best, 0)
                    if ok:
                        scored[cat].append(
                            {
                                **c,
                                "score": score,
                                "board": board,
                                "dists": dists,
                                "sf": sf,
                                "best": best,
                                "gap": g,
                                "pb": pb,
                                "tops": tops,
                            }
                        )

    labels = {
        "quiet": "Quiet middlegame: human vs engine taste",
        "tactic": "A tactic a 1100 player would likely miss",
        "engine": "Engine move that humans rarely play",
        "endgame": "Endgame",
        "own": f"From my own game ({user})",
    }
    order = ["quiet", "tactic", "engine", "endgame", "own"]
    used_fens, positions = set(), []
    ranking = Table(
        title="Candidate positions ranked by Maia/Stockfish contrast",
        box=box.ROUNDED,
        header_style="bold",
        title_style="bold",
    )
    for col in (
        "Category",
        "Source",
        "Stockfish best",
        "Gap to 2nd",
        *[f"Maia {r} top (p SF best)" for r in SHOW],
        "Score",
    ):
        ranking.add_column(col, justify="left" if col in ("Category", "Source") else "right")
    for i, cat in enumerate(order, 1):
        items = sorted(scored[cat], key=lambda c: -c["score"])
        pick = next((c for c in items if c["fen"] not in used_fens), None)
        for j, c in enumerate(items[:3]):
            b = c["board"]
            ranking.add_row(
                cat if j == 0 else "",
                c["source"],
                f"{b.san(c['best'])} ({c['sf'][0].eval_str})",
                f"{c['gap']} cp",
                *[f"{b.san(c['tops'][r])} ({c['pb'][r]:.0%})" for r in SHOW],
                f"[bold]{c['score']:.2f}" if c is pick else f"{c['score']:.2f}",
            )
        ranking.add_section()
        if pick is None:
            console.print(f"[yellow]No verified candidate for {cat}[/]")
            continue
        used_fens.add(pick["fen"])
        b = pick["board"]
        top = {r: b.san(pick["tops"][r]) for r in SHOW}
        entry = {
            "id": i,
            "category": cat,
            "label": labels[cat],
            "fen": pick["fen"],
            "source": pick["source"],
            "stockfish_best": b.san(pick["best"]),
            "stockfish_eval": pick["sf"][0].eval_str,
            "maia_top": top,
            "maia_p_stockfish_best": {str(r): round(pick["pb"][r], 4) for r in SHOW},
            "history": pick.get("history", []),
            "note": (
                f"Stockfish plays {b.san(pick['best'])} ({pick['sf'][0].eval_str}); Maia 1100 prefers {top[1100]}, "
                f"Maia 1900 prefers {top[1900]}. Maia 1100 gives the engine move "
                f"{pick['pb'][1100]:.0%}, Maia 1900 {pick['pb'][1900]:.0%}."
            ),
        }
        if cat == "own":
            entry["my_move"] = b.san(chess.Move.from_uci(pick["human"]))
            entry["how_to_replace"] = (
                "Replace 'fen' with any position from your games (Lichess: analysis board → "
                "FEN box), set 'my_move' to the SAN you played, then re-open the GUI."
            )
        positions.append(entry)
    console.print(ranking)

    OUT.write_text(json.dumps(positions, indent=2, ensure_ascii=False))
    t = Table(title="positions.json", box=box.HEAVY_HEAD, header_style="bold", title_style="bold")
    t.add_column("#")
    t.add_column("Label")
    t.add_column("Contrast")
    for p in positions:
        t.add_row(str(p["id"]), p["label"], p["note"])
    console.print(t)
    console.print(f"[green]✓[/] wrote {OUT.relative_to(C.ROOT)}")
