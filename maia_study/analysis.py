"""The analysis pass: run every Maia model and Stockfish on every sampled position.

Each record holds, for one position: the move the player made, each Maia model's top move,
top 5 and the probability/rank it gave the player's move, Stockfish's top-3 moves with evals, and
the centipawn loss of the player's move. Records are written to results/cache/analysis_<key>.jsonl,
where <key> hashes the sampling settings, so a rerun with the same settings reuses the file.
"""

import hashlib
import json

import chess
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from . import config as C
from .data import sample_positions
from .engines import Engines

console = Console()


def cache_path(meta: dict):
    # "v2-history" marks the run in which Maia was given the real move history; it is part of the
    # key so the published cache (analysis_462a5efce6.jsonl) is found again by a rerun.
    key = hashlib.sha1(json.dumps([meta, C.RATINGS, "v2-history"], sort_keys=True).encode()).hexdigest()[:10]
    return C.CACHE / f"analysis_{key}.jsonl"


def analyse(user=None, n=C.N_POSITIONS, seed=C.SEED, depth=C.SF_DEPTH, refresh=False):
    """Analyse the sampled positions (cached). Returns (records, meta)."""
    samples, meta = sample_positions(user=user, n=n, seed=seed)
    meta["sf_depth"] = depth
    cache = cache_path(meta)
    if cache.exists() and not refresh:
        records = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
        if len(records) == len(samples):
            console.print(f"[dim]using cached analysis {cache.relative_to(C.ROOT)} ({len(records)} positions)[/]")
            return records, meta

    C.CACHE.mkdir(parents=True, exist_ok=True)
    console.print(
        Panel.fit(
            f"[bold]Analysing {len(samples)} positions[/] from {meta['games']} games of {meta['user']}\n"
            f"Maia ratings {C.RATINGS[0]}–{C.RATINGS[-1]} (nodes={C.LC0_NODES})  ·  Stockfish depth {depth}",
            border_style="cyan",
        )
    )
    records = []
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
    )
    with Engines() as eng, progress, open(cache, "w") as out:
        task = progress.add_task("engines", total=len(samples))
        for s in samples:
            # Replay the game so python-chess sends `position startpos moves ...` and lc0 builds
            # Maia's history planes from the real previous positions, not from a bare FEN.
            board = chess.Board()
            for u in s.history:
                board.push_uci(u)
            assert board.fen() == s.fen
            human = chess.Move.from_uci(s.move_uci)
            rec = {
                k: getattr(s, k)
                for k in ("game_id", "ply", "fen", "move_uci", "move_san", "player", "player_elo", "event")
            }
            rec["n_legal"] = board.legal_moves.count()
            rec["maia"] = {}
            for r in C.RATINGS:
                dist = eng.maia_distribution(board, r)
                moves = list(dist)
                rec["maia"][str(r)] = {
                    "top1": moves[0].uci(),
                    "top1_san": board.san(moves[0]),
                    "p_human": dist.get(human, 0.0),
                    "rank_human": moves.index(human) + 1,
                    "top5": [[m.uci(), round(p, 5)] for m, p in list(dist.items())[:5]],
                }
            lines = eng.stockfish_analysis(board, depth=depth, multipv=C.SF_MULTIPV)
            best = lines[0]
            rec["sf_best"] = best.move.uci()
            rec["sf_best_san"] = board.san(best.move)
            rec["sf_best_cp"] = best.cp
            rec["sf_eval"] = best.eval_str
            rec["sf_top3"] = [line.move.uci() for line in lines]
            rec["sf_top3_cp"] = [line.cp for line in lines]
            human_cp = next((line.cp for line in lines if line.move == human), None)
            if human_cp is None:
                human_cp = eng.stockfish_move_cp(board, human, depth=depth)
            rec["human_cp"] = human_cp
            rec["cp_loss"] = 0 if human == best.move else max(0, best.cp - human_cp)
            records.append(rec)
            out.write(json.dumps(rec) + "\n")
            progress.advance(task)
    return records, meta
