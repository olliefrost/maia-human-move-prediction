"""Engine top-up: evaluate Maia's top-1 moves that the saved analysis did not cover.

The saved run only has Stockfish evals for Stockfish's own top-3 moves, so the centipawn loss of
Maia's top-1 move is missing whenever that move is outside Stockfish's top 3. This searches exactly
those moves (same Stockfish, same depth) and caches the result. Nothing else is recomputed.
The published run needed 426 searches, about 30 s of search time.
"""

import json
import time

import chess
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from . import config as C
from . import stats
from .engines import Engines


def main():
    console = Console()
    df, _, _ = stats.clean()
    done = set()
    if C.TOPUP.exists():
        done = {(t["fen"], t["move"]) for t in map(json.loads, C.TOPUP.read_text().splitlines()) if t}
    todo = [x for x in stats.topup_needed(df) if x not in done]
    console.print(f"{len(todo)} Stockfish searches needed (depth {C.SF_DEPTH})")
    t0 = time.time()
    with (
        Engines() as eng,
        open(C.TOPUP, "a") as out,
        Progress(
            TextColumn("[cyan]stockfish"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console
        ) as prog,
    ):
        task = prog.add_task("", total=len(todo))
        for fen, mv in todo:
            board = chess.Board(fen)
            cp = eng.stockfish_move_cp(board, chess.Move.from_uci(mv), depth=C.SF_DEPTH)
            out.write(json.dumps({"fen": fen, "move": mv, "cp": cp}) + "\n")
            prog.advance(task)
    console.print(f"done in {time.time() - t0:.0f}s -> {C.TOPUP.relative_to(C.ROOT)}")
