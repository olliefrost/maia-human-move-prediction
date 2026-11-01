"""Maia (through lc0) and Stockfish behind python-chess's UCI interface.

    with Engines() as eng:
        dist = eng.maia_distribution(board, 1500)     # {chess.Move: probability}
        lines = eng.stockfish_analysis(board)         # top moves with evals

How Maia's probabilities are read, and why each lc0 option is set, is described in docs/method.md.
"""

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.engine

from . import config as C

# One line of lc0's VerboseMoveStats output looks like:
#   info string e2e4  (322 ) N:       0 (+ 0) (P: 50.22%) (WL:  -.-----) ...
# Only the move and its policy prior P are needed.
VERBOSE_LINE = re.compile(r"^([a-h][1-8][a-h][1-8][qrbn]?)\s.*?\(P:\s*([\d.]+)%\)")


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        sys.exit(f"{what} not found at {path}. See README 'Setup', or set the environment variable.")
    return path


def find_lc0() -> Path:
    return _require(C.LC0, "lc0 binary (MAIA_LC0)")


def find_stockfish() -> Path:
    return _require(C.STOCKFISH, "Stockfish binary (MAIA_STOCKFISH)")


def find_weights(rating: int) -> Path:
    return _require(C.WEIGHTS_DIR / f"maia-{rating}.pb.gz", f"Maia {rating} weights (MAIA_WEIGHTS_DIR)")


def score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> int:
    """Centipawns from `turn`'s point of view, mates mapped to +-MATE_SCORE and capped."""
    cp = score.pov(turn).score(mate_score=C.MATE_SCORE)
    return max(-C.CP_CAP, min(C.CP_CAP, cp))


def format_score(score: chess.engine.PovScore, turn: chess.Color = chess.WHITE) -> str:
    """Human readable eval from White's point of view: '+0.35', '-1.20', '#3', '#-2'."""
    s = score.pov(chess.WHITE)
    if s.is_mate():
        return f"#{s.mate()}"
    return f"{s.score() / 100:+.2f}"


@dataclass
class SFLine:
    move: chess.Move
    score: chess.engine.PovScore
    cp: int  # from the side-to-move's point of view, capped
    pv: list = field(default_factory=list)

    @property
    def eval_str(self):
        return format_score(self.score)


class Engines:
    """Owns one lc0 process per Maia rating (started lazily) and one Stockfish process."""

    def __init__(self, sf_threads=C.SF_THREADS, sf_hash=C.SF_HASH_MB, lc0_backend=C.LC0_BACKEND):
        self.sf_threads = sf_threads
        self.sf_hash = sf_hash
        self.lc0_backend = lc0_backend
        self._maia = {}
        self._sf = None

    # ---------- lifecycle ----------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for eng in list(self._maia.values()) + ([self._sf] if self._sf else []):
            try:
                eng.quit()
            except Exception:
                pass
        self._maia.clear()
        self._sf = None

    # ---------- Maia ----------
    def _maia_engine(self, rating):
        if rating not in self._maia:
            # stderr carries lc0's startup banner, which would clutter the demo output
            eng = chess.engine.SimpleEngine.popen_uci([str(find_lc0())], stderr=subprocess.DEVNULL)
            eng.configure(
                {
                    "WeightsFile": str(find_weights(rating)),
                    "Backend": self.lc0_backend,
                    "VerboseMoveStats": True,
                    "PolicyTemperature": C.LC0_POLICY_TEMPERATURE,
                    "Threads": 1,
                }
            )
            self._maia[rating] = eng
        return self._maia[rating]

    def maia_distribution(self, board: chess.Board, rating: int) -> dict:
        """Maia's policy: probability of each legal move, normalised to sum to 1.

        Runs `go nodes 1`, a single forward pass of the network with no search, and reads the
        per-move policy prior P from lc0's VerboseMoveStats lines. P is printed to 0.01%, and is
        renormalised over the legal moves. Pass a board with its move stack (replayed from the
        start of the game) so lc0 sees the real history rather than a synthesised one.
        """
        legal = list(board.legal_moves)
        if len(legal) == 1:
            return {legal[0]: 1.0}
        eng = self._maia_engine(rating)
        raw = {}
        # game=object() forces a `ucinewgame` so no search tree is reused between calls
        with eng.analysis(board, chess.engine.Limit(nodes=C.LC0_NODES), game=object()) as analysis:
            for info in analysis:
                m = VERBOSE_LINE.match(info.get("string", ""))
                if m:
                    raw[m.group(1)] = float(m.group(2)) / 100.0
        dist = {}
        for move in legal:
            # lc0 prints castling as king-takes-rook (e1h1) in some modes; accept both.
            p = raw.get(move.uci())
            if p is None and board.is_castling(move):
                rook_sq = chess.H1 if chess.square_file(move.to_square) == 6 else chess.A1
                if board.turn == chess.BLACK:
                    rook_sq ^= 56
                p = raw.get(chess.Move(move.from_square, rook_sq).uci())
            dist[move] = p or 0.0
        total = sum(dist.values())
        if total <= 0:
            raise RuntimeError(f"lc0 returned no policy values for {board.fen()}")
        return {m: p / total for m, p in sorted(dist.items(), key=lambda kv: -kv[1])}

    # ---------- Stockfish ----------
    def _stockfish(self):
        if self._sf is None:
            self._sf = chess.engine.SimpleEngine.popen_uci([str(find_stockfish())])
            self._sf.configure({"Threads": self.sf_threads, "Hash": self.sf_hash})
        return self._sf

    def stockfish_analysis(self, board: chess.Board, depth=C.SF_DEPTH, multipv=C.SF_MULTIPV, time=None) -> list:
        """Top `multipv` moves, best first, each with score and principal variation."""
        limit = chess.engine.Limit(time=time) if time else chess.engine.Limit(depth=depth)
        infos = self._stockfish().analyse(board, limit, multipv=multipv)
        lines = []
        for info in infos:
            if "pv" not in info or not info["pv"]:
                continue
            lines.append(
                SFLine(
                    move=info["pv"][0], score=info["score"], cp=score_to_cp(info["score"], board.turn), pv=info["pv"]
                )
            )
        return lines

    def stockfish_move_cp(self, board: chess.Board, move: chess.Move, depth=C.SF_DEPTH) -> int:
        """Eval (side-to-move POV, capped) if `move` is played, searched to the same depth."""
        info = self._stockfish().analyse(board, chess.engine.Limit(depth=depth), root_moves=[move])
        return score_to_cp(info["score"], board.turn)
