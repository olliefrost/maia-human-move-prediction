"""Load the player's Lichess games and sample positions from them.

Positions are sampled where the player is to move, and each one carries the full move list from
the start of the game so that lc0 can be given the real history (see docs/method.md).
"""

import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import requests

from . import config as C


@dataclass
class Sample:
    game_id: str
    ply: int
    fen: str
    move_uci: str
    move_san: str
    player: str
    player_elo: int | None
    event: str
    history: list  # UCI moves from the start of the game


def fetch_lichess(user: str = C.PLAYER, dest: Path = C.PGN, max_games: int = C.LICHESS_MAX_GAMES) -> Path:
    """Download the user's games as PGN (no clocks, no evals). Overwrites `dest`."""
    url = f"https://lichess.org/api/games/user/{user}"
    headers = {"User-Agent": C.LICHESS_USER_AGENT, "Accept": "application/x-chess-pgn"}
    resp = requests.get(
        url, params={"max": max_games, "clocks": "false", "evals": "false"}, headers=headers, timeout=120
    )
    resp.raise_for_status()
    if not resp.text.lstrip().startswith("["):
        raise RuntimeError(f"Lichess did not return PGN for user {user!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text)
    return dest


def anonymise_pgn(path: Path = C.PGN, keep: str = C.PLAYER) -> int:
    """Replace every White/Black name other than `keep` with "Opponent", in place.

    Only the two name headers change. Titles (e.g. BOT, used to filter bot games), ratings and moves
    are kept, so sampling and every result are unaffected. Returns the number of names replaced.
    """
    pattern = re.compile(r'^\[(White|Black) "([^"]*)"\]$', re.M)
    count = 0

    def repl(m):
        nonlocal count
        if m.group(2).lower() == keep.lower():
            return m.group(0)
        count += 1
        return f'[{m.group(1)} "Opponent"]'

    path.write_text(pattern.sub(repl, path.read_text()))
    return count


def resolve_pgn(user: str | None = None) -> tuple[Path, str]:
    """The PGN to read and whose moves to sample. Downloads the games if the file is missing."""
    user = user or C.PLAYER
    path = C.PGN if user == C.PLAYER else C.DATA_DIR / f"lichess_{user.lower()}.pgn"
    if not path.exists():
        fetch_lichess(user, path)
        anonymise_pgn(path, keep=user)
    return path, user


def read_games(path: Path):
    with open(path, encoding="utf-8", errors="replace") as f:
        while (game := chess.pgn.read_game(f)) is not None:
            yield game


def _usable(game) -> bool:
    h = game.headers
    if h.get("Variant", "Standard") != "Standard" or "FEN" in h:
        return False
    if any(s in h.get("Event", "") for s in C.SKIP_EVENTS):
        return False
    # Moves against bots are still human moves, but bots play strangely; skip them.
    return not (h.get("WhiteTitle") == "BOT" or h.get("BlackTitle") == "BOT")


def _elo(h, color):
    try:
        return int(h.get("WhiteElo" if color == chess.WHITE else "BlackElo", ""))
    except ValueError:
        return None


def sample_positions(
    user=None, n=C.N_POSITIONS, min_ply=C.SAMPLE_MIN_PLY, max_ply=C.SAMPLE_MAX_PLY, seed=C.SEED
) -> tuple[list[Sample], dict]:
    """Sample up to n middlegame positions where `user` is to move, with the move they played."""
    path, user = resolve_pgn(user)
    candidates, games = [], 0
    for game in read_games(path):
        if not _usable(game):
            continue
        h = game.headers
        side = None
        if user:
            if h.get("White", "").lower() == user.lower():
                side = chess.WHITE
            elif h.get("Black", "").lower() == user.lower():
                side = chess.BLACK
            else:
                continue
        games += 1
        board = game.board()
        played = []
        game_id = h.get("GameId") or h.get("Site", "").rsplit("/", 1)[-1] or f"g{games}"
        for node in game.mainline():
            move = node.move
            ply = board.ply()
            if (
                min_ply <= ply <= max_ply
                and (side is None or board.turn == side)
                and not board.is_check()
                and board.legal_moves.count() > 1
            ):
                candidates.append(
                    Sample(
                        game_id=game_id,
                        ply=ply,
                        fen=board.fen(),
                        move_uci=move.uci(),
                        move_san=board.san(move),
                        player=h.get("White" if board.turn == chess.WHITE else "Black", "?"),
                        player_elo=_elo(h, board.turn),
                        event=h.get("Event", ""),
                        history=list(played),
                    )
                )
            board.push(move)
            played.append(move.uci())
    rng = random.Random(seed)
    rng.shuffle(candidates)
    picked = sorted(candidates[:n], key=lambda s: (s.game_id, s.ply))
    elos = [s.player_elo for s in picked if s.player_elo]
    meta = {
        "source": str(path.relative_to(C.ROOT)),
        "user": user,
        "games": games,
        "candidates": len(candidates),
        "sampled": len(picked),
        "player_elo_median": int(statistics.median(elos)) if elos else None,
        "player_elo_range": [min(elos), max(elos)] if elos else None,
        "min_ply": min_ply,
        "max_ply": max_ply,
        "seed": seed,
    }
    return picked, meta


def board_from_entry(entry: dict) -> chess.Board:
    """Board for a positions.json entry, replaying its game 'history' (UCI moves) when present
    so Maia receives real previous positions instead of lc0's synthesised history."""
    if entry.get("history"):
        board = chess.Board()
        for u in entry["history"]:
            board.push_uci(u)
        if board.fen() == entry["fen"]:
            return board
    return chess.Board(entry["fen"])


def history_index(user=None) -> dict:
    """(game_id, fen) -> list of UCI moves leading to that position, for every game in the PGN."""
    path, _ = resolve_pgn(user)
    index = {}
    for game in read_games(path):
        if not _usable(game):
            continue
        h = game.headers
        game_id = h.get("GameId") or h.get("Site", "").rsplit("/", 1)[-1]
        board, played = game.board(), []
        for move in game.mainline_moves():
            index[(game_id, board.fen())] = list(played)
            board.push(move)
            played.append(move.uci())
    return index
