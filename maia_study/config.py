"""Every setting the pipeline depends on, in one place.

Paths are relative to the repository root. Engine locations can be overridden with the
environment variables MAIA_LC0, MAIA_STOCKFISH and MAIA_WEIGHTS_DIR.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- inputs
PLAYER = "osovv"  # my Lichess account: the moves being predicted
DATA_DIR = ROOT / "data"
# This exact relative path is part of the analysis cache key (see analysis.py), so keep it.
PGN = DATA_DIR / "lichess_osovv.pgn"
DEMO_POSITIONS = DATA_DIR / "positions.json"  # positions shown in the GUI demo
# I exported the PGN on 2026-10-28; it covers my games from 2022-10-01 to 2026-10-28.
GAMES_DATE_RANGE = ("2022-10-01", "2026-10-28")
# Lichess returns 404 to API requests without a descriptive User-Agent.
LICHESS_USER_AGENT = "maia-human-move-prediction/1.0 (university project)"
LICHESS_MAX_GAMES = 300
# Ultra-bullet and correspondence are skipped: neither matches the time controls Maia was trained on.
SKIP_EVENTS = ("ultraBullet", "correspondence")

# ---------------------------------------------------------------- position sampling
N_POSITIONS = 1000
SAMPLE_MIN_PLY = 16  # sample from move 8 onwards ...
SAMPLE_MAX_PLY = 70  # ... up to move 35
SEED = 7  # sampling shuffle and bootstrap resampling

# ---------------------------------------------------------------- engines
RATINGS = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
MAIA_URL = "https://github.com/CSSLab/maia-chess/releases/download/v1.0/maia-{}.pb.gz"
LC0_BACKEND = "blas"  # CPU backend; the numbers were produced with blas
# 1.0 gives the network's raw softmax. lc0's default (1.359) flattens the policy for search.
LC0_POLICY_TEMPERATURE = 1.0
LC0_NODES = 1  # one forward pass, no search: Maia's policy is the prediction

STOCKFISH_TAG = "sf_17.1"
STOCKFISH_REPO = "https://github.com/official-stockfish/Stockfish.git"
SF_DEPTH = 14
# MultiPV 3 gives Stockfish's top-3 for the top-3 comparison. It also gives the human move's eval
# for free whenever the human move is among them; otherwise one extra search is run.
SF_MULTIPV = 3
SF_THREADS = 4  # multi-threaded search is not bit-reproducible; reuse the cache for exact numbers
SF_HASH_MB = 128
# Evals are capped so one missed mate does not count as a 10000 cp loss.
CP_CAP = 1000
MATE_SCORE = 10000

LC0 = Path(os.environ.get("MAIA_LC0", ROOT / "lc0" / "build" / "release" / "lc0"))
STOCKFISH = Path(os.environ.get("MAIA_STOCKFISH", ROOT / "engines" / "stockfish" / "stockfish"))
WEIGHTS_DIR = Path(os.environ.get("MAIA_WEIGHTS_DIR", ROOT / "weights"))
ENGINES_DIR = ROOT / "engines"  # where setup_engines.py builds Stockfish

# ---------------------------------------------------------------- clean analysis set
MIN_PLY = 20  # drop the first 10 moves: opening/book play says little about my own decisions
MIN_LEGAL = 2  # forced moves carry no information
MAX_ABS_EVAL = 500  # drop positions already decided (|Stockfish eval before the move| > 5 pawns)

# A mistake is a move that loses at least MISTAKE_CP centipawns against Stockfish's best move
# (and is not that move). 50 and 200 are reported as sensitivity checks.
MISTAKE_CP = 100
MISTAKE_SENSITIVITY = [50, 100, 200]

# lc0 prints P to 0.01%, so moves below that read as 0; floor p before taking -log p.
EPS = 1e-4
N_BOOT = 2000  # bootstrap resamples (whole games, see stats.py)

# Median of my in-game ratings across the games used (995 to 1748, mostly provisional).
MY_RATING = 1359

# ---------------------------------------------------------------- outputs
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
# The analysis run behind every number: 1000 positions, Maia given the real move history.
ANALYSIS = CACHE / "analysis_462a5efce6.jsonl"
TOPUP = CACHE / "topup_maia_top1_cp.jsonl"  # Stockfish evals of Maia's top-1 moves (topup.py)
SHOWCASE_DISTS = CACHE / "showcase_full_dists.json"  # full Maia distributions (showcase.py)
TOY_CURVES = CACHE / "toy_model_curves.json"  # toy_model.py
HEADLINE = RESULTS / "headline_numbers.json"
NO_BULLET = RESULTS / "no_bullet_check.json"
FIG_DATA = RESULTS / "figure_data"
SHOWCASE_DIR = RESULTS / "showcase"
SHOWCASE_JSON = SHOWCASE_DIR / "showcase.json"

FIGURES = ROOT / "figures"
FIG_DIR = FIGURES / "charts"
POS_DIR = FIGURES / "positions"
VIDEO_DIR = FIGURES / "video"

# ---------------------------------------------------------------- colours
# Colour-blind-safe categorical set: Maia blue, Stockfish green, me orange.
THEMES = {
    "light": {
        "bg": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "grid": "#e4e3df",
        "maia": "#2a78d6",
        "maia_soft": "#86b6ef",
        "sf": "#1baf7a",
        "me": "#eb6834",
        "neutral": "#9a9890",
    },
    "dark": {
        "bg": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "grid": "#33322f",
        "maia": "#3987e5",
        "maia_soft": "#1c5cab",
        "sf": "#199e70",
        "me": "#d95926",
        "neutral": "#8a897f",
    },
}
