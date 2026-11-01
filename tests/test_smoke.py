"""Smoke tests.

The cache tests need only the repository. The engine test runs when lc0, the Maia weights and
Stockfish are available (see README), and is skipped otherwise.
"""

import json

import chess
import pytest

from maia_study import config as C
from maia_study import stats
from maia_study.analysis import cache_path
from maia_study.data import sample_positions


def test_sampling_reproduces_cached_positions():
    samples, meta = sample_positions()
    meta["sf_depth"] = C.SF_DEPTH
    assert cache_path(meta) == C.ANALYSIS
    cached = [json.loads(line) for line in C.ANALYSIS.read_text().splitlines() if line.strip()]
    assert [(s.game_id, s.fen, s.move_uci) for s in samples] == [
        (r["game_id"], r["fen"], r["move_uci"]) for r in cached
    ]


def test_clean_set_counts():
    df, steps, full = stats.clean()
    assert len(full) == 1000
    assert [remaining for _, _, remaining in steps] == [1000, 902, 902, 527]
    assert df.game_id.nunique() == 67


def test_headline_numbers_match_saved_results():
    saved = json.loads(C.HEADLINE.read_text())
    match = saved["match"]
    assert (match["1100"]["top1"]["k"], match["1400"]["top1"]["k"], match["stockfish"]["top1"]["k"]) == (268, 260, 152)
    assert round(match["random"]["top1_expected"], 3) == 0.031
    assert saved["maia_vs_stockfish_agreement"]["1100"]["k"] == 182
    mistakes = saved["mistakes"]["100"]
    assert mistakes["n_mistakes"] == 146
    assert (mistakes["maia_1100"]["top1"]["k"], mistakes["maia_1100"]["top3"]["k"]) == (39, 88)
    assert mistakes["stockfish"]["top3"]["k"] == 28
    # recomputing from the cache gives the saved file exactly
    assert json.dumps(stats.jsonable(stats.headline()), indent=2) == C.HEADLINE.read_text()


def test_no_bullet_check():
    nb = stats.no_bullet_check()
    assert nb["clean_set"]["n_games_with_mistakes"] == 58
    assert nb["clean_set"]["n_bullet_positions"] == 131
    assert (nb["no_bullet"]["n_positions"], nb["no_bullet"]["n_games"]) == (396, 51)


ENGINES_PRESENT = (
    C.LC0.exists() and C.STOCKFISH.exists() and all((C.WEIGHTS_DIR / f"maia-{r}.pb.gz").exists() for r in C.RATINGS)
)


@pytest.mark.skipif(not ENGINES_PRESENT, reason="lc0, Maia weights or Stockfish not found")
def test_engines_return_sane_output():
    from maia_study.engines import Engines

    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    with Engines() as eng:
        for rating in (1100, 1500, 1900):
            dist = eng.maia_distribution(board, rating)
            assert set(dist) == set(board.legal_moves)
            assert abs(sum(dist.values()) - 1) < 1e-9
        lines = eng.stockfish_analysis(board, depth=12, multipv=3)
        assert len(lines) == 3
        assert board.san(lines[0].move) == "Qxf7#"
