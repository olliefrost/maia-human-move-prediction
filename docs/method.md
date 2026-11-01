# Method details

This document describes exactly what my code does. Settings live in `maia_study/config.py`.

## Maia

Maia (McIlroy-Young, Sen, Kleinberg, Anderson, KDD 2020) is a family of nine neural networks in the
Leela Chess Zero (lc0) format, one per rating band from 1100 to 1900. Each is a residual convolutional
network whose policy head ends in a softmax over moves. Each network was trained with supervised
learning on Lichess games between players in its rating band: the loss is the cross-entropy between
the softmax output and the move the human actually played. According to the paper, the training data
is games from January 2016 to November 2019 (about 12 million games per band), with December 2019
reserved for testing. The test sets drop Bullet and HyperBullet games, the first 10 plies of each game
and moves made with under 30 seconds left.

The nine networks are separate models, not one model with a rating input, so their predictions do not
change smoothly from one rating to the next.

I never train Maia. All my games (2022 to 2026) postdate Maia's training data, so every Maia number here
is a test result.

## How Maia's probabilities are read (`engines.py`, `Engines.maia_distribution`)

1. lc0 is started with these UCI options:
   - `WeightsFile = weights/maia-<rating>.pb.gz`
   - `Backend = blas` (CPU)
   - `VerboseMoveStats = true`: lc0 prints one `info string` line per legal move
   - `PolicyTemperature = 1.0`
   - `Threads = 1`
2. The position is sent with its full move list (`position startpos moves ...`), followed by
   `go nodes 1`. With one node, lc0 evaluates the network once on the root position and stops, with no
   tree search. The paper's move-prediction results also use the policy network without search.
3. Each line looks like
   `info string e2e4  (322 ) N: 0 (+ 0) (P: 65.13%) (WL: -.-----) ...`.
   `P` is the move's policy prior, the softmax probability the network gives it after illegal moves are
   masked. The move and P are extracted with a regular expression.
4. The moves are matched to python-chess's legal moves (castling is accepted as both `e1g1` and `e1h1`),
   then divided by their sum so the distribution adds up to exactly 1. The raw P values sum to 1 ± about
   0.001 because they are rounded to 0.01%.
5. A new game (`ucinewgame`) is started for every call, so nothing is reused between positions.

### Why these settings

- **Policy temperature.** lc0's default `PolicyTemperature` is 1.359, which flattens the policy to
  widen tree search. At the default, Maia 1500 gives 1.e4 50.2% from the starting position; at 1.0 (the
  network's own softmax) it gives 65.1%. Other Maia front-ends may use different settings, so their
  percentages can differ.
- **Move history.** lc0's input holds the current position and the 7 before it. Given only a FEN, lc0
  fills that history by repeating the current position, which is not what Maia saw in training.
  Replaying each game from the start gives the network the real previous positions. On the 1000
  sampled positions this raised Maia 1100's top-1 match from 48.5% to 51.3%. The paper describes
  giving Maia the previous 12 plies, so the history here is shorter than in the paper.
- **Resolution.** lc0 prints P to 0.01%, so very unlikely moves read as 0. Where a log is taken (the
  supplementary cross-entropy), p is floored at 1e-4. The floor affected 4 of the 9000
  (position, rating) pairs.
- **No search.** With more nodes, lc0 would run tree search with the Maia network and play stronger,
  less human-like moves. Every Maia number here uses one node.

## Stockfish

- Stockfish 17.1 built from the official source (`ARCH=armv8-dotprod`), 4 threads, 128 MB hash.
- The analysis uses depth 14 with MultiPV 3. That gives Stockfish's top-3 moves for the top-3 comparison,
  and my move's eval whenever it is one of them. Otherwise my move is searched on its own
  (`searchmoves`) to the same depth.
- Evals are from the mover's point of view. Mates are scored as ±10000 and every eval is capped at
  ±1000 cp, so one missed mate counts as at most a 2000 cp loss.
- **Centipawn loss** = eval of Stockfish's best move minus eval of the move I played (0 if they are the
  same move).
- **Mistake** = centipawn loss ≥ 100 (one pawn) and not Stockfish's best move. The paper defines
  blunders through a drop in win probability instead; the centipawn threshold is my choice.
  50 and 200 cp are reported as sensitivity checks.
- Stockfish's evals of Maia's top-1 moves outside its own top 3 come from a separate top-up run
  (`topup.py`, 426 searches at the same depth). They are used only for the supplementary centipawn-loss
  comparison.
- Multi-threaded search is not deterministic: evals vary by a few centipawns between runs. The published
  numbers come from the saved analysis in `results/cache/`.

## Data and sampling

- My games, from my Lichess account `osovv` via the Lichess API (`/api/games/user/osovv`, no clocks, no
  evals). Lichess returns 404 unless the request has a descriptive `User-Agent`.
- Kept: standard variant, normal starting position, not ultra-bullet or correspondence, no bot opponent.
  That leaves 78 of the 93 games.
- Candidates: plies 16 to 70 where I was to move, not in check, with more than one legal move. That
  gives 1413 candidates; I draw 1000 at random (Python `random.Random(7)`). By time control: 621 blitz,
  252 bullet, 108 rapid, 19 classical.
- The analysis cache is named by a hash of the sampling settings
  (`results/cache/analysis_462a5efce6.jsonl`). Re-sampling from the committed PGN gives the same 1000
  positions, which `tests/test_smoke.py` checks.
- Opponent names in the PGN are replaced with `Opponent` (`data.anonymise_pgn`). Only the two name headers
  change, so sampling and every result are unaffected.

## Clean set and statistics (`stats.py`)

Filters, in order: ply ≥ 20 (98 removed), at least 2 legal moves (0 removed), |eval before the move| ≤
500 cp (375 removed). That leaves 527 positions from 67 games. Positions with |eval| > 500 cp are dropped
because, once a game is decided, the moves say little about normal decision-making.

- Proportions: Wilson 95% intervals (positions treated as independent).
- Means, medians and their differences: bootstrap 95% intervals with 2000 resamples of whole games
  (seed 7), because positions from the same game are correlated.
- Paired top-1 comparisons on the same positions: exact two-sided McNemar test.
- Rating trend: least-squares slope of top-1 against Maia rating, with a game-level bootstrap interval.
- My rating: 1359, the median of my in-game ratings across the games used (range 995 to 1748, mostly
  provisional). The Maia model closest to it is 1400.
