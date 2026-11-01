# How well does Maia predict my moves?

Maia is a set of chess networks trained to predict the move a human of a given rating will play,
rather than the best move. In this project I test how well Maia predicts my own moves (my Lichess
account is `osovv`), and compare it with Stockfish, an engine built to find the best move. It is the
code behind my mid-term video for COMS20018 (Introduction to AI). Every number and figure in the video
can be reproduced from this repository.

**Question.** On positions from my own games, how often does each model's first choice match the move I
actually played? And when I made a mistake, did the model see it coming?

## Headline results

527 positions from 67 of my rated and casual Lichess games (2022 to 2026). My median in-game rating is
about 1359.

| Model | Top-1 match | 95% CI | Top-3 match |
|---|---:|---:|---:|
| Maia 1100 | 50.9% (268/527) | [46.6, 55.1] | 78.0% |
| Maia 1400 (nearest my rating) | 49.3% (260/527) | [45.1, 53.6] | 78.6% |
| Maia 1900 | 42.9% (226/527) | [38.7, 47.1] | 71.9% |
| Stockfish 17.1 (depth 14) | 28.8% (152/527) | [25.1, 32.9] | 55.8% |
| Random legal move (mean 1/n_legal) | 3.1% | | |

- **Every Maia model predicts my moves better than Stockfish.** Maia 1100 vs Stockfish on the same
  positions: exact McNemar p = 3e-17.
- **Maia 1100 to 1400 are statistically tied** (1100 vs 1400: p = 0.46). Top-1 falls by about 0.9
  percentage points per 100 Maia rating points (95% CI -1.5 to -0.4).
- **Maia 1100 and Stockfish agree with each other on 34.5%** of positions [30.6, 38.7].

**Mistakes** (a move that lost at least 100 cp against Stockfish's best move): 146 of my 527 moves, from
58 games.

| On my 146 mistakes | Ranked the mistake first | Mistake in its top 3 |
|---|---:|---:|
| Maia 1100 | 26.7% [20.2, 34.4] | 60.3% [52.2, 67.9] |
| Stockfish | 0% (by definition) | 19.2% [13.6, 26.3] |

**Without bullet games** (396 positions from 51 games): Maia 1100 48.2%, Maia 1400 49.0%, Stockfish 29.0%,
random 3.1%. The conclusions don't change.

All numbers: [`results/headline_numbers.json`](results/headline_numbers.json) and
[`results/no_bullet_check.json`](results/no_bullet_check.json).

## Method

**Data.** My games from my Lichess account `osovv`, 2022-10-01 to 2026-10-28, exported from the Lichess
API without clock times and stored in [`data/lichess_osovv.pgn`](data/lichess_osovv.pgn) (opponent names
replaced with `Opponent`). I keep standard-variant games and exclude ultra-bullet, correspondence, bot
games and games from a custom starting position. That leaves 78 games.

**Sampling.** Every position from ply 16 to 70 (move 8 to 35) where I was to move, was not in check and
had more than one legal move: 1413 candidates. I draw 1000 of them at random (seed 7).

**Filters** (applied in order, to every number and figure):

| Rule | Why | Removed | Left |
|---|---|---:|---:|
| sampled | | | 1000 |
| ply ≥ 20 (drop the first 10 moves) | opening play is mostly memorised (the paper drops the first 10 *plies*) | 98 | 902 |
| at least 2 legal moves | forced moves carry no information | 0 | 902 |
| \|Stockfish eval before the move\| ≤ 500 cp | drop positions where the game is already decided | 375 | 527 |

The 527 positions come from 67 games. The |eval| ≤ 500 cp rule and the 20-ply cut are my choices, not
the paper's.

**Engines.**
- **Maia 1100 to 1900**: the nine official Maia networks, run in lc0 with `go nodes 1` (one forward pass,
  no search), `PolicyTemperature = 1.0` (the raw softmax; lc0's default 1.359 flattens it) and
  `VerboseMoveStats`. The per-move policy prior P is parsed and renormalised over the legal moves. Each
  position is sent with the full move list from the start of the game, so lc0 builds the history input
  from real previous positions rather than from a bare FEN.
- **Stockfish 17.1**: depth 14, MultiPV 3, 4 threads, 128 MB hash. Evals are from the mover's point of
  view and capped at ±1000 cp. If my move is not in Stockfish's top 3, it is searched separately to the
  same depth.

**Metrics.**
- *Top-1 match*: the model's most likely (Maia) or best (Stockfish) move equals the move I played.
- *Top-3 match*: the move I played is among the model's top 3.
- *Centipawn loss*: eval of Stockfish's best move minus eval of the move I played, both at depth 14.
- *Mistake*: centipawn loss ≥ 100 and not Stockfish's best move. 50 and 200 cp are reported as
  sensitivity checks. Stockfish's top-1 on mistakes is 0% by construction.
- *Random baseline*: the mean of 1/(number of legal moves).
- *Intervals*: Wilson 95% intervals for proportions; exact McNemar tests for paired top-1 comparisons;
  for means, a bootstrap (2000 resamples) that resamples whole games, because positions from one game
  are correlated.

Full details of the probability extraction and engine settings: [`docs/method.md`](docs/method.md).

## Limitations

- **The paper's clock filter could not be applied.** Maia's test sets drop moves made with under 30
  seconds left. I exported my games without clock times, so I could not apply this filter.
- **Bullet games are included.** The paper's test sets exclude Bullet and HyperBullet games. Here, 131 of
  my 527 positions come from bullet games. Removing them (396 positions from 51 games) gives similar
  results (see above).
- **I highlighted Maia 1100 after seeing the results.** It scored highest, so I use it in the figures,
  but I did not decide this in advance. I therefore also report Maia 1400, the model nearest my median
  rating (about 1359), and 1100 to 1400 cannot be separated statistically. No Maia
  model below 1100 exists, so the best fit could be lower.
- **One player (me), a few hundred positions.** The intervals are wide (about ±4 percentage points on
  top-1), and the results describe how well Maia fits me, not humans in general.
- **History length differs from the paper.** The paper describes giving Maia the previous 12 plies. lc0's
  input format holds the current position and the 7 before it, so here the network sees up to 7
  previous plies.
- **Only top-k match is used for the headline analysis.** Cross-entropy was also computed, as a
  supplementary check ([`docs/supplementary.md`](docs/supplementary.md)). Calibration was not measured.
- **Stockfish is the measuring stick, not ground truth.** Centipawn losses and mistake labels depend
  on a depth-14 search. Multi-threaded Stockfish is also not deterministic, so a fresh engine run
  moves the numbers by a fraction of a point. The exact published numbers come from the saved analysis.
- **My ratings are provisional and mixed.** My ratings in these games range from 995 to 1748
  across bullet, blitz and rapid.

## Repository layout

```
maia_study/            the code (one package, one command-line entry point)
  config.py            every setting: player, sampling, engine options, filters, thresholds, paths
  data.py              Lichess download and anonymisation, position sampling with move history
  engines.py           Maia (lc0) and Stockfish wrappers; how Maia's probabilities are read
  analysis.py          the engine pass over the 1000 sampled positions (cached)
  topup.py             extra Stockfish evals for Maia's top-1 moves (cached)
  stats.py             clean-set filter, headline numbers, intervals, tests, no-bullet check
  showcase.py          the four showcase positions (boards and cards)
  figures.py           chart figures fig1 to fig10
  video_assets.py      video clips and stills
  toy_model.py         supplementary toy model (figure 10)
  terminal_views.py    terminal tables for screenshots and recordings
  gui.py, pick_positions.py   interactive demo and its positions
  setup_engines.py     downloads the Maia weights and builds Stockfish
data/                  my games (PGN) and the GUI demo positions
results/               headline numbers, per-figure data (CSV), showcase cards, engine-output cache
figures/charts/        fig1 to fig10, light and dark, PNG (1920x1080) and SVG
figures/positions/     showcase boards and cards
figures/video/         rating-sweep clips, top-1-vs-rating clip, reveal stills (MP4 + GIF)
docs/                  method details and supplementary results
tests/                 smoke tests
```

## Setup

Tested on Ubuntu 24.04 (WSL2, ARM64) with Python 3.12.3.

```bash
sudo apt install python3.12-venv libcairo2 ffmpeg python3-tk   # ffmpeg: videos; python3-tk: GUI only
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

That is enough to reproduce every number and figure from the saved analysis. Running the engines again
also needs the following. Binaries and weights are not included in the repository.

| Component | Version used | Where to get it |
|---|---|---|
| Maia weights | v1.0 release: `maia-1100.pb.gz` … `maia-1900.pb.gz` (nine files) | [CSSLab/maia-chess releases](https://github.com/CSSLab/maia-chess/releases/tag/v1.0) or `python -m maia_study setup --weights`; see [maiachess.com](https://maiachess.com) |
| lc0 | `v0.34.0-dev+git.d91bd14` (commit `d91bd1443ac1a4f367de96d3569dce7bbd35920a`), CPU `blas` backend | [LeelaChessZero/lc0](https://github.com/LeelaChessZero/lc0): build from source (below), or use a [release](https://github.com/LeelaChessZero/lc0/releases) (not tested) |
| Stockfish | 17.1 (tag `sf_17.1`), built with `ARCH=armv8-dotprod` | `python -m maia_study setup --stockfish` builds it, or download 17.1 from [stockfishchess.org](https://stockfishchess.org/download/) |

```bash
python -m maia_study setup          # Maia weights -> weights/, Stockfish 17.1 -> engines/stockfish/stockfish
git clone https://github.com/LeelaChessZero/lc0.git && cd lc0 && git checkout d91bd1443ac1
./build.sh                          # needs meson, ninja and OpenBLAS; binary at lc0/build/release/lc0
```

Engines are looked for at `lc0/build/release/lc0`, `engines/stockfish/stockfish` and `weights/`. To use
other locations, set `MAIA_LC0`, `MAIA_STOCKFISH` and `MAIA_WEIGHTS_DIR`.

SHA-256 of the weight files used (identical to the v1.0 release downloads):

```
e1cf1cd0c96b8a4fa6a275f4b9fd54ed1ffebf9fe44641b9fceded310e9619c4  maia-1100.pb.gz
ead4ba953f233ae732999ebc1e2b675378148527ebcfad2f0acbc5e4c224d98e  maia-1200.pb.gz
36195f87bf4761834baa0bf87472b18509a7261a9d7d6f1a8443261369a733f2  maia-1300.pb.gz
d5353ea6766356dad2d28920c6692f37a5f30963767f1a3105d33b4d0af011e8  maia-1400.pb.gz
35ab6f20421d59e1df3b17c5a5016947af4c6761368ef84044a9a9c7619a9a00  maia-1500.pb.gz
d2c9e5948581acf4b9fc0b1e720c5dc0fe64ce80cfc4a239d3f8a42e1176c876  maia-1600.pb.gz
d277eacd792d340a30abb464dc65127254e65cac57abca17facc469889b96478  maia-1700.pb.gz
0031ad7c4256b1fd09fbebd28418d644d68b26cd2a45df4967ccf5c7ec9c4965  maia-1800.pb.gz
e2f565f42d7cd9f122557e6dc4eb84e5bbaedceda1d404dc485d3611c7c97a12  maia-1900.pb.gz
```

## Reproducing the results

### Shortcut: from the saved analysis (no engines)

The engine output is saved in `results/cache/` (2.4 MB). From it:

```bash
.venv/bin/python -m maia_study reproduce     # about 3 minutes; --skip-video: about 1 minute
.venv/bin/python -m pytest                   # smoke tests (engine test is skipped without engines)
```

This rewrites `results/` and `figures/` and gives byte-identical numbers and pixel-identical figures.

### Each output

| Output | Command | Time |
|---|---|---|
| `results/headline_numbers.json`, `results/no_bullet_check.json` | `python -m maia_study stats` | 35 s |
| `results/showcase/`, `figures/positions/` | `python -m maia_study showcase` | 5 s |
| `figures/charts/` fig1–fig10, `results/figure_data/` | `python -m maia_study figures` (or e.g. `figures fig4`) | 15 s |
| fig10 data (`results/cache/toy_model_curves.json`) | `python -m maia_study toy-model` | 25 s |
| `figures/video/` (rating sweeps, top-1-vs-rating, reveal stills) | `python -m maia_study video` | 2.5 min |
| terminal screenshots | `python -m maia_study terminal headline` (also `mistakes`, `xent`, `position 2`, `summary`) | instant |

`showcase` and `figures` read `results/headline_numbers.json`, so run `stats` first. `reproduce` runs
them in order.

| Figure | File(s) |
|---|---|
| Top-1 match by Maia rating vs Stockfish | `figures/charts/fig1_match_by_rating_*`, `figures/video/top1_vs_rating.*` |
| Mistakes: Maia vs Stockfish | `figures/charts/fig4_mistakes_*` |
| Rating sweep, Maia 1100 vs 1900 | `figures/video/rating_sweep_1100_vs_1900.*`, `figures/charts/fig6_rating_sweep_*` |
| Rating sweep where every Maia agrees with me | `figures/video/rating_sweep_agree.*` |
| A mistake Maia expected | `figures/positions/pos_2_mistake_*`, `figures/video/rating_sweep_mistake.*` |
| The engine move Maia did not expect | `figures/positions/pos_3_engine_move_*` |
| Supplementary (cross-entropy, probabilities, toy model) | `fig2`, `fig3`, `fig5`, `fig7`, `fig8`, `fig9`, `fig10`; see [`docs/supplementary.md`](docs/supplementary.md) |

### Full rerun (engines)

```bash
python -m maia_study analyse     # 1000 positions x (9 Maia + Stockfish): about 6 minutes on a 12-core ARM64 laptop
python -m maia_study topup       # 426 extra Stockfish searches: about 30 s of search
python -m maia_study reproduce
```

`analyse` reuses `results/cache/analysis_462a5efce6.jsonl` if it exists; pass `--refresh` to rerun the
engines. A rerun samples exactly the same 1000 positions (checked by the tests), but Stockfish's
multi-threaded search can change a few evals.

`python -m maia_study fetch` downloads my latest games from Lichess and anonymises them. Newer
games change the sample, so use the committed PGN to reproduce the published results.

## Licences and data

- **Code in this repository:** MIT ([LICENSE](LICENSE)).
- **Game data:** my own games, exported through the
  [Lichess API](https://lichess.org/api) and subject to the
  [Lichess terms of service](https://lichess.org/terms-of-service). Lichess publishes its game database
  under CC0. Opponent usernames are replaced with `Opponent`; Lichess game IDs are kept so each game can
  be looked up.
- **Maia** (code and weights): [CSSLab/maia-chess](https://github.com/CSSLab/maia-chess), GPL-3.0. The
  weights are not redistributed here.
- **lc0**: GPL-3.0. **Stockfish**: GPL-3.0. Neither is redistributed. This code only talks to them as
  separate programs over the UCI protocol.

## Citation

Reid McIlroy-Young, Siddhartha Sen, Jon Kleinberg and Ashton Anderson. 2020. *Aligning Superhuman AI
with Human Behavior: Chess as a Model System.* In Proceedings of the 26th ACM SIGKDD International
Conference on Knowledge Discovery & Data Mining (KDD '20). https://doi.org/10.1145/3394486.3403219

## AI assistance

The code in this repository was written with the help of AI coding tools (Claude Code) and reviewed by me.
