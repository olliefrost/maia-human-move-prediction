# Supplementary results

The headline analysis in the README uses top-1 and top-3 match. This document collects the other
measurements made on the same 527 positions, with notes on how far each one can be trusted. All
numbers come from `results/headline_numbers.json` and `results/figure_data/*.csv`.

## Every Maia rating

Top-1 match with the player's move:

| Maia | 1100 | 1200 | 1300 | 1400 | 1500 | 1600 | 1700 | 1800 | 1900 |
|---|---|---|---|---|---|---|---|---|---|
| Top-1 | 50.9% | 49.9% | 50.3% | 49.3% | 46.5% | 46.9% | 47.4% | 44.6% | 42.9% |

- Maia 1100 vs 1200, 1300 and 1400: all within noise (McNemar p = 0.59, 0.81, 0.46).
- Eight pairwise comparisons against 1100 were run. 1100 vs 1500 (p = 0.028) and 1100 vs 1600
  (p = 0.042) are below 0.05, but with a Bonferroni correction (0.05 / 8) only 1100 vs 1800 (p = 0.002)
  and 1100 vs 1900 (p = 3e-5) survive. The overall downward trend is the supported claim, not
  individual pairs.
- Maia's agreement with Stockfish rises with rating, from 34.5% (1100) to 41.9% (1900), consistent
  with stronger players choosing more engine-like moves.
- Agreement breakdown (Maia 1100, Stockfish, the player): all three agree on 21.1% of positions; Maia and
  the player only, 29.8%; Stockfish and the player only, 7.8%; Maia and Stockfish only, 13.5%; all
  different, 27.9% (figure 7).

## Cross-entropy (figures 2, 8 and 9)

Mean -log p of the move played, with p floored at 1e-4:

- Maia 1100: 1.565 nats [1.445, 1.694], rising steadily to Maia 1900: 1.865 [1.694, 2.046]. A uniform
  guess over legal moves: 3.510.
- As perplexity, Maia 1100 was on average as unsure as a choice between 4.8 equally likely moves.
- Paired game-level bootstrap against 1100: 1200 and 1300 are within noise; 1400 is worse by 0.04 nats
  (interval just above 0); from 1500 up the gap grows from 0.09 to 0.30 nats.
- This is the same formula as Maia's training loss, but evaluated on games the networks never saw, so it
  is a test loss.
- Figure 9 shows that top-1 falls with rating more clearly than the probability on the player's move
  does: top-1 is all-or-nothing, while probability gives partial credit.

Calibration (whether a predicted 30% comes true 30% of the time) was not measured.

## Probability on the player's move (figures 3 and 5)

- Median probability Maia 1100 gave the move played: 29.1% (mean 35.3%), about nine times the random
  baseline of 3.1%.
- Non-mistakes: median 36.0%. Mistakes (≥ 100 cp): median 11.1% [8.6, 17.9]. Mean gap 19 points
  [14, 25].
- Spearman correlation between centipawn loss and Maia 1100's probability: -0.44.
- 76% of the 83 moves that lost 200 cp or more still got more probability than a random guess would.
- Mistake threshold sensitivity (Maia 1100 top-3): 50 cp, 225 mistakes, 66.7%; 100 cp, 146 mistakes,
  60.3%; 200 cp, 83 mistakes, 55.4%.

## Centipawn loss of the top-1 move

The player's moves lost 100 cp on average [84, 117]. Maia 1100's top-1 move would have lost 66 [54, 79];
Maia 1900's, 51 [41, 64]. Stockfish loses 0 by definition, because its best move is the reference. Maia's
most likely move is a typical move, and typical moves are on average better than a player's worst
moments. These numbers use the 426 extra Stockfish searches from `topup.py`.

## Toy model (figure 10)

Maia was not trained on the player's games, so there is no Maia training curve for this data. To show
real training and test loss curves, `toy_model.py` trains a small model with the same objective as
Maia's policy head: score every legal move, softmax over the legal moves, minimise cross-entropy against
the move played.

- The model is linear on 145 hand-made move features (piece type, from and to squares, capture, check,
  hanging piece and others), not a CNN. Adam, full batch, learning rate 0.05, 300 epochs, no
  regularisation.
- Data: the clean set, split by game (seed 7): 54 games / 391 positions for training, 13 games / 136
  positions for testing.
- Training loss falls to 2.31 nats. Test loss is lowest at epoch 43 (2.75) and then rises to 3.00: the
  model overfits square-specific patterns from 54 games. Maia 1100 scores 1.48 on the same test positions,
  and a uniform guess 3.41.
- Across 10 random game splits (seeds 1 to 10, which include the main split), the lowest test loss falls
  between epoch 42 and epoch 239. Test loss ends more than 0.1 nats above its minimum in 5 of the 10;
  in the others the test curve flattens. With 13 test games, the exact numbers are
  noisy (`results/figure_data/fig10_other_splits.csv`).

## Showcase positions and video assets

- Showcase positions (`showcase.py`, `results/showcase/`) are chosen by fixed rules from the clean set:
  (1) Stockfish's move is outside the top 5 of both Maia 1100 and 1900 and Stockfish's top move stands
  well clear of its second choice; (2) a move losing at least 150 cp that was Maia 1100's first choice;
  (3) the player found Stockfish's move but Maia 1100's top choice differed; (4) a worked example for
  cross-entropy. Distinct games where possible. The notes on each card were written after seeing the
  numbers.
- The figure 6 position is the one, among positions where Maia 1100 and 1900 pick different moves, whose
  distributions differ most (total-variation distance 0.58).
- Video assets (`video_assets.py`) run no engines. In the rating-sweep clips, each held rating shows the
  exact stored values. The frames between ratings are interpolated for the animation only and are not
  real model outputs. The agreement clip uses the clean-set position where every Maia rating's top move
  was the player's move, Stockfish preferred another move and the move lost under 50 cp, choosing the
  one with the highest minimum probability across ratings.
- The running example is game 4hGkJ3rK, move 20, where Nf8 was played (Stockfish: Nf4; Maia 1100: Nxh5).
  Bxb4, Maia 1500–1700's top choice there, was played one move later.
  `python -m maia_study video --running-move 21` builds the assets for that position instead.

## Unfiltered comparison

On all 1000 sampled positions, before the clean-set filters, Maia 1100's top-1 match is 51.3%, against
50.9% on the clean set. The filters do not drive the conclusions.
