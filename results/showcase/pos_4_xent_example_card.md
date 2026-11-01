# Position 4: Cross-entropy worked example

- Source: osovv game 42CHHlG2, move 17 (rated blitz game)
- White to move
- FEN: `rn1qk2r/pp3p2/2p1p1p1/3pP1Pp/3P3P/bPNQBN2/PnP2P2/1K1R3R w kq - 3 17`

| Maia 1100 top 3 | p |
|---|---|
| Qe2 | 34.4% |
| Qd2 | 31.3% |
| Rd2 | 9.5% |

| Stockfish top 3 (depth 14) | eval (White's view) |
|---|---|
| Qe2 | -1.98 |
| Qd2 | -2.20 |
| Qf1 | -2.34 |

- My move: **Qd2**
- Maia 1100 probability on my move: 31.3% (rank 2 of 34)
- My centipawn loss: 22

**Why it is interesting**

Maia 1100 split its prediction between Qe2 (34%) and Qd2 (31%); I played Qd2, so the loss is -log 0.313 = 1.16 nats even though Maia's top choice was a different move. The two moves are only 22 centipawns apart for Stockfish, so this is a fair, low-drama example of the loss at work.
