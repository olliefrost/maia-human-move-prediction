# Position 1: Maia and Stockfish want completely different moves

- Source: osovv game VQBP2Pua, move 15 (casual blitz game)
- White to move
- FEN: `r6k/ppp3p1/2np3p/2b5/2B1P2B/2N2r1q/PPP2P2/R2Q1RK1 w - - 0 15`

| Maia 1100 top 3 | p |
|---|---|
| Bg3 | 53.5% |
| Qxf3 | 26.0% |
| Nd5 | 3.3% |

| Stockfish top 3 (depth 14) | eval (White's view) |
|---|---|
| Be6 | +1.85 |
| Qxf3 | -5.30 |
| Ne2 | -7.47 |

- My move: **Bg3**
- Maia 1100 probability on my move: 53.5% (rank 1 of 37)
- My centipawn loss: 1185

**Why it is interesting**

Stockfish sees exactly one good move here, Be6, which keeps a clear advantage; its next-best choice is already about seven pawns worse. Every Maia model from 1100 to 1900 gives Be6 under 2%, while Maia 1100 put 53% on Bg3, the move I actually played, and Bg3 threw the game away.
