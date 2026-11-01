"""Supplementary: a small move-prediction model trained on my games (figure 10).

Maia was trained by its authors on Lichess games up to November 2019, not on these games, so there is
no Maia training curve on this data. This trains a small model with the same objective as Maia's
policy head (score every legal move, softmax over the legal moves, cross-entropy against the move
actually played, gradient descent) to show real training and test loss curves. It is a linear model
on hand-made move features, not a CNN.

The split is by game (all positions from one game go to the same side), so the test set is unseen games.
"""

import json

import chess
import numpy as np

from . import config as C
from . import stats

OUT = C.TOY_CURVES
TEST_FRAC = 0.2
EPOCHS = 300
LR = 0.05
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

FEATURES = (
    [f"piece_{chess.piece_name(p)}" for p in chess.PIECE_TYPES]
    + [f"from_{chess.square_name(s)}" for s in chess.SQUARES]
    + [f"to_{chess.square_name(s)}" for s in chess.SQUARES]
    + [
        "capture",
        "captured_value",
        "capture_gain",
        "gives_check",
        "promotion",
        "castling",
        "to_attacked",
        "to_defended",
        "hangs",
        "from_attacked",
        "forward_ranks",
    ]
)


def move_features(board: chess.Board, mv: chess.Move) -> np.ndarray:
    """Features from the mover's point of view (squares mirrored for Black)."""
    me = board.turn
    orient = (lambda s: s) if me == chess.WHITE else chess.square_mirror
    pt = board.piece_type_at(mv.from_square)
    victim = board.piece_type_at(mv.to_square) or (chess.PAWN if board.is_en_passant(mv) else None)
    f = np.zeros(len(FEATURES))
    f[pt - 1] = 1
    f[6 + orient(mv.from_square)] = 1
    f[70 + orient(mv.to_square)] = 1
    i = 134
    from_attacked = board.is_attacked_by(not me, mv.from_square)
    gives_check = board.gives_check(mv)
    after = board.copy(stack=False)
    after.push(mv)
    to_att = after.is_attacked_by(not me, mv.to_square)
    to_def = after.is_attacked_by(me, mv.to_square)
    fwd = chess.square_rank(orient(mv.to_square)) - chess.square_rank(orient(mv.from_square))
    f[i:] = [
        victim is not None,
        (VALUE[victim] if victim else 0) / 9,
        ((VALUE[victim] if victim else 0) - (VALUE[pt] if to_att else 0)) / 9,
        gives_check,
        mv.promotion == chess.QUEEN,
        board.is_castling(mv),
        to_att,
        to_def,
        to_att and not to_def,
        from_attacked,
        fwd / 7,
    ]
    return f


def build(df):
    """One (n_legal x n_features) matrix per position + the index of the move I played."""
    data = []
    for r in df.itertuples():
        board = chess.Board(r.fen)
        moves = list(board.legal_moves)
        X = np.stack([move_features(board, m) for m in moves])
        y = [m.uci() for m in moves].index(r.move_uci)
        data.append((X, y))
    return data


def loss_and_grad(w, data):
    tot, g = 0.0, np.zeros_like(w)
    for X, y in data:
        z = X @ w
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        tot -= np.log(p[y])
        g += X.T @ p - X[y]  # gradient of cross-entropy w.r.t. w: sum_j p_j x_j - x_played
    return tot / len(data), g / len(data)


def top1(w, data):
    return float(np.mean([int(np.argmax(X @ w) == y) for X, y in data]))


def run(df, seed):
    games = sorted(df.game_id.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_test = round(len(games) * TEST_FRAC)
    test_games = set(games[:n_test])
    tr_df, te_df = df[~df.game_id.isin(test_games)], df[df.game_id.isin(test_games)]
    train, test = build(tr_df), build(te_df)

    # Adam, full batch, no regularisation (so overfitting is visible)
    w = np.zeros(len(FEATURES))
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    b1, b2, eps = 0.9, 0.999, 1e-8
    curve = []
    for epoch in range(EPOCHS + 1):
        tr_loss, g = loss_and_grad(w, train)
        te_loss, _ = loss_and_grad(w, test)
        curve.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "test_loss": te_loss,
                "train_top1": top1(w, train),
                "test_top1": top1(w, test),
            }
        )
        if epoch == EPOCHS:
            break
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w -= LR * (m / (1 - b1 ** (epoch + 1))) / (np.sqrt(v / (1 - b2 ** (epoch + 1))) + eps)

    best = min(curve, key=lambda c: c["test_loss"])
    ref = {}
    for name, d in (("train", tr_df), ("test", te_df)):
        ref[name] = {
            "maia_1100": float(stats.xent(d["p_1100"]).mean()),
            "uniform": float(np.log(d.n_legal).mean()),
            "positions": len(d),
            "games": int(d.game_id.nunique()),
        }
    return {
        "model": "linear softmax over legal moves (NOT Maia, NOT a CNN)",
        "features": len(FEATURES),
        "epochs": EPOCHS,
        "lr": LR,
        "optimizer": "Adam, full batch, no regularisation",
        "split": f"by game, {TEST_FRAC:.0%} of games held out, seed {seed}",
        "test_games": sorted(test_games),
        "reference": ref,
        "best_epoch": best,
        "curve": curve,
    }


def main():
    df, _, _ = stats.clean()
    out = run(df, C.SEED)
    ref, curve, best = out["reference"], out["curve"], out["best_epoch"]
    # robustness: does the same shape appear with other random game splits?
    out["other_splits"] = []
    for seed in range(1, 11):
        r = run(df, seed)
        out["other_splits"].append(
            {
                "seed": seed,
                "best_epoch": r["best_epoch"]["epoch"],
                "min_test_loss": r["best_epoch"]["test_loss"],
                "final_test_loss": r["curve"][-1]["test_loss"],
                "final_train_loss": r["curve"][-1]["train_loss"],
                "maia_1100_test": r["reference"]["test"]["maia_1100"],
            }
        )
    OUT.write_text(json.dumps(out, indent=1))
    print(
        f"train: {ref['train']['positions']} positions / {ref['train']['games']} games · "
        f"test: {ref['test']['positions']} positions / {ref['test']['games']} games"
    )
    for c in curve[:: EPOCHS // 10]:
        print(
            f"epoch {c['epoch']:4d}  train {c['train_loss']:.3f}  test {c['test_loss']:.3f}  "
            f"top1 {c['train_top1']:.1%}/{c['test_top1']:.1%}"
        )
    print(
        f"best test loss {best['test_loss']:.3f} at epoch {best['epoch']}; "
        f"Maia 1100 on the same test positions {ref['test']['maia_1100']:.3f}; "
        f"uniform {ref['test']['uniform']:.3f}"
    )
    for o in out["other_splits"]:
        print(
            f"  seed {o['seed']:2d}: best epoch {o['best_epoch']:3d}  min test {o['min_test_loss']:.3f}  "
            f"final test {o['final_test_loss']:.3f}  final train {o['final_train_loss']:.3f}  "
            f"Maia1100 {o['maia_1100_test']:.3f}"
        )
    print(f"wrote {OUT.relative_to(C.ROOT)}")
