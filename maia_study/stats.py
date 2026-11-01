"""Apply the clean-set filter to the saved analysis and compute every reported number.

Intervals:
  * proportions: Wilson 95% interval (treats positions as independent)
  * means/medians and differences: bootstrap 95% interval that resamples whole games, not positions,
    because positions from the same game are correlated (67 games, so the intervals are wider)
  * paired comparisons on the same positions: exact McNemar test (top-1 hits) or a paired game-level
    bootstrap of the difference (losses)
"""

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C


# ------------------------------------------------------------------ loading + filtering
@lru_cache(maxsize=1)
def load_records():
    return [json.loads(line) for line in C.ANALYSIS.read_text().splitlines() if line.strip()]


def to_frame(records):
    rows = []
    for i, r in enumerate(records):
        row = {k: v for k, v in r.items() if k not in ("maia",)}
        row["idx"] = i
        row["sf_gap"] = r["sf_top3_cp"][0] - r["sf_top3_cp"][1] if len(r["sf_top3_cp"]) > 1 else np.nan
        for rt in C.RATINGS:
            m = r["maia"][str(rt)]
            row[f"top1_{rt}"] = m["top1"]
            row[f"p_{rt}"] = m["p_human"]
            row[f"rank_{rt}"] = m["rank_human"]
        rows.append(row)
    return pd.DataFrame(rows)


def filter_steps(df):
    """Apply the clean-set rules in order; return (clean_df, list of (rule, removed, remaining))."""
    steps = [("all analysed positions", 0, len(df))]
    rules = [
        (f"ply >= {C.MIN_PLY} (drop first {C.MIN_PLY // 2} moves)", df.ply >= C.MIN_PLY),
        (f"legal moves >= {C.MIN_LEGAL}", df.n_legal >= C.MIN_LEGAL),
        (f"|eval before move| <= {C.MAX_ABS_EVAL} cp (game not decided)", df.sf_best_cp.abs() <= C.MAX_ABS_EVAL),
    ]
    keep = pd.Series(True, index=df.index)
    for name, mask in rules:
        before = int(keep.sum())
        keep &= mask
        steps.append((name, before - int(keep.sum()), int(keep.sum())))
    return df[keep].copy(), steps


@lru_cache(maxsize=1)
def clean():
    df = to_frame(load_records())
    cdf, steps = filter_steps(df)
    return add_topup(cdf), steps, df


def add_topup(df):
    """Attach Stockfish evals of Maia's top-1 move (from SF top-3 where possible, else the top-up file)."""
    top = {}
    if C.TOPUP.exists():
        for line in C.TOPUP.read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                top[(t["fen"], t["move"])] = t["cp"]
    for rt in C.RATINGS:
        cps = []
        for _, r in df.iterrows():
            mv = r[f"top1_{rt}"]
            if mv in r["sf_top3"]:
                cp = r["sf_top3_cp"][r["sf_top3"].index(mv)]
            else:
                cp = top.get((r["fen"], mv))
            cps.append(np.nan if cp is None else max(0, r["sf_best_cp"] - cp) if mv != r["sf_best"] else 0)
        df[f"cpl_maia_{rt}"] = cps
    return df


def topup_needed(df):
    """(fen, move) pairs whose Stockfish eval is not in the saved analysis."""
    need = set()
    for _, r in df.iterrows():
        for rt in C.RATINGS:
            mv = r[f"top1_{rt}"]
            if mv not in r["sf_top3"]:
                need.add((r["fen"], mv))
    return sorted(need)


# ------------------------------------------------------------------ statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, centre - half, centre + half


def boot_ci(df, fn, n_boot=C.N_BOOT, seed=C.SEED):
    """Game-cluster bootstrap: resample games with replacement, recompute fn on the resampled rows."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    groups = [np.asarray(ix) for ix in df.groupby("game_id").indices.values()]
    est = fn(df)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        vals.append(fn(df.take(np.concatenate([groups[i] for i in pick]))))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def mcnemar(a_hits, b_hits):
    """Exact two-sided McNemar test on paired booleans. Returns (b, c, p)."""
    a_hits, b_hits = np.asarray(a_hits, bool), np.asarray(b_hits, bool)
    b = int((a_hits & ~b_hits).sum())
    c = int((~a_hits & b_hits).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return b, c, min(1.0, p)


def xent(p):
    return -np.log(np.maximum(p, C.EPS))


def prop(mask):
    mask = np.asarray(mask, bool)
    k, n = int(mask.sum()), len(mask)
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": p, "ci95": [lo, hi]}


def is_mistake(d, thr=C.MISTAKE_CP):
    """Lost at least `thr` cp against Stockfish's best move, and was not that move."""
    return (d.cp_loss >= thr) & (d.move_uci != d.sf_best)


def verdict(p_value, label):
    if p_value < 0.05:
        return f"{label}: real difference (p = {p_value:.3g})"
    return f"{label}: within noise, do not claim a difference (p = {p_value:.3g})"


# ------------------------------------------------------------------ headline numbers
def headline():
    df, steps, full = clean()
    n = len(df)
    H = {
        "meta": {
            "analysis_file": str(C.ANALYSIS.relative_to(C.ROOT)),
            "player": C.PLAYER,
            "my_rating": C.MY_RATING,
            "n_positions_raw": len(full),
            "n_positions_clean": n,
            "n_games_clean": int(df.game_id.nunique()),
            "filter_steps": [{"rule": s, "removed": r, "remaining": k} for s, r, k in steps],
            "maia_ratings": C.RATINGS,
            "stockfish": f"Stockfish 17.1, depth {C.SF_DEPTH}, MultiPV 3",
            "maia": "lc0 + Maia weights, nodes=1, PolicyTemperature=1.0, real move history",
            "ci_method": "Wilson 95% for proportions; game-level bootstrap 95% (2000 resamples) for means/medians",
            "player_elo_in_games": {
                "median": float(df.player_elo.median()),
                "min": int(df.player_elo.min()),
                "max": int(df.player_elo.max()),
            },
        }
    }

    # top-1 / top-3
    match = {}
    for rt in C.RATINGS:
        match[str(rt)] = {"top1": prop(df[f"top1_{rt}"] == df.move_uci), "top3": prop(df[f"rank_{rt}"] <= 3)}
    match["stockfish"] = {
        "top1": prop(df.sf_best == df.move_uci),
        "top3": prop([m in t for m, t in zip(df.move_uci, df.sf_top3)]),
    }
    match["random"] = {"top1_expected": float(np.mean(1 / df.n_legal))}
    H["match"] = match
    best_top1 = max(C.RATINGS, key=lambda r: match[str(r)]["top1"]["rate"])

    # cross-entropy + perplexity (bootstrap by game)
    ce = {}
    for rt in C.RATINGS:
        est, lo, hi = boot_ci(df, lambda d, rt=rt: xent(d[f"p_{rt}"]).mean())
        ce[str(rt)] = {
            "mean_xent": est,
            "ci95": [lo, hi],
            "perplexity": math.exp(est),
            "perplexity_ci95": [math.exp(lo), math.exp(hi)],
        }
    ce["random"] = {"mean_xent": float(np.log(df.n_legal).mean())}
    H["cross_entropy"] = ce
    best_ce = min(C.RATINGS, key=lambda r: ce[str(r)]["mean_xent"])
    H["best_rating"] = {
        "by_top1": best_top1,
        "by_cross_entropy": best_ce,
        "my_rating": C.MY_RATING,
        "closest_maia_to_my_rating": min(C.RATINGS, key=lambda r: abs(r - C.MY_RATING)),
    }

    # paired tests
    tests = {}
    hits = {rt: (df[f"top1_{rt}"] == df.move_uci).values for rt in C.RATINGS}
    sf_hit = (df.sf_best == df.move_uci).values
    b, c, p = mcnemar(hits[best_top1], sf_hit)
    tests["maia_best_vs_stockfish_top1"] = {
        "maia": best_top1,
        "maia_only": b,
        "sf_only": c,
        "p": p,
        "verdict": verdict(p, f"Maia {best_top1} vs Stockfish top-1"),
    }
    for rt in C.RATINGS:
        if rt == best_top1:
            continue
        b, c, p = mcnemar(hits[best_top1], hits[rt])
        tests[f"maia_{best_top1}_vs_{rt}_top1"] = {
            "a_only": b,
            "b_only": c,
            "p": p,
            "verdict": verdict(p, f"Maia {best_top1} vs Maia {rt} top-1"),
        }
    for rt in C.RATINGS:
        if rt == best_ce:
            continue
        est, lo, hi = boot_ci(df, lambda d, rt=rt: xent(d[f"p_{rt}"]).mean() - xent(d[f"p_{best_ce}"]).mean())
        noise = lo <= 0 <= hi
        tests[f"xent_{rt}_minus_{best_ce}"] = {
            "diff": est,
            "ci95": [lo, hi],
            "verdict": (
                f"Maia {rt} vs {best_ce} loss: "
                + ("within noise, do not claim a difference" if noise else f"real difference ({est:+.3f} nats)")
            ),
        }

    # rating trend: does accuracy fall with rating? (slope of top-1 rate vs rating, bootstrap by game)
    def slope(d):
        y = [(d[f"top1_{rt}"] == d.move_uci).mean() for rt in C.RATINGS]
        return np.polyfit(C.RATINGS, y, 1)[0] * 100 * 100  # percentage points per 100 rating points

    est, lo, hi = boot_ci(df, slope)
    tests["top1_trend_per_100_rating"] = {
        "slope": est,
        "ci95": [lo, hi],
        "verdict": "falls with rating" if hi < 0 else ("rises with rating" if lo > 0 else "no clear trend"),
    }
    H["tests"] = tests

    # Maia vs Stockfish agreement
    H["maia_vs_stockfish_agreement"] = {str(rt): prop(df[f"top1_{rt}"] == df.sf_best) for rt in C.RATINGS}

    # mistakes (sensitivity)
    mist = {}
    for thr in C.MISTAKE_SENSITIVITY:
        m = df[is_mistake(df, thr)]
        entry = {"threshold_cp": thr, "n_mistakes": len(m), "share_of_moves": prop(is_mistake(df, thr))}
        for rt in sorted({best_top1, best_ce}):
            entry[f"maia_{rt}"] = {
                "top1": prop(m[f"rank_{rt}"] == 1),
                "top3": prop(m[f"rank_{rt}"] <= 3),
                "median_p": boot_ci(m, lambda d, rt=rt: d[f"p_{rt}"].median()) if len(m) else None,
                "mean_p": boot_ci(m, lambda d, rt=rt: d[f"p_{rt}"].mean()) if len(m) else None,
            }
        entry["stockfish"] = {
            "top1": prop(m.sf_best == m.move_uci),
            "top3": prop([mv in t for mv, t in zip(m.move_uci, m.sf_top3)]),
            "note": "Stockfish top-1 is 0 by construction: a mistake is defined relative to Stockfish's best move.",
        }
        entry["random_expected_p"] = float(np.mean(1 / m.n_legal)) if len(m) else None
        mist[str(thr)] = entry
    H["mistakes"] = mist

    # probability on my move: overall, mistakes vs non-mistakes (at best-by-top1 rating)
    rt = best_top1
    is_m = is_mistake(df)
    H["prob_on_my_move"] = {"rating": rt}
    for name, d in (("all", df), ("mistakes", df[is_m]), ("non_mistakes", df[~is_m])):
        H["prob_on_my_move"][name] = {
            "n": len(d),
            "median": boot_ci(d, lambda x: x[f"p_{rt}"].median()),
            "mean": boot_ci(d, lambda x: x[f"p_{rt}"].mean()),
        }
    est, lo, hi = boot_ci(
        df, lambda d: d.loc[~is_mistake(d), f"p_{rt}"].mean() - d.loc[is_mistake(d), f"p_{rt}"].mean()
    )
    H["prob_on_my_move"]["non_mistake_minus_mistake_mean"] = {"diff": est, "ci95": [lo, hi]}

    # centipawn loss: me, Maia top-1, Stockfish
    cpl = {
        "me": boot_ci(df, lambda d: d.cp_loss.mean()),
        "stockfish": {"mean": 0.0, "note": "0 by definition: Stockfish's own best move is the measuring stick"},
    }
    for r2 in C.RATINGS:
        col = f"cpl_maia_{r2}"
        if df[col].isna().any():
            cpl[f"maia_{r2}"] = {"missing": int(df[col].isna().sum()), "note": "run `python -m maia_study topup`"}
        else:
            cpl[f"maia_{r2}"] = boot_ci(df, lambda d, col=col: d[col].mean())
    H["centipawn_loss"] = cpl
    return H


# ------------------------------------------------------------------ printing
def fmt_p(d):
    return f"{d['rate']:.1%} [{d['ci95'][0]:.1%}, {d['ci95'][1]:.1%}]"


def fmt_short(d):
    """Compact form for narrow tables: 34.2% (28-41)."""
    return f"{d['rate']:.1%} ({d['ci95'][0] * 100:.0f}-{d['ci95'][1] * 100:.0f})"


def print_headline(H, console):
    from rich import box
    from rich.table import Table

    m = H["meta"]
    console.print(
        f"[bold]Clean set:[/] {m['n_positions_clean']} of {m['n_positions_raw']} positions, "
        f"{m['n_games_clean']} games of {m['player']}  ·  my rating (config) {m['my_rating']}"
    )
    t = Table(title="Match with my actual move (95% Wilson CI)", box=box.SIMPLE_HEAVY, header_style="bold")
    for c in ("Model", "Top-1", "Top-3", "Agrees w/ SF", "n"):
        t.add_column(c, justify="left" if c == "Model" else "right")
    for rt in C.RATINGS:
        d = H["match"][str(rt)]
        style = "bold #5598e7" if rt == H["best_rating"]["by_top1"] else ""
        t.add_row(
            f"[{style}]Maia {rt}[/]" if style else f"Maia {rt}",
            fmt_p(d["top1"]),
            fmt_p(d["top3"]),
            f"{H['maia_vs_stockfish_agreement'][str(rt)]['rate']:.1%}",
            str(d["top1"]["n"]),
        )
    d = H["match"]["stockfish"]
    t.add_row("[bold #1baf7a]Stockfish[/]", fmt_p(d["top1"]), fmt_p(d["top3"]), "", str(d["top1"]["n"]))
    t.add_row("[dim]Random move[/]", f"[dim]{H['match']['random']['top1_expected']:.1%}[/]", "", "", "")
    console.print(t)

    t = Table(title="Cross-entropy of my moves (95% game-bootstrap CI)", box=box.SIMPLE_HEAVY, header_style="bold")
    for c in ("Model", "Mean -log p (nats)", "Perplexity"):
        t.add_column(c, justify="left" if c == "Model" else "right")
    for rt in C.RATINGS:
        d = H["cross_entropy"][str(rt)]
        style = "bold #5598e7" if rt == H["best_rating"]["by_cross_entropy"] else ""
        name = f"[{style}]Maia {rt}[/]" if style else f"Maia {rt}"
        t.add_row(
            name,
            f"{d['mean_xent']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]",
            f"{d['perplexity']:.2f} [{d['perplexity_ci95'][0]:.2f}, {d['perplexity_ci95'][1]:.2f}]",
        )
    t.add_row(
        "[dim]Random move[/]",
        f"[dim]{H['cross_entropy']['random']['mean_xent']:.3f}[/]",
        f"[dim]{math.exp(H['cross_entropy']['random']['mean_xent']):.1f}[/]",
    )
    console.print(t)

    b = H["best_rating"]
    console.print(
        f"[bold]Best Maia:[/] top-1 -> {b['by_top1']}, cross-entropy -> {b['by_cross_entropy']}; "
        f"my rating {b['my_rating']} (closest model {b['closest_maia_to_my_rating']})"
    )
    for k, v in H["tests"].items():
        if "verdict" in v and (
            k.startswith("maia_best")
            or k.startswith("xent")
            or k.startswith("top1_trend")
            or k.endswith(f"_vs_{b['closest_maia_to_my_rating']}_top1")
        ):
            extra = (
                f" ({v['slope']:+.2f} pts per 100 rating, CI [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}])"
                if "slope" in v
                else ""
            )
            console.print(f"  • {v['verdict']}{extra}")

    t = Table(title="Mistakes test (my move lost >= X cp vs Stockfish best)", box=box.SIMPLE_HEAVY, header_style="bold")
    for c in ("X", "n", "Maia top-1", "Maia top-3", "Maia median p", "SF top-1", "SF top-3", "Random p"):
        t.add_column(c, justify="right")
    rt = H["best_rating"]["by_top1"]
    for thr, e in H["mistakes"].items():
        mm = e[f"maia_{rt}"]
        t.add_row(
            thr,
            str(e["n_mistakes"]),
            fmt_short(mm["top1"]),
            fmt_short(mm["top3"]),
            f"{mm['median_p'][0]:.1%} ({mm['median_p'][1] * 100:.0f}-{mm['median_p'][2] * 100:.0f})",
            f"{e['stockfish']['top1']['rate']:.0%}",
            fmt_short(e["stockfish"]["top3"]),
            f"{e['random_expected_p']:.1%}",
        )
    console.print(t)
    console.print(f"  [dim]Maia column = Maia {rt}. Stockfish top-1 is 0 by construction.[/]")

    pm = H["prob_on_my_move"]
    console.print(f"[bold]Maia {pm['rating']} probability on my move[/]  (median / mean, game-bootstrap CI)")
    for k in ("all", "mistakes", "non_mistakes"):
        d = pm[k]
        console.print(
            f"  {k:<13} n={d['n']:<4} median {d['median'][0]:.1%} [{d['median'][1]:.1%}, {d['median'][2]:.1%}]"
            f"   mean {d['mean'][0]:.1%} [{d['mean'][1]:.1%}, {d['mean'][2]:.1%}]"
        )
    cp = H["centipawn_loss"]
    line = f"[bold]Average centipawn loss[/]: me {cp['me'][0]:.0f} [{cp['me'][1]:.0f}, {cp['me'][2]:.0f}]"
    for r2 in (H["best_rating"]["by_top1"], 1900):
        v = cp[f"maia_{r2}"]
        line += f"  ·  Maia {r2} top-1 " + (
            f"{v[0]:.0f} [{v[1]:.0f}, {v[2]:.0f}]" if isinstance(v, tuple) else "missing"
        )
    console.print(line + "  ·  Stockfish 0 (measuring stick)")


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def no_bullet_check():
    """Robustness check: the main top-1 numbers with bullet games removed.

    The paper drops moves made with under 30 s on the clock. The PGN has no clock data, so that
    filter cannot be applied; removing bullet games entirely is the closest available check.
    """
    df, _, _ = clean()
    mistakes = df[is_mistake(df)]
    nb = df[~df.event.str.contains("bullet")]
    top1 = {str(rt): prop(nb[f"top1_{rt}"] == nb.move_uci) for rt in C.RATINGS}
    return {
        "clean_set": {
            "n_positions": len(df),
            "n_games": int(df.game_id.nunique()),
            "n_bullet_positions": int(df.event.str.contains("bullet").sum()),
            "n_mistakes": len(mistakes),
            "n_games_with_mistakes": int(mistakes.game_id.nunique()),
        },
        "no_bullet": {
            "n_positions": len(nb),
            "n_games": int(nb.game_id.nunique()),
            "maia_top1": top1,
            "stockfish_top1": prop(nb.sf_best == nb.move_uci),
            "random_top1_expected": float((1 / nb.n_legal).mean()),
        },
    }


def main():
    from rich.console import Console

    console = Console(width=110)
    H = headline()
    print_headline(H, console)
    C.RESULTS.mkdir(exist_ok=True)
    C.HEADLINE.write_text(json.dumps(jsonable(H), indent=2))
    console.print(f"[dim]saved {C.HEADLINE.relative_to(C.ROOT)}[/]")
    nb = no_bullet_check()
    C.NO_BULLET.write_text(json.dumps(jsonable(nb), indent=2))
    b = nb["no_bullet"]
    console.print(
        f"[bold]No-bullet check[/]: {b['n_positions']} positions from {b['n_games']} games · "
        f"Maia 1100 {b['maia_top1']['1100']['rate']:.1%} · Maia 1400 {b['maia_top1']['1400']['rate']:.1%} · "
        f"Stockfish {b['stockfish_top1']['rate']:.1%} · random {b['random_top1_expected']:.1%}"
    )
    console.print(f"[dim]saved {C.NO_BULLET.relative_to(C.ROOT)}[/]")
