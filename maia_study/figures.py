"""The chart figures (fig1 to fig10), built from the saved analysis.

Every figure is written as a 1920x1080 PNG (200 dpi) and an SVG, in light and dark versions, to
figures/charts/. The numbers behind each figure are printed and saved to results/figure_data/<fig>.csv.
"""

import json
import math
import textwrap

import chess
import matplotlib

matplotlib.use("Agg")
# Fixed SVG element ids, so regenerated SVGs are byte-identical.
matplotlib.rcParams["svg.hashsalt"] = "maia-study"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

from . import config as C
from . import stats

console = Console(width=110)
W, H_IN, DPI = 9.6, 5.4, 200  # 1920 x 1080 px


# ------------------------------------------------------------------ styling helpers
def style(theme):
    t = C.THEMES[theme]
    return {
        "figure.facecolor": t["bg"],
        "axes.facecolor": t["bg"],
        "savefig.facecolor": t["bg"],
        "text.color": t["ink"],
        "axes.labelcolor": t["ink"],
        "xtick.color": t["ink2"],
        "ytick.color": t["ink2"],
        "axes.edgecolor": t["grid"],
        "grid.color": t["grid"],
        "font.size": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "legend.frameon": False,
        "svg.fonttype": "none",
    }


def new_fig(theme, title, subtitle, rect=(0.11, 0.15, 0.86, None)):
    """Title + subtitle header (wrapped, min 18pt); the plot area starts below whatever height they need.
    rect = (left, bottom, width, top-or-None); None top means 'just under the header'."""
    t = C.THEMES[theme]
    fig = plt.figure(figsize=(W, H_IN), dpi=DPI)
    tl = textwrap.wrap(title, 50)
    sl = textwrap.wrap(subtitle, 66)
    y = 0.965
    fig.text(0.03, y, "\n".join(tl), fontsize=22, weight="bold", color=t["ink"], va="top", linespacing=1.05)
    y -= 0.068 * len(tl) + 0.012
    fig.text(0.03, y, "\n".join(sl), fontsize=18, color=t["ink2"], va="top", linespacing=1.1)
    y -= 0.056 * len(sl)
    fig.header_bottom = y
    ax = None
    if rect:
        left, bottom, width, top = rect
        top = min(top or 1, y - 0.06)
        ax = fig.add_axes([left, bottom, width, top - bottom])
    return fig, ax, t


def save(fig, name, theme):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / f"{name}_{theme}.png", dpi=DPI)
    fig.savefig(C.FIG_DIR / f"{name}_{theme}.svg", metadata={"Date": None})  # no timestamp: reproducible files
    plt.close(fig)


def dump(name, frame):
    C.FIG_DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(C.FIG_DATA / f"{name}.csv", index=False)
    console.rule(f"[bold]{name}")
    console.print(frame.to_string(index=False))


def both(fn):
    for theme in ("light", "dark"):
        with plt.rc_context(style(theme)):
            fn(theme)


# ------------------------------------------------------------------ data
df, steps, _ = stats.clean()
HL = json.loads((C.HEADLINE).read_text())
BEST = HL["best_rating"]["by_top1"]
N = len(df)
NG = df.game_id.nunique()
SHOW = json.loads((C.SHOWCASE_JSON).read_text())
DISTS = json.loads(C.SHOWCASE_DISTS.read_text())
NOTE_N = f"n = {N} positions from {NG} of my games"


def is_mistake(d, thr=C.MISTAKE_CP):
    return (d.cp_loss >= thr) & (d.move_uci != d.sf_best)


# ------------------------------------------------------------------ fig1
def fig1():
    rows = []
    for rt in C.RATINGS:
        m = HL["match"][str(rt)]["top1"]
        rows.append(
            {
                "model": f"Maia {rt}",
                "rating": rt,
                "top1": m["rate"],
                "ci_lo": m["ci95"][0],
                "ci_hi": m["ci95"][1],
                "n": m["n"],
            }
        )
    s = HL["match"]["stockfish"]["top1"]
    rows.append(
        {
            "model": "Stockfish",
            "rating": None,
            "top1": s["rate"],
            "ci_lo": s["ci95"][0],
            "ci_hi": s["ci95"][1],
            "n": s["n"],
        }
    )
    d = pd.DataFrame(rows)
    dump("fig1_match_by_rating", d)
    m = d[d.model != "Stockfish"].assign(rating=lambda x: x.rating.astype(int))

    def draw(theme):
        fig, ax, t = new_fig(
            theme,
            "Maia predicts my moves far better than Stockfish",
            f"Top-1 match with the move I played · {NOTE_N} · error bars: 95% CI",
        )
        x = np.arange(len(m))
        cols = [t["maia"] if r == BEST else t["maia_soft"] for r in m.rating]
        ax.bar(x, m.top1 * 100, color=cols, width=0.7)
        ax.errorbar(
            x,
            m.top1 * 100,
            yerr=[(m.top1 - m.ci_lo) * 100, (m.ci_hi - m.top1) * 100],
            fmt="none",
            ecolor=t["ink"],
            elinewidth=2,
            capsize=5,
        )
        for xi, v, hi in zip(x, m.top1, m.ci_hi):
            ax.text(xi, hi * 100 + 1.2, f"{v * 100:.0f}", ha="center", va="bottom", fontsize=18, color=t["ink"])
        sf = d[d.model == "Stockfish"].iloc[0]
        ax.axhspan(sf.ci_lo * 100, sf.ci_hi * 100, color=t["sf"], alpha=0.18, lw=0)
        ax.axhline(sf.top1 * 100, color=t["sf"], lw=3)
        ax.text(
            len(m) - 0.5,
            sf.ci_lo * 100 - 1,
            f"Stockfish {sf.top1 * 100:.1f}%",
            color=t["sf"],
            ha="right",
            va="top",
            fontsize=18,
            weight="bold",
        )
        ax.set_xticks(x, [str(r) for r in m.rating])
        ax.set_ylim(0, 65)
        ax.set_ylabel("Top-1 match (%)")
        ax.set_xlabel("Maia model (training rating)")
        save(fig, "fig1_match_by_rating", theme)

    both(draw)


# ------------------------------------------------------------------ fig2
def fig2():
    rows = [
        {
            "model": f"Maia {rt}",
            "rating": rt,
            "mean_xent_nats": v["mean_xent"],
            "ci_lo": v["ci95"][0],
            "ci_hi": v["ci95"][1],
            "perplexity": v["perplexity"],
        }
        for rt, v in ((int(k), HL["cross_entropy"][k]) for k in map(str, C.RATINGS))
    ]
    d = pd.DataFrame(rows)
    rnd = HL["cross_entropy"]["random"]["mean_xent"]
    d = pd.concat(
        [
            d,
            pd.DataFrame(
                [{"model": "Random legal move", "rating": None, "mean_xent_nats": rnd, "perplexity": math.exp(rnd)}]
            ),
        ],
        ignore_index=True,
    )
    dump("fig2_loss_by_rating", d)
    m = d[d.rating.notna()]
    best = int(m.loc[m.mean_xent_nats.idxmin(), "rating"])

    def draw(theme):
        fig, ax, t = new_fig(
            theme,
            "The lowest-rated Maia models fit my moves best",
            f"Average cross-entropy -log p(my move), lower is better · {NOTE_N} · 95% CI",
        )
        x = np.arange(len(m))
        ax.errorbar(
            x,
            m.mean_xent_nats,
            yerr=[m.mean_xent_nats - m.ci_lo, m.ci_hi - m.mean_xent_nats],
            fmt="none",
            ecolor=t["maia_soft"],
            elinewidth=3,
            capsize=6,
        )
        ax.plot(x, m.mean_xent_nats, color=t["maia"], lw=2.5, marker="o", ms=11, zorder=3)
        bi = list(m.rating).index(best)
        ax.plot([bi], [m.mean_xent_nats.iloc[bi]], marker="o", ms=22, color=t["maia"], mec=t["ink"], mew=3, zorder=4)
        ax.annotate(
            f"lowest: Maia {best}\n{m.mean_xent_nats.iloc[bi]:.2f} nats",
            (bi, m.mean_xent_nats.iloc[bi]),
            xytext=(bi + 0.6, m.ci_hi.max() + 0.02),
            fontsize=18,
            color=t["ink"],
            arrowprops=dict(arrowstyle="-", color=t["ink2"]),
        )
        ax.set_xticks(x, [str(int(r)) for r in m.rating])
        ax.set_ylabel("Cross-entropy (nats)")
        ax.set_xlabel("Maia model (training rating)")
        ax.set_ylim(1.3, 2.2)
        ax.text(
            0.99,
            0.04,
            f"random-move baseline: {rnd:.2f} nats (off the chart)",
            transform=ax.transAxes,
            ha="right",
            fontsize=18,
            color=t["ink2"],
        )
        save(fig, "fig2_loss_by_rating", theme)

    both(draw)


# ------------------------------------------------------------------ fig3
def fig3():
    p = df[f"p_{BEST}"]
    med = p.median()
    rnd = float(np.mean(1 / df.n_legal))
    edges = np.linspace(0, 1, 21)
    counts, _ = np.histogram(p, edges)
    d = pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:], "count": counts})
    dump("fig3_prob_hist", d)
    console.print(f"median p = {med:.4f}, mean p = {p.mean():.4f}, random-guess mean = {rnd:.4f}, n = {len(p)}")
    pd.DataFrame([{"rating": BEST, "median_p": med, "mean_p": p.mean(), "random_mean_p": rnd, "n": len(p)}]).to_csv(
        C.FIG_DATA / "fig3_prob_hist_summary.csv", index=False
    )

    def draw(theme, log=False):
        fig, ax, t = new_fig(
            theme,
            f"Maia gave my actual move a median {med:.0%} probability",
            f"Probability Maia {BEST} assigned to the move I played · {NOTE_N}" + (" · log scale" if log else ""),
        )
        if log:
            e = np.logspace(-4, 0, 25)
            ax.hist(np.maximum(p, 1e-4), bins=e, color=t["maia"], edgecolor=t["bg"], lw=1.5)
            ax.set_xscale("log")
            ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ["0.01%", "0.1%", "1%", "10%", "100%"])
        else:
            ax.hist(p, bins=edges, color=t["maia"], edgecolor=t["bg"], lw=1.5)
            ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
        top = ax.get_ylim()[1]
        ax.axvline(med, color=t["me"], lw=3)
        ax.set_ylim(0, top * 1.12)
        ax.text(
            med * 0.9 if log else med + 0.015,
            top * 1.08,
            f"median {med:.1%}",
            color=t["me"],
            fontsize=18,
            weight="bold",
            va="top",
            ha="right" if log else "left",
        )
        ax.axvline(rnd, color=t["neutral"], lw=2.5, ls="--")
        ax.text(
            rnd * (0.85 if log else 1) + (0 if log else 0.012),
            top * 0.72,
            f"random guess\n{rnd:.1%}",
            color=t["ink2"],
            fontsize=18,
            va="top",
            ha="right" if log else "left",
        )
        ax.set_xlabel("Probability Maia gave my move")
        ax.set_ylabel("Positions")
        save(fig, "fig3_prob_hist_log" if log else "fig3_prob_hist", theme)

    both(draw)
    both(lambda th: draw(th, log=True))


# ------------------------------------------------------------------ fig4
def fig4():
    e = HL["mistakes"][str(C.MISTAKE_CP)]
    mm = e[f"maia_{BEST}"]
    rows = [
        {
            "model": f"Maia {BEST}",
            "metric": "top1",
            **{k: v for k, v in zip(("rate", "ci_lo", "ci_hi"), [mm["top1"]["rate"], *mm["top1"]["ci95"]])},
        },
        {
            "model": f"Maia {BEST}",
            "metric": "top3",
            **{k: v for k, v in zip(("rate", "ci_lo", "ci_hi"), [mm["top3"]["rate"], *mm["top3"]["ci95"]])},
        },
        {
            "model": "Stockfish",
            "metric": "top1",
            "rate": e["stockfish"]["top1"]["rate"],
            "ci_lo": e["stockfish"]["top1"]["ci95"][0],
            "ci_hi": e["stockfish"]["top1"]["ci95"][1],
        },
        {
            "model": "Stockfish",
            "metric": "top3",
            "rate": e["stockfish"]["top3"]["rate"],
            "ci_lo": e["stockfish"]["top3"]["ci95"][0],
            "ci_hi": e["stockfish"]["top3"]["ci95"][1],
        },
    ]
    m = df[is_mistake(df)]
    rows += [
        {"model": "Random guess", "metric": "top1", "rate": float(np.mean(1 / m.n_legal))},
        {"model": "Random guess", "metric": "top3", "rate": float(np.mean(np.minimum(1, 3 / m.n_legal)))},
    ]
    d = pd.DataFrame(rows)
    d["n_mistakes"] = e["n_mistakes"]
    sens = []
    for thr, v in HL["mistakes"].items():
        sens.append(
            {
                "threshold_cp": int(thr),
                "n": v["n_mistakes"],
                "maia_top1": v[f"maia_{BEST}"]["top1"]["rate"],
                "maia_top3": v[f"maia_{BEST}"]["top3"]["rate"],
                "maia_median_p": v[f"maia_{BEST}"]["median_p"][0],
                "sf_top1": v["stockfish"]["top1"]["rate"],
                "sf_top3": v["stockfish"]["top3"]["rate"],
            }
        )
    dump("fig4_mistakes", d)
    dump("fig4_mistakes_sensitivity", pd.DataFrame(sens))

    def draw(theme):
        fig, ax, t = new_fig(
            theme,
            "Maia expects my mistakes; Stockfish rarely does",
            f"{e['n_mistakes']} moves where I lost at least {C.MISTAKE_CP} centipawns · "
            "Stockfish #1 is 0% by definition · 95% CI",
        )
        groups = ["Ranked my mistake #1", "My mistake in its top 3"]
        x = np.arange(2)
        wdt = 0.26
        for i, (model, col) in enumerate(
            ((f"Maia {BEST}", t["maia"]), ("Stockfish", t["sf"]), ("Random guess", t["neutral"]))
        ):
            sub = d[d.model == model].set_index("metric").loc[["top1", "top3"]]
            xs = x + (i - 1) * wdt
            ax.bar(xs, sub.rate * 100, width=wdt * 0.92, color=col, label=model)
            if sub.ci_lo.notna().all():
                ax.errorbar(
                    xs,
                    sub.rate * 100,
                    yerr=[(sub.rate - sub.ci_lo) * 100, (sub.ci_hi - sub.rate) * 100],
                    fmt="none",
                    ecolor=t["ink"],
                    elinewidth=2,
                    capsize=5,
                )
            for xi, v, hi in zip(xs, sub.rate, sub.ci_hi.fillna(sub.rate)):
                ax.text(xi, hi * 100 + 1.5, f"{v * 100:.0f}%", ha="center", fontsize=18, color=t["ink"])
        ax.set_xticks(x, groups)
        ax.set_ylim(0, 80)
        ax.set_ylabel("Share of my mistakes (%)")
        ax.legend(loc="upper left", ncol=1)
        save(fig, "fig4_mistakes", theme)

    both(draw)


# ------------------------------------------------------------------ fig5
def fig5():
    p = df[f"p_{BEST}"]
    cpl = df.cp_loss.clip(upper=1500)
    rho = df.cp_loss.rank().corr(p.rank())
    bins = [-1, 0, 25, 50, 100, 200, 500, 3000]
    labels = ["0", "1-25", "26-50", "51-100", "101-200", "201-500", "500+"]
    df["_bin"] = pd.cut(df.cp_loss, bins, labels=labels)
    rows = []
    for lab, mid in zip(labels, [0, 12, 38, 75, 150, 350, 900]):
        g = df[df._bin == lab]
        est, lo, hi = stats.boot_ci(g, lambda x: x[f"p_{BEST}"].mean()) if len(g) > 2 else (np.nan,) * 3
        rows.append(
            {
                "cp_loss_bin": lab,
                "x_mid": mid,
                "n": len(g),
                "mean_p": est,
                "ci_lo": lo,
                "ci_hi": hi,
                "median_p": g[f"p_{BEST}"].median(),
                "random_mean_p": float(np.mean(1 / g.n_legal)) if len(g) else np.nan,
            }
        )
    d = pd.DataFrame(rows)
    dump("fig5_cpl_vs_prob", d)
    console.print(f"Spearman rank correlation (cp loss vs Maia p): {rho:.3f}, n = {N}")
    pd.DataFrame(df[["game_id", "ply", "move_san", "cp_loss", f"p_{BEST}"]]).to_csv(
        C.FIG_DATA / "fig5_points.csv", index=False
    )
    big = df[df.cp_loss >= 200]
    share = (big[f"p_{BEST}"] >= 1 / big.n_legal).mean()

    def draw(theme):
        fig, ax, t = new_fig(
            theme,
            "Worse moves get less Maia probability, not zero",
            f"Each dot is one of my moves (n = {N}) · line: mean per bin, 95% CI · Spearman {rho:.2f}",
        )
        ax.grid(axis="x")
        ax.scatter(cpl + 1, p * 100, s=40, color=t["me"], alpha=0.45, lw=0)
        ok = d.dropna(subset=["mean_p"])
        ax.fill_between(ok.x_mid + 1, ok.ci_lo * 100, ok.ci_hi * 100, color=t["maia"], alpha=0.2, lw=0)
        ax.plot(ok.x_mid + 1, ok.mean_p * 100, color=t["maia"], lw=3.5, marker="o", ms=9)
        ax.axvline(C.MISTAKE_CP + 1, color=t["neutral"], ls="--", lw=2)
        ax.text(C.MISTAKE_CP * 1.08, 1, "mistake (100 cp)", color=t["ink2"], fontsize=18, va="bottom")
        ax.text(
            0.985,
            0.97,
            f"{share:.0%} of my 200+ cp blunders got\nmore probability than a random guess",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=18,
            color=t["ink"],
            bbox=dict(boxstyle="round,pad=0.3", fc=t["bg"], ec=t["grid"], alpha=0.92),
        )
        ax.set_xscale("log")
        ax.set_xticks([1, 11, 51, 101, 201, 501, 1501], ["0", "10", "50", "100", "200", "500", "1500+"])
        ax.set_xlim(0.8, 2000)
        ax.set_ylim(-2, 100)
        ax.set_xlabel("Centipawns I lost with the move (log scale)")
        ax.set_ylabel(f"Maia {BEST} probability (%)")
        save(fig, "fig5_cpl_vs_prob", theme)

    both(draw)


# ------------------------------------------------------------------ fig6
def fig6():
    sw = SHOW["sweep"]
    fen = sw["fen"]
    dist = DISTS[fen]
    board = chess.Board(fen)
    avg = {}
    for rt in C.RATINGS:
        for u, pr in dist[str(rt)].items():
            avg[u] = avg.get(u, 0) + pr / len(C.RATINGS)
    moves = sorted(avg, key=lambda u: -avg[u])[:5]
    for must in (sw["sf_best"], sw["my_move"]):
        if must not in moves:
            moves[-1] = must if moves[-1] not in (sw["sf_best"], sw["my_move"]) else moves[-2]
    rows = [
        {
            "move": board.san(chess.Move.from_uci(u)),
            "uci": u,
            **{str(rt): dist[str(rt)].get(u, 0.0) for rt in C.RATINGS},
            "is_stockfish_best": u == sw["sf_best"],
            "is_my_move": u == sw["my_move"],
        }
        for u in moves
    ]
    d = pd.DataFrame(rows)
    dump("fig6_rating_sweep", d)
    tops = {rt: board.san(chess.Move.from_uci(max(dist[str(rt)], key=dist[str(rt)].get))) for rt in C.RATINGS}
    console.print("top move by rating: " + ", ".join(f"{k}:{v}" for k, v in tops.items()))

    # board image for the left side
    from . import showcase

    stem = C.POS_DIR / "sweep_board"
    maia_top = chess.Move.from_uci(max(dist[str(BEST)], key=dist[str(BEST)].get))
    showcase.render_board(
        board, maia_top, chess.Move.from_uci(sw["sf_best"]), chess.Move.from_uci(sw["my_move"]), str(stem)
    )

    def draw(theme):
        fig, _, t = new_fig(
            theme,
            "The favourite move shifts across Maia ratings",
            f"One position from my game {sw['game_id']} (move {sw['move_number']}): "
            f"my move {board.san(chess.Move.from_uci(sw['my_move']))}"
            f" was also Stockfish's choice",
            rect=None,
        )
        hb = fig.header_bottom - 0.03
        axb = fig.add_axes([0.0, 0.03, 0.36, hb - 0.03])
        axb.imshow(plt.imread(f"{stem}.png"))
        axb.axis("off")
        ax = fig.add_axes([0.47, 0.14, 0.29, hb - 0.2])
        other_cols = [t["maia"], t["maia_soft"], t["neutral"], t["ink2"], t["grid"]]
        oc = iter(other_cols)
        labels = []
        for _, r in d.iterrows():
            ys = [r[str(rt)] * 100 for rt in C.RATINGS]
            if r.is_my_move and r.is_stockfish_best:
                col, tag, lw = t["sf"], " (me, SF)", 4
            elif r.is_my_move:
                col, tag, lw = t["me"], " (me)", 4
            elif r.is_stockfish_best:
                col, tag, lw = t["sf"], " (SF)", 4
            else:
                col, tag, lw = next(oc), "", 2.5
            ax.plot(C.RATINGS, ys, color=col, lw=lw, marker="o", ms=7)
            labels.append([ys[-1], r.move + tag, col, bool(tag)])
        # spread the end labels so they never overlap (min 7 points apart)
        ymax = max(d[[str(rt) for rt in C.RATINGS]].max()) * 100 * 1.2
        gap = 0.1 * ymax
        labels.sort(key=lambda lab: lab[0])
        for i in range(1, len(labels)):
            labels[i][0] = max(labels[i][0], labels[i - 1][0] + gap)
        for y, text, col, bold in labels:
            ax.text(1935, y, text, color=col, fontsize=18, va="center", weight="bold" if bold else "normal")
        ax.set_xticks([1100, 1500, 1900], ["1100", "1500", "1900"])
        ax.set_xlim(1060, 1940)
        ax.set_ylim(0, max(ymax, labels[-1][0] + gap / 2))
        ax.set_xlabel("Maia model")
        ax.set_ylabel("Probability (%)")
        save(fig, "fig6_rating_sweep", theme)

    both(draw)


# ------------------------------------------------------------------ fig7
def fig7():
    mt = df[f"top1_{BEST}"]
    me, sf = df.move_uci, df.sf_best
    cats = {
        "All three agree": (mt == me) & (sf == me),
        "Maia + me only": (mt == me) & (sf != me),
        "Stockfish + me only": (sf == me) & (mt != me),
        "Maia + Stockfish only": (mt == sf) & (mt != me),
        "All different": (mt != me) & (sf != me) & (mt != sf),
    }
    rows = [
        {
            "category": k,
            **{kk: vv for kk, vv in stats.prop(v).items() if kk in ("k", "n", "rate")},
            "ci_lo": stats.prop(v)["ci95"][0],
            "ci_hi": stats.prop(v)["ci95"][1],
        }
        for k, v in cats.items()
    ]
    d = pd.DataFrame(rows)
    assert d.k.sum() == N, "categories must partition the positions"
    dump("fig7_agreement_matrix", d)

    def draw(theme):
        fig, ax, t = new_fig(
            theme,
            "Maia agrees with me far more often than Stockfish",
            f"Maia {BEST}, Stockfish and my move compared on the same {N} positions",
            rect=(0.33, 0.15, 0.6, None),
        )
        cols = [t["ink2"], t["maia"], t["sf"], t["neutral"], t["grid"]]
        y = np.arange(len(d))[::-1]
        ax.barh(y, d.rate * 100, color=cols, height=0.65)
        for yi, r in zip(y, d.itertuples()):
            ax.text(r.rate * 100 + 1, yi, f"{r.rate:.0%}  (n = {r.k})", va="center", fontsize=18, color=t["ink"])
        ax.set_yticks(y, d.category)
        ax.set_xlim(0, 60)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x")
        ax.set_xlabel("Share of positions (%)")
        save(fig, "fig7_agreement_matrix", theme)

    both(draw)


# ------------------------------------------------------------------ fig8
def fig8():
    pos = next(p for p in SHOW["positions"] if p["slug"] == "xent_example")
    dist = DISTS[pos["fen"]][str(BEST)]
    board = chess.Board(pos["fen"])
    ranked = sorted(dist.items(), key=lambda kv: -kv[1])
    show = ranked[:7]
    if pos["my_uci"] not in [u for u, _ in show]:
        show[-1] = (pos["my_uci"], dist[pos["my_uci"]])
    rest = 1 - sum(p for _, p in show)
    p_me = dist[pos["my_uci"]]
    loss = -math.log(max(p_me, C.EPS))
    rows = [
        {
            "move": board.san(chess.Move.from_uci(u)),
            "p": pr,
            "target_y": int(u == pos["my_uci"]),
            "minus_y_log_p": -math.log(max(pr, C.EPS)) if u == pos["my_uci"] else 0.0,
        }
        for u, pr in show
    ]
    rows.append({"move": f"other {len(dist) - len(show)} moves", "p": rest, "target_y": 0, "minus_y_log_p": 0.0})
    d = pd.DataFrame(rows)
    dump("fig8_xent_worked_example", d)
    console.print(f"loss = -ln({p_me:.4f}) = {loss:.4f} nats; my move {pos['my_move']}; Maia {BEST}; FEN {pos['fen']}")

    stem = C.POS_DIR / f"pos_{pos['id']}_{pos['slug']}_board"

    def draw(theme):
        fig, _, t = new_fig(
            theme,
            "Cross-entropy only scores the move I played",
            f"Maia {BEST}'s probabilities for one of my positions · target = the move I actually played",
            rect=None,
        )
        hb = fig.header_bottom - 0.03
        axb = fig.add_axes([0.01, 0.03, 0.28, hb - 0.03])
        axb.imshow(plt.imread(f"{stem}.png"))
        axb.axis("off")
        ax = fig.add_axes([0.40, 0.2, 0.58, hb - 0.24])
        x = np.arange(len(d))
        cols = [
            t["me"] if y == 1 else (t["neutral"] if m.startswith("other") else t["maia"])
            for m, y in zip(d.move, d.target_y)
        ]
        ax.bar(x, d.p * 100, color=cols, width=0.7)
        for xi, pr in zip(x, d.p):
            ax.text(xi, pr * 100 + 1.5, f"{pr * 100:.0f}", ha="center", fontsize=18, color=t["ink"])
        ax.set_xticks(x, [m if not m.startswith("other") else "rest" for m in d.move], fontsize=18)
        for xi, y in zip(x, d.target_y):
            ax.text(
                xi,
                -0.2,
                f"y={y}",
                ha="center",
                va="top",
                fontsize=18,
                transform=ax.get_xaxis_transform(),
                color=t["me"] if y else t["ink2"],
                weight="bold" if y else "normal",
            )
        ax.set_ylim(0, max(d.p) * 100 * 1.35)
        ax.set_ylabel("Maia probability (%)")
        ax.text(
            0.97,
            0.95,
            f"loss = -log p({pos['my_move']})\n= -log {p_me:.3f} = {loss:.2f} nats",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=20,
            weight="bold",
            color=t["me"],
        )
        ax.text(
            0.97,
            0.66,
            "moves with y = 0 add nothing",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=18,
            color=t["ink2"],
        )
        save(fig, "fig8_xent_worked_example", theme)

    both(draw)


# ------------------------------------------------------------------ fig9
def fig9():
    rows = []
    for rt in C.RATINGS:
        m = HL["match"][str(rt)]["top1"]
        est, lo, hi = stats.boot_ci(df, lambda x, rt=rt: x[f"p_{rt}"].mean())
        rows.append(
            {
                "rating": rt,
                "top1": m["rate"],
                "top1_lo": m["ci95"][0],
                "top1_hi": m["ci95"][1],
                "mean_p_my_move": est,
                "mean_p_lo": lo,
                "mean_p_hi": hi,
                "mean_xent": HL["cross_entropy"][str(rt)]["mean_xent"],
            }
        )
    d = pd.DataFrame(rows)
    dump("fig9_proxy_warning", d)
    hit = df[f"top1_{BEST}"] == df.move_uci
    extra = pd.DataFrame(
        [
            {
                "rating": BEST,
                "n_hits": int(hit.sum()),
                "n_misses": int((~hit).sum()),
                "median_p_on_my_move_when_hit": df.loc[hit, f"p_{BEST}"].median(),
                "median_p_on_my_move_when_miss": df.loc[~hit, f"p_{BEST}"].median(),
                "share_of_misses_where_my_move_was_rank2": (df.loc[~hit, f"rank_{BEST}"] == 2).mean(),
                "share_of_hits_with_p_below_40pct": (df.loc[hit, f"p_{BEST}"] < 0.4).mean(),
            }
        ]
    )
    extra.to_csv(C.FIG_DATA / "fig9_hits_vs_misses.csv", index=False)
    console.print(extra.T.to_string(header=False))

    def draw(theme):
        fig, ax, t = new_fig(
            theme,
            "Top-1 drops with rating; probability does not",
            f"Same {N} positions · top-1 is all-or-nothing; "
            "the probability Maia puts on my move keeps partial credit · 95% CI",
        )
        ax.fill_between(d.rating, d.top1_lo * 100, d.top1_hi * 100, color=t["maia"], alpha=0.18, lw=0)
        ax.plot(d.rating, d.top1 * 100, color=t["maia"], lw=3.5, marker="o", ms=9)
        ax.fill_between(d.rating, d.mean_p_lo * 100, d.mean_p_hi * 100, color=t["me"], alpha=0.18, lw=0)
        ax.plot(d.rating, d.mean_p_my_move * 100, color=t["me"], lw=3.5, marker="s", ms=9)
        ax.text(
            1100,
            57,
            f"top-1 match: {d.top1.iloc[0]:.0%} at 1100, {d.top1.iloc[-1]:.0%} at 1900",
            color=t["maia"],
            fontsize=18,
            weight="bold",
            ha="left",
            va="bottom",
        )
        ax.text(
            1100,
            30,
            f"avg probability on my move: {d.mean_p_my_move.iloc[0]:.0%} at 1100, "
            f"{d.mean_p_my_move.iloc[-1]:.0%} at 1900",
            color=t["me"],
            fontsize=18,
            weight="bold",
            ha="left",
            va="top",
        )
        ax.set_xticks(C.RATINGS, [str(r) for r in C.RATINGS])
        ax.set_ylim(20, 62)
        ax.set_xlabel("Maia model (training rating)")
        ax.set_ylabel("Percent")
        save(fig, "fig9_proxy_warning", theme)

    both(draw)


# ------------------------------------------------------------------ fig10
TOY = C.TOY_CURVES


def fig10():
    """Training vs test loss of the supplementary toy model (toy_model.py). Not Maia."""
    if not TOY.exists():
        console.print("[yellow]fig10 skipped: run `python -m maia_study toy-model` first[/]")
        return
    R = json.loads(TOY.read_text())
    d = pd.DataFrame(R["curve"])
    ref, best = R["reference"], R["best_epoch"]
    d["maia_1100_test"] = ref["test"]["maia_1100"]
    d["uniform_test"] = ref["test"]["uniform"]
    dump("fig10_loss_curves", d)
    dump("fig10_other_splits", pd.DataFrame(R["other_splits"]))

    def draw(theme):
        fig, ax, t = new_fig(
            theme,
            "Training vs test loss: a small model trained on my games",
            f"Linear softmax over legal moves, cross-entropy loss (not Maia) · train "
            f"{ref['train']['positions']} positions / {ref['train']['games']} games, test "
            f"{ref['test']['positions']} / {ref['test']['games']} unseen games",
            rect=(0.10, 0.21, 0.64, None),
        )
        ax.plot(d.epoch, d.train_loss, color=t["neutral"], lw=3.5)
        ax.plot(d.epoch, d.test_loss, color=t["me"], lw=3.5)
        ax.plot([best["epoch"]], [best["test_loss"]], marker="o", ms=13, color=t["me"], mec=t["ink"], mew=2.5, zorder=4)
        ax.annotate(
            f"lowest test loss\nepoch {best['epoch']}: {best['test_loss']:.2f}",
            (best["epoch"], best["test_loss"]),
            xytext=(best["epoch"] + 30, 1.72),
            fontsize=16,
            color=t["ink"],
            arrowprops=dict(arrowstyle="-", color=t["ink2"]),
        )
        uni, maia = ref["test"]["uniform"], ref["test"]["maia_1100"]
        ax.axhline(uni, color=t["ink2"], lw=2, ls=":")
        ax.axhline(maia, color=t["maia"], lw=3, ls="--")
        ax.set_xlim(0, R["epochs"])
        ax.set_ylim(1.2, 3.8)
        ax.set_yticks([1.5, 2.0, 2.5, 3.0, 3.5])
        ax.set_xlabel("Training epoch (full passes over the training games)")
        ax.set_ylabel("Cross-entropy (nats)")
        # right-hand labels for the reference lines and the end of each curve
        x_lab = 1.02
        tr = ax.get_yaxis_transform()
        ax.text(
            x_lab,
            uni + 0.03,
            f"random guess {uni:.2f}",
            transform=tr,
            fontsize=16,
            color=t["ink2"],
            va="bottom",
            weight="bold",
        )
        for y, txt, col in (
            (d.test_loss.iloc[-1], f"TEST loss\n{d.test_loss.iloc[-1]:.2f}", t["me"]),
            (d.train_loss.iloc[-1], f"TRAINING loss\n{d.train_loss.iloc[-1]:.2f}", t["neutral"]),
            (maia, f"Maia 1100, same\ntest games {maia:.2f}", t["maia"]),
        ):
            ax.text(x_lab, y, txt, transform=tr, fontsize=16, color=col, va="center", weight="bold")
        fig.text(
            0.03,
            0.025,
            "The gap between the two curves is overfitting. Maia never saw my games.",
            fontsize=15,
            color=t["ink2"],
        )
        save(fig, "fig10_loss_curves", theme)

    both(draw)


def main(only=()):
    """Build all figures, or only those named in `only` (e.g. ["fig10"])."""
    console.print(f"[bold]Clean set[/]: {N} positions, {NG} games, best Maia (top-1) = {BEST}")
    figs = (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10)
    chosen = [f for f in figs if f.__name__ in only] or figs
    for f in chosen:
        f()
    console.print(f"[green]figures written to {C.FIG_DIR.relative_to(C.ROOT)}[/]")
