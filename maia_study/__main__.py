"""Command-line entry point: python -m maia_study <command> [options].

Reproduce every number and figure from the saved analysis (no engines needed):

    python -m maia_study reproduce

Full pipeline from scratch (needs lc0, the Maia weights and Stockfish; see README):

    python -m maia_study fetch       # download my games from Lichess and anonymise them
    python -m maia_study analyse     # run Maia 1100-1900 and Stockfish on 1000 sampled positions
    python -m maia_study topup       # Stockfish evals for Maia top-1 moves outside Stockfish's top 3
    python -m maia_study reproduce   # statistics, showcase positions, figures, video assets
"""

import argparse
import time

from rich.console import Console

console = Console()


def cmd_fetch(args):
    from . import config as C
    from .data import anonymise_pgn, fetch_lichess

    path = fetch_lichess(args.user, max_games=args.max_games)
    n = anonymise_pgn(path, keep=args.user)
    console.print(f"wrote {path.relative_to(C.ROOT)} ({n} opponent names replaced)")
    console.print("[yellow]Note: this downloads the latest games, so the sample can differ from the published one.[/]")


def cmd_analyse(args):
    from .analysis import analyse

    records, meta = analyse(user=args.user, n=args.n, seed=args.seed, depth=args.depth, refresh=args.refresh)
    console.print(f"{len(records)} positions analysed · {meta}")


def cmd_topup(_):
    from . import topup

    topup.main()


def cmd_stats(_):
    from . import stats

    stats.main()


def cmd_showcase(_):
    from . import showcase

    showcase.main()


def cmd_figures(args):
    from . import figures

    figures.main(args.only)


def cmd_toy_model(_):
    from . import toy_model

    toy_model.main()


def cmd_video(args):
    from . import video_assets

    video_assets.run(args.rest)


def cmd_terminal(args):
    from . import terminal_views

    terminal_views.main(args.rest)


def cmd_gui(args):
    from . import gui

    gui.main(args.rest)


def cmd_pick_positions(args):
    from . import pick_positions

    pick_positions.main(args.rest)


def cmd_setup(args):
    from . import setup_engines

    setup_engines.main(args.rest)


def cmd_reproduce(args):
    """Everything downstream of the cached engine output, in dependency order."""
    steps = [("stats", cmd_stats), ("showcase", cmd_showcase), ("figures", cmd_figures)]
    if not args.skip_video:
        steps.append(("video", cmd_video))
    for name, fn in steps:
        t0 = time.time()
        console.rule(f"[bold]{name}")
        fn(argparse.Namespace(only=[], rest=[]))
        console.print(f"[dim]{name}: {time.time() - t0:.0f} s[/]")


def main(argv=None):
    from . import config as C

    ap = argparse.ArgumentParser(
        prog="python -m maia_study", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")

    sp = sub.add_parser("fetch", help="download and anonymise my Lichess games")
    sp.add_argument("--user", default=C.PLAYER)
    sp.add_argument("--max-games", type=int, default=C.LICHESS_MAX_GAMES)
    sp.set_defaults(fn=cmd_fetch)

    sp = sub.add_parser("analyse", help="run all engines on the sampled positions (cached)")
    sp.add_argument("--user", default=C.PLAYER)
    sp.add_argument("--n", type=int, default=C.N_POSITIONS)
    sp.add_argument("--seed", type=int, default=C.SEED)
    sp.add_argument("--depth", type=int, default=C.SF_DEPTH)
    sp.add_argument("--refresh", action="store_true", help="ignore the cache and rerun the engines")
    sp.set_defaults(fn=cmd_analyse)

    sub.add_parser("topup", help="extra Stockfish evals for Maia's top-1 moves (cached)").set_defaults(fn=cmd_topup)
    sub.add_parser("stats", help="headline numbers -> results/*.json").set_defaults(fn=cmd_stats)
    sub.add_parser("showcase", help="showcase positions, boards and cards").set_defaults(fn=cmd_showcase)
    sp = sub.add_parser("figures", help="chart figures -> figures/charts/")
    sp.add_argument("only", nargs="*", help="figure names, e.g. fig1 fig10 (default: all)")
    sp.set_defaults(fn=cmd_figures)
    sub.add_parser("toy-model", help="supplementary toy model (figure 10 data)").set_defaults(fn=cmd_toy_model)
    sp = sub.add_parser("reproduce", help="stats + showcase + figures + video from the saved analysis")
    sp.add_argument("--skip-video", action="store_true", help="skip the video assets (about 2.5 min, needs ffmpeg)")
    sp.set_defaults(fn=cmd_reproduce)

    # Commands with their own options: everything after the command name is passed through.
    for name, fn, hlp in (
        ("video", cmd_video, "video clips and stills -> figures/video/"),
        ("terminal", cmd_terminal, "terminal views for screenshots"),
        ("gui", cmd_gui, "interactive tkinter demo"),
        ("pick-positions", cmd_pick_positions, "choose the GUI demo positions"),
        ("setup", cmd_setup, "download Maia weights and build Stockfish"),
    ):
        sub.add_parser(name, help=hlp, add_help=False).set_defaults(fn=fn, passthrough=True)

    args, rest = ap.parse_known_args(argv)
    if getattr(args, "passthrough", False):
        args.rest = rest
    elif rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    args.fn(args)


if __name__ == "__main__":
    main()
