"""Download the Maia weights and build Stockfish 17.1 from source. Safe to re-run.

    python -m maia_study setup              # weights + Stockfish
    python -m maia_study setup --weights    # weights only

lc0 is not built here; see the README for the exact commit and build command.
"""

import argparse
import platform
import shutil
import subprocess
from pathlib import Path

import requests
from rich.console import Console

from . import config as C

console = Console()


def download_weights():
    C.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for r in C.RATINGS:
        dest = C.WEIGHTS_DIR / f"maia-{r}.pb.gz"
        if dest.exists():
            console.print(f"  [green]✓[/] {dest.name} already present")
            continue
        url = C.MAIA_URL.format(r)
        console.print(f"  [cyan]↓[/] {url}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        console.print(f"  [green]✓[/] {dest.name} ({len(resp.content) / 1e6:.1f} MB)")


def stockfish_arch():
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists() and "asimddp" in cpuinfo.read_text():
            return "armv8-dotprod"
        return "armv8"
    return "x86-64-avx2"


def build_stockfish():
    binary = C.STOCKFISH
    if binary.exists():
        console.print(f"  [green]✓[/] Stockfish already built at {binary}")
        return
    src = C.ENGINES_DIR / "stockfish-src"
    if not src.exists():
        C.ENGINES_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", C.STOCKFISH_TAG, C.STOCKFISH_REPO, str(src)], check=True
        )
    arch = stockfish_arch()
    console.print(f"  [cyan]⚙[/] building Stockfish {C.STOCKFISH_TAG} ARCH={arch} (takes a few minutes)")
    subprocess.run(["make", "-j", "build", f"ARCH={arch}"], cwd=src / "src", check=True, stdout=subprocess.DEVNULL)
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "src" / "stockfish", binary)
    console.print(f"  [green]✓[/] Stockfish built at {binary}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m maia_study setup", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", action="store_true", help="only download Maia weights")
    parser.add_argument("--stockfish", action="store_true", help="only build Stockfish")
    args = parser.parse_args(argv)
    both = not (args.weights or args.stockfish)
    if both or args.weights:
        console.rule("Maia weights")
        download_weights()
    if both or args.stockfish:
        console.rule("Stockfish")
        build_stockfish()
