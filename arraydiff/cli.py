"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from collections import Counter


def _load_backend(name):
    from .backends import mlx_backend, torch_backend

    if name == "mlx":
        try:
            import mlx.core as mx
        except ImportError:
            print(
                "mlx is not importable; set PYTHONPATH to an mlx build",
                file=sys.stderr,
            )
            return None
        return mlx_backend(mx)
    try:
        import torch
    except ImportError:
        print("torch is not importable", file=sys.stderr)
        return None
    return torch_backend(torch)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="arraydiff",
        description="Differential numerical testing for array libraries.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="restrict to these op names")
    ap.add_argument(
        "--random", type=int, default=256, help="random draws per magnitude band"
    )
    ap.add_argument("--check", nargs="*", help="restrict to these check names")
    ap.add_argument(
        "--all",
        action="store_true",
        help="also print divergences that known.py already accounts for",
    )
    ap.add_argument(
        "--backend", default="mlx", choices=("mlx", "torch"), help="library to test"
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    be = _load_backend(args.backend)
    if be is None:
        return 2

    from .known import known_for, partition
    from .runner import run

    findings = run(
        be, seed=args.seed, only=args.only, n_random=args.random, verbose=args.verbose
    )
    if args.check:
        findings = [f for f in findings if f.check in args.check]

    new, already_known = partition(findings, known_for(be.name))

    for f in new:
        print(f)
        print()

    print(f"{len(new)} new findings")
    for check, n in sorted(Counter(f.check for f in new).items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {check}")

    # Say what was suppressed and why. A silently shrinking count is how a real
    # regression hides behind an entry someone added months ago.
    if already_known:
        print(f"\n{len(already_known)} already accounted for in known.py")
        for ref, n in sorted(
            Counter(k.ref for _, k in already_known).items(), key=lambda kv: -kv[1]
        ):
            print(f"  {n:4d}  {ref}")
        if args.all:
            print()
            for f, k in already_known:
                print(f"{f}\n    known:  {k.ref}\n")

    return 1 if new else 0


if __name__ == "__main__":
    raise SystemExit(main())
