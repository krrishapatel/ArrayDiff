"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from collections import Counter


# Module to import, then the adapter to hand it to. A table rather than a chain
# of ifs, so adding a library is one line here and one in backends.py.
BACKENDS = {
    "mlx": ("mlx.core", "mlx_backend"),
    "torch": ("torch", "torch_backend"),
    "jax": ("jax", "jax_backend"),
    "numpy": ("numpy", None),
}


def _load_backend(name):
    from importlib import import_module

    from . import backends

    module, adapter = BACKENDS[name]
    if adapter is None:
        return backends.numpy_backend()
    try:
        lib = import_module(module)
    except ImportError:
        hint = "; set PYTHONPATH to an mlx build" if name == "mlx" else ""
        print(f"{module} is not importable{hint}", file=sys.stderr)
        return None
    return getattr(backends, adapter)(lib)


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
        "--backend",
        default="mlx",
        choices=tuple(BACKENDS),
        help="library to test",
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
