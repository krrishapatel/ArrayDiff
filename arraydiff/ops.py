"""The op table: how to call each op in MLX and in NumPy.

`domain` restricts inputs where the op is only defined on part of the line, so
that a NaN-vs-NaN comparison does not stand in for a real check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Op:
    name: str
    arity: int
    mlx: Callable
    numpy: Callable
    # Restrict the input domain, e.g. positives only for log.
    domain: Callable[[np.ndarray], np.ndarray] | None = None
    # Second-argument domain for binary ops, e.g. no zero divisors.
    domain_b: Callable[[np.ndarray], np.ndarray] | None = None
    kinds: tuple[str, ...] = ("f",)
    exact_key: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


def positive(x):
    return np.abs(x) + 0.5


def nonzero(x):
    out = x.copy()
    out[out == 0] = 1
    return out


def build_ops(mx):
    """Built lazily so importing this module does not require MLX."""
    ops = [
        # --- unary, exact or near-exact ---
        Op("negative", 1, mx.negative, np.negative, kinds=("f", "i"), tags=("exact",)),
        Op("abs", 1, mx.abs, np.abs, kinds=("f", "i"), tags=("exact",)),
        Op("sign", 1, mx.sign, np.sign, kinds=("f", "i"), tags=("exact",)),
        Op("floor", 1, mx.floor, np.floor, tags=("exact",)),
        Op("ceil", 1, mx.ceil, np.ceil, tags=("exact",)),
        Op("square", 1, mx.square, np.square, exact_key="square", tags=("exact",)),
        Op("sqrt", 1, mx.sqrt, np.sqrt, domain=positive, exact_key="sqrt",
           tags=("exact",)),
        Op("reciprocal", 1, mx.reciprocal, np.reciprocal, domain=nonzero,
           exact_key="reciprocal", tags=("exact",)),
        Op("rsqrt", 1, mx.rsqrt, lambda x: 1.0 / np.sqrt(x), domain=positive,
           exact_key="rsqrt"),
        # --- unary transcendentals, judged by ULP not bit equality ---
        Op("exp", 1, mx.exp, np.exp, exact_key="exp"),
        Op("log", 1, mx.log, np.log, domain=positive, exact_key="log"),
        Op("log1p", 1, mx.log1p, np.log1p, domain=positive, exact_key="log1p"),
        Op("expm1", 1, mx.expm1, np.expm1, exact_key="expm1"),
        Op("sin", 1, mx.sin, np.sin, exact_key="sin"),
        Op("cos", 1, mx.cos, np.cos, exact_key="cos"),
        Op("tan", 1, mx.tan, np.tan, exact_key="tan"),
        Op("tanh", 1, mx.tanh, np.tanh, exact_key="tanh"),
        Op("sinh", 1, mx.sinh, np.sinh, exact_key="sinh"),
        Op("cosh", 1, mx.cosh, np.cosh, exact_key="cosh"),
        Op("arctan", 1, mx.arctan, np.arctan, exact_key="arctan"),
        Op("erf", 1, mx.erf, None),
        Op("logical_not", 1, mx.logical_not, np.logical_not, kinds=("f", "i")),
        # --- binary ---
        Op("add", 2, mx.add, np.add, kinds=("f", "i"), tags=("exact",)),
        Op("subtract", 2, mx.subtract, np.subtract, kinds=("f", "i"),
           tags=("exact",)),
        Op("multiply", 2, mx.multiply, np.multiply, kinds=("f", "i"),
           tags=("exact",)),
        Op("divide", 2, mx.divide, np.divide, domain_b=nonzero, tags=("exact",)),
        Op("maximum", 2, mx.maximum, np.maximum, kinds=("f", "i"), tags=("exact",)),
        Op("minimum", 2, mx.minimum, np.minimum, kinds=("f", "i"), tags=("exact",)),
        Op("power", 2, mx.power, np.power, domain=positive),
        Op("remainder", 2, mx.remainder, np.remainder, domain_b=nonzero,
           kinds=("f", "i"), tags=("exact",)),
        Op("floor_divide", 2, mx.floor_divide, np.floor_divide, domain_b=nonzero,
           kinds=("f", "i"), tags=("exact",)),
        Op("arctan2", 2, mx.arctan2, np.arctan2),
        Op("logaddexp", 2, mx.logaddexp, np.logaddexp),
        # --- comparisons, must be exactly right ---
        Op("equal", 2, mx.equal, np.equal, kinds=("f", "i"), tags=("exact",)),
        Op("less", 2, mx.less, np.less, kinds=("f", "i"), tags=("exact",)),
        Op("greater", 2, mx.greater, np.greater, kinds=("f", "i"), tags=("exact",)),
        Op("less_equal", 2, mx.less_equal, np.less_equal, kinds=("f", "i"),
           tags=("exact",)),
    ]
    return {op.name: op for op in ops if op.mlx is not None}
