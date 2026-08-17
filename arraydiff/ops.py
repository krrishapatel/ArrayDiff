"""The op table: how to call each op in the library under test, and in NumPy.

There is one spec list, not one per library. Two tables drift: an op added for
MLX and forgotten for Torch silently reduces coverage without failing anything.
Instead each spec names the op, and a library that spells it differently gets an
entry in that library's name map.

`domain` restricts inputs where the op is only defined on part of the line, so
that a NaN-vs-NaN comparison does not stand in for a real check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Op:
    name: str
    arity: int
    fn: Callable  # the op in the library under test
    numpy: Callable
    # Restrict the input domain, e.g. positives only for log.
    domain: Callable[[np.ndarray], np.ndarray] | None = None
    # Second-argument domain for binary ops, e.g. no zero divisors.
    domain_b: Callable[[np.ndarray], np.ndarray] | None = None
    kinds: tuple[str, ...] = ("f",)
    exact_key: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Spec:
    name: str
    arity: int
    numpy: Callable | None
    domain: Callable | None = None
    domain_b: Callable | None = None
    kinds: tuple[str, ...] = ("f",)
    exact_key: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


def positive(x):
    return np.abs(x) + 0.5


def nonzero(x):
    out = x.copy()
    out[out == 0] = 1
    return out


SPECS = [
    # --- unary, exact or near-exact ---
    Spec("negative", 1, np.negative, kinds=("f", "i"), tags=("exact",)),
    Spec("abs", 1, np.abs, kinds=("f", "i"), tags=("exact",)),
    Spec("sign", 1, np.sign, kinds=("f", "i"), tags=("exact",)),
    Spec("floor", 1, np.floor, tags=("exact",)),
    Spec("ceil", 1, np.ceil, tags=("exact",)),
    Spec("square", 1, np.square, exact_key="square", tags=("exact",)),
    Spec("sqrt", 1, np.sqrt, domain=positive, exact_key="sqrt", tags=("exact",)),
    Spec("reciprocal", 1, np.reciprocal, domain=nonzero, exact_key="reciprocal",
         tags=("exact",)),
    Spec("rsqrt", 1, lambda x: 1.0 / np.sqrt(x), domain=positive,
         exact_key="rsqrt"),
    # --- unary transcendentals, judged by ULP not bit equality ---
    Spec("exp", 1, np.exp, exact_key="exp"),
    Spec("log", 1, np.log, domain=positive, exact_key="log"),
    Spec("log1p", 1, np.log1p, domain=positive, exact_key="log1p"),
    Spec("expm1", 1, np.expm1, exact_key="expm1"),
    Spec("sin", 1, np.sin, exact_key="sin"),
    Spec("cos", 1, np.cos, exact_key="cos"),
    Spec("tan", 1, np.tan, exact_key="tan"),
    Spec("tanh", 1, np.tanh, exact_key="tanh"),
    Spec("sinh", 1, np.sinh, exact_key="sinh"),
    Spec("cosh", 1, np.cosh, exact_key="cosh"),
    Spec("arctan", 1, np.arctan, exact_key="arctan"),
    Spec("erf", 1, None),
    Spec("logical_not", 1, np.logical_not, kinds=("f", "i")),
    # --- binary ---
    Spec("add", 2, np.add, kinds=("f", "i"), tags=("exact",)),
    Spec("subtract", 2, np.subtract, kinds=("f", "i"), tags=("exact",)),
    Spec("multiply", 2, np.multiply, kinds=("f", "i"), tags=("exact",)),
    Spec("divide", 2, np.divide, domain_b=nonzero, tags=("exact",)),
    Spec("maximum", 2, np.maximum, kinds=("f", "i"), tags=("exact",)),
    Spec("minimum", 2, np.minimum, kinds=("f", "i"), tags=("exact",)),
    Spec("power", 2, np.power, domain=positive),
    Spec("remainder", 2, np.remainder, domain_b=nonzero, kinds=("f", "i"),
         tags=("exact",)),
    Spec("floor_divide", 2, np.floor_divide, domain_b=nonzero, kinds=("f", "i"),
         tags=("exact",)),
    Spec("arctan2", 2, np.arctan2),
    Spec("logaddexp", 2, np.logaddexp),
    # --- comparisons, must be exactly right ---
    Spec("equal", 2, np.equal, kinds=("f", "i"), tags=("exact",)),
    Spec("less", 2, np.less, kinds=("f", "i"), tags=("exact",)),
    Spec("greater", 2, np.greater, kinds=("f", "i"), tags=("exact",)),
    Spec("less_equal", 2, np.less_equal, kinds=("f", "i"), tags=("exact",)),
]

# Where a library spells an op differently. `torch.equal` is the trap: it reduces
# the whole tensor to one bool rather than comparing elementwise, so using it
# would make every comparison check silently vacuous.
TORCH_NAMES = {"power": "pow", "equal": "eq"}


def _build(lib, names):
    ops = {}
    for s in SPECS:
        fn = getattr(lib, names.get(s.name, s.name), None)
        if fn is None:
            continue
        ops[s.name] = Op(
            name=s.name,
            arity=s.arity,
            fn=fn,
            numpy=s.numpy,
            domain=s.domain,
            domain_b=s.domain_b,
            kinds=s.kinds,
            exact_key=s.exact_key,
            tags=s.tags,
        )
    return ops


def build_ops(mx):
    """MLX. Built lazily so importing this module does not require MLX."""
    return _build(mx, {})


def build_torch_ops(torch):
    return _build(torch, TORCH_NAMES)


def build_jax_ops(jax):
    """jnp uses NumPy's spelling for almost everything.

    rsqrt and erf are the two exceptions and live elsewhere in JAX, so they are
    pulled in by hand. Letting them fall through as missing would quietly test
    JAX on a smaller op set than the other libraries.
    """
    import jax.numpy as jnp
    import jax.scipy.special as jsp

    ns = SimpleNamespace(
        **{s.name: getattr(jnp, s.name) for s in SPECS if hasattr(jnp, s.name)}
    )
    ns.rsqrt = jax.lax.rsqrt
    ns.erf = jsp.erf
    return _build(ns, {})
