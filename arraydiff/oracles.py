"""Reference oracles and ULP measurement.

NumPy answers "what should this op return", mpmath answers "what is the exactly
correct real number", which is what ULP error has to be measured against. A
NumPy result is itself only correctly rounded for the basic arithmetic ops, so
comparing one implementation's transcendentals against NumPy's measures
disagreement, not error.
"""

from __future__ import annotations

import numpy as np

try:
    import mpmath as mp

    HAVE_MPMATH = True
except ImportError:  # pragma: no cover - optional dependency
    HAVE_MPMATH = False

# Exact real-valued references, used only for ULP measurement.
EXACT = {}

# Relative condition number of each function, cond(x) = |x f'(x) / f(x)|.
#
# This is what separates "the implementation is inaccurate" from "the problem is
# ill-conditioned". exp has cond = |x|, so at x = 700 an error of several hundred
# ULP is forced by the input's own rounding and says nothing about the code. Near
# a zero of sin, cond blows up for the same reason. Holding every op to a flat 1
# ULP budget reports arithmetic that is working correctly.
CONDITION = {}

# Functions whose argument must determine a phase. One ULP of 1e308 spans about
# 1e291 periods, so sin/cos/tan of a huge argument have no meaningful answer to
# be accurate to, no matter how good the implementation is.
PERIODIC = {"sin", "cos", "tan"}

# How many ULP each op is actually expected to be within, before the condition
# number scaling in check_ulp is applied.
#
# IEEE-754 requires sqrt and division to be correctly rounded, so half a ULP is
# the real contract and anything above it is a genuine defect. No standard
# requires that of the transcendentals: vector math libraries typically promise a
# few ULP, and Accelerate, the Metal shading language and CUDA all document
# theirs in the low single digits. Demanding one ULP of expm1 reports a working
# implementation as broken, which is the failure mode that makes a tool like this
# get ignored. DEFAULT_BUDGET is deliberately loose enough that a finding means
# an implementation nobody would defend.
CORRECTLY_ROUNDED = {"sqrt", "reciprocal", "square"}
DEFAULT_BUDGET = 4.0
STRICT_BUDGET = 0.5


def budget_for(exact_key: str) -> float:
    return STRICT_BUDGET if exact_key in CORRECTLY_ROUNDED else DEFAULT_BUDGET

# Above this input spacing the phase of a periodic function is indeterminate.
PHASE_SPACING_LIMIT = 1e-2

if HAVE_MPMATH:
    mp.mp.dps = 60
    EXACT.update(
        {
            "sqrt": mp.sqrt,
            "exp": mp.exp,
            "log": mp.log,
            "sin": mp.sin,
            "cos": mp.cos,
            "tan": mp.tan,
            "tanh": mp.tanh,
            "sinh": mp.sinh,
            "cosh": mp.cosh,
            "arctan": mp.atan,
            "log1p": mp.log1p,
            "expm1": mp.expm1,
            "square": lambda x: x * x,
            "reciprocal": lambda x: 1 / x,
            "rsqrt": lambda x: 1 / mp.sqrt(x),
        }
    )

    def _safe(fn):
        """Condition numbers are only advisory, so fall back to 1 on failure."""

        def wrapped(x):
            try:
                v = fn(x)
                return float(abs(v)) if mp.isfinite(v) else 1.0
            except (ValueError, ZeroDivisionError, OverflowError):
                return 1.0

        return wrapped

    CONDITION.update(
        {
            "sqrt": _safe(lambda x: mp.mpf(0.5)),
            "square": _safe(lambda x: mp.mpf(2)),
            "reciprocal": _safe(lambda x: mp.mpf(1)),
            "rsqrt": _safe(lambda x: mp.mpf(0.5)),
            "exp": _safe(lambda x: x),
            "expm1": _safe(lambda x: x * mp.exp(x) / mp.expm1(x)),
            "log": _safe(lambda x: 1 / mp.log(x)),
            "log1p": _safe(lambda x: x / ((1 + x) * mp.log1p(x))),
            "sin": _safe(lambda x: x * mp.cos(x) / mp.sin(x)),
            "cos": _safe(lambda x: x * mp.sin(x) / mp.cos(x)),
            "tan": _safe(lambda x: x / (mp.sin(x) * mp.cos(x))),
            "tanh": _safe(lambda x: 2 * x / mp.sinh(2 * x)),
            "sinh": _safe(lambda x: x * mp.cosh(x) / mp.sinh(x)),
            "cosh": _safe(lambda x: x * mp.tanh(x)),
            "arctan": _safe(lambda x: x / ((1 + x * x) * mp.atan(x))),
        }
    )


def ulp_error(got: np.ndarray, inputs: np.ndarray, exact_fn, dtype=None) -> np.ndarray:
    """|got - exact(input)| measured in units of the last place at `got`.

    `dtype` is the precision the op actually ran in, and it is required for the
    answer to mean anything: one ULP of float16 near 1.0 is about 1e-3 while one
    ULP of float64 is about 2e-16. Measuring a float16 result against float64
    spacing overstates the error by ~1e12 and makes every op look broken.

    Non-finite expectations and non-finite results are reported as 0 so that
    special-value handling is judged by the semantic checks instead, which can
    actually describe what went wrong.
    """
    if not HAVE_MPMATH:
        raise RuntimeError("ULP measurement needs mpmath")
    work = np.dtype(dtype) if dtype is not None else np.asarray(got).dtype
    if work.kind != "f":
        work = np.dtype(np.float64)
    got = np.asarray(got, dtype=np.float64)
    inputs = np.asarray(inputs, dtype=np.float64)
    out = np.zeros(got.shape, dtype=np.float64)
    for i in range(got.size):
        g, x = got.flat[i], inputs.flat[i]
        if not np.isfinite(g) or not np.isfinite(x):
            continue
        try:
            want = exact_fn(mp.mpf(float(x)))
        except (ValueError, ZeroDivisionError):
            continue
        if not mp.isfinite(want):
            continue
        # spacing() at the computed value is the ULP width in that binade, taken
        # in the precision the op ran in.
        step = float(np.spacing(work.type(abs(g)))) or float(np.finfo(work).tiny)
        out.flat[i] = abs(float((mp.mpf(float(g)) - want) / mp.mpf(float(step))))
    return out


def bitwise_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise mask of where `a` and `b` differ, counting -0.0 != 0.0 and
    NaN == NaN.

    Plain == says 0.0 == -0.0 and NaN != NaN, so it hides exactly the two classes
    of bug that show up most often, signed zero and NaN propagation. Every place
    that reports a difference has to use this one definition: an earlier version
    compared with bitwise_equal but counted and located the differing element
    with ==, so a signed-zero disagreement was reported as "0 elements differ" at
    index 0, which points at the wrong input.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if a.dtype.kind == "c":
        return bitwise_diff(a.real, b.real) | bitwise_diff(a.imag, b.imag)
    if a.dtype.kind == "f" and a.dtype == b.dtype:
        both_nan = np.isnan(a) & np.isnan(b)
        # view as ints so -0.0 and 0.0 compare unequal
        same = a.view(_int_view(a.dtype)) == b.view(_int_view(b.dtype))
        return ~(both_nan | same)
    if a.dtype.kind in "iub" and b.dtype.kind in "iub":
        # Compare integers as integers; a float64 detour would lose int64 bits.
        return a != b
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    both_nan = np.isnan(a64) & np.isnan(b64)
    same = (a64 == b64) & (np.signbit(a64) == np.signbit(b64))
    return ~(both_nan | same)


def bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether `a` and `b` agree bit for bit. See bitwise_diff."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return not bool(bitwise_diff(a, b).any())


def _int_view(dtype: np.dtype):
    return {2: np.int16, 4: np.int32, 8: np.int64}[dtype.itemsize]
