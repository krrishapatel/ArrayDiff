"""Input spaces: dtypes, layouts, and the values worth testing.

The value sets are deliberately weighted toward the places floating point
implementations actually disagree: signs, zeros, denormals, and the boundaries
of each dtype's range. Uniform random sampling almost never lands on those.
"""

from __future__ import annotations

import numpy as np

# Values that break implementations. Each entry is exactly representable in
# every float dtype we test, so the same list is reusable across precisions.
SPECIAL_FLOATS = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    3.0,
    -3.0,
    7.0,
    -7.0,
    float("inf"),
    float("-inf"),
    float("nan"),
]

SPECIAL_INTS = [0, 1, -1, 2, -2, 3, -3, 7, -7, 127, -128]


def denormals(dtype: np.dtype) -> list[float]:
    """Smallest subnormal, a mid subnormal, and the smallest normal."""
    info = np.finfo(dtype)
    tiny = float(info.smallest_subnormal)
    return [tiny, -tiny, tiny * 3, float(info.smallest_normal)]


def extremes(dtype: np.dtype) -> list[float]:
    info = np.finfo(dtype)
    return [float(info.max), float(-info.max), float(info.eps), float(1 + info.eps)]


def float_values(dtype: np.dtype, *, rng: np.random.Generator, n_random: int = 256):
    """Special values, denormals, extremes, then random draws to fill in."""
    vals = list(SPECIAL_FLOATS) + denormals(dtype) + extremes(dtype)
    if n_random:
        # Spread across several magnitudes rather than one uniform band, since
        # accuracy bugs are usually magnitude dependent.
        for scale in (1e-3, 1.0, 1e3):
            vals.extend((rng.uniform(-3, 3, n_random // 3) * scale).tolist())
    return np.array(vals, dtype=dtype)


def int_values(dtype: np.dtype, *, rng: np.random.Generator, n_random: int = 128):
    info = np.iinfo(dtype)
    vals = [v for v in SPECIAL_INTS if info.min <= v <= info.max]
    vals += [int(info.min), int(info.max), int(info.min) + 1, int(info.max) - 1]
    if n_random:
        vals.extend(
            rng.integers(info.min, info.max, n_random, endpoint=True).tolist()
        )
    return np.array(vals, dtype=dtype)


# Layouts. Each takes a backend and a 1-D reference array, and returns
# (native_array, numpy_array) holding the SAME logical values reached through a
# different memory layout. Any op whose result depends on which of these it got
# is a bug: the layout carries no numerical information.
#
# Not every backend can express every layout. Torch has no negative strides, so
# `reversed` does not exist there; `Backend.layouts` says which are available
# rather than substituting a copy and comparing a value against itself.


def _pad(values: np.ndarray, factor: int) -> np.ndarray:
    """Interleave values with filler so a stride can pick them back out."""
    out = np.zeros(len(values) * factor, dtype=values.dtype)
    out[::factor] = values
    return out


LAYOUTS = {}


def layout(name):
    def deco(fn):
        LAYOUTS[name] = fn
        return fn

    return deco


@layout("contiguous")
def _contiguous(be, values, dtype=None):
    return be.array(values, dtype=dtype), values


@layout("strided")
def _strided(be, values, dtype=None):
    padded = _pad(values, 2)
    return be.array(padded, dtype=dtype)[::2], padded[::2]


@layout("strided3")
def _strided3(be, values, dtype=None):
    padded = _pad(values, 3)
    return be.array(padded, dtype=dtype)[::3], padded[::3]


@layout("reversed")
def _reversed(be, values, dtype=None):
    flipped = values[::-1].copy()
    return be.array(flipped, dtype=dtype)[::-1], flipped[::-1]


@layout("transposed")
def _transposed(be, values, dtype=None):
    """A transposed view, reading back in the original order.

    The values go in row 0 of a (3, n) grid, so after the transpose column 0 of
    the (n, 3) result is the original sequence. Taking the transpose of a square
    grid instead would permute the values, which makes any resulting difference
    the harness's fault rather than the library's.
    """
    n = len(values)
    grid = np.zeros((3, n), dtype=values.dtype)
    grid[0] = values
    return (
        be.array(grid, dtype=dtype).T[:, 0],
        grid.T[:, 0],
    )


@layout("column")
def _column(be, values, dtype=None):
    # A column of a 2-D array: non-contiguous, original order.
    n = len(values)
    grid = np.zeros((n, 3), dtype=values.dtype)
    grid[:, 0] = values
    return be.array(grid, dtype=dtype)[:, 0], grid[:, 0]


@layout("offset")
def _offset(be, values, dtype=None):
    # A slice that does not start at element zero.
    padded = np.concatenate([np.zeros(3, dtype=values.dtype), values])
    return be.array(padded, dtype=dtype)[3:], padded[3:]


@layout("broadcast")
def _broadcast(be, values, dtype=None):
    # Broadcasting a length-1 axis gives a zero-stride view.
    col = values.reshape(-1, 1)
    return (
        be.broadcast_to(be.array(col, dtype=dtype), (len(values), 2))[:, 0],
        np.broadcast_to(col, (len(values), 2))[:, 0],
    )
