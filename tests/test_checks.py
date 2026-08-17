"""Tests that the checks themselves are valid.

A check is only worth running if a correct implementation passes it. NumPy is the
stand-in for correct here, so every check that NumPy is allowed to satisfy gets
run against NumPy. When one of these fails, the check is wrong, not the library
under test. This caught a real mistake: `divmod-identity` demanded exact
reconstruction, which NumPy misses on 54 of 400 float32 pairs, because the
floored remainder is `fmod(a, b) + b` and that addition rounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from arraydiff.checks import check_divmod_identity

FLOATS = [("float16", np.float16), ("float32", np.float32), ("float64", np.float64)]
INTS = [("int8", np.int8), ("int16", np.int16), ("int32", np.int32), ("int64", np.int64)]

# Deliberately mixed: values whose decimal form is not representable (0.1, 0.3,
# 0.7), both signs, and magnitudes far apart. The inexact ones are the whole
# point, since exact operands like 7.5 / 2.5 hide quotient rounding.
HARD = [0.1, -0.1, 1.0, -1.0, 2.5, -2.5, 7.0, -7.0, 3.0, -3.0,
        0.3, -0.3, 0.7, -0.7, 13.0, -13.0, 1e3, -1e3]


class NumpyAsMX:
    """Presents NumPy through the small surface the checks call."""

    float16, float32, float64 = np.float16, np.float32, np.float64
    int8, int16, int32, int64 = np.int8, np.int16, np.int32, np.int64
    # NumPy has no bfloat16; a sentinel keeps `_to_numpy`'s dtype test falsy.
    bfloat16 = object()

    @staticmethod
    def array(values, dtype=None):
        return np.asarray(values, dtype=dtype)

    @staticmethod
    def divmod(a, b):
        return np.divmod(a, b)

    @staticmethod
    def floor_divide(a, b):
        return np.floor_divide(a, b)

    @staticmethod
    def remainder(a, b):
        return np.remainder(a, b)

    @staticmethod
    def eval(*args):
        return None


def _all_pairs(values, dtype):
    v = np.asarray(values, dtype=dtype)
    return np.repeat(v, v.size).astype(dtype), np.tile(v, v.size).astype(dtype)


class TestDivmodIdentity:
    """`q*b + r == a` is exact for integers and only near-exact for floats."""

    @pytest.mark.parametrize("name,dtype", FLOATS)
    def test_numpy_passes_in_every_float_precision(self, name, dtype):
        a, b = _all_pairs(HARD, dtype)
        with np.errstate(all="ignore"):
            found = check_divmod_identity(NumpyAsMX, a, b, a, b, name)
        assert found == [], f"the check flags NumPy in {name}: {found}"

    @pytest.mark.parametrize("name,dtype", INTS)
    def test_numpy_passes_for_integers(self, name, dtype):
        vals = [-128, -13, -7, -3, -1, 1, 3, 7, 13, 127]
        a, b = _all_pairs(vals, dtype)
        with np.errstate(all="ignore"):
            found = check_divmod_identity(NumpyAsMX, a, b, a, b, name)
        assert found == [], f"the check flags NumPy in {name}: {found}"

    def test_integers_are_still_judged_exactly(self):
        """The float tolerance must not leak into the integer path.

        Off by one in an integer quotient is a real bug, and it is smaller than
        one ULP of the reconstruction, so a tolerance would hide it.
        """
        a = np.array([7, -7], np.int32)
        b = np.array([2, 2], np.int32)

        class OffByOne(NumpyAsMX):
            @staticmethod
            def divmod(x, y):
                q, r = np.divmod(x, y)
                return q + 1, r

        with np.errstate(all="ignore"):
            found = check_divmod_identity(OffByOne, a, b, a, b, "int32")
        assert "divmod-identity" in {f.check for f in found}

    def test_a_wrong_float_quotient_is_still_caught(self):
        """The tolerance is a few ULP, so a whole integer step still fails.

        This is the mlx case: `floor(x / y)` rounds up to 10 for `1.0 / 0.1` in
        float32 while the remainder is right, so reconstruction lands at 1.1.
        """
        a = np.array([1.0], np.float32)
        b = np.array([0.1], np.float32)

        class DividesTwice(NumpyAsMX):
            @staticmethod
            def divmod(x, y):
                return np.floor(x / y), np.fmod(x, y)

        with np.errstate(all="ignore"):
            found = check_divmod_identity(DividesTwice, a, b, a, b, "float32")
        checks = {f.check for f in found}
        assert "divmod-identity" in checks
        # The same input also trips the oracle-free cross-check, which is fine.
        assert "divmod-vs-floor_divide" in checks
