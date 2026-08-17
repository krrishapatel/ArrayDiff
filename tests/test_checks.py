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

from dataclasses import replace

from arraydiff.backends import numpy_backend
from arraydiff.checks import check_device_invariance, check_divmod_identity
from arraydiff.ops import Op

FLOATS = [("float16", np.float16), ("float32", np.float32), ("float64", np.float64)]
INTS = [("int8", np.int8), ("int16", np.int16), ("int32", np.int32), ("int64", np.int64)]

# Deliberately mixed: values whose decimal form is not representable (0.1, 0.3,
# 0.7), both signs, and magnitudes far apart. The inexact ones are the whole
# point, since exact operands like 7.5 / 2.5 hide quotient rounding.
HARD = [0.1, -0.1, 1.0, -1.0, 2.5, -2.5, 7.0, -7.0, 3.0, -3.0,
        0.3, -0.3, 0.7, -0.7, 13.0, -13.0, 1e3, -1e3]


# The real NumPy backend, not a stand-in. An earlier version of this file used a
# hand-written shim, and because it was missing the `ops` table the check caught
# the AttributeError, returned no findings, and two tests passed for that reason
# instead of for the right one.
NUMPY = numpy_backend()


def with_divmod(fn):
    """The same backend with a deliberately broken divmod, to prove the check
    fires. Everything else stays real."""
    return replace(NUMPY, divmod=fn)


def _all_pairs(values, dtype):
    v = np.asarray(values, dtype=dtype)
    return np.repeat(v, v.size).astype(dtype), np.tile(v, v.size).astype(dtype)


class TestDivmodIdentity:
    """`q*b + r == a` is exact for integers and only near-exact for floats."""

    @pytest.mark.parametrize("name,dtype", FLOATS)
    def test_numpy_passes_in_every_float_precision(self, name, dtype):
        a, b = _all_pairs(HARD, dtype)
        with np.errstate(all="ignore"):
            found = check_divmod_identity(NUMPY, a, b, a, b, name)
        assert found == [], f"the check flags NumPy in {name}: {found}"

    @pytest.mark.parametrize("name,dtype", INTS)
    def test_numpy_passes_for_integers(self, name, dtype):
        vals = [-128, -13, -7, -3, -1, 1, 3, 7, 13, 127]
        a, b = _all_pairs(vals, dtype)
        with np.errstate(all="ignore"):
            found = check_divmod_identity(NUMPY, a, b, a, b, name)
        assert found == [], f"the check flags NumPy in {name}: {found}"

    def test_integers_are_still_judged_exactly(self):
        """The float tolerance must not leak into the integer path.

        Off by one in an integer quotient is a real bug, and it is smaller than
        one ULP of the reconstruction, so a tolerance would hide it.
        """
        a = np.array([7, -7], np.int32)
        b = np.array([2, 2], np.int32)

        def off_by_one(x, y):
            q, r = np.divmod(x, y)
            return q + 1, r

        with np.errstate(all="ignore"):
            found = check_divmod_identity(
                with_divmod(off_by_one), a, b, a, b, "int32"
            )
        assert "divmod-identity" in {f.check for f in found}

    def test_a_wrong_float_quotient_is_still_caught(self):
        """The tolerance is a few ULP, so a whole integer step still fails.

        This is the mlx case: `floor(x / y)` rounds up to 10 for `1.0 / 0.1` in
        float32 while the remainder is right, so reconstruction lands at 1.1.
        """
        a = np.array([1.0], np.float32)
        b = np.array([0.1], np.float32)

        def divides_twice(x, y):
            return np.floor(x / y), np.fmod(x, y)

        with np.errstate(all="ignore"):
            found = check_divmod_identity(
                with_divmod(divides_twice), a, b, a, b, "float32"
            )
        checks = {f.check for f in found}
        assert "divmod-identity" in checks
        # The same input also trips the oracle-free cross-check, which is fine.
        assert "divmod-vs-floor_divide" in checks


def two_devices(first, second, *, tags=(), arity=1):
    """A backend with two devices whose kernels differ.

    NumPy has only one implementation per op, so a second one has to be built:
    the op returns `first` on the reference device and `second` on the other,
    which is what a real second device is. `to_device` is the identity, so the
    values reaching both kernels are provably the same and any difference is the
    kernel's.
    """
    calls = []

    def fn(*args):
        calls.append(1)
        return (first if len(calls) == 1 else second)(*args)

    be = replace(NUMPY, devices=("here", "there"), to_device=lambda a, d: a)
    return be, Op(name="probe", arity=arity, fn=fn, numpy=None, tags=tags)


VALUES = np.array([1.0, 2.0, 3.0, 4.0])


class TestDeviceInvariance:
    """Two devices, and the bar for reporting a difference between them."""

    def test_agreeing_devices_report_nothing(self):
        be, op = two_devices(np.abs, np.abs)
        found, err = check_device_invariance(be, op, VALUES, "float64")
        assert (found, err) == ([], None)

    def test_a_backend_with_one_device_skips(self):
        """No second device means nothing to compare. It has to skip rather than
        compare the reference against itself and pass for free."""
        be, op = two_devices(np.abs, np.negative)
        be = replace(be, devices=("here",))
        assert check_device_invariance(be, op, VALUES, "float64") == ([], None)
        assert check_device_invariance(
            replace(be, to_device=None), op, VALUES, "float64"
        ) == ([], None)

    def test_an_exact_op_is_held_to_the_last_bit(self):
        """add is correctly rounded, so the two devices must agree exactly."""
        be, op = two_devices(
            np.abs, lambda a: np.nextafter(np.abs(a), np.inf), tags=("exact",)
        )
        found, _ = check_device_invariance(be, op, VALUES, "float64")
        assert len(found) == 1
        assert found[0].check == "device-invariance"
        assert "there differs from here" in found[0].detail

    def test_a_transcendental_gets_the_last_bit(self):
        """Nobody promises a Metal exp matches an Accelerate exp bit for bit, so
        reporting one ULP here would bury the real findings."""
        be, op = two_devices(np.exp, lambda a: np.nextafter(np.exp(a), np.inf))
        assert check_device_invariance(be, op, VALUES, "float64") == ([], None)

    def test_a_transcendental_does_not_get_a_nan(self):
        """The other half of the same rule: rounding cannot turn a number into a
        NaN, so that difference is a finding on any device."""
        be, op = two_devices(np.exp, lambda a: np.full(a.shape, np.nan))
        found, _ = check_device_invariance(be, op, VALUES, "float64")
        assert len(found) == 1
        assert "differs in kind" in found[0].detail

    def test_a_binary_op_moves_both_operands(self):
        be, op = two_devices(
            np.minimum, lambda a, b: np.maximum(a, b), tags=("exact",), arity=2
        )
        found, _ = check_device_invariance(
            be, op, VALUES, "float64", values_b=VALUES[::-1].copy()
        )
        assert len(found) == 1
        assert len(found[0].inputs) == 2

    def test_an_unsupported_dtype_is_not_a_finding(self):
        """float64 on MPS raises rather than returning a wrong answer. That is a
        documented limit, so it has to skip."""

        def unsupported(_):
            raise TypeError("not implemented for this device")

        be, op = two_devices(np.abs, unsupported)
        assert check_device_invariance(be, op, VALUES, "float64") == ([], None)


class TestAgainstNumpyItself:
    """Run the whole sweep against NumPy through the backend adapter.

    NumPy is what the oracle-based checks compare to, so it has to come back
    clean. Anything reported here is a bug in a check, and it covers the
    oracle-free checks too: NumPy has one code path per op, so any layout or
    length difference reported against it would be the harness inventing one.
    """

    def test_the_sweep_reports_nothing(self):
        from arraydiff.runner import run

        with np.errstate(all="ignore"):
            found = run(NUMPY, n_random=64)
        assert found == [], f"a check flags NumPy: {found[:3]}"
