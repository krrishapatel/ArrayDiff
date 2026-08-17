"""Self-validation for the measurement machinery.

A differential tester is only worth as much as its own correctness. These tests
pin down the properties the measurements must have, using facts that are true
independently of any library under test: IEEE-754 requires sqrt and division to
be correctly rounded, so a working ULP measurement has to report them under half
a ULP in every precision.

The dtype-awareness test exists because the first version of ulp_error measured
every result against float64 spacing. That inflated float16 errors by about a
factor of 1e12 and made every op look broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from arraydiff.oracles import (
    EXACT,
    HAVE_MPMATH,
    STRICT_BUDGET,
    bitwise_equal,
    budget_for,
    ulp_error,
)

pytestmark = pytest.mark.skipif(not HAVE_MPMATH, reason="needs mpmath")

DTYPES = [np.float16, np.float32, np.float64]


@pytest.mark.parametrize("dtype", DTYPES)
def test_sqrt_is_correctly_rounded(dtype):
    """IEEE-754 requires sqrt to be correctly rounded, so error <= 0.5 ULP."""
    x = np.array([0.5, 1.0, 1.5, 2.0, 3.7, 100.0, 1e-3], dtype=dtype)
    got = np.sqrt(x)
    err = ulp_error(got, x, EXACT["sqrt"], dtype)
    assert err.max() <= 0.5, f"{dtype.__name__} sqrt measured at {err.max()} ULP"


@pytest.mark.parametrize("dtype", DTYPES)
def test_reciprocal_is_correctly_rounded(dtype):
    """Division is also required to be correctly rounded."""
    x = np.array([0.5, 1.0, 1.5, 3.0, 7.0, 100.0], dtype=dtype)
    got = (dtype(1.0) / x).astype(dtype)
    err = ulp_error(got, x, EXACT["reciprocal"], dtype)
    assert err.max() <= 0.5


def test_ulp_is_dtype_aware():
    """The same absolute error must be more ULP in float64 than in float16.

    One ULP near 1.0 is about 1e-3 in float16 and 2e-16 in float64, so an error
    of a fixed size is roughly 1e13 times more significant in float64. If this
    test fails, the measurement is ignoring the working precision.
    """
    x = np.array([1.0])
    nudged = np.array([1.0 + 1e-3])
    err16 = ulp_error(nudged.astype(np.float16), x, EXACT["sqrt"], np.float16)
    err64 = ulp_error(nudged, x, EXACT["sqrt"], np.float64)
    assert err16.max() < 10, "float16 error should be about 1 ULP"
    assert err64.max() > 1e11, "the same error is enormous in float64"


def test_exactly_correct_result_is_zero_ulp():
    x = np.array([4.0, 9.0, 16.0], dtype=np.float64)
    err = ulp_error(np.sqrt(x), x, EXACT["sqrt"], np.float64)
    assert err.max() == 0.0


def test_nonfinite_inputs_are_skipped():
    x = np.array([np.inf, -np.inf, np.nan, 4.0])
    err = ulp_error(np.sqrt(np.abs(x)), x, EXACT["sqrt"], np.float64)
    assert err.max() == 0.0


class TestBudgets:
    """The budget has to pass a known-good implementation.

    NumPy's transcendentals are the reference point for "nobody would call this
    broken". If the default budget flags them, the budget is wrong, and the tool
    will spend its findings on implementations that are working fine.
    """

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("name", ["exp", "log", "sin", "tan", "tanh", "expm1"])
    def test_numpy_passes_its_own_budget(self, dtype, name):
        # A well-conditioned range, so the condition-number scaling is not what
        # is doing the work here.
        x = np.array([0.1, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0], dtype=dtype)
        if name in ("log",):
            x = np.abs(x)
        got = getattr(np, name)(x).astype(dtype)
        err = ulp_error(got, x, EXACT[name], dtype)
        assert err.max() <= budget_for(name), f"{name} {dtype.__name__} {err.max()}"

    def test_correctly_rounded_ops_are_held_to_half_a_ulp(self):
        assert budget_for("sqrt") == STRICT_BUDGET
        assert budget_for("reciprocal") == STRICT_BUDGET

    def test_transcendentals_get_more_room(self):
        """No standard requires correct rounding for these."""
        for name in ("exp", "log", "sin", "tan", "erf", "rsqrt"):
            assert budget_for(name) > STRICT_BUDGET


class TestBitwiseEqual:
    def test_signed_zero_is_not_equal(self):
        """Plain == says 0.0 == -0.0, which hides signed-zero bugs."""
        assert np.float64(0.0) == np.float64(-0.0)
        assert not bitwise_equal(np.array([0.0]), np.array([-0.0]))

    def test_nan_is_equal_to_nan(self):
        """Plain == says NaN != NaN, which reports noise on every NaN input."""
        assert not (np.nan == np.nan)
        assert bitwise_equal(np.array([np.nan]), np.array([np.nan]))

    def test_ordinary_values(self):
        assert bitwise_equal(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert not bitwise_equal(np.array([1.0]), np.array([1.0000001]))

    def test_dtype_and_shape_mismatch(self):
        assert not bitwise_equal(np.array([1.0]), np.array([1.0], dtype=np.float32))
        assert not bitwise_equal(np.array([1.0]), np.array([1.0, 1.0]))

    def test_diff_mask_agrees_with_the_boolean_verdict(self):
        """The count and the verdict must come from one definition.

        An earlier version compared with bitwise_equal but counted with ==, so a
        signed-zero disagreement was reported as "0 elements differ" and pointed
        at element 0, which is not even the element that differed.
        """
        from arraydiff.oracles import bitwise_diff

        a = np.array([1.0, 0.0, np.nan, 3.0])
        b = np.array([1.0, -0.0, np.nan, 3.0])
        mask = bitwise_diff(a, b)
        assert mask.tolist() == [False, True, False, False]
        assert mask.any() == (not bitwise_equal(a, b))
        assert int(np.argmax(mask)) == 1

    def test_int64_low_bits_are_not_lost(self):
        """A float64 detour would equate these two."""
        from arraydiff.oracles import bitwise_diff

        a = np.array([2**62 + 1], dtype=np.int64)
        b = np.array([2**62 + 2], dtype=np.int64)
        assert bitwise_diff(a, b).all()

    def test_complex(self):
        assert bitwise_equal(np.array([1 + 2j]), np.array([1 + 2j]))
        assert not bitwise_equal(np.array([complex(0.0, 1.0)]),
                                 np.array([complex(-0.0, 1.0)]))


class TestCategoricalDiff:
    """The looser definition used on the device axis, where bit equality is not
    the contract but a difference in kind still is.

    Every case here is a subset of bitwise_diff: categorical_diff may never
    report something bitwise_diff does not.
    """

    def _only(self, a, b):
        from arraydiff.oracles import bitwise_diff, categorical_diff

        cat, bit = categorical_diff(np.array(a), np.array(b)), bitwise_diff(
            np.array(a), np.array(b)
        )
        assert not (cat & ~bit).any(), "categorical is not a subset of bitwise"
        return cat.tolist()

    def test_rounding_is_not_categorical(self):
        """The whole reason this function exists. Two devices are allowed to
        differ in the last bit of a transcendental, so a 1-ULP disagreement must
        not be reported."""
        a = np.array([0.8337300251311491, 1e-30, 700.0])
        b = np.array([0.8337300251311493, 1.0000001e-30, 700.0000000000001])
        assert self._only(a, b) == [False, False, False]

    def test_nan_against_a_number(self):
        assert self._only([np.nan, 1.0], [1.0, 1.0]) == [True, False]

    def test_infinity_against_a_finite_value(self):
        """A huge-but-finite result is not an infinity, however large it is."""
        assert self._only([np.inf], [1e308]) == [True]

    def test_infinity_sign(self):
        assert self._only([np.inf], [-np.inf]) == [True]

    def test_zero_sign(self):
        assert self._only([0.0, -0.0], [-0.0, -0.0]) == [True, False]

    def test_exact_zero_against_something_nonzero(self):
        """erf on arm64 returns exactly 0.0 for tiny inputs where the correct
        answer is a small nonzero number. Underflowing all the way to zero is a
        change in kind, not a rounding difference, however small the value."""
        assert self._only([0.0], [1.3264033e-38]) == [True]

    def test_integers_have_no_accuracy_axis(self):
        """Nothing about an integer result is approximate, so every difference
        counts. Rounding a float64 detour through int64 is the bug this guards."""
        a = np.array([2**62 + 1, 5], dtype=np.int64)
        b = np.array([2**62 + 2, 5], dtype=np.int64)
        assert self._only(a, b) == [True, False]
