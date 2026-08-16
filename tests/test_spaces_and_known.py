"""Tests for the input spaces and the known-divergence suppression."""

from __future__ import annotations

import numpy as np
import pytest

from arraydiff.checks import Finding
from arraydiff.known import Known, partition
from arraydiff.spaces import LAYOUTS, SPECIAL_FLOATS, float_values, int_values


class FakeMX:
    """Stands in for mlx.core so layouts can be tested without a build."""

    @staticmethod
    def array(values, dtype=None):
        return np.array(values)

    @staticmethod
    def broadcast_to(a, shape):
        return np.broadcast_to(a, shape)


class TestLayouts:
    """Every layout must present the SAME logical values as the contiguous one.

    If a layout helper silently changes the values, then a layout-invariance
    finding is the harness's own bug rather than the library's, which is the
    worst possible failure for this tool.
    """

    @pytest.mark.parametrize("name", sorted(LAYOUTS))
    def test_values_are_preserved(self, name):
        values = np.array([1.0, -2.0, 3.5, -0.0, 7.25, 11.0, 13.5, 17.0])
        _, got = LAYOUTS[name](FakeMX, values, None)
        assert np.array_equal(np.asarray(got), values, equal_nan=True), name

    @pytest.mark.parametrize("name", sorted(LAYOUTS))
    def test_specials_are_preserved(self, name):
        values = np.array(SPECIAL_FLOATS, dtype=np.float64)
        _, got = LAYOUTS[name](FakeMX, values, None)
        assert np.array_equal(np.asarray(got), values, equal_nan=True), name

    def test_signed_zero_survives_the_round_trip(self):
        values = np.array([0.0, -0.0])
        for name in LAYOUTS:
            _, got = LAYOUTS[name](FakeMX, values, None)
            assert np.signbit(np.asarray(got)[1]), f"{name} lost the sign of -0.0"


class TestValueSpaces:
    def test_floats_include_the_hard_cases(self):
        vals = float_values(np.float32, rng=np.random.default_rng(0), n_random=0)
        assert np.isnan(vals).any(), "NaN must be covered"
        assert np.isinf(vals).any(), "infinities must be covered"
        assert (np.signbit(vals) & (vals == 0)).any(), "-0.0 must be covered"
        # subnormals are where flush-to-zero bugs live
        assert (np.abs(vals) < np.finfo(np.float32).smallest_normal).any()

    def test_ints_stay_in_range(self):
        for dtype in (np.int8, np.uint8, np.int32):
            vals = int_values(dtype, rng=np.random.default_rng(0))
            info = np.iinfo(dtype)
            assert vals.min() >= info.min and vals.max() <= info.max
            assert info.min in vals.tolist() and info.max in vals.tolist()


def _finding(check="numpy-semantics", op="sign", dtype="float32"):
    return Finding(check, op, dtype, "detail", (1.0,), "got", "want")


class TestKnownSuppression:
    def test_known_divergence_is_separated(self):
        new, seen = partition([_finding()])
        assert new == [] and len(seen) == 1
        assert "3644" in seen[0][1].ref

    def test_unknown_finding_survives(self):
        new, seen = partition([_finding(op="cos", check="ulp")])
        assert len(new) == 1 and seen == []

    def test_dtype_scoped_entry_only_matches_that_dtype(self):
        known = [Known("ulp", "cos", "reason", "ref", dtype="float16")]
        new, seen = partition([_finding(check="ulp", op="cos", dtype="float16")], known)
        assert len(seen) == 1
        new, seen = partition([_finding(check="ulp", op="cos", dtype="float64")], known)
        assert len(new) == 1

    def test_every_entry_carries_a_reason_and_a_link(self):
        """An unexplained suppression is how a real bug gets buried."""
        from arraydiff.known import MLX_KNOWN

        for k in MLX_KNOWN:
            assert len(k.reason) > 30, k
            assert k.ref.startswith("https://github.com/"), k
