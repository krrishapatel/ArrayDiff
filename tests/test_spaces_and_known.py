"""Tests for the input spaces and the known-divergence suppression."""

from __future__ import annotations

import numpy as np
import pytest

from arraydiff import known
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

    def test_every_special_pairing_is_actually_tried(self):
        """The regression guard for a real coverage hole.

        The second operand used to be a rotation of the first, so only the
        pairings that a shift of 3 happens to produce were ever tested. `0.0`
        against `-0.0` was not one of them, which is why the sweep did not see
        torch's #193781 until this existed.
        """
        from arraydiff.spaces import SPECIAL_FLOATS, special_pairs

        a, b = special_pairs(np.float32)
        seen = {
            (float(x), bool(np.signbit(x)), float(y), bool(np.signbit(y)))
            for x, y in zip(a, b)
            if not (np.isnan(x) or np.isnan(y))
        }
        assert (0.0, False, 0.0, True) in seen, "0.0 against -0.0 must be tried"
        assert (0.0, True, 0.0, False) in seen, "and in the other order"
        assert len(a) == len(b) == len(SPECIAL_FLOATS) ** 2

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

    def test_a_predicate_narrows_an_entry_to_part_of_the_input_range(self):
        """`when` exists so that suppressing a documented underflow in exp does
        not also suppress every other accuracy bug in exp."""
        known = [
            Known("ulp", "exp", "reason", "ref", when=lambda f: f.inputs[0] < -708.0)
        ]
        new, seen = partition([_finding(check="ulp", op="exp")], known)
        assert len(new) == 1, "an in-range input is not covered by this entry"
        deep = Finding("ulp", "exp", "float64", "d", (-720.0,), "got", "want")
        new, seen = partition([deep], known)
        assert len(seen) == 1

    def test_the_real_exp_underflow_predicate_is_dtype_aware(self):
        """float32 subnormals start about 1e-38 and float64 about 1e-308, so the
        same input is a documented flush in one precision and a real bug in the
        other."""
        from arraydiff.known import _exp_underflows_to_subnormal as under

        assert under(Finding("ulp", "exp", "float32", "d", (-95.0,), "g", "w"))
        assert not under(Finding("ulp", "exp", "float64", "d", (-95.0,), "g", "w"))
        assert under(Finding("ulp", "exp", "float64", "d", (-715.0,), "g", "w"))
        # Below the smallest subnormal, returning 0.0 is correct rather than a
        # flush, so the entry must not cover it.
        assert not under(Finding("ulp", "exp", "float64", "d", (-800.0,), "g", "w"))

    def test_numpy_is_not_a_sign_aware_oracle_in_float16(self):
        """The fact the min/max oracle entry rests on.

        NumPy's float16 min/max return whichever argument came first for a pair
        of zeros, so a library that is right by IEEE 754 disagrees with NumPy
        there, and `numpy-semantics` has to blame the oracle rather than the
        library. float16 is the case that holds on every machine: it has no
        ISA-specific override, unlike float32 and float64, so it always takes
        the sign-blind C path. If NumPy ever fixes it, this fails and the entry
        should be deleted rather than kept as a permanent excuse.
        """
        for op in (np.minimum, np.maximum):
            first = op(np.float16(0.0), np.float16(-0.0))
            second = op(np.float16(-0.0), np.float16(0.0))
            assert np.signbit(first) != np.signbit(second), (
                f"np.{op.__name__} in float16 is order independent now, so the "
                "float16 known entry is obsolete"
            )
        assert "float16" in known.NUMPY_SIGN_BLIND_FLOATS

    def test_the_oracle_split_is_probed_not_assumed(self):
        """The regression guard for a genuine portability bug in this file.

        The sign-aware dtypes used to be a hardcoded ("float32", "float64",
        "bfloat16"). That is true on aarch64, where NumPy's wide loops use
        FMIN/FMAX, and false on x86, where they use MINSS/MAXSS and return the
        second operand. CI on x86 Linux failed on it. So the split has to be
        measured on the host, and the two halves have to stay complementary.
        """
        aware = set(known.NUMPY_SIGN_AWARE_FLOATS)
        blind = set(known.NUMPY_SIGN_BLIND_FLOATS)
        assert aware.isdisjoint(blind)
        assert aware | blind == set(known.FLOATS)
        for dtype in known.FLOATS:
            probe = np.float32 if dtype == "bfloat16" else getattr(np, dtype)
            order_independent = np.signbit(
                np.minimum(probe(0.0), probe(-0.0))
            ) == np.signbit(np.minimum(probe(-0.0), probe(0.0)))
            assert (dtype in aware) == bool(order_independent), (
                f"{dtype} is on the wrong side of the split"
            )

    def test_the_float16_entry_does_not_swallow_the_wider_dtypes(self):
        """Narrowing the oracle entry must not also narrow the real bug.

        #193781 is a genuine library bug in every float dtype on the oracle-free
        axes. Only `numpy-semantics` in float16 is the oracle's fault.
        """
        from arraydiff.known import known_for

        entries = known_for("torch")
        for check in ("layout-invariance", "size-invariance"):
            f = Finding(check, "minimum", "float16", "d", (0.0, -0.0), "g", "w")
            new, seen = partition([f], entries)
            assert len(seen) == 1, f"{check} float16 must still be attributed"
            assert "193781" in seen[0][1].ref

    def test_every_entry_carries_a_reason_and_a_link(self):
        """An unexplained suppression is how a real bug gets buried."""
        from arraydiff.known import KNOWN

        for name, entries in KNOWN.items():
            for k in entries:
                assert len(k.reason) > 30, (name, k)
                assert k.ref.startswith("https://"), (name, k)
