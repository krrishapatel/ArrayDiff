"""Runs the real differential sweep, so the bugs the README names are checked
against real library builds rather than trusted.

What this pins is the *regression signal*, and only that. Each filed bug below
is asserted to still reproduce, so the day one is fixed upstream its known.py
entry stops matching and this turns red. That is the point of the job: it is how
the README learns a PR landed instead of the claims here quietly going stale.

What this deliberately does NOT assert is a zero-new-findings count. That count
is platform-dependent by design and cannot be pinned in a portable test. NumPy
runs its transcendental SIMD loops on contiguous data only, so on x86 a reversed
view takes the scalar loop and lands about 1 ULP away on `arctan`, `cosh`,
`sqrt` and friends, while on aarch64 it is bit-identical. torch's CPU `sqrt` is
likewise about 0.7 ULP off on x86 and exact on aarch64. Those are rounding, not
defects, and `tests/test_checks.py` already handles them with a measured ULP
tolerance rather than by suppressing them. Re-asserting "zero new" here would
just duplicate that check and break on whichever architecture the runner
happens to be.

The bugs pinned below are all categorical or sign errors, or a library
disagreeing with itself across layouts. None of those is a rounding difference
in any precision, so they reproduce on any architecture, which is what makes
them safe to assert in CI.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import Counter

import numpy as np
import pytest

from arraydiff import backends
from arraydiff.known import known_for, partition
from arraydiff.runner import run

pytestmark = pytest.mark.sweep


@pytest.fixture(autouse=True)
def _quiet_numpy():
    """The sweep feeds infinities and NaNs through the ops on purpose, so the
    RuntimeWarnings NumPy raises on overflow are expected, not a problem."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = np.seterr(all="ignore")
        try:
            yield
        finally:
            np.seterr(**old)


def _torch_cpu(torch):
    """Torch forced onto CPU, so the result is identical whether or not the
    machine running it has a GPU."""
    return dataclasses.replace(backends.torch_backend(torch), devices=("cpu",))


def _known_refs(be):
    """How many findings this backend's sweep attributes to each known ref."""
    _, known = partition(run(be), known_for(be.name))
    return Counter(k.ref for _, k in known), known


@pytest.mark.parametrize(
    "ref, what",
    [
        (
            "https://github.com/pytorch/pytorch/issues/193753",
            "fmod/remainder return NaN on the vectorized CPU path",
        ),
        (
            "https://github.com/pytorch/pytorch/issues/193754",
            "floor_divide depends on tensor length in half precision",
        ),
        (
            "https://github.com/pytorch/pytorch/issues/193755",
            "remainder gives a zero result the sign of the dividend",
        ),
        (
            "https://github.com/pytorch/pytorch/issues/193781",
            "minimum/maximum give a zero result a layout-dependent sign",
        ),
    ],
)
def test_named_torch_cpu_bugs_still_reproduce(ref, what):
    """The filed CPU bugs the README names have to still be observable. When one
    is fixed upstream its known.py entry stops matching and this turns red.

    All four are torch-vs-itself or a categorical/sign disagreement, so they do
    not depend on the ISA or on a GPU being present.
    """
    torch = pytest.importorskip("torch")
    refs, known = _known_refs(_torch_cpu(torch))
    assert known, "the torch sweep found nothing, so it is not actually running"
    assert refs[ref] > 0, (
        f"no finding is still attributed to {ref} ({what}); if torch fixed it, "
        "remove that entry from known.py and update the README"
    )


def test_jax_remainder_zero_sign_still_reproduces():
    """jax's filed bug: remainder gives a zero result the sign of the dividend.
    A sign error, so it is architecture-independent and safe to pin."""
    jax = pytest.importorskip("jax")
    refs, known = _known_refs(backends.jax_backend(jax))
    assert known, "the jax sweep found nothing, so it is not actually running"
    assert refs["https://github.com/jax-ml/jax/issues/40028"] > 0, (
        "jnp.remainder no longer gives a zero the wrong sign; jax may have fixed "
        "#40028, so update known.py and the README"
    )


@pytest.mark.parametrize(
    "ref, what",
    [
        (
            "recorded: tensorflow min/max give a zero the wrong sign",
            "minimum/maximum give a zero result the wrong IEEE sign",
        ),
        (
            "recorded: tensorflow remainder zero-sign",
            "remainder gives a zero result the sign of the dividend",
        ),
        (
            "recorded: tensorflow floor_divide at infinity",
            "1.0 // -inf is -0.0 rather than -1.0",
        ),
    ],
)
def test_recorded_tensorflow_bug_classes_still_reproduce(ref, what):
    """The three bug classes the README says recur in TensorFlow.

    Only these three are pinned. All are sign or categorical errors, so they hold
    on any ISA. The subnormal flush and the transcendental ULP gap are
    deliberately left out: flush-to-zero is a property of the CPU kernels and the
    ULP gap is rounding, so both could legitimately differ between x86 and
    aarch64, exactly like the numpy and torch cases in the module docstring.
    """
    tf = pytest.importorskip("tensorflow")
    refs, known = _known_refs(backends.tf_backend(tf))
    assert known, "the tensorflow sweep found nothing, so it is not actually running"
    assert refs[ref] > 0, (
        f"nothing is still attributed to {ref} ({what}); if TensorFlow fixed it, "
        "remove that entry from known.py and update the README"
    )


@pytest.mark.parametrize(
    "op, dtype",
    [(op, dt) for op in ("minimum", "maximum") for dt in ("float32", "float64")],
)
def test_torch_mps_zero_sign_bug_when_available(op, dtype):
    """The MPS half of the torch table only exists with a GPU, so it is checked
    only where one is present. minimum/maximum on (-0.0, 0.0) disagree between
    CPU and MPS: the README's device-invariance table. Skipped on x86 CI."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device on this machine")
    new, _ = partition(run(backends.torch_backend(torch)), known_for("torch"))
    hits = [f for f in new if f.check == "device-invariance" and f.op == op]
    assert hits, f"{op} no longer disagrees across devices; MPS may be fixed"
