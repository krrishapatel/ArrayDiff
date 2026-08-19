"""Runs the real differential sweep, so the README's counts are tested rather
than trusted.

The other test modules check the checks against synthetic cases. This runs the
tool the way the README reports it, against whatever real library builds are
installed, and pins the two things that have to stay true:

  - the NumPy oracle never contradicts itself. It is the reference for
    ``numpy-semantics``, so a single finding against it means a check is wrong,
    not that NumPy is. This holds on any machine and needs nothing installed.

  - every finding torch and jax produce is already accounted for in ``known.py``.
    A *new* finding here is either a real regression in the library or a stale
    ``known.py`` entry, and both are things to look at. The specific bugs the
    README names are pinned individually, so the day one is fixed upstream the
    matching entry stops firing and this goes red, rather than the README
    quietly going out of date.

``device-invariance`` needs a second device, so the MPS half of the torch table
only exists when MPS is present. On CI (x86, no MPS) it skips and the CPU
assertions carry the load; the tool skips the axis on its own, so nothing here
has to know which machine it is on. The torch sweep is forced onto CPU only, so
the count is identical whether or not the machine running it happens to have a
GPU.
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


@pytest.fixture(autouse=True)
def _quiet_numpy():
    """The sweep deliberately feeds infinities and NaNs through the ops, so the
    RuntimeWarnings NumPy raises on overflow are expected, not a problem."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = np.seterr(all="ignore")
        try:
            yield
        finally:
            np.seterr(**old)


def _sweep(be):
    return partition(run(be), known_for(be.name))


def _refs(known):
    return Counter(k.ref for _, k in known)


def test_numpy_oracle_never_contradicts_itself():
    """NumPy is the oracle, so nothing at all should be reported against it. A
    finding here means a check is encoding a bug rather than finding one."""
    new, known = _sweep(backends.numpy_backend())
    assert new == [], (
        "the NumPy backend produced findings, which means a check disagrees "
        f"with the oracle it uses: {[str(f) for f in new[:3]]}"
    )
    assert known == [], "nothing should need suppressing on the oracle backend"


def test_torch_cpu_findings_are_all_accounted_for():
    """Every divergence torch shows on CPU is in known.py. A new one is a real
    regression or a stale entry."""
    torch = pytest.importorskip("torch")
    be = dataclasses.replace(backends.torch_backend(torch), devices=("cpu",))
    new, known = _sweep(be)
    assert new == [], (
        "unaccounted torch CPU findings; either torch regressed or known.py is "
        f"stale: {[str(f) for f in new[:5]]}"
    )
    assert known, "the torch sweep found nothing, so it is not actually running"


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
    is fixed upstream its known.py entry stops matching and this turns red, which
    is the point: it is how the README learns a PR landed.

    All four are torch-vs-itself or torch-vs-a-trusted-oracle on CPU, so they do
    not depend on the ISA or on a GPU being present.
    """
    torch = pytest.importorskip("torch")
    be = dataclasses.replace(backends.torch_backend(torch), devices=("cpu",))
    _, known = _sweep(be)
    assert _refs(known)[ref] > 0, (
        f"no finding is still attributed to {ref} ({what}); if torch fixed it, "
        "remove that entry from known.py and update the README"
    )


def test_jax_sweep_is_clean():
    """The README's headline for jax is zero new findings. Pin it."""
    jax = pytest.importorskip("jax")
    new, known = _sweep(backends.jax_backend(jax))
    assert new == [], (
        f"jax produced new findings the README does not claim: "
        f"{[str(f) for f in new[:5]]}"
    )
    assert known, "the jax sweep found nothing, so it is not actually running"


@pytest.mark.parametrize(
    "op, dtype",
    [(op, dt) for op in ("minimum", "maximum") for dt in ("float32", "float64")],
)
def test_torch_mps_zero_sign_bug_when_available(op, dtype):
    """The MPS half of the torch table only exists with a GPU, so it is checked
    only where one is present. minimum/maximum on (-0.0, 0.0) disagree between
    CPU and MPS: the README's device-invariance table. Skipped on CI."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device on this machine")
    be = backends.torch_backend(torch)
    new, _ = _sweep(be)
    hits = [f for f in new if f.check == "device-invariance" and f.op == op]
    assert hits, f"{op} no longer disagrees across devices; MPS may be fixed"
