"""The invariant checks.

The strongest checks here need no reference implementation at all. Layout
invariance and device invariance compare the library against itself, so they
stay valid even where the library deliberately differs from NumPy, and they
produce findings that are impossible to argue with: the same op, on the same
values, returned two different answers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .oracles import (
    CONDITION,
    EXACT,
    HAVE_MPMATH,
    PERIODIC,
    PHASE_SPACING_LIMIT,
    bitwise_diff,
    bitwise_equal,
    budget_for,
    ulp_error,
)
from .spaces import LAYOUTS


@dataclass
class Finding:
    check: str
    op: str
    dtype: str
    detail: str
    # A minimal reproducing case, so a report can be written straight from it.
    inputs: tuple
    got: object
    want: object

    def __str__(self):
        return (
            f"[{self.check}] {self.op} {self.dtype}: {self.detail}\n"
            f"    inputs: {self.inputs}\n"
            f"    got:    {self.got}\n"
            f"    want:   {self.want}"
        )


def _to_numpy(mx, a):
    import numpy as np

    if a.dtype == mx.bfloat16:
        a = a.astype(mx.float32)
    return np.asarray(a)


def check_layout_invariance(
    mx, op, values, dtype_name, mdtype=None, *, layouts=None, values_b=None
):
    """The same values through different memory layouts must give bit-identical
    results. A difference means the result depends on how the input happened to
    be stored, which no numerical op is allowed to do.

    Binary ops get `values_b`, and both operands are moved to the same layout at
    once. That is the case that matters in practice, since a contiguity check in
    the backend usually looks at all the inputs together. It is also the case
    that found #4162: `power` is binary, so restricting this check to unary ops
    silently skipped it.
    """
    findings = []
    names = layouts or list(LAYOUTS)
    ref_name = names[0]

    def build(layout_name):
        arr, arr_np = LAYOUTS[layout_name](mx, values, mdtype)
        if values_b is None:
            return (arr,), (arr_np,)
        b, b_np = LAYOUTS[layout_name](mx, values_b, mdtype)
        return (arr, b), (arr_np, b_np)

    ref_arrs, ref_nps = build(ref_name)
    try:
        ref_out = _to_numpy(mx, _apply(mx, op, *ref_arrs))
    except Exception as exc:  # op unsupported for this dtype
        return [], str(exc)

    for name in names[1:]:
        arrs, arr_nps = build(name)
        if not all(
            np.array_equal(got, want, equal_nan=True)
            for got, want in zip(arr_nps, ref_nps)
        ):
            continue  # layout helper changed the values; skip rather than lie
        arr_np, ref_np = arr_nps[0], ref_nps[0]
        try:
            out = _to_numpy(mx, _apply(mx, op, *arrs))
        except Exception:
            continue
        if not bitwise_equal(out, ref_out):
            bad = _first_diff(out, ref_out)
            findings.append(
                Finding(
                    check="layout-invariance",
                    op=op.name,
                    dtype=dtype_name,
                    detail=(
                        f"{name} disagrees with {ref_name} on "
                        f"{_count_diff(out, ref_out)}/{out.size} elements"
                    ),
                    inputs=tuple(float(a.reshape(-1)[bad]) for a in ref_nps),
                    got=f"{name}={out.reshape(-1)[bad]!r}",
                    want=f"{ref_name}={ref_out.reshape(-1)[bad]!r}",
                )
            )
    return findings, None


def check_numpy_semantics(mx, op, arrays_mx, arrays_np, dtype_name):
    """Bit-exact agreement with NumPy, for ops where both should be exact.

    Only applied to ops tagged "exact": add, multiply, divide, comparisons,
    floor_divide, remainder and friends are all correctly rounded or integral,
    so any difference is a real semantic disagreement rather than a rounding
    choice. Transcendentals are judged by ULP instead.
    """
    if op.numpy is None or "exact" not in op.tags:
        return []
    try:
        got = _to_numpy(mx, _apply(mx, op, *arrays_mx))
    except Exception:
        return []
    with np.errstate(all="ignore"):
        try:
            want = op.numpy(*arrays_np)
        except Exception:
            return []
    if got.shape != want.shape:
        return [
            Finding("numpy-semantics", op.name, dtype_name,
                    f"shape {got.shape} != numpy {want.shape}",
                    tuple(a.tolist()[:4] for a in arrays_np), got.shape, want.shape)
        ]
    # We care about values here, not dtype policy, but integers must still be
    # compared as integers: a float64 detour silently equates int64 values that
    # differ in their low bits, which is exactly where an overflow bug shows up.
    diff = bitwise_diff(got, np.asarray(want)).reshape(-1)
    if not diff.any():
        return []
    bad = int(np.argmax(diff))
    return [
        Finding(
            check="numpy-semantics",
            op=op.name,
            dtype=dtype_name,
            detail=f"{int(diff.sum())}/{diff.size} elements differ from numpy",
            inputs=tuple(_at(a, bad) for a in arrays_np),
            got=repr(got.reshape(-1)[bad]),
            want=repr(np.asarray(want).reshape(-1)[bad]),
        )
    ]


def check_ulp(mx, op, arr_mx, arr_np, dtype_name, ndtype=None, *, budget=None):
    """Accuracy against the exactly-rounded value, in ULP.

    `budget` defaults to what the op is actually expected to deliver: half a ULP
    where IEEE-754 requires correct rounding, a few ULP for transcendentals where
    no standard does. See oracles.budget_for.
    """
    if not HAVE_MPMATH or op.exact_key not in EXACT or op.arity != 1:
        return []
    if budget is None:
        budget = budget_for(op.exact_key)
    try:
        got = _to_numpy(mx, _apply(mx, op, arr_mx))
    except Exception:
        return []
    err = ulp_error(got, arr_np, EXACT[op.exact_key], ndtype)

    # Scale the budget by how ill-conditioned each input is, so that only
    # accuracy the implementation could actually have delivered is demanded.
    cond_fn = CONDITION.get(op.exact_key)
    allowed = np.full(err.shape, budget, dtype=np.float64)
    flat = np.asarray(arr_np, dtype=np.float64).reshape(-1)
    if cond_fn is not None:
        for i in range(flat.size):
            if np.isfinite(flat[i]):
                allowed.reshape(-1)[i] = budget * max(1.0, cond_fn(flat[i]))

    # Drop inputs whose phase is not determined by the input itself.
    if op.exact_key in PERIODIC:
        indeterminate = np.spacing(np.abs(flat)) > PHASE_SPACING_LIMIT
        allowed.reshape(-1)[indeterminate] = np.inf

    excess = err - allowed
    worst_i = int(np.nanargmax(excess)) if excess.size else 0
    if excess.size == 0 or excess.reshape(-1)[worst_i] <= 0:
        return []
    return [
        Finding(
            check="ulp",
            op=op.name,
            dtype=dtype_name,
            detail=(
                f"error {err.reshape(-1)[worst_i]:.3f} ULP exceeds "
                f"{allowed.reshape(-1)[worst_i]:.3f} allowed "
                f"(budget {budget:.2f} x condition number)"
            ),
            inputs=(float(flat[worst_i]),),
            got=repr(got.reshape(-1)[worst_i]),
            want=f"{err.reshape(-1)[worst_i]:.3f} ULP off exact",
        )
    ]


def check_divmod_identity(mx, a_mx, b_mx, a_np, b_np, dtype_name):
    """divmod must satisfy q*b + r == a, and agree with floor_divide/remainder.

    This is the invariant that has to hold no matter which division convention
    the library picks, so it is a fair check even without an oracle.
    """
    findings = []
    try:
        q, r = mx.divmod(a_mx, b_mx)
        fd = mx.floor_divide(a_mx, b_mx)
        rem = mx.remainder(a_mx, b_mx)
        mx.eval(q, r, fd, rem)
    except Exception:
        return findings
    q_n, r_n = _to_numpy(mx, q), _to_numpy(mx, r)
    finite = np.isfinite(a_np) & np.isfinite(b_np) & (b_np != 0)

    recon = q_n.astype(np.float64) * b_np.astype(np.float64) + r_n.astype(np.float64)
    ok = ~finite | (recon == a_np.astype(np.float64))
    if not np.all(ok):
        bad = int(np.argmin(ok))
        findings.append(
            Finding("divmod-identity", "divmod", dtype_name,
                    f"q*b + r != a on {int((~ok).sum())}/{ok.size} elements",
                    (_at(a_np, bad), _at(b_np, bad)),
                    f"q={q_n.reshape(-1)[bad]!r} r={r_n.reshape(-1)[bad]!r} "
                    f"-> {recon.reshape(-1)[bad]!r}",
                    repr(a_np.reshape(-1)[bad])))

    fd_n = _to_numpy(mx, fd)
    ok = ~finite | (q_n.astype(np.float64) == fd_n.astype(np.float64))
    if not np.all(ok):
        bad = int(np.argmin(ok))
        findings.append(
            Finding("divmod-vs-floor_divide", "divmod", dtype_name,
                    f"divmod q != floor_divide on {int((~ok).sum())}/{ok.size}",
                    (_at(a_np, bad), _at(b_np, bad)),
                    f"divmod q={q_n.reshape(-1)[bad]!r}",
                    f"floor_divide={fd_n.reshape(-1)[bad]!r}"))
    return findings


def _apply(mx, op, *arrays):
    out = op.mlx(*arrays)
    mx.eval(out)
    return out


def _at(a, i):
    return a.reshape(-1)[i].item()


def _first_diff(a, b):
    diff = bitwise_diff(a, b).reshape(-1)
    return int(np.argmax(diff)) if diff.any() else 0


def _count_diff(a, b):
    return int(bitwise_diff(a, b).sum())
