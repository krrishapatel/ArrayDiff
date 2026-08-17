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
    categorical_diff,
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


def _to_numpy(be, a):
    """Kept as a shim so callers read the same as before; the conversion itself
    is the backend's, since bfloat16 has no NumPy equivalent to convert to."""
    return be.to_numpy(a)


def check_layout_invariance(
    be, op, values, dtype_name, mdtype=None, *, layouts=None, values_b=None
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
    # A backend that cannot express a layout is skipped for it rather than given
    # a copy to compare against itself.
    names = layouts or [n for n in LAYOUTS if be.layouts is None or n in be.layouts]
    if len(names) < 2:
        return [], None
    ref_name = names[0]

    def build(layout_name):
        arr, arr_np = LAYOUTS[layout_name](be, values, mdtype)
        if values_b is None:
            return (arr,), (arr_np,)
        b, b_np = LAYOUTS[layout_name](be, values_b, mdtype)
        return (arr, b), (arr_np, b_np)

    ref_arrs, ref_nps = build(ref_name)
    try:
        ref_out = _to_numpy(be, _apply(be, op, *ref_arrs))
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
            out = _to_numpy(be, _apply(be, op, *arrs))
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


# Lengths short enough to miss the vectorized kernel. 1 is the pure scalar path;
# 3 and 7 sit just under the common 4-wide float64 and 8-wide float32 registers.
SIZES = (1, 3, 7)


def check_size_invariance(be, op, values, dtype_name, mdtype=None, *, values_b=None):
    """The same values in a shorter array must give the same results.

    Array length carries no numerical information, so this is the same argument
    as layout invariance. It is a separate axis because length is what picks the
    kernel: a long array goes through the vectorized body, and a short one goes
    through the scalar path, which is often a different algorithm rather than the
    same one unrolled. Layout invariance cannot see this, since every layout it
    builds is full length.

    Each length is exercised by running the op over consecutive chunks of that
    length and stitching the results back together, so every value is covered at
    every length instead of only the ones that land in the first few slots.
    """
    findings = []
    ref_arr, _ = LAYOUTS["contiguous"](be, values, mdtype)
    arrs = [ref_arr]
    if values_b is not None:
        arrs.append(LAYOUTS["contiguous"](be, values_b, mdtype)[0])
    try:
        ref_out = _to_numpy(be, _apply(be, op, *arrs))
    except Exception as exc:
        return [], str(exc)

    n = len(values)
    for size in SIZES:
        if size >= n:
            continue
        try:
            pieces = [
                _to_numpy(be, _apply(be, op, *(a[i : i + size] for a in arrs)))
                for i in range(0, n, size)
            ]
        except Exception:
            continue
        out = np.concatenate([np.asarray(p).reshape(-1) for p in pieces])[:n]
        flat_ref = ref_out.reshape(-1)[:n]
        if bitwise_equal(out, flat_ref):
            continue
        bad = _first_diff(out, flat_ref)
        findings.append(
            Finding(
                check="size-invariance",
                op=op.name,
                dtype=dtype_name,
                detail=(
                    f"length {size} disagrees with length {n} on "
                    f"{_count_diff(out, flat_ref)}/{n} elements"
                ),
                inputs=tuple(
                    float(_to_numpy(be, a).reshape(-1)[bad]) for a in arrs
                ),
                got=f"length {size}={out[bad]!r}",
                want=f"length {n}={flat_ref[bad]!r}",
            )
        )
    return findings, None


def check_device_invariance(be, op, values, dtype_name, mdtype=None, *, values_b=None):
    """The same values on another device must give the same answer.

    The third axis that carries no numerical information, after layout and
    length. It is worth its own check because a second device is a second
    implementation: CPU and GPU kernels are written separately, often years
    apart, and the special-case handling is where they drift.

    The bar is not bit equality here, and that distinction is the whole check.
    Nobody promises a Metal transcendental matches Accelerate in the last bit,
    and torch documents as much, so demanding it would bury real findings under
    a wall of rounding. Ops tagged "exact" are held to bit equality, since
    IEEE 754 or the op's own definition pins their result on any device. For the
    rest only `categorical_diff` counts: a NaN, an infinity, or a zero sign
    cannot be a rounding difference no matter which device produced it.
    """
    if be.to_device is None or len(be.devices) < 2:
        return [], None
    ref_dev = be.devices[0]
    exact = "exact" in op.tags
    findings = []

    def build(device):
        arrs = [LAYOUTS["contiguous"](be, values, mdtype)[0]]
        if values_b is not None:
            arrs.append(LAYOUTS["contiguous"](be, values_b, mdtype)[0])
        return [be.to_device(a, device) for a in arrs]

    try:
        ref_arrs = build(ref_dev)
        ref_out = _to_numpy(be, _apply(be, op, *ref_arrs))
    except Exception as exc:
        return [], str(exc)

    for device in be.devices[1:]:
        try:
            arrs = build(device)
            out = _to_numpy(be, _apply(be, op, *arrs))
        except Exception:
            # The dtype or the op is missing on this device. float64 on MPS is
            # the common one, and it is a documented limit rather than a finding.
            continue
        if out.shape != ref_out.shape:
            continue
        diff = bitwise_diff(out, ref_out) if exact else categorical_diff(out, ref_out)
        if not diff.any():
            continue
        bad = int(np.argmax(diff.reshape(-1)))
        kind = "differs" if exact else "differs in kind"
        findings.append(
            Finding(
                check="device-invariance",
                op=op.name,
                dtype=dtype_name,
                detail=(
                    f"{device} {kind} from {ref_dev} on "
                    f"{int(diff.sum())}/{out.size} elements"
                ),
                inputs=tuple(
                    float(_to_numpy(be, a).reshape(-1)[bad]) for a in ref_arrs
                ),
                got=f"{device}={out.reshape(-1)[bad]!r}",
                want=f"{ref_dev}={ref_out.reshape(-1)[bad]!r}",
            )
        )
    return findings, None


def check_numpy_semantics(be, op, arrays_lib, arrays_np, dtype_name):
    """Bit-exact agreement with NumPy, for ops where both should be exact.

    Only applied to ops tagged "exact": add, multiply, divide, comparisons,
    floor_divide, remainder and friends are all correctly rounded or integral,
    so any difference is a real semantic disagreement rather than a rounding
    choice. Transcendentals are judged by ULP instead.
    """
    if op.numpy is None or "exact" not in op.tags:
        return []
    try:
        got = _to_numpy(be, _apply(be, op, *arrays_lib))
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


def check_ulp(be, op, arr_lib, arr_np, dtype_name, ndtype=None, *, budget=None):
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
        got = _to_numpy(be, _apply(be, op, arr_lib))
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


# In floating point `q*b + r == a` does not hold exactly, and demanding that it
# does is a bug in the test: the floored remainder is `fmod(a, b) + b`, which
# rounds. NumPy misses exact reconstruction on 54 of 400 float32 pairs.
#
# The tolerance is measured at the scale of `q*b`, not of `a`. Forming `q*b + r`
# can cancel almost everything, so the error belongs to the larger intermediate:
# for `divmod(0.1, -2.5)` NumPy returns q=-1, r=-2.4 and reconstructs 2.5 - 2.4,
# which is 13 ULP away from 0.1 but well under 1 ULP of 2.5. Scaling by |a| would
# flag NumPy; scaling by the larger term does not, and still catches a quotient
# that is off by a whole step, which is the mlx bug this was written for.
DIVMOD_RECON_ULP = 4.0


def check_divmod_identity(be, a_lib, b_lib, a_np, b_np, dtype_name):
    """divmod must satisfy q*b + r == a, and agree with floor_divide/remainder.

    This is the invariant that has to hold no matter which division convention
    the library picks, so it is a fair check even without an oracle.
    """
    findings = []
    if be.divmod is None:
        return findings
    try:
        q, r = be.divmod(a_lib, b_lib)
        fd = be.ops["floor_divide"].fn(a_lib, b_lib)
        rem = be.ops["remainder"].fn(a_lib, b_lib)
        be.evaluate(q, r, fd, rem)
    except Exception:
        return findings
    q_n, r_n = _to_numpy(be, q), _to_numpy(be, r)
    finite = np.isfinite(a_np) & np.isfinite(b_np) & (b_np != 0)

    if a_np.dtype.kind in "iub":
        # Reconstruct with the dtype's own wraparound. A fixed width integer only
        # satisfies the identity modulo 2**n, and int8 -128 // -1 is the case that
        # proves it: the true quotient 128 is not representable, so NumPy returns
        # -128. Reconstructing in float64 would call that a bug.
        with np.errstate(over="ignore"):
            recon = (q_n * b_np + r_n).astype(a_np.dtype)
        close = recon == a_np
    else:
        a64, b64 = a_np.astype(np.float64), b_np.astype(np.float64)
        # A quotient too large for the dtype has to come back infinite, and then
        # nothing can reconstruct `a`. NumPy fails the identity here too, on
        # divmod(65504, -6e-08) in float16 for one, so judging it would be
        # judging the format. The true quotient is taken in float64 rather than
        # from the library, so a library that overflows early is still caught.
        with np.errstate(all="ignore"):
            true_q = np.floor(a64 / b64)
        finite = finite & np.isfinite(true_q)
        finite = finite & (np.abs(true_q) <= float(np.finfo(a_np.dtype).max))

        prod = q_n.astype(np.float64) * b64
        recon = prod + r_n.astype(np.float64)
        scale = np.maximum(np.abs(prod), np.abs(a64))
        close = np.abs(recon - a64) <= (
            DIVMOD_RECON_ULP * _spacing_in(scale.astype(a_np.dtype), dtype_name)
        )
    ok = ~finite | close
    if not np.all(ok):
        bad = int(np.argmin(ok))
        findings.append(
            Finding("divmod-identity", "divmod", dtype_name,
                    f"q*b + r != a on {int((~ok).sum())}/{ok.size} elements",
                    (_at(a_np, bad), _at(b_np, bad)),
                    f"q={q_n.reshape(-1)[bad]!r} r={r_n.reshape(-1)[bad]!r} "
                    f"-> {recon.reshape(-1)[bad]!r}",
                    repr(a_np.reshape(-1)[bad])))

    fd_n = _to_numpy(be, fd)
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


def _apply(be, op, *arrays):
    out = op.fn(*arrays)
    be.evaluate(out)
    return out


def _at(a, i):
    return a.reshape(-1)[i].item()


# bfloat16 keeps 8 mantissa bits against float32's 24, and _to_numpy widens it to
# float32, so float32 spacing understates a bfloat16 ULP by this much.
_BF16_ULP_RATIO = 2**16


def _spacing_in(a_np, dtype_name=None):
    """ULP width at each element, in the precision the op actually ran in.

    Same reason as `ulp_error`: float64 spacing would understate a float16 ULP by
    about 1e12 and turn a passing implementation into a wall of findings.
    """
    a = np.abs(np.asarray(a_np))
    if a.dtype.kind != "f":
        return np.ones(a.shape, dtype=np.float64)
    step = np.spacing(a).astype(np.float64)
    if dtype_name == "bfloat16":
        step = step * _BF16_ULP_RATIO
    return np.where(step > 0, step, float(np.finfo(a.dtype).tiny))


def _first_diff(a, b):
    diff = bitwise_diff(a, b).reshape(-1)
    return int(np.argmax(diff)) if diff.any() else 0


def _count_diff(a, b):
    return int(bitwise_diff(a, b).sum())
