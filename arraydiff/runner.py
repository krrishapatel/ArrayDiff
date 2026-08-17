"""Sweep driver."""

from __future__ import annotations

import numpy as np

from .checks import (
    _to_numpy,
    check_divmod_identity,
    check_layout_invariance,
    check_numpy_semantics,
    check_size_invariance,
    check_ulp,
    Finding,
)
from .spaces import LAYOUTS, float_values, int_values

# NumPy has no bfloat16, so its entry has no NumPy dtype and the checks that
# need an oracle skip it. Everything oracle-free still runs on it.
FLOAT_DTYPES = [
    ("float32", np.float32),
    ("float16", np.float16),
    ("float64", np.float64),
    ("bfloat16", None),
]

INT_DTYPES = [
    ("int8", np.int8),
    ("int16", np.int16),
    ("int32", np.int32),
    ("int64", np.int64),
    ("uint8", np.uint8),
    ("uint32", np.uint32),
]


def float_dtypes(be):
    """The float dtypes this backend actually has. A missing one is dropped
    rather than substituted: MLX has no float64 on Metal, and torch has no
    uint32, and testing a stand-in would prove nothing about either."""
    return [(n, be.dtype(n), nd) for n, nd in FLOAT_DTYPES if be.has(n)]


def int_dtypes(be):
    return [(n, be.dtype(n), nd) for n, nd in INT_DTYPES if be.has(n)]


def run(be, *, seed=0, only=None, n_random=256, verbose=False):
    """Run every check over every op/dtype/layout. Returns a Finding list."""
    rng = np.random.default_rng(seed)
    ops = be.ops
    findings: list[Finding] = []
    skipped: list[str] = []

    for dname, mdtype, ndtype in float_dtypes(be):
        # bfloat16 has no NumPy equivalent, so generate in float32 and cast.
        gen_dtype = ndtype or np.float32
        base = float_values(gen_dtype, rng=rng, n_random=n_random)

        for name, op in sorted(ops.items()):
            if only and name not in only:
                continue
            if "f" not in op.kinds:
                continue
            vals = base if op.domain is None else op.domain(base).astype(gen_dtype)

            # 1. layout invariance (no oracle needed)
            vals_b = (
                None
                if op.arity == 1
                else _second_operand(base, op, gen_dtype, rng)[: len(vals)]
            )
            got, err = check_layout_invariance(
                be, op, vals, dname, mdtype, values_b=vals_b
            )
            findings += got
            if err:
                skipped.append(f"{name}/{dname}: {err[:60]}")
                continue

            # 1b. size invariance (no oracle needed either)
            got, err = check_size_invariance(
                be, op, vals, dname, mdtype, values_b=vals_b
            )
            findings += got

            # 2. numpy semantics + 3. ULP, on the contiguous layout
            arr, arr_np = LAYOUTS["contiguous"](be, vals, mdtype)
            if op.arity == 1:
                if ndtype is not None:
                    findings += check_numpy_semantics(be, op, (arr,), (arr_np,), dname)
                    findings += check_ulp(be, op, arr, arr_np, dname, ndtype)
            else:
                b = _second_operand(base, op, gen_dtype, rng)
                b_arr, b_np = LAYOUTS["contiguous"](be, b, mdtype)
                if ndtype is not None:
                    findings += check_numpy_semantics(
                        be, op, (arr, b_arr), (arr_np, b_np), dname
                    )

        # 4. divmod invariants
        if not only or "divmod" in only:
            a = base
            b = _second_operand(base, None, gen_dtype, rng)
            a_arr = LAYOUTS["contiguous"](be, a, mdtype)[0]
            b_arr = LAYOUTS["contiguous"](be, b, mdtype)[0]
            # Read the operands back out, so the reference is what the kernel
            # actually holds rather than what was generated. bfloat16 keeps 8
            # mantissa bits, so casting moves the float32 values, and 1.4e-45
            # lands on exactly 0. Comparing against the pre-cast value reported
            # a correct divide by zero as a bug.
            findings += check_divmod_identity(
                be, a_arr, b_arr, _to_numpy(be, a_arr), _to_numpy(be, b_arr), dname
            )

    for dname, mdtype, ndtype in int_dtypes(be):
        base = int_values(ndtype, rng=rng, n_random=128)
        for name, op in sorted(ops.items()):
            if only and name not in only:
                continue
            if "i" not in op.kinds:
                continue
            b = None if op.arity == 1 else _nonzero_int(base, ndtype, rng)
            findings += check_size_invariance(
                be, op, base, dname, mdtype, values_b=b
            )[0]
            arr, arr_np = LAYOUTS["contiguous"](be, base, mdtype)
            if op.arity == 1:
                findings += check_numpy_semantics(be, op, (arr,), (arr_np,), dname)
            else:
                b_arr, b_np = LAYOUTS["contiguous"](be, b, mdtype)
                findings += check_numpy_semantics(
                    be, op, (arr, b_arr), (arr_np, b_np), dname
                )

    if verbose and skipped:
        print(f"skipped {len(skipped)} op/dtype pairs (unsupported)")
    return findings


def _second_operand(base, op, gen_dtype, rng):
    """A divisor/exponent array: a rotation of the inputs so signs mix, with
    zeros left in only where the op tolerates them."""
    b = np.roll(base, 3).astype(gen_dtype)
    if op is not None and op.domain_b is not None:
        b = op.domain_b(b).astype(gen_dtype)
    return b


def _nonzero_int(base, ndtype, rng):
    b = np.roll(base, 3).astype(ndtype)
    b[b == 0] = 1
    return b
