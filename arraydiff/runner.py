"""Sweep driver."""

from __future__ import annotations

import numpy as np

from .checks import (
    _to_numpy,
    check_divmod_identity,
    check_layout_invariance,
    check_numpy_semantics,
    check_ulp,
    Finding,
)
from .ops import build_ops
from .spaces import LAYOUTS, float_values, int_values



def float_dtypes(mx):
    out = [("float32", mx.float32, np.float32), ("float16", mx.float16, np.float16)]
    if hasattr(mx, "float64"):
        out.append(("float64", mx.float64, np.float64))
    out.append(("bfloat16", mx.bfloat16, None))
    return out


def int_dtypes(mx):
    return [
        ("int8", mx.int8, np.int8),
        ("int16", mx.int16, np.int16),
        ("int32", mx.int32, np.int32),
        ("int64", mx.int64, np.int64),
        ("uint8", mx.uint8, np.uint8),
        ("uint32", mx.uint32, np.uint32),
    ]


def run(mx, *, seed=0, only=None, n_random=256, verbose=False):
    """Run every check over every op/dtype/layout. Returns a Finding list."""
    rng = np.random.default_rng(seed)
    ops = build_ops(mx)
    findings: list[Finding] = []
    skipped: list[str] = []

    for dname, mdtype, ndtype in float_dtypes(mx):
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
                mx, op, vals, dname, mdtype, values_b=vals_b
            )
            findings += got
            if err:
                skipped.append(f"{name}/{dname}: {err[:60]}")
                continue

            # 2. numpy semantics + 3. ULP, on the contiguous layout
            arr_mx, arr_np = LAYOUTS["contiguous"](mx, vals, mdtype)
            if op.arity == 1:
                if ndtype is not None:
                    findings += check_numpy_semantics(
                        mx, op, (arr_mx,), (arr_np,), dname
                    )
                    findings += check_ulp(mx, op, arr_mx, arr_np, dname, ndtype)
            else:
                b = _second_operand(base, op, gen_dtype, rng)
                b_mx, b_np = LAYOUTS["contiguous"](mx, b, mdtype)
                if ndtype is not None:
                    findings += check_numpy_semantics(
                        mx, op, (arr_mx, b_mx), (arr_np, b_np), dname
                    )

        # 4. divmod invariants
        if not only or "divmod" in only:
            a = base
            b = _second_operand(base, None, gen_dtype, rng)
            a_mx = LAYOUTS["contiguous"](mx, a, mdtype)[0]
            b_mx = LAYOUTS["contiguous"](mx, b, mdtype)[0]
            # Read the operands back out, so the reference is what the kernel
            # actually holds rather than what was generated. bfloat16 keeps 8
            # mantissa bits, so casting moves the float32 values, and 1.4e-45
            # lands on exactly 0. Comparing against the pre-cast value reported
            # a correct divide by zero as a bug.
            findings += check_divmod_identity(
                mx, a_mx, b_mx, _to_numpy(mx, a_mx), _to_numpy(mx, b_mx), dname
            )

    for dname, mdtype, ndtype in int_dtypes(mx):
        base = int_values(ndtype, rng=rng, n_random=128)
        for name, op in sorted(ops.items()):
            if only and name not in only:
                continue
            if "i" not in op.kinds:
                continue
            arr_mx, arr_np = LAYOUTS["contiguous"](mx, base)
            if op.arity == 1:
                findings += check_numpy_semantics(mx, op, (arr_mx,), (arr_np,), dname)
            else:
                b = _nonzero_int(base, ndtype, rng)
                b_mx, b_np = LAYOUTS["contiguous"](mx, b)
                findings += check_numpy_semantics(
                    mx, op, (arr_mx, b_mx), (arr_np, b_np), dname
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
