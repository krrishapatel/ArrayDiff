"""Divergences that are already known, intentional, or filed.

This list is the most important file in the project. A differential tester that
re-reports settled decisions is worse than useless: it burns maintainer goodwill
and buries the findings that are actually new. Every entry needs a reason and a
link, so that "we already talked about this" is checkable rather than folklore.

Add an entry when a maintainer has ruled on the behaviour, or when an issue or
PR already covers it. Remove one when upstream changes its mind.
"""

from __future__ import annotations

from dataclasses import dataclass


FLOATS = ("float16", "float32", "float64", "bfloat16")
INTS = ("int8", "int16", "int32", "int64", "uint8", "uint32")


@dataclass(frozen=True)
class Known:
    check: str
    # One op name, or a tuple of them when a single root cause covers a family.
    op: str | tuple[str, ...]
    reason: str
    ref: str
    # None means "every dtype". A tuple restricts to those dtypes, which matters
    # when the same check fires on ints and floats for unrelated reasons.
    dtype: str | tuple[str, ...] | None = None

    def matches(self, finding) -> bool:
        ops = (self.op,) if isinstance(self.op, str) else self.op
        dtypes = (self.dtype,) if isinstance(self.dtype, str) else self.dtype
        return (
            self.check == finding.check
            and finding.op in ops
            and (dtypes is None or finding.dtype in dtypes)
        )


MLX_KNOWN = [
    Known(
        check="numpy-semantics",
        op="sign",
        reason=(
            "sign(nan) returns 0 by design. torch.sign does the same, and "
            "propagating NaN costs extra instructions on the hot path, so the "
            "maintainers declined to match NumPy here."
        ),
        ref="https://github.com/ml-explore/mlx/issues/3644",
    ),
    Known(
        check="numpy-semantics",
        op="floor_divide",
        dtype=INTS,
        reason=(
            "Integer floor_divide truncates toward zero instead of flooring. "
            "Filed as #4119. PR #4108 is approved but its fix subtracts the "
            "remainder first, which overflows the small dtypes, so int8 and "
            "int16 come out with the wrong sign. Raised on the #4108 thread."
        ),
        ref="https://github.com/ml-explore/mlx/issues/4119",
    ),
    Known(
        check="numpy-semantics",
        op="floor_divide",
        dtype=FLOATS,
        reason=(
            "The float path is floor(a / b), which is not what // means at "
            "infinity: inf // -3.0 gives -inf where NumPy and Python both give "
            "nan, and 1.0 // -inf gives -0.0 where both give -1.0. Filed as "
            "#4317."
        ),
        ref="https://github.com/ml-explore/mlx/issues/4317",
    ),
    Known(
        check="divmod-identity",
        op="divmod",
        dtype=FLOATS,
        reason=(
            "The quotient is floor(x / y), a second division, so it can round "
            "up past its own floor and then q*b + r is off by a whole divisor: "
            "in bfloat16 2144 / 358 is 5.9888, which rounds to exactly 6.0. "
            "PR #4003 fixed the half-precision half of this by widening the "
            "division and was closed, on the grounds that it adds overhead and "
            "that PyTorch behaves the same way. Recorded as ruled-on rather "
            "than new. Re-raised on the #4108 thread, because PyTorch actually "
            "derives the quotient from the remainder rather than dividing "
            "twice, and because float64 breaks the same way with no wider type "
            "to escape to."
        ),
        ref="https://github.com/ml-explore/mlx/pull/4003",
    ),
    Known(
        check="divmod-vs-floor_divide",
        op="divmod",
        reason=(
            "divmod truncates the quotient while floor_divide floors, so the "
            "two disagree. Filed as #4119 with PR #4108 approved."
        ),
        ref="https://github.com/ml-explore/mlx/issues/4119",
    ),
    Known(
        check="divmod-identity",
        op="divmod",
        dtype=INTS,
        reason=(
            "For integers q*b + r != a follows from the same truncation as "
            "#4119; the remainder is not adjusted to match the quotient."
        ),
        ref="https://github.com/ml-explore/mlx/issues/4119",
    ),
    Known(
        check="layout-invariance",
        op="power",
        reason=(
            "Contiguous inputs use Accelerate's vectorized pow while strided "
            "inputs fall back to scalar std::pow, so results depend on layout."
        ),
        ref="https://github.com/ml-explore/mlx/issues/4162",
    ),
    Known(
        check="layout-invariance",
        op=(
            "arctan", "cosh", "erf", "expm1", "log", "log1p",
            "rsqrt", "sinh", "tan", "tanh",
        ),
        reason=(
            "One root cause for the whole unary family: on Apple Silicon the CPU "
            "backend runs Accelerate SIMD on contiguous inputs and scalar libm on "
            "strided views and on the contiguous residual, so the two disagree by "
            "about 1 ULP. Filed as #4163 and #4161; PR #4187 fixes it but is "
            "closed pending the highway migration in #3019."
        ),
        ref="https://github.com/ml-explore/mlx/issues/4163",
    ),
    Known(
        check="layout-invariance",
        op=("arctan2", "logaddexp"),
        reason=(
            "Same root cause as #4163, one level out: the binary ops split the "
            "same way, Accelerate SIMD on contiguous inputs and scalar libm on "
            "strided views and the residual, so they differ by about 1 ULP. #4163 "
            "is written up as a unary problem, so this is recorded here rather "
            "than filed separately while #4187 waits on #3019."
        ),
        ref="https://github.com/ml-explore/mlx/issues/4163",
    ),
    Known(
        check="ulp",
        op=("sin", "cos", "exp"),
        dtype="float64",
        reason=(
            "These three evaluate the float32 approximation and widen the result, "
            "so a float64 input gets float32-class accuracy. Filed as #4158, and "
            "#3047 is the open umbrella issue for the fp32 fallback."
        ),
        ref="https://github.com/ml-explore/mlx/issues/4158",
    ),
]


def partition(findings, known=None):
    """Split findings into (new, already_known)."""
    known = MLX_KNOWN if known is None else known
    new, seen = [], []
    for f in findings:
        hit = next((k for k in known if k.matches(f)), None)
        (seen if hit else new).append((f, hit) if hit else f)
    return new, seen
