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


@dataclass(frozen=True)
class Known:
    check: str
    # One op name, or a tuple of them when a single root cause covers a family.
    op: str | tuple[str, ...]
    reason: str
    ref: str
    # None means "every dtype".
    dtype: str | None = None

    def matches(self, finding) -> bool:
        ops = (self.op,) if isinstance(self.op, str) else self.op
        return (
            self.check == finding.check
            and finding.op in ops
            and (self.dtype is None or self.dtype == finding.dtype)
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
        reason=(
            "Integer floor_divide truncates toward zero instead of flooring. "
            "Already filed, and PR #4108 is open against it."
        ),
        ref="https://github.com/ml-explore/mlx/pull/4108",
    ),
    Known(
        check="divmod-vs-floor_divide",
        op="divmod",
        reason=(
            "divmod truncates the quotient while floor_divide floors, so the "
            "two disagree. Filed as #4119 with PRs #4108 and #4311 open."
        ),
        ref="https://github.com/ml-explore/mlx/issues/4119",
    ),
    Known(
        check="divmod-identity",
        op="divmod",
        reason=(
            "q*b + r != a follows from the same truncation as #4119; the "
            "remainder is not adjusted to match the quotient."
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
