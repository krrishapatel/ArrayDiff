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
        check="size-invariance",
        op=(
            "arctan", "arctan2", "cosh", "erf", "expm1", "log", "log1p",
            "logaddexp", "rsqrt", "sinh", "tan", "tanh",
        ),
        reason=(
            "The same #4163 split seen on the length axis instead of the layout "
            "axis. #4163 names the contiguous residual directly as one of the two "
            "paths, and a short array is nothing but residual, so this is that "
            "bug rather than a second one."
        ),
        ref="https://github.com/ml-explore/mlx/issues/4163",
    ),
    Known(
        check="size-invariance",
        op="power",
        reason=(
            "Same as the #4162 layout entry: a short array does not reach "
            "Accelerate's vectorized pow and falls back to scalar std::pow."
        ),
        ref="https://github.com/ml-explore/mlx/issues/4162",
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


# Ops whose vectorized kernel is Sleef and whose scalar kernel is libm. The two
# agree to about a ULP but not bit for bit, so every layout and length check
# fires on them. Kept as one list because it is one root cause.
_TORCH_SLEEF_OPS = (
    "exp",
    "expm1",
    "log",
    "log1p",
    "sin",
    "cos",
    "tan",
    "tanh",
    "sinh",
    "cosh",
    "arctan",
    "arctan2",
    "logaddexp",
    "power",
    "erf",
    "sqrt",
    "rsqrt",
    "reciprocal",
)

TORCH_KNOWN = [
    Known(
        check="layout-invariance",
        op=_TORCH_SLEEF_OPS,
        reason=(
            "The vectorized kernel calls Sleef and the scalar tail calls libm, "
            "so a non-contiguous input that falls back to the scalar path can "
            "differ in the last bit. Torch documents that results are not "
            "bitwise identical across vectorization, so this is expected rather "
            "than a bug. Reported only when the difference is more than "
            "rounding, which is what #193753 and #193754 are."
        ),
        ref="https://pytorch.org/docs/stable/notes/numerical_accuracy.html",
    ),
    Known(
        check="size-invariance",
        op=_TORCH_SLEEF_OPS,
        reason=(
            "Same root cause as the layout entry above: array length decides "
            "whether Sleef or libm runs, and the two differ in the last bit."
        ),
        ref="https://pytorch.org/docs/stable/notes/numerical_accuracy.html",
    ),
    Known(
        check="numpy-semantics",
        op="sign",
        reason=(
            "sign(nan) returns 0 rather than propagating, because the kernel is "
            "(0 < x) - (x < 0) and both comparisons are false for NaN. Open as "
            "#41245 since 2020 and again as #187295. PR #187558 fixed it across "
            "CPU, CUDA and Inductor and was closed unmerged, so this is known "
            "and unfixed rather than new."
        ),
        ref="https://github.com/pytorch/pytorch/issues/187295",
    ),
    Known(
        check="numpy-semantics",
        op="remainder",
        dtype=FLOATS,
        reason=(
            "Two separate causes, both filed. A zero remainder keeps the sign of "
            "the dividend instead of the divisor, because the sign fixup is "
            "guarded on the remainder being nonzero: filed as #193755. And the "
            "vectorized Sleef_fmod returns NaN once the division overflows, "
            "where libm returns a finite value: filed as #193753."
        ),
        ref="https://github.com/pytorch/pytorch/issues/193755",
    ),
    Known(
        check="size-invariance",
        op="remainder",
        dtype=FLOATS,
        reason=(
            "Sleef_fmod is undefined once abs(a / b) reaches 1e300 for double or "
            "1e38 for float, so a long array gets NaN and a short one gets the "
            "right answer. Diagnosed on #77742, which was closed by a fix that "
            "only covered div with rounding_mode='floor'. Filed as #193753."
        ),
        ref="https://github.com/pytorch/pytorch/issues/193753",
    ),
    Known(
        check="layout-invariance",
        op="remainder",
        dtype=FLOATS,
        reason=(
            "Same Sleef_fmod overflow as the size entry above, reached the other "
            "way: a strided input falls back to the scalar libm path and gets "
            "the right answer, while the contiguous one gets NaN. Filed as "
            "#193753."
        ),
        ref="https://github.com/pytorch/pytorch/issues/193753",
    ),
    Known(
        check="size-invariance",
        op=("floor_divide", "divmod"),
        dtype=("float16", "bfloat16"),
        reason=(
            "div_floor_floating_vec computes in Vectorized<Half>, so fmod and "
            "the division round to 8 or 11 mantissa bits at each step, while "
            "the scalar path promotes to float32. The vectorized result can "
            "exceed the quotient it is meant to floor: bfloat16 560 // 3 gives "
            "187. Filed as #193754."
        ),
        ref="https://github.com/pytorch/pytorch/issues/193754",
    ),
    Known(
        check="numpy-semantics",
        op="floor_divide",
        dtype=("float16", "bfloat16"),
        reason=(
            "Same reduced-precision vectorized path as the size-invariance "
            "entry above. Filed as #193754."
        ),
        ref="https://github.com/pytorch/pytorch/issues/193754",
    ),
]


# Each library gets its own list. A shared list would let an MLX ruling suppress
# a real torch bug that happens to be the same op and dtype.
KNOWN = {"mlx": MLX_KNOWN, "torch": TORCH_KNOWN}


def known_for(backend_name):
    """Empty for a library with no list yet, so its findings all read as new
    rather than being silently matched against another library's rulings."""
    return KNOWN.get(backend_name, [])


def partition(findings, known=None):
    """Split findings into (new, already_known)."""
    known = MLX_KNOWN if known is None else known
    new, seen = [], []
    for f in findings:
        hit = next((k for k in known if k.matches(f)), None)
        (seen if hit else new).append((f, hit) if hit else f)
    return new, seen
