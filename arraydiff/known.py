"""Divergences that are already known, intentional, or filed.

This list is the most important file in the project. A differential tester that
re-reports settled decisions is worse than useless: it burns maintainer goodwill
and buries the findings that are actually new. Every entry needs a reason and a
link, so that "we already talked about this" is checkable rather than folklore.

Add an entry when a maintainer has ruled on the behaviour, or when an issue or
PR already covers it. Remove one when upstream changes its mind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


FLOATS = ("float16", "float32", "float64", "bfloat16")
INTS = ("int8", "int16", "int32", "int64", "uint8", "uint32")

def _numpy_minmax_is_sign_aware(dtype: str) -> bool:
    """Whether NumPy's own min/max pick a zero's sign in this dtype, here.

    Probed rather than written down, because the answer depends on the machine.
    NumPy's float32 and float64 loops replace their scalar macro with an ISA
    instruction: aarch64 uses FMIN/FMAX, which are sign-aware, and x86 uses
    MINSS/MAXSS, which return the second operand and so are order-dependent.
    The plain C fallback returns the first operand. float16 has no such override
    and always takes the C path.

    So a hardcoded list of sign-aware dtypes is right on one host and wrong on
    another. CI caught exactly that: a list naming float32 held on Apple Silicon
    and failed on x86 Linux.
    """
    # bfloat16 has no NumPy dtype, so its reference is computed in float32 and
    # that is the loop whose behaviour decides this.
    probe = np.float32 if dtype == "bfloat16" else getattr(np, dtype)
    zero, negzero = probe(0.0), probe(-0.0)
    return bool(
        np.signbit(np.minimum(zero, negzero))
        and np.signbit(np.minimum(negzero, zero))
        and not np.signbit(np.maximum(zero, negzero))
        and not np.signbit(np.maximum(negzero, zero))
    )


# Where NumPy is a usable oracle for a zero's sign, a disagreement is the
# library's. Where it is not, the disagreement is NumPy's and says nothing about
# the library under test.
NUMPY_SIGN_AWARE_FLOATS = tuple(f for f in FLOATS if _numpy_minmax_is_sign_aware(f))
NUMPY_SIGN_BLIND_FLOATS = tuple(f for f in FLOATS if f not in NUMPY_SIGN_AWARE_FLOATS)


@dataclass(frozen=True)
class Known:
    # One check name, or a tuple of them when the same cause surfaces through
    # several checks. A flush to zero, for instance, is reported by
    # numpy-semantics against the oracle and by size-invariance against the
    # library itself, and splitting that into two entries would duplicate the
    # reason and let the copies drift apart.
    check: str | tuple[str, ...]
    # One op name, or a tuple of them when a single root cause covers a family.
    op: str | tuple[str, ...]
    reason: str
    # A link to the upstream thread that settled this, or `recorded: <what>` when
    # nothing has been filed. The second form is deliberately clumsy to write, so
    # that "nobody has looked at this" never reads like "someone ruled on it".
    ref: str
    # None means "every dtype". A tuple restricts to those dtypes, which matters
    # when the same check fires on ints and floats for unrelated reasons.
    dtype: str | tuple[str, ...] | None = None
    # An extra condition on the finding itself, for a cause that only applies to
    # part of an op's input range. Without it, suppressing a documented underflow
    # in `exp` would also suppress every other accuracy bug in `exp`, which is
    # how a known-list starts hiding regressions.
    when: Callable[[object], bool] | None = None

    def matches(self, finding) -> bool:
        checks = (self.check,) if isinstance(self.check, str) else self.check
        ops = (self.op,) if isinstance(self.op, str) else self.op
        dtypes = (self.dtype,) if isinstance(self.dtype, str) else self.dtype
        return (
            finding.check in checks
            and finding.op in ops
            and (dtypes is None or finding.dtype in dtypes)
            and (self.when is None or self.when(finding))
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
    # minimum/maximum on a pair of zeros. std::min(a, b) is b < a ? b : a, which
    # is sign-blind on zeros and returns whichever operand it was handed first,
    # while vminq_f32 is sign-aware. So the answer depends on which kernel ran,
    # which is why this fires on all three oracle-free axes at once.
    *[
        Known(
            check=check,
            op=("minimum", "maximum"),
            dtype=FLOATS if check != "numpy-semantics" else NUMPY_SIGN_AWARE_FLOATS,
            reason=(
                "A zero result gets a sign that depends on which kernel ran. The "
                "scalar path is std::min, which returns b < a ? b : a and so "
                "ignores the sign of a zero, and the NEON path is vminq_f32, "
                "which is sign-aware. minimum(0.0, -0.0) is therefore -0.0 in a "
                "long tensor and 0.0 in a short one. Filed as #193781, fixed in "
                "PR #193851."
            ),
            ref="https://github.com/pytorch/pytorch/issues/193781",
        )
        for check in ("layout-invariance", "size-invariance", "numpy-semantics")
    ],
    # The oracle is the thing that is wrong here, which is why this is a separate
    # entry from #193781 rather than more dtypes on it.
    Known(
        check="numpy-semantics",
        op=("minimum", "maximum"),
        dtype=NUMPY_SIGN_BLIND_FLOATS,
        reason=(
            "NumPy is not a valid oracle for a signed zero in these dtypes on "
            "this machine, so a disagreement is NumPy's rather than the "
            "library's. Its min/max return whichever operand a comparison "
            "reached first, and which dtypes that affects is ISA dependent: the "
            "float32 and float64 loops swap in FMIN/FMAX on aarch64, which are "
            "sign-aware, and MINSS/MAXSS on x86, which return the second "
            "operand, while float16 has no override anywhere. IEEE 754 minimum "
            "and maximum, std::fmin/std::fmax and vminq_f32 all give -0.0 for "
            "the minimum in either argument order, so a library that does the "
            "same is right. The dtype list is probed at import, not written "
            "down, because a fixed list is wrong on half the machines."
        ),
        ref="https://numpy.org/doc/stable/reference/generated/numpy.minimum.html",
    ),
    Known(
        check="device-invariance",
        op="abs",
        dtype=INTS,
        reason=(
            "MPS computed integer abs by converting to float32, so abs rounded "
            "above 2^24: abs(int32(-123456789)) came back 123456792, and int64 "
            "lost even more. Fixed in main by #190053, which added an integer "
            "overload to abs_functor, so this reproduces on 2.13.0 only."
        ),
        ref="https://github.com/pytorch/pytorch/pull/190053",
    ),
    Known(
        check="device-invariance",
        op=(
            "add", "subtract", "multiply", "divide", "reciprocal",
            "ceil", "floor", "sign", "logical_not",
            "expm1", "logaddexp", "arctan2",
        ),
        dtype=("float32",),
        reason=(
            "Metal flushes float32 subnormals to zero, so every one of these "
            "reports on an input or a result near 1e-38. The consequences look "
            "unrelated but are one cause: sign(1.4e-45) is 0, ceil(1.4e-45) is "
            "0.0 rather than 1.0, and 1.4e-45 * inf is NaN rather than inf "
            "because the operand became a zero first. Documented in the Metal "
            "Shading Language Specification, which permits flushing denormals "
            "on input and output. float16 and bfloat16 do not report, since "
            "their subnormals are far larger than the flush threshold."
        ),
        ref=(
            "https://developer.apple.com/metal/"
            "Metal-Shading-Language-Specification.pdf"
        ),
    ),
]


# XLA on CPU treats subnormals as zero in arithmetic. Storage keeps them, so a
# subnormal survives a round trip and only changes when it is operated on, which
# is why this shows up as a NumPy mismatch rather than a bad input. Ruled expected
# twice and documented, so it is one entry reused rather than nine findings.
_JAX_DAZ = (
    "XLA flushes subnormals to zero in arithmetic on CPU. 1 / DBL_MAX gives "
    "0.0 and a subnormal operand multiplies as if it were zero. Closed as "
    "expected on #37670 and #37761, and documented in Sharp Bits."
)
_JAX_DAZ_REF = (
    "https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html"
    "#double-64bit-precision"
)

def _exp_underflows_to_subnormal(finding) -> bool:
    """Whether `exp` of this input lands in the subnormal range of its dtype.

    That band is narrow, about x in [-745.1, -708.4] for float64, and it is
    exactly where a flush to zero turns into an unbounded ULP error: below it the
    true result is smaller than the smallest subnormal, so returning 0.0 is
    correct and the measured error is already 0. Whether the sweep happens to
    draw a value inside the band depends on the sample size, which is why this
    has to be an entry rather than left to luck.
    """
    if finding.dtype not in ("float32", "float64"):
        return False
    try:
        y = math.exp(finding.inputs[0])
    except (OverflowError, ValueError):
        return False
    return 0 < y < float(np.finfo(finding.dtype).smallest_normal)


JAX_KNOWN = [
    Known(
        check="numpy-semantics",
        op=(
            "add", "subtract", "multiply", "divide",
            "minimum", "maximum", "floor", "ceil", "reciprocal",
        ),
        dtype=FLOATS,
        reason=_JAX_DAZ,
        ref=_JAX_DAZ_REF,
    ),
    Known(
        check="ulp",
        op="reciprocal",
        dtype=FLOATS,
        reason=(
            _JAX_DAZ + " Here the flush is on the output: the true reciprocal of "
            "a near-max input is subnormal, so it comes back as 0.0."
        ),
        ref=_JAX_DAZ_REF,
    ),
    Known(
        check="divmod-identity",
        op="divmod",
        dtype=FLOATS,
        reason=(
            _JAX_DAZ + " A subnormal divisor acts as zero, so the quotient is "
            "inf and q*b + r cannot recover the dividend."
        ),
        ref=_JAX_DAZ_REF,
    ),
    Known(
        check="size-invariance",
        op=("exp", "log", "arctan"),
        dtype=FLOATS,
        reason=(
            "Every jnp op is jit-decorated, so each shape is a separate XLA "
            "compilation and can pick a different vector width. The results "
            "differ by a ULP. JAX documents that jit changes the exact numerics "
            "of outputs, so length dependence is expected here in a way it is "
            "not in an eager library."
        ),
        ref="https://docs.jax.dev/en/latest/faq.html",
    ),
    Known(
        check="ulp",
        op="sinh",
        dtype=FLOATS,
        reason=(
            "About 4 ULP at small arguments, where NumPy is under 1. sinh is "
            "(exp(x) - exp(-x)) / 2, so XLA's exp error shows through. Not "
            "filed: JAX's own test tolerance is 1e-6 relative for float32 and "
            "1e-15 for float64 (_default_tolerance in public_test_util.py), and "
            "4 ULP is 4.8e-7, inside both. Worth reporting only if it grows."
        ),
        ref="https://github.com/jax-ml/jax/blob/main/jax/_src/public_test_util.py",
    ),
    Known(
        check="numpy-semantics",
        op="sign",
        dtype=FLOATS,
        reason=(
            "sign(-0.0) returns -0.0 where NumPy returns 0.0. The documented "
            "value for a zero input is 0, and -0.0 == 0, so this meets the "
            "contract. Recorded rather than filed."
        ),
        ref="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sign.html",
    ),
    Known(
        check="numpy-semantics",
        op="remainder",
        dtype=FLOATS,
        reason=(
            "A zero remainder keeps the sign of the dividend, but the docstring "
            "says the result has the sign of the divisor. The sign fixup in "
            "remainder is guarded on the remainder being nonzero, so lax.rem's "
            "sign survives. Filed as #40028."
        ),
        ref="https://github.com/jax-ml/jax/issues/40028",
    ),
    Known(
        check="numpy-semantics",
        op="floor_divide",
        dtype=FLOATS,
        reason=(
            "A zero quotient gets the sign of the divisor alone, not of a / b, "
            "so -0.0 // 3.0 is 0.0 where Python and NumPy give -0.0. "
            "_float_divmod computes x1 - mod first, and for a negative zero "
            "dividend that is -0.0 - -0.0, which is +0.0. Reported on the "
            "#40028 thread, since the fix is in the same file as the remainder "
            "sign. Some elements in this finding are the DAZ flush instead."
        ),
        ref="https://github.com/jax-ml/jax/issues/40028",
    ),
    Known(
        check="ulp",
        op="exp",
        dtype=("float32", "float64"),
        when=_exp_underflows_to_subnormal,
        reason=(
            "The same flush to zero as the arithmetic entries, reached through "
            "the accuracy check instead. exp(-715.4) is 2.1e-311, a subnormal, "
            "and XLA returns 0.0, which is an error of 4e12 ULP rather than a "
            "small one. Restricted to inputs whose exact result is actually "
            "subnormal, so a real accuracy bug in exp anywhere else is still "
            "reported."
        ),
        ref=_JAX_DAZ_REF,
    ),
]


def _tf_input_is_subnormal(finding) -> bool:
    """True when a finding's input is a nonzero subnormal in its own dtype.

    TensorFlow's CPU kernels flush subnormals to zero, so a divergence on a
    subnormal input is that flush rather than a separate bug. Scoped this tightly
    on purpose: minimum, maximum and floor_divide have their own zero-sign and
    infinity bugs on ordinary inputs, and a blanket suppression would hide them.
    """
    name = "float32" if finding.dtype == "bfloat16" else finding.dtype
    try:
        tiny = float(np.finfo(getattr(np, name)).smallest_normal)
    except (AttributeError, ValueError):
        return False
    return any(v == v and v != 0.0 and abs(v) < tiny for v in finding.inputs)


# TensorFlow, on CPU. Nothing is filed against it, in line with the rest of this
# file, so the refs describe the cause and point at the same bug in the libraries
# where it is filed. Confirmed to persist with TF_ENABLE_ONEDNN_OPTS=0, so it is
# the CPU kernels themselves, not oneDNN.
_TF_FLUSH = (
    "TensorFlow flushes subnormals to zero in its CPU kernels: ceil of the "
    "smallest subnormal is 0.0 rather than 1.0, and add(x, x) on it is 0.0. Same "
    "class as XLA's flush in JAX above; persists with TF_ENABLE_ONEDNN_OPTS=0, so "
    "it is the kernels, not oneDNN. Recorded rather than filed."
)
_TF_FLUSH_REF = "recorded: tensorflow CPU flushes subnormals to zero"
_TF_TRANSC = (
    "About 1 ULP between a short array and a long one. TensorFlow's transcendental "
    "kernels vectorize over length, so a length-1 or length-3 array takes the "
    "scalar path and rounds a bit differently from the vectorized body. These ops "
    "are not correctly rounded in any library, so this is allowed, the same as the "
    "size-invariance entries for torch and jax above. Recorded because it is not a "
    "defect, rather than filed."
)
_TF_TRANSC_REF = "recorded: tensorflow vectorized-vs-scalar transcendental rounding"

TF_KNOWN: list[Known] = [
    # Subnormal flush, on whichever ops happen to see a subnormal input. The
    # predicate keeps this from swallowing the sign and infinity bugs below.
    Known(
        check="numpy-semantics",
        op=("add", "subtract", "multiply", "divide", "floor", "ceil", "sign",
            "minimum", "maximum"),
        dtype=FLOATS,
        when=_tf_input_is_subnormal,
        reason=_TF_FLUSH,
        ref=_TF_FLUSH_REF,
    ),
    Known(
        check="size-invariance",
        op="sign",
        dtype=FLOATS,
        reason=(
            _TF_FLUSH + " Here it shows as length dependence: the scalar residual "
            "keeps a subnormal that the vectorized body flushed, so sign disagrees "
            "with itself across lengths."
        ),
        ref=_TF_FLUSH_REF,
    ),
    # reciprocal's findings are all the flush on the output: the reciprocal of a
    # near-max input is subnormal and comes back as 0.0.
    Known(
        check=("numpy-semantics", "ulp", "size-invariance"),
        op="reciprocal",
        dtype=FLOATS,
        reason=(
            _TF_FLUSH + " Here the flush is on the output: the reciprocal of a "
            "near-max value is subnormal, so it returns 0.0 (numpy-semantics/ulp) "
            "and disagrees across lengths (size-invariance)."
        ),
        ref=_TF_FLUSH_REF,
    ),
    # minimum/maximum give a zero the wrong IEEE sign. Same bug as pytorch#193781
    # and mlx. Where numpy can be trusted for a zero's sign the disagreement is
    # TensorFlow's; on the sign-blind dtypes it is numpy's, exactly as for torch.
    Known(
        check="numpy-semantics",
        op=("minimum", "maximum"),
        dtype=NUMPY_SIGN_AWARE_FLOATS,
        reason=(
            "minimum(0.0, -0.0) is +0.0 and maximum(-0.0, 0.0) is -0.0, the "
            "opposite of the IEEE result, which numpy gives correctly on these "
            "dtypes. Same bug as pytorch#193781 in a fourth library. Recorded "
            "rather than filed."
        ),
        ref="recorded: tensorflow min/max give a zero the wrong sign",
    ),
    # The same zero-sign bug reached through the check that needs no oracle, so
    # it covers float16 too, where numpy is no use as a reference. This is where
    # the cause is visible: the vectorized body returns the IEEE sign and the
    # scalar tail does not, so max(-0.0, 0.0) is +0.0 in a length-503 array and
    # -0.0 in a length-1 one. A separate entry from the one above because the
    # dtype scope is genuinely different, not because the wording is.
    Known(
        check="size-invariance",
        op=("minimum", "maximum"),
        dtype=FLOATS,
        reason=(
            "TensorFlow contradicts itself on the sign of a zero across lengths: "
            "the vectorized kernel picks the IEEE sign and the scalar tail returns "
            "one operand regardless. No oracle is needed for this one, so it holds "
            "on float16 as well. Some elements are the subnormal flush instead. "
            "Recorded rather than filed."
        ),
        ref="recorded: tensorflow min/max give a zero the wrong sign",
    ),
    Known(
        check="numpy-semantics",
        op=("minimum", "maximum"),
        dtype=NUMPY_SIGN_BLIND_FLOATS,
        reason=(
            "On the dtypes where numpy's own min/max are not sign-aware, a zero-"
            "sign disagreement is numpy's, not TensorFlow's, so it says nothing "
            "about the library under test. Same split as the torch entry above."
        ),
        ref="https://numpy.org/doc/stable/reference/generated/numpy.minimum.html",
    ),
    # floor_divide at infinity: 1.0 // -inf is -0.0 where numpy and Python give
    # -1.0. Same bug as mlx#4317 and torch, now in TensorFlow.
    Known(
        check="numpy-semantics",
        op="floor_divide",
        dtype=FLOATS,
        reason=(
            "1.0 // -inf is -0.0 where numpy and Python give -1.0: floor(a / b) "
            "is not what // means once an operand is infinite. Same bug as "
            "mlx#4317 and the torch device split. Recorded rather than filed."
        ),
        ref="recorded: tensorflow floor_divide at infinity",
    ),
    # remainder gives a zero result the wrong sign. Same class as jax#40028,
    # torch#193755, mlx#4315.
    Known(
        check=("numpy-semantics", "size-invariance"),
        op="remainder",
        dtype=FLOATS,
        reason=(
            "remainder(0.0, -1.0) is +0.0 where the contract is the sign of the "
            "divisor, -0.0, which numpy and Python both give. The zero case is "
            "not sign-corrected. Same bug as jax#40028, torch#193755 and "
            "mlx#4315, found by one check in a fourth library. Recorded rather "
            "than filed."
        ),
        ref="recorded: tensorflow remainder zero-sign",
    ),
    # divmod here is floordiv and floormod called together, so a divmod-identity
    # finding is those two TF ops disagreeing with each other. Two causes, and
    # they need separate entries because one is the flush and one is not.
    Known(
        check="divmod-identity",
        op="divmod",
        dtype=FLOATS,
        when=_tf_input_is_subnormal,
        reason=(
            _TF_FLUSH + " A subnormal divisor acts as zero, so the quotient is inf "
            "and q*b + r cannot recover the dividend. Same as the jax entry above."
        ),
        ref=_TF_FLUSH_REF,
    ),
    Known(
        check="divmod-identity",
        op="divmod",
        dtype=FLOATS,
        reason=(
            "floordiv is floor(a / b), a division whose own rounding can carry the "
            "result up past the floor, and floormod does not follow it there: in "
            "bfloat16 2144 / 358 is 5.9888, which rounds to exactly 6.0, so "
            "floordiv gives 6 while floormod gives 354, the remainder for 5. The "
            "two ops are inconsistent, and composing divmod from them is what "
            "surfaces it. This is the same defect as mlx#4003, which was closed on "
            "the grounds that widening the division costs performance and that "
            "PyTorch rounds the same way, so it is recorded as ruled-on rather "
            "than filed a fourth time."
        ),
        ref="https://github.com/ml-explore/mlx/pull/4003",
    ),
    # Vectorized-vs-scalar rounding on the transcendentals, about 1 ULP, allowed.
    Known(
        check="size-invariance",
        op=("arctan", "arctan2", "cos", "sin", "tanh", "exp", "expm1",
            "log", "log1p", "power", "erf", "rsqrt"),
        dtype=FLOATS,
        reason=_TF_TRANSC,
        ref=_TF_TRANSC_REF,
    ),
]

# Each library gets its own list. A shared list would let an MLX ruling suppress
# a real torch bug that happens to be the same op and dtype.
KNOWN = {
    "mlx": MLX_KNOWN,
    "torch": TORCH_KNOWN,
    "jax": JAX_KNOWN,
    "tensorflow": TF_KNOWN,
}


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
