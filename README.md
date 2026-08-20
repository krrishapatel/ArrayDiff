# arraydiff

[![CI](https://github.com/krrishapatel/ArrayDiff/actions/workflows/ci.yml/badge.svg)](https://github.com/krrishapatel/ArrayDiff/actions/workflows/ci.yml)

Differential numerical testing for array libraries. It looks for places where an
elementwise op returns the wrong answer, an inconsistent answer, or an answer
that depends on something it should not depend on.

## Why the checks are shaped this way

Most numerical test suites compare against NumPy with a loose tolerance. That
misses two whole classes of bug and invents a third.

**The strongest checks need no reference at all.** If the same op, on the same
values, returns different results depending on how the input happened to be
stored in memory, that is a bug regardless of what NumPy would have said. It is
also impossible to argue with, since the library is disagreeing with itself:

```
$ arraydiff --only tan
[layout-invariance] tan float64: strided disagrees with contiguous on 41/83 elements
    inputs: (0.6931471805599453,)
    got:    strided=np.float64(0.8337300251311491)
    want:   contiguous=np.float64(0.8337300251311493)
```

This is what `layout-invariance` does, across eight memory layouts: contiguous,
two strides, reversed, transposed, a non-zero offset, a zero-stride broadcast,
and a 2-D column. The usual cause is a vectorized kernel that only runs on
contiguous data, with a scalar fallback that rounds differently. Binary ops are
covered too, with both operands moved to the same layout at once; an earlier
version checked only unary ops, which silently skipped `power`.

`size-invariance` is the same argument on the length axis, and it is a separate
check because length is what picks the kernel: a long array goes through the
vectorized body and a short one goes through the scalar path, which is usually a
different algorithm rather than the same one unrolled. Layout invariance cannot
see this, since every layout it builds is full length. It is the check that found
most of the torch bugs below.

`device-invariance` is the third axis, and it is the strongest of the three,
because a second device is not a second code path through one algorithm but a
separate implementation: the CPU and GPU kernels for an op are written by
different people, often years apart, and the special cases are where they drift.

The bar there cannot be bit equality, and getting that right is the whole check.
Nobody promises that a Metal `exp` matches an Accelerate `exp` in the last bit,
and torch documents as much, so demanding it would bury every real finding under
a wall of rounding. Ops that are correctly rounded or integral are still held to
bit equality, since IEEE 754 or the op's own definition pins their result on any
device. For the rest only a difference in *kind* counts, which is
`categorical_diff`: a NaN against a number, an infinity against a finite value,
an infinity or a zero with the other sign, or an exact zero where the other
device returned something nonzero. None of those can be a rounding difference,
in any precision, so they are findings wherever they appear.

That split is what makes the axis usable. Held to bit equality, torch reports on
most of the op table; held to differences in kind, it reports 18 findings with
five causes.

**A binary op has to be tried against the combinations, not the values.** The
second operand used to be a rotation of the first, which only ever produces the
pairings that the rotation happens to make. With a shift of 3, `0.0` never lines
up with `-0.0`, so no number of random draws could have found a `minimum` that
returns the wrong zero. Every special value is now tried against every other,
which is 225 pairs and costs nothing. That single change is what surfaced torch
[#193781](https://github.com/pytorch/pytorch/issues/193781) and the 42 MLX
findings below, both of which had been sitting under the previous generator the
whole time.

**The budget has to be one a working implementation passes.** Two things go wrong
here. A flat budget ignores conditioning: `exp` has relative condition number
`|x|`, so at `x = 700` the input's own rounding forces an error of several hundred
ULP, and demanding 1 ULP there is a bug in the test. Every budget is therefore
scaled by `|x f'(x) / f(x)|`. For the same reason `sin`/`cos`/`tan` inputs are
dropped once one ULP of the input spans a meaningful fraction of a period: one
ULP of `1e308` is about `1e291` periods, so there is no correct answer to be
accurate to.

The other half is the base budget. IEEE-754 requires `sqrt` and division to be
correctly rounded, so half a ULP is the real contract there. No standard requires
that of the transcendentals, and Accelerate, Metal and CUDA all document theirs
in the low single digits, so those get 4 ULP. `tests/test_oracles.py` checks the
budget against NumPy's own transcendentals in every precision: if a budget flags
NumPy, it is the budget that is wrong.

`tests/test_checks.py` applies the same rule to the checks. `divmod-identity`
originally demanded that `q*b + r` equal `a` exactly, which NumPy fails on 54 of
400 float32 pairs, because the floored remainder is `fmod(a, b) + b` and that
addition rounds. Forming `q*b + r` can also cancel almost everything, so the
tolerance is measured at the scale of `q*b` rather than of `a`. Integers are
still judged exactly, reconstructed with the dtype's own wraparound so that
`int8(-128) // -1` is not blamed for an overflow the format cannot avoid.

**Accuracy has to be measured in the right precision.** One ULP near 1.0 is
about `1e-3` in float16 and `2e-16` in float64. Measuring a float16 result
against float64 spacing overstates the error by about `1e12` and makes every op
look broken. `tests/test_oracles.py` pins this down by asserting that `sqrt` and
division measure under half a ULP in every precision, which IEEE-754 guarantees
independently of any library.

**Reporting a settled decision is worse than reporting nothing.** It wastes
maintainer attention and buries the findings that are new. Every known or
intentional divergence lives in `known.py` with a reason and a link, and
`partition()` splits a run into new versus already-known. For example
`mx.sign(nan)` returns 0 rather than propagating, which was
[declined deliberately](https://github.com/ml-explore/mlx/issues/3644) because
torch does the same and propagating costs instructions on the hot path.

An entry can also carry a `when` predicate, so that a cause which only applies to
part of an op's input range suppresses only that part. XLA flushing subnormals
makes `jnp.exp(-715.4)` return `0.0` instead of `2.1e-311`, which the accuracy
check sees as an error of 4e12 ULP. Suppressing that by op and dtype alone would
hide every other accuracy bug in `exp`, so the entry is restricted to inputs
whose exact result is genuinely subnormal, a band about 37 wide out of the whole
line. Below it, returning `0.0` is correct rather than a flush, and the measured
error is already zero.

An entry can name several checks, for the opposite reason: one cause often
surfaces through more than one. A flush to zero is reported by `numpy-semantics`
against the oracle and by `size-invariance` against the library itself, and
splitting that into two entries duplicates the reason and lets the copies drift.
Naming a check it did not name is still a miss, so the entry cannot widen past
what it claims.

## Checks

| check | needs an oracle | what it means |
|---|---|---|
| `layout-invariance` | no | same values, different memory layout, different result |
| `size-invariance` | no | same values, shorter array, different result |
| `device-invariance` | no | same values, another device, a difference rounding cannot explain |
| `divmod-identity` | no | `q*b + r == a` fails |
| `divmod-vs-floor_divide` | no | `divmod`'s quotient disagrees with `floor_divide` |
| `numpy-semantics` | NumPy | bit-exact disagreement on an op that should be exact |
| `ulp` | mpmath | error exceeds the condition-scaled budget |

`numpy-semantics` is only applied to ops that are correctly rounded or integral
(`add`, `multiply`, `divide`, comparisons, `remainder`, `floor_divide`), where a
difference is a real semantic disagreement rather than a rounding choice.
Transcendentals are judged by ULP instead.

**The reference has to be the values the kernel actually got.** bfloat16 keeps 8
mantissa bits, so casting a float32 input moves it, and `1.4e-45` lands on exactly
`0`. Operands are therefore read back out of the array before being used as the
reference, otherwise a correct divide by zero is reported as a bug.

Every difference, in every check, is decided by one function, `bitwise_diff`.
Signed zero counts as a difference, NaN compares equal to NaN, and integers are
compared as integers rather than through float64. Plain `==` hides the first two
bug classes and loses the low bits of int64 in the third. Sharing one definition
also matters for the report: an earlier version decided with `bitwise_diff` but
counted with `==`, so a signed-zero disagreement was reported as "0 elements
differ" and pointed at the wrong input.

## Usage

```bash
pip install -e '.[dev]'

# no library under test is a dependency; point at whatever build you want
PYTHONPATH=/path/to/mlx/python arraydiff
PYTHONPATH=/path/to/mlx/python arraydiff --only tan power --check layout-invariance

arraydiff --backend torch
arraydiff --backend jax
arraydiff --backend tensorflow
```

Exit status is 1 when there are findings, so it can gate CI. This repo's own CI
runs the sweep against real torch, jax and tensorflow builds on every push and
asserts that the specific bugs below still reproduce, so the day one is fixed
upstream the run goes red instead of the counts here going stale. It does not assert a
zero-new count, because that count is platform-dependent: NumPy runs its
transcendental SIMD loops on contiguous data only, so on x86 a reversed view
lands about 1 ULP away where aarch64 is bit-identical, and torch's CPU `sqrt` is
similarly off by under a ULP on x86. Those are rounding, handled by the ULP
tolerance in `tests/test_checks.py`, not defects. The bugs the job pins are all
categorical, sign, or self-contradiction errors, which reproduce on any
architecture. The GPU half of the torch table needs a second device, so it is
checked on Apple Silicon locally and skipped on the x86 CI runner.

```python
from arraydiff.backends import mlx_backend
from arraydiff.known import known_for, partition
from arraydiff.runner import run
import mlx.core as mx

be = mlx_backend(mx)
new, already_known = partition(run(be), known_for(be.name))
```

## Status

Five libraries, four of them under test and NumPy mostly serving as the oracle.
`known.py` accounts for everything already settled, so the new count below is
only the part that is not, and every line of it is attributed to a cause before
it is called a finding.

Nothing is being filed against MLX, PyTorch, JAX or TensorFlow at the moment. Everything
already open there is waiting on a maintainer, and adding to that pile is not
what makes any of it get read. The one recent exception is
[numpy#32341](https://github.com/numpy/numpy/pull/32341), which went to a
project with nothing else outstanding.

### MLX

Against MLX main at `c2bcf47`, with the pending `remainder` fix below applied:

```
42 new findings
    24  layout-invariance
    16  size-invariance
     2  numpy-semantics

228 already accounted for in known.py
   196  https://github.com/ml-explore/mlx/issues/4163
    14  https://github.com/ml-explore/mlx/issues/4162
     8  https://github.com/ml-explore/mlx/issues/4119
     3  https://github.com/ml-explore/mlx/issues/4317
     3  https://github.com/ml-explore/mlx/issues/3644
     3  https://github.com/ml-explore/mlx/issues/4158
     1  https://github.com/ml-explore/mlx/pull/4003
```

The 228 are the main evidence that the checks work: the tool rediscovers the
whole known numerical bug cluster without being told about any of it.

All 42 new findings are one bug, and the special-value pairing change above is
what surfaced it. It had been reachable the whole time and nothing reached it.

**`minimum` and `maximum` give a zero result a sign that depends on the array's
length.** There are two implementations. The vectorized one calls Accelerate's
`simd::min`, which compiles to `vminq_f32` and is sign aware. The scalar one, in
`base_simd.h`, is `return (a < b) ? a : b`, and `-0.0 < 0.0` is false, so it
returns whichever operand came second no matter what the signs were. Which one
runs is decided by how many elements are left.

That makes it visible with no oracle, as a single array contradicting itself:

```
mx.minimum(mx.full((n,), -0.0), mx.full((n,), 0.0))

n = 7    every lane  0.0     scalar throughout, all wrong
n = 8    every lane -0.0     one full register, all right
n = 9    lanes 0-7  -0.0     the vectorized body
         lane 8      0.0     the scalar residual, same inputs, other answer
```

NumPy and Python both give `-0.0`. `maximum` is the mirror image, `(a > b) ? a : b`,
so `mx.maximum(0.0, -0.0)` is `-0.0` where it should be `0.0`.

This is the same shape as the `remainder` bug below, which is the argument for
`layout-invariance` and `size-invariance` being separate checks rather than one:
a residual lane disagreeing with the lanes beside it needs no reference to be
wrong. Recorded here rather than filed. It only moves a zero's sign, and it is
queued behind threads in these projects that are still unanswered.

Two bugs were filed out of earlier runs, and one rediscovery is recorded.

**`remainder` gives a zero result a path dependent sign.** Found by
`layout-invariance` with no oracle at all, which is what that check is for. It
reduces to a self contradiction: lane 8 of
`mx.remainder(mx.full((9,), -0.0), mx.full((9,), 3.0))` disagrees with lanes 0 to
7 on identical inputs, because the Accelerate SIMD body and the scalar residual
produce zeros of different signs and the floored fixup skips zero. Filed as
[#4315](https://github.com/ml-explore/mlx/issues/4315), with a fix across all
four backends proposed in
[#4316](https://github.com/ml-explore/mlx/pull/4316), still open. The numbers
above are measured with that patch applied, which is why it has no entry in the
list.

**`floor_divide` disagrees with both NumPy and Python at infinity.** `inf // -3.0`
gives `-inf` where both references give `nan`, and `1.0 // -inf` gives `-0.0`
where both give `-1.0`. `floor(a / b)` is not what `//` means once an operand is
infinite. Filed as
[#4317](https://github.com/ml-explore/mlx/issues/4317), and it is the three
`numpy-semantics` entries above.

**`divmod` computes the quotient by dividing a second time**, so `q*b + r` can be
off by a whole divisor. `divmod(mx.array([2144.0], mx.bfloat16), 358.0)` returns
`q = 6` where the floor is 5, because `5.9888` rounds to exactly `6.0` in
bfloat16 and `floor` cannot undo it.

This one is a rediscovery, and it is the more useful entry in `known.py` for it.
[PR #4003](https://github.com/ml-explore/mlx/pull/4003) found the
half-precision half of it months earlier and was closed: widening the division
adds overhead, and PyTorch was cited as behaving the same way. So the tool was
reporting a decision that had already been made, which is the failure mode this
project is supposed to avoid, and it is suppressed on those grounds rather than
on my disagreeing with it.

Two things about it did seem worth raising on the open PR in this area. PyTorch's
floor division is not `floor(a / b)`: `div_floor_floating` derives the quotient
from the remainder, the same as NumPy, and is vectorized. And float64 breaks the
same way, `divmod(1.0, 0.1)` giving `q = 10`, where widening has nothing to widen
to. Also raised there: an integer overflow in that PR's `floor_divide` shortcut,
where `a - remainder(a, b)` leaves the dtype range and `mx.int8(120) // -27`
comes back `4` instead of `-5`.

### PyTorch

Against torch 2.13.0, CPU and MPS:

```
18 new findings
    18  device-invariance

122 already accounted for in known.py
    46  https://github.com/pytorch/pytorch/issues/193781
    38  https://pytorch.org/docs/stable/notes/numerical_accuracy.html
    12  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
    12  https://github.com/pytorch/pytorch/issues/193753
     4  https://github.com/pytorch/pytorch/pull/190053
     3  https://github.com/pytorch/pytorch/issues/193755
     3  https://github.com/pytorch/pytorch/issues/187295
     2  https://github.com/pytorch/pytorch/issues/193754
     2  https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
```

The 38 are the Sleef-versus-libm last-bit differences, which torch documents as
allowed and which are therefore not findings. The 12 Metal entries are float32
subnormals: Metal flushes them, the specification says so, and it shows up on 12
ops at once. float16 and bfloat16 are unaffected because their subnormals start
far above the threshold. The 4 are integer `abs` on MPS rounding through float32,
so `abs(int32(-123456789))` came back `123456792`; that one is a rediscovery, and
the prior-work check is what caught it, since `d0458e4559` had already fixed it
in main. It reproduces on 2.13.0 only.

The 18 new findings are 6 ops in 3 dtypes each, and five causes:

| op | CPU | MPS |
|---|---|---|
| `arctan(-0.0)` | `-0.0` | `0.0` |
| `erf(1.18e-38)` | `0.0` | `1.33e-38` |
| `remainder(0.0, inf)` | `0.0` | `nan`, on 75 of 503 elements |
| `1.0 // -inf` | `-1.0` | `-0.0`, on 38 of 503 elements |
| `minimum(-0.0, 0.0)` | `-0.0` | `0.0` |

`erf` is the one worth pointing at, because the wrong device is the reference one.
The arm64 CPU kernel uses the Abramowitz and Stegun 7.1.26 approximation, which
has a bounded absolute error and no relative guarantee, so for a tiny input it
returns exactly `0.0` where the answer is a small nonzero number. `erf(x) ~ 2x/sqrt(pi)`
there, so the result is a normal float and there is nothing to underflow.
`IMPLEMENT_FLOAT_KERNEL` routes every length through the vector path, so there is
no scalar fallback to disagree with it and neither oracle-free axis on one device
can see it. Two devices can. This is the case that `categorical_diff` is written
for: an exact zero against a nonzero value is not a rounding difference at any
precision, which is why the check reports it at all instead of measuring the gap
in ULP and calling it noise.

`floor_divide` at infinity is the same bug as MLX's
[#4317](https://github.com/ml-explore/mlx/issues/4317), now in a second library,
and `minimum` on zeros is the MPS kernel repeating the CPU bug in
[#193781](https://github.com/pytorch/pytorch/issues/193781) in a separate
codebase. All 18 are held rather than filed, for the reason at the top of Status.

Four bugs were filed out of earlier runs.

**`minimum` and `maximum` give a zero result a sign that depends on the layout and
the length.** `std::min(a, b)` is specified as `b < a ? b : a`, so on two zeros it
ignores the signs and returns whichever argument the expansion happens to name,
while the vectorized kernel uses `vminq_f32` and is sign aware. So
`torch.minimum` on `(-0.0, 0.0)` gives `-0.0` on a contiguous tensor and `0.0`
through a stride, on the same values. Filed as
[#193781](https://github.com/pytorch/pytorch/issues/193781) and fixed in
[#193851](https://github.com/pytorch/pytorch/pull/193851), and it is the 46
entries above, spread over `layout-invariance`, `size-invariance` and
`numpy-semantics` because all three axes can see it. MLX has the same bug from the
same cause in its own scalar path, above; it was invisible to this tool in both
libraries until the operand pairing was fixed.

Writing the test for that fix turned up the last two entries, and they are the
more interesting pair, because the oracle is what is wrong. **NumPy's `minimum`
and `maximum` are not sign aware on a pair of zeros, and which dtypes that
affects depends on the ISA.** They return whichever operand a comparison reached
first, so `np.minimum(float16(0.0), float16(-0.0))` is `0.0`. float32 and float64
substitute an instruction for that comparison: `FMIN`/`FMAX` on aarch64, which
are sign-aware, and `MINSS`/`MAXSS` on x86, which return the second operand and
are therefore order-dependent too. float16 has no override on any machine. IEEE
754, `std::fmin` and `vminq_f32` all give `-0.0` for the minimum in either
argument order, so a library that does the same is correct and the disagreement
belongs to NumPy. `numpy-semantics` scopes the `#193781` attribution to the
dtypes where NumPy can be trusted here, and the rest get an entry pointing at
NumPy. Getting that split wrong is how a tool starts filing an oracle's bug
against the library under test.

That split is **probed at import rather than written down**, and it is worth
saying why, because this project got it wrong first. It was a hardcoded
`("float32", "float64", "bfloat16")`, which passes on Apple Silicon and fails on
x86 Linux. CI caught it on the first run. A cross-platform differential tester
that hardcodes one platform's answer is the same mistake it exists to find.

**`fmod` and `remainder` return NaN where the scalar path returns the right
answer.** `Sleef_fmod` is documented as undefined once `abs(a / b)` reaches
`1e300` for double or `1e38` for float, and every architecture's
`Vectorized<T>::fmod` calls it directly, so a long array gets NaN and a short one
gets a finite value. A maintainer diagnosed this on
[#77742](https://github.com/pytorch/pytorch/issues/77742) in 2022 and posted the
`fmod` case himself; that issue was closed by a fix that only covered
`div(rounding_mode="floor")`, so `fmod_kernel` and `remainder_kernel` still call
Sleef unguarded. Filed as
[#193753](https://github.com/pytorch/pytorch/issues/193753).

**`floor_divide` on float16 and bfloat16 depends on the tensor's length, and the
vectorized result can exceed the quotient it is meant to be the floor of.**
`torch.floor_divide` on a length-2 bfloat16 tensor gives `560 // 3 == 187`, and
`187 * 3` is `561`. A length-1 tensor gives `186`. The scalar path promotes to
float32 through `std::fmod`, while `div_floor_floating_vec` computes in
`Vectorized<Half>` and rounds to 8 or 11 mantissa bits at every step. 39 of 4000
random float16 pairs and 65 of 4000 bfloat16 pairs disagree between the two;
float32 and float64 are clean, which is what pins the cause. Filed as
[#193754](https://github.com/pytorch/pytorch/issues/193754).

**`remainder` gives a zero result the sign of the dividend, not the divisor.**
`torch.remainder(torch.tensor([-1.0]), 1.0)` is `-0.0`, where the documented
contract is that the result has the sign of the divisor, and Python gives `0.0`.
The sign fixup is guarded on the remainder being nonzero, so the zero case is
never corrected. Filed as
[#193755](https://github.com/pytorch/pytorch/issues/193755). This is the same bug
class as MLX's [#4315](https://github.com/ml-explore/mlx/issues/4315) above,
found independently in a second library by the same check.

### JAX

Against jax 0.11.0 on CPU:

```
0 new findings

40 already accounted for in known.py
    24  https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
     6  https://docs.jax.dev/en/latest/faq.html
     6  https://github.com/jax-ml/jax/issues/40028
     3  https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sign.html
     1  https://github.com/jax-ml/jax/blob/main/jax/_src/public_test_util.py
```

JAX is the one that comes back empty, and it stays empty at a larger sample. It
did not at first: raising `--random` turned up an `exp` accuracy finding, which is
the subnormal flush described above and is the reason the `when` predicate exists.
Suppressing `exp` outright would have made this section clean by making the check
useless.

JAX has no layout axis to test. XLA materializes every view, so a strided input
reaches the kernel as a fresh contiguous buffer and the check would be comparing a
copy against itself. Its adapter declares only `contiguous`, which makes
`layout-invariance` skip rather than pass for free. Length still picks the kernel,
so `size-invariance` still applies, but for a different reason than elsewhere:
every `jnp` op is `jit` decorated, so each shape is a separate XLA compilation, and
JAX documents that `jit` changes the exact numerics of outputs. Those 6 are that.

The 24 are one root cause. XLA flushes subnormals to zero in arithmetic on CPU, so
`1 / DBL_MAX` is `0.0` and a subnormal operand multiplies as if it were zero.
Storage keeps subnormals, which is why this shows up as a NumPy mismatch on
ordinary inputs rather than as a bad input. Closed as expected twice and documented
in Sharp Bits.

One bug was filed out of this run.

**`remainder` gives a zero result the sign of the dividend, not the divisor.**
`jnp.remainder` documents that the result has the sign of `x2`, and all four of
`jnp.remainder([-1, -4, 0, 4], [1, 2, -3, -2])` disagree with that, with NumPy, and
with Python's `%`. `jnp.fmod` is right on the same inputs, which localizes it to the
floored fixup: it is guarded on the remainder being nonzero, so `lax.rem`'s sign
survives. Filed as [#40028](https://github.com/jax-ml/jax/issues/40028).

`floor_divide` is a sibling in the same file and is on that thread rather than a
second issue. A zero quotient gets the sign of the divisor alone, so `-0.0 // 3.0`
is `0.0` where Python and NumPy give `-0.0`. `_float_divmod` computes `x1 - mod`
before dividing, and for a negative zero dividend that is `-0.0 - -0.0`, which is
`+0.0`, so the sign is gone before the division happens.

The zero remainder sign is now the third independent instance of one bug class
found by one check, after MLX and torch. That is the argument for a differential
tester over a per library test suite: the bug is in the shape of the algorithm, not
in any one codebase.

### TensorFlow

Against tensorflow 2.21.0, eager, on CPU:

```
0 new findings

130 already accounted for in known.py
    65  recorded: tensorflow vectorized-vs-scalar transcendental rounding
    31  recorded: tensorflow CPU flushes subnormals to zero
    19  recorded: tensorflow min/max give a zero the wrong sign
     9  recorded: tensorflow remainder zero-sign
     3  recorded: tensorflow floor_divide at infinity
     2  https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
     1  https://github.com/ml-explore/mlx/pull/4003
```

This is the fourth library, and the point of adding it was to find out whether
the bug classes above are properties of these codebases or of the algorithms.
They are properties of the algorithms. Three of the four classes filed elsewhere
came straight back, on inputs nobody chose for them, with no new check code:

| what | also filed as |
|---|---|
| `minimum(0.0, -0.0)` is `+0.0`, `maximum(-0.0, 0.0)` is `-0.0` | pytorch [#193781](https://github.com/pytorch/pytorch/issues/193781), MLX above |
| `remainder(0.0, -1.0)` is `+0.0`, not the divisor's `-0.0` | jax [#40028](https://github.com/jax-ml/jax/issues/40028), pytorch [#193755](https://github.com/pytorch/pytorch/issues/193755), MLX [#4315](https://github.com/ml-explore/mlx/issues/4315) |
| `1.0 // -inf` is `-0.0`, not `-1.0` | MLX [#4317](https://github.com/ml-explore/mlx/issues/4317), pytorch MPS above |
| `floordiv` rounds up past its own floor and `floormod` does not follow | MLX [#4003](https://github.com/ml-explore/mlx/pull/4003) |

The last of those is the same arithmetic in the same dtype as the MLX entry:
`floordiv(2144.0, 358.0)` in bfloat16 gives `6` because `5.9888` rounds to
exactly `6.0`, while `floormod` gives `354`, the remainder for `5`. TensorFlow has
no `divmod`, so the adapter calls both ops and compares them, which is what makes
the inconsistency between the two visible at all. That was closed upstream in MLX
on performance grounds, so it is recorded as ruled-on rather than filed a fourth
time.

The `min`/`max` case is the one that shows why `size-invariance` is worth having
as a separate check. On float16 NumPy is not a usable oracle for a zero's sign, so
`numpy-semantics` has to stay quiet there. `size-invariance` does not need an
oracle, and TensorFlow contradicts itself: `maximum(-0.0, 0.0)` is `+0.0` in a
length-503 array and `-0.0` in a length-1 one. The vectorized kernel is the one
that gets IEEE right and the scalar tail is the one that does not, the same
split as MLX's `base_simd.h` and torch's `std::min`, in a third unrelated
codebase.

The 31 are one root cause: **TensorFlow's CPU kernels flush subnormals to zero.**
`tf.math.ceil` of the smallest float32 subnormal is `0.0` rather than `1.0`, and
`add(x, x)` on it is `0.0`. This survives `TF_ENABLE_ONEDNN_OPTS=0`, so it is the
kernels themselves rather than oneDNN dispatch. Same class as XLA's flush in JAX,
which is documented and was closed as expected twice, and it is recorded on the
same reasoning. It is scoped by a `when` predicate to inputs that are genuinely
subnormal in their own dtype, because a blanket suppression on those ops would
have swallowed the zero-sign and infinity bugs above, which happen on ordinary
inputs.

The 65 are about 1 ULP between a short array and a long one on the
transcendentals. Not a defect: no library rounds those correctly, and it is the
same allowance the torch and jax sections make.

Two things had to be scoped rather than tested, and both are recorded in the
adapter rather than left to look like passes. `tf.constant` materializes a fresh
contiguous buffer, so a strided NumPy view arrives contiguous and
`layout-invariance` would be comparing a copy against itself, exactly as in JAX;
the adapter declares only `contiguous` so the check skips. And there is one device
here, so `device-invariance` skips too. `logaddexp` has no TensorFlow equivalent
and `logical_not` takes a bool tensor rather than acting on a float, so both are
left out of the op table instead of being mapped to something close, which would
have tested a different op and reported coverage that does not exist.

Only the three sign and infinity classes are pinned in CI. The subnormal flush is
a property of the CPU kernels and the ULP gap is rounding, so either could
legitimately differ between x86 and aarch64, which is the same reasoning that
keeps the zero-new count out of the CI assertions.

### NumPy

NumPy is also wired up as a backend, and the sweep against it has to come back
almost empty. It is the oracle for `numpy-semantics`, so a disagreement there
would be the harness contradicting itself rather than finding anything.

The oracle-free checks are a different matter. This section used to say NumPy
has one code path per op, so any layout difference reported against it had to be
the harness inventing one. That was wrong, and it was wrong in the direction
that hides bugs. NumPy applies its transcendental SIMD loops to contiguous data
only, so a reversed view takes the scalar loop and can land a ULP away. Measured
on both architectures: 0 ULP on aarch64, 1 on x86 for most of the affected ops,
2 for `cosh` in float32 on NumPy 2.2. That is NumPy's accuracy budget for ops it
does not promise to round correctly, not a defect.

So `tests/test_checks.py` exempts `layout-invariance` on ten transcendentals and
nothing else, and rather than trusting the exemption it measures the gap and
fails over `MAX_LAYOUT_ULP`. A wrong answer cannot hide under a bound that
tight, and the untouched half of the assertion is machine independent: rounding
cannot turn a finite number into a NaN, so the two layouts must at least agree
on which values are finite.

One real NumPy bug came out of this work, though the sweep did not find it and
could not have. `maximum` and `minimum` returned a zero whose sign depended on
the dtype, the argument order and the CPU, so on x86 both
`np.maximum(0.0, -0.0)` and `np.minimum(0.0, -0.0)` gave `-0.0`. NumPy is the
oracle here, so nothing that compares against it can see this. It turned up by
carrying the signed-zero question from
[#4317](https://github.com/ml-explore/mlx/issues/4317) and
[#193781](https://github.com/pytorch/pytorch/issues/193781) into a third library
by hand. Fixed in
[numpy#32341](https://github.com/numpy/numpy/pull/32341), verified against real
aarch64 and x86 builds, where the same test class fails 25 times unpatched.

## Adding a library

An adapter in `backends.py` and a `known.py` list. `backends.py` is the whole
contract: dtypes, array construction, conversion to NumPy, broadcast, and the op
callables. `ops.py` has one spec list rather than one table per library, because
two tables drift and an op added for one library and forgotten for another
silently reduces coverage without failing anything.

Nothing in `checks.py` needs to change, and when it did, that was the useful
part. Adding torch turned up two assumptions about MLX that had been sitting
inside supposedly library-agnostic code: `Backend.layouts` exists because torch
has no negative strides, so `reversed` cannot be built there and comparing a copy
against itself would have been a silently vacuous check, and `TORCH_NAMES` maps
`equal` to `torch.eq` because `torch.equal` reduces a whole tensor to one bool,
which would have made every comparison check pass for free.

TensorFlow, the fourth, needed no change to `checks.py` at all, which is the
result the contract is for. Its adapter is 30 lines and its op table is a name
map, and the ops it does not have are left out rather than mapped to something
close. The only thing it changed was `known.py`, which learned to let one entry
name several checks, because a single TensorFlow cause was being reported through
two of them.
