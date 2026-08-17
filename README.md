# arraydiff

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
see this, since every layout it builds is full length. This is the check that
found the three torch bugs below.

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

## Checks

| check | needs an oracle | what it means |
|---|---|---|
| `layout-invariance` | no | same values, different memory layout, different result |
| `size-invariance` | no | same values, shorter array, different result |
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
```

Exit status is 1 when there are findings, so it can gate CI.

```python
from arraydiff.backends import mlx_backend
from arraydiff.known import known_for, partition
from arraydiff.runner import run
import mlx.core as mx

be = mlx_backend(mx)
new, already_known = partition(run(be), known_for(be.name))
```

## Status

Three libraries. All three come back clean, which is the goal state rather than an
empty run: everything either check finds is fixed, filed, or ruled on.

### MLX

Against MLX main at `c2bcf47` plus the pending `remainder` fix below:

```
0 new findings

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

Two bugs were filed out of running this, and one rediscovery is recorded.

**`remainder` gives a zero result a path dependent sign.** Found by
`layout-invariance` with no oracle at all, which is what that check is for. It
reduces to a self contradiction: lane 8 of
`mx.remainder(mx.full((9,), -0.0), mx.full((9,), 3.0))` disagrees with lanes 0 to
7 on identical inputs, because the Accelerate SIMD body and the scalar residual
produce zeros of different signs and the floored fixup skips zero. Filed as
[#4315](https://github.com/ml-explore/mlx/issues/4315) and fixed across all four
backends in [#4316](https://github.com/ml-explore/mlx/pull/4316); the sweep
against that build is what the numbers above are measured on.

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

Against torch 2.13.0 on CPU:

```
0 new findings

52 already accounted for in known.py
    32  https://pytorch.org/docs/stable/notes/numerical_accuracy.html
    12  https://github.com/pytorch/pytorch/issues/193753
     3  https://github.com/pytorch/pytorch/issues/193755
     3  https://github.com/pytorch/pytorch/issues/187295
     2  https://github.com/pytorch/pytorch/issues/193754
```

The 32 are the Sleef-versus-libm last-bit differences, which torch documents as
allowed and which are therefore not findings. The rest are three bugs filed out
of this run.

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

38 already accounted for in known.py
    22  https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
     6  https://docs.jax.dev/en/latest/faq.html
     6  https://github.com/jax-ml/jax/issues/40028
     3  https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sign.html
     1  https://github.com/jax-ml/jax/blob/main/jax/_src/public_test_util.py
```

JAX has no layout axis to test. XLA materializes every view, so a strided input
reaches the kernel as a fresh contiguous buffer and the check would be comparing a
copy against itself. Its adapter declares only `contiguous`, which makes
`layout-invariance` skip rather than pass for free. Length still picks the kernel,
so `size-invariance` still applies, but for a different reason than elsewhere:
every `jnp` op is `jit` decorated, so each shape is a separate XLA compilation, and
JAX documents that `jit` changes the exact numerics of outputs. Those 6 are that.

The 22 are one root cause. XLA flushes subnormals to zero in arithmetic on CPU, so
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

### NumPy

NumPy is also wired up as a backend, and the sweep against it has to come back
empty. It is the oracle for `numpy-semantics`, and it has one code path per op,
so any layout or length difference reported against it would be the harness
inventing one rather than finding one. `tests/test_checks.py` asserts this.

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
