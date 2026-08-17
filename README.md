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

# MLX is not a dependency; point at whatever build you want to test
PYTHONPATH=/path/to/mlx/python arraydiff
PYTHONPATH=/path/to/mlx/python arraydiff --only tan power --check layout-invariance
```

Exit status is 1 when there are findings, so it can gate CI.

```python
from arraydiff.runner import run
from arraydiff.known import partition
import mlx.core as mx

new, already_known = partition(run(mx))
```

## Status

The MLX op table is the only one implemented so far. Against MLX main at
`c2bcf47` plus the pending `remainder` fix below:

```
4 new findings
     3  numpy-semantics
     1  divmod-identity

134 already accounted for in known.py
   112  https://github.com/ml-explore/mlx/issues/4163
     8  https://github.com/ml-explore/mlx/issues/4162
     8  https://github.com/ml-explore/mlx/issues/4119
     3  https://github.com/ml-explore/mlx/issues/3644
     3  https://github.com/ml-explore/mlx/issues/4158
```

The 134 are the main evidence that the checks work: the tool rediscovers the
whole known numerical bug cluster without being told about any of it. They are
suppressed because each one is already filed or already declined, so re-reporting
them would be noise.

Three bugs came out of running this.

**`remainder` gives a zero result a path dependent sign.** Found by
`layout-invariance` with no oracle at all, which is what that check is for. It
reduces to a self contradiction: lane 8 of
`mx.remainder(mx.full((9,), -0.0), mx.full((9,), 3.0))` disagrees with lanes 0 to
7 on identical inputs, because the Accelerate SIMD body and the scalar residual
produce zeros of different signs and the floored fixup skips zero. Fixed across
all four backends, and the sweep against the fixed build is what the numbers above
are measured on.

**`divmod` computes the quotient by dividing a second time**, so it can disagree
with its own remainder. `divmod(mx.array([2144.0], mx.bfloat16), 358.0)` returns
`q = 6` where the floor is 5, because `5.9888` rounds to exactly `6.0` in
bfloat16 and `floor` cannot undo it. This is the one `divmod-identity` above.
Reported on the thread of the approved PR in this area, along with an integer
overflow in the same PR's `floor_divide` shortcut, where `a - remainder(a, b)`
leaves the dtype range and `mx.int8(120) // -27` comes back `4` instead of `-5`.

**`floor_divide` disagrees with both NumPy and Python at infinity.** `inf // -3.0`
gives `-inf` where both references give `nan`, and `1.0 // -inf` gives `-0.0`
where both give `-1.0`. That is the three `numpy-semantics` findings.

Adding a library means writing an op table like `ops.py` and a `known.py` list.
The checks themselves are library-agnostic.
