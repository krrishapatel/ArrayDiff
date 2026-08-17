"""What a library has to provide in order to be tested.

The checks are written against this and nothing else. Keeping the surface this
small is the point: as soon as a check reaches for a library-specific call it
stops being a differential tester and becomes a test for one library.

Adding a library is an adapter here plus an op table in `ops.py`. Nothing in
`checks.py` should need to change, and if it does, that is a sign the check was
encoding an assumption about MLX rather than about arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Backend:
    name: str
    # Dtype names to the library's own dtype objects. A missing name means the
    # library does not have that dtype, and it is skipped rather than faked.
    dtypes: dict[str, object]
    array: Callable  # (np.ndarray, dtype | None) -> native array
    to_numpy: Callable  # native array -> np.ndarray
    broadcast_to: Callable  # (native array, shape) -> native array
    ops: dict = field(default_factory=dict)
    # `divmod` as a single call returning (quotient, remainder), or None when the
    # library has no such op. Torch has none, so its adapter composes one.
    divmod: Callable | None = None
    # Forces a lazy graph so timing-independent results are actually computed.
    # A no-op on eager libraries.
    evaluate: Callable = lambda *_: None
    # Layouts this backend can express, or None for all of them. Torch has no
    # negative strides, so a reversed view does not exist there at all.
    layouts: tuple[str, ...] | None = None
    # Devices to cross-check, most trusted first. Fewer than two means there is
    # nothing to compare and `device-invariance` skips. MLX has one unified
    # device, and JAX here is CPU only, so torch is currently the only backend
    # with a second entry.
    devices: tuple[str, ...] = ()
    to_device: Callable | None = None  # (native array, device name) -> array

    def dtype(self, name):
        return self.dtypes.get(name)

    def has(self, name) -> bool:
        return name in self.dtypes


def numpy_backend() -> Backend:
    """NumPy as a backend, so the checks can be run against the thing they use
    as an oracle. Nothing should ever be reported here, which is what makes it
    useful: a finding against this backend means the check is wrong."""
    from .ops import build_ops

    dtypes = {
        n: getattr(np, n)
        for n in (
            "float16", "float32", "float64",
            "int8", "int16", "int32", "int64", "uint8", "uint32",
        )
    }
    return Backend(
        name="numpy",
        dtypes=dtypes,
        array=lambda v, dtype=None: np.asarray(v, dtype=dtype),
        to_numpy=np.asarray,
        broadcast_to=np.broadcast_to,
        ops=build_ops(np),
        divmod=np.divmod,
    )


def mlx_backend(mx) -> Backend:
    from .ops import build_ops

    dtypes = {
        n: getattr(mx, n)
        for n in (
            "float16", "float32", "float64", "bfloat16",
            "int8", "int16", "int32", "int64", "uint8", "uint32",
        )
        if hasattr(mx, n)
    }

    def to_numpy(a):
        # NumPy has no bfloat16, so widen before handing it over. float32 holds
        # every bfloat16 value exactly, so this loses nothing.
        if a.dtype == mx.bfloat16:
            a = a.astype(mx.float32)
        return np.asarray(a)

    return Backend(
        name="mlx",
        dtypes=dtypes,
        array=lambda v, dtype=None: mx.array(v, dtype=dtype),
        to_numpy=to_numpy,
        broadcast_to=mx.broadcast_to,
        ops=build_ops(mx),
        divmod=mx.divmod,
        evaluate=mx.eval,
    )


def jax_backend(jax) -> Backend:
    """JAX.

    Layout invariance does not apply here, and saying so is the point. XLA
    materializes every view, so a strided input reaches the kernel as a fresh
    contiguous buffer and the check would be comparing a copy against itself.
    Only `contiguous` is declared, which makes the check skip rather than pass
    for free. Length still selects the kernel, so size invariance does apply.
    """
    import jax.numpy as jnp

    from .ops import build_jax_ops

    # float64 is opt-in in JAX. Without this a float64 request silently gets
    # float32, so every float64 finding would be mislabelled.
    jax.config.update("jax_enable_x64", True)

    dtypes = {
        n: getattr(jnp, n)
        for n in (
            "float16", "float32", "float64", "bfloat16",
            "int8", "int16", "int32", "int64", "uint8", "uint32",
        )
        if hasattr(jnp, n)
    }

    def to_numpy(a):
        if a.dtype == jnp.bfloat16:
            a = a.astype(jnp.float32)
        return np.asarray(a)

    return Backend(
        name="jax",
        dtypes=dtypes,
        array=lambda v, dtype=None: jnp.array(v, dtype=dtype),
        to_numpy=to_numpy,
        broadcast_to=jnp.broadcast_to,
        ops=build_jax_ops(jax),
        divmod=jnp.divmod,
        # JAX dispatches asynchronously, so without this a check could compare
        # results that have not been computed yet.
        evaluate=lambda *xs: jax.block_until_ready(list(xs)),
        layouts=("contiguous",),
    )


def torch_backend(torch) -> Backend:
    """Torch, on every device it can reach.

    The device list is what makes `device-invariance` run here. MPS is added only
    when it is actually available, so the check skips on a machine without it
    rather than reporting the whole op table as broken.
    """
    from .ops import build_torch_ops

    dtypes = {
        n: getattr(torch, n)
        for n in (
            "float16", "float32", "float64", "bfloat16",
            "int8", "int16", "int32", "int64", "uint8",
        )
        if hasattr(torch, n)
    }

    def array(v, dtype=None):
        # from_numpy would alias the caller's buffer, and the layout helpers
        # reuse theirs, so copy.
        return torch.tensor(np.asarray(v), dtype=dtype)

    def to_numpy(a):
        if a.dtype == torch.bfloat16:
            a = a.to(torch.float32)
        return a.detach().cpu().numpy()

    def divmod_(a, b):
        # Torch has no divmod. Composing it from the two ops is not a shortcut
        # here, it is the check: if the quotient and the remainder disagree,
        # that is exactly the bug divmod-identity looks for.
        return (
            torch.div(a, b, rounding_mode="floor"),
            torch.remainder(a, b),
        )

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    def to_device(a, device):
        return a.to(device)

    return Backend(
        name="torch",
        dtypes=dtypes,
        array=array,
        to_numpy=to_numpy,
        broadcast_to=torch.broadcast_to,
        ops=build_torch_ops(torch),
        divmod=divmod_,
        devices=tuple(devices),
        to_device=to_device,
        # No `evaluate` is needed even for MPS: every check reads its results
        # through to_numpy, and the .cpu() there synchronizes.
        # Torch has no negative strides, so a reversed view cannot be built.
        # torch.flip copies, which would make the layout contiguous and turn
        # the check into a comparison of a value against itself.
        layouts=tuple(
            n
            for n in (
                "contiguous", "strided", "strided3", "transposed",
                "column", "offset", "broadcast",
            )
        ),
    )
