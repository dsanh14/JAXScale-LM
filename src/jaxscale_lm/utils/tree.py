"""PyTree helpers: parameter counting, byte sizing, dtype mapping."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxscale_lm.config import DTypeName
from jaxscale_lm.types import PyTree

_DTYPES: dict[DTypeName, jnp.dtype] = {
    "float32": jnp.dtype(jnp.float32),
    "bfloat16": jnp.dtype(jnp.bfloat16),
    "float16": jnp.dtype(jnp.float16),
}


def dtype_from_name(name: DTypeName) -> jnp.dtype:
    """Map a config dtype string to a jnp dtype."""
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(_DTYPES)}") from None


def count_params(tree: PyTree) -> int:
    """Total number of scalar elements across all array leaves."""
    leaves = jax.tree.leaves(tree)
    return sum(int(np.prod(leaf.shape)) for leaf in leaves if hasattr(leaf, "shape"))


def tree_bytes(tree: PyTree) -> int:
    """Total in-memory size of array leaves in bytes."""
    leaves = jax.tree.leaves(tree)
    return sum(
        int(np.prod(leaf.shape)) * leaf.dtype.itemsize
        for leaf in leaves
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype")
    )


def tree_allclose(a: PyTree, b: PyTree, *, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    """Whether two pytrees match in structure and values."""
    a_leaves, a_def = jax.tree.flatten(a)
    b_leaves, b_def = jax.tree.flatten(b)
    if a_def != b_def:
        return False
    return all(
        np.allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol)
        for x, y in zip(a_leaves, b_leaves, strict=True)
    )
