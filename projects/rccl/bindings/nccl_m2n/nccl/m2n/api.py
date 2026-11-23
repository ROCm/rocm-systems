# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information.

"""High-level public entry points for NCCL M2N resharding."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from nccl.core.typing import NcclInvalid, NcclStreamSpec
from nccl._extensions.bindings import nccl_m2n as _m2n_bindings
from nccl._extensions._runtime import NATIVE_CALL_LOCK
from nccl.m2n.config import Config
from nccl.m2n.constants import MESH_NDIMS, REPLICATE, shard
from nccl.m2n.handle import (
    Handle,
    _reshard as _reshard_with_handle,
    _reshard_with_window as _reshard_with_window_with_handle,
)
from nccl.m2n.mesh import Mesh
from nccl.m2n.placement import Replicate, Shard
from nccl.m2n.tensor import DistTensor, normalize_dtype

if TYPE_CHECKING:
    import torch
    import torch.distributed.tensor._api as dtensor


def _finalize_default_handle() -> None:
    with NATIVE_CALL_LOCK:
        _m2n_bindings.finalize(0)


def init(config: Config | None = None) -> Handle:
    """Create and return an explicit M2N handle.

    Keep the returned handle alive, pass it to ``reshard(handle=...)``, and
    release it explicitly with :func:`finalize`, :meth:`Handle.destroy`, or a
    handle context manager. Bare ``reshard()`` calls use the separate native
    default handle without this configuration.
    """

    return Handle.create(config)


def finalize(handle: Handle | None = None) -> None:
    """Finalize an explicit handle, or the native default when omitted."""

    if handle is None:
        _finalize_default_handle()
    elif isinstance(handle, Handle):
        handle.destroy()
    else:
        raise TypeError(f"handle must be nccl.m2n.Handle or None, got {type(handle).__name__}")


def group_start() -> None:
    """Begin recording M2N reshard calls on the current host thread."""

    _m2n_bindings.group_start()


def group_end() -> None:
    """Close one M2N group level and submit the outermost group."""

    _m2n_bindings.group_end()


def group_abort() -> None:
    """Discard every active M2N group level on the current host thread."""

    _m2n_bindings.group_abort()


@contextmanager
def group() -> Iterator[None]:
    """Record reshard calls and submit them when the context exits."""

    group_start()
    try:
        yield
    except BaseException:
        group_abort()
        raise
    else:
        group_end()


def _is_dtensor_like(tensor: object) -> bool:
    return (
        hasattr(tensor, "device_mesh")
        and hasattr(tensor, "placements")
        and callable(getattr(tensor, "to_local", None))
    )


def _has_cuda_array_interface(buffer: object) -> bool:
    return getattr(buffer, "__cuda_array_interface__", None) is not None


def _has_data_ptr_buffer(buffer: object) -> bool:
    return (
        callable(getattr(buffer, "data_ptr", None))
        and hasattr(buffer, "shape")
        and hasattr(buffer, "dtype")
    )


def _check_buffer_like(buffer: object | None, name: str, api: str) -> object | None:
    if (
        buffer is None
        or isinstance(buffer, int)
        or _has_data_ptr_buffer(buffer)
        or _has_cuda_array_interface(buffer)
    ):
        return buffer
    raise TypeError(
        f"{api} expects {name} to be None, a raw device pointer integer, "
        "a CUDA Array Interface object, or an object with data_ptr(), shape, "
        f"and dtype; got {type(buffer).__name__}"
    )


def _check_dtensor_input(value: object | None, name: str, api: str) -> object | None:
    if value is None or _is_dtensor_like(value):
        return value
    return _check_buffer_like(value, name, api)


def _check_local_shape(shape: object, name: str) -> tuple[int, ...]:
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise TypeError(f"{name}_local_shape must be a sequence of dimensions")
    shape = tuple(int(dim) for dim in shape)
    if not shape:
        raise ValueError(f"{name}_local_shape must describe at least one dimension")
    if any(dim < 0 for dim in shape):
        raise ValueError(f"{name}_local_shape must be non-negative, got {shape}")
    return shape


def _local_shape(
    buffer: object,
    explicit_shape: Sequence[int] | None,
    name: str,
) -> tuple[int, ...]:
    if explicit_shape is not None:
        return _check_local_shape(explicit_shape, name)
    if isinstance(buffer, int):
        raise ValueError(f"{name}_local_shape is required for raw pointer buffers")
    if _has_data_ptr_buffer(buffer):
        return _check_local_shape(getattr(buffer, "shape"), name)

    cai = getattr(buffer, "__cuda_array_interface__", None)
    if cai is not None:
        return _check_local_shape(cai["shape"], name)

    raise TypeError(f"cannot infer {name}_local_shape from {type(buffer).__name__}")


def _dtype(buffer: object, explicit_dtype: object | None, name: str) -> object:
    if explicit_dtype is not None:
        return normalize_dtype(explicit_dtype)
    if isinstance(buffer, int):
        raise ValueError(f"{name}_dtype is required for raw pointer buffers")
    if _has_data_ptr_buffer(buffer):
        return normalize_dtype(getattr(buffer, "dtype"))

    cai = getattr(buffer, "__cuda_array_interface__", None)
    if cai is not None:
        typestr = cai.get("typestr")
        if typestr is None:
            raise ValueError(
                "__cuda_array_interface__ is missing typestr for "
                f"{type(buffer).__name__}"
            )
        return normalize_dtype(typestr)

    raise TypeError(f"cannot infer {name}_dtype from {type(buffer).__name__}")


def _check_arg(value: object | None, name: str, api: str) -> object:
    if value is None:
        raise ValueError(f"{name} is needed for nccl.m2n.{api}")
    return value


def _mesh_ndim(mesh: object) -> int | None:
    if isinstance(mesh, Mesh):
        # Preserve the existing permissive placement input for the padded
        # size-1 form, including native 1-D meshes' compatibility view.
        active_dims = mesh.dims[: mesh.ndims]
        return None if all(dim == 1 for dim in active_dims[1:]) else mesh.ndims
    return None


def _check_mesh(mesh: object | None, name: str, api: str) -> Mesh:
    value = _check_arg(mesh, name, api)
    if isinstance(value, Mesh):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return Mesh.from_ranks(value)
    raise TypeError(f"{name} must be a Mesh or 1-D/2-D rank sequence")


def _placement_to_m2n(placement: object, tensor_ndim: int, name: str) -> int:
    if isinstance(placement, Replicate):
        return REPLICATE
    if isinstance(placement, Shard):
        dim = int(placement.dim)
    elif isinstance(placement, bool):
        raise TypeError("boolean placement is invalid; use Replicate() or Shard(dim)")
    elif isinstance(placement, int):
        dim = int(placement)
        if dim < 0:
            raise ValueError(
                f"{name}_placements contains negative integer {dim}; "
                "use Replicate() for replication"
            )
    else:
        cls_name = placement.__class__.__name__
        if cls_name == "Replicate":
            return REPLICATE
        if cls_name == "Partial":
            raise TypeError("NCCL Reshard does not support Partial placements")
        if cls_name != "Shard":
            raise TypeError(
                f"Unsupported placement for NCCL Reshard: {placement!r}"
            )
        dim = int(placement.dim)

    if dim < 0:
        dim += tensor_ndim
    if dim < 0 or dim >= tensor_ndim:
        raise ValueError(
            f"{name}_placements contains Shard({dim}), but tensor rank is {tensor_ndim}"
        )
    return shard(dim)


def _placements_to_m2n(
    placements: Sequence[object] | None,
    mesh_ndim: int | None,
    tensor_ndim: int,
    name: str,
    api: str,
) -> tuple[int, ...]:
    values = _check_arg(placements, f"{name}_placements", api)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name}_placements must be a sequence")
    if not 1 <= len(values) <= MESH_NDIMS:
        raise ValueError(
            f"{name}_placements must have length 1..{MESH_NDIMS}, got {len(values)}"
        )
    if mesh_ndim is not None and len(values) != mesh_ndim:
        raise ValueError(
            f"{name}_placements length ({len(values)}) must match "
            f"{name}_mesh rank ({mesh_ndim})"
        )

    normalized = tuple(
        _placement_to_m2n(placement, tensor_ndim, name) for placement in values
    )
    return normalized + (REPLICATE,) * (MESH_NDIMS - len(normalized))


def _placement_objects(values: tuple[int, ...]) -> tuple[object, ...]:
    return tuple(
        Replicate() if value == REPLICATE else Shard(value) for value in values
    )


def _shard_factors(
    mesh: Mesh,
    placements: tuple[int, ...],
    tensor_ndim: int,
    name: str,
) -> tuple[int, ...]:
    factors = [1] * tensor_ndim
    for axis, placement in enumerate(placements):
        if placement == REPLICATE:
            continue
        if placement < 0 or placement >= tensor_ndim:
            raise ValueError(
                f"{name}_placements contains Shard({placement}), "
                f"but tensor rank is {tensor_ndim}"
            )
        factors[placement] *= int(mesh.dims[axis])
    return tuple(factors)


def _global_shape_from_local(
    local_shape: tuple[int, ...],
    mesh: Mesh,
    placements: tuple[int, ...],
    name: str,
) -> tuple[int, ...]:
    factors = _shard_factors(mesh, placements, len(local_shape), name)
    return tuple(dim * factor for dim, factor in zip(local_shape, factors))


def _local_shape_from_global(
    global_shape: tuple[int, ...],
    mesh: Mesh,
    placements: tuple[int, ...],
    name: str,
    api: str,
) -> tuple[int, ...]:
    factors = _shard_factors(mesh, placements, len(global_shape), name)
    local_shape = []
    for dim_index, (dim, factor) in enumerate(zip(global_shape, factors)):
        if dim % factor != 0:
            raise ValueError(
                f"{api} cannot derive {name}_local_shape: "
                f"global_shape[{dim_index}]={dim} is not divisible by "
                f"shard factor {factor}"
            )
        local_shape.append(dim // factor)
    return tuple(local_shape)


def _resolve_global_shape(
    candidates: Sequence[tuple[str, tuple[int, ...]]],
    api: str,
) -> tuple[int, ...]:
    if not candidates:
        raise AssertionError("unreachable")
    source_name, global_shape = candidates[0]
    for name, shape in candidates[1:]:
        if shape != global_shape:
            raise ValueError(
                f"{api} expects src and dst layouts to describe the same "
                f"global shape; {source_name} gives {global_shape}, "
                f"but {name} gives {shape}"
            )
    return global_shape


def _dtensor_local_tensor(value: object) -> object:
    to_local = getattr(value, "to_local", None)
    if callable(to_local):
        return to_local()
    raise TypeError("DTensor input must expose to_local()")


def _dtensor_global_shape(value: object) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(value, "shape"))


def _resolve_buffer_role(
    value: object | None,
    mesh: object | None,
    placements: Sequence[object] | None,
    local_shape: Sequence[int] | None,
    dtype: object | None,
    name: str,
    api: str,
) -> tuple[
    object | None,
    object | None,
    int | None,
    Sequence[object] | None,
    tuple[int, ...] | None,
    object | None,
]:
    checked = _check_buffer_like(value, name, api)
    mesh_ndim = _mesh_ndim(mesh) if mesh is not None else None
    if checked is None:
        shape = _check_local_shape(local_shape, name) if local_shape is not None else None
        dtype_value = normalize_dtype(dtype) if dtype is not None else None
        return None, mesh, mesh_ndim, placements, shape, dtype_value
    return (
        checked,
        mesh,
        mesh_ndim,
        placements,
        _local_shape(checked, local_shape, name),
        _dtype(checked, dtype, name),
    )


def _resolve_dtensor_role(
    value: object | None,
    mesh: object | None,
    placements: Sequence[object] | None,
    name: str,
    api: str,
) -> tuple[
    object | None,
    object | None,
    int | None,
    Sequence[object] | None,
    tuple[int, ...] | None,
    object | None,
    tuple[int, ...] | None,
]:
    checked = _check_dtensor_input(value, name, api)
    if checked is None:
        mesh_ndim = _mesh_ndim(mesh) if mesh is not None else None
        return None, mesh, mesh_ndim, placements, None, None, None
    if _is_dtensor_like(checked):
        local_tensor = _check_buffer_like(
            _dtensor_local_tensor(checked), f"{name} local tensor", api
        )
        if local_tensor is None:
            raise TypeError("DTensor local tensor must not be None")
        device_mesh = getattr(checked, "device_mesh")
        mesh_arg = mesh if mesh is not None else Mesh.from_device_mesh(device_mesh)
        mesh_ndim = _mesh_ndim(mesh_arg)
        placement_arg = (
            placements if placements is not None else getattr(checked, "placements")
        )
        return (
            local_tensor,
            mesh_arg,
            mesh_ndim,
            placement_arg,
            _local_shape(local_tensor, None, name),
            _dtype(local_tensor, None, name),
            _dtensor_global_shape(checked),
        )

    buffer, mesh_arg, mesh_ndim, placement_arg, shape, dtype = _resolve_buffer_role(
        checked, mesh, placements, None, None, name, api
    )
    return buffer, mesh_arg, mesh_ndim, placement_arg, shape, dtype, None


def _resolve_common_dtype(
    src_dtype: object | None,
    dst_dtype: object | None,
    api: str,
) -> object:
    if src_dtype is not None and dst_dtype is not None and src_dtype != dst_dtype:
        raise NcclInvalid(
            f"{api} expects src and dst to have the same dtype, "
            f"got {src_dtype!r} and {dst_dtype!r}"
        )
    dtype = src_dtype if src_dtype is not None else dst_dtype
    if dtype is None:
        raise ValueError(
            f"{api} requires src_dtype or dst_dtype when both src and dst are None"
        )
    return dtype


def _reshard(
    src: object | None,
    dst: object | None,
    comm: Any,
    stream: NcclStreamSpec | None = None,
    *,
    src_mesh: object | None = None,
    src_placements: Sequence[object] | None = None,
    src_local_shape: Sequence[int] | None = None,
    src_dtype: object | None = None,
    dst_mesh: object | None = None,
    dst_placements: Sequence[object] | None = None,
    dst_local_shape: Sequence[int] | None = None,
    dst_dtype: object | None = None,
    handle: Handle | None = None,
    window: object | None = None,
    allow_dtensor: bool = False,
    api: str,
) -> None:
    if allow_dtensor:
        (
            src_tensor,
            src_mesh_arg,
            src_mesh_ndim,
            src_placements_arg,
            src_shape,
            src_dtype_value,
            src_global_shape,
        ) = _resolve_dtensor_role(src, src_mesh, src_placements, "src", api)
        (
            dst_tensor,
            dst_mesh_arg,
            dst_mesh_ndim,
            dst_placements_arg,
            dst_shape,
            dst_dtype_value,
            dst_global_shape,
        ) = _resolve_dtensor_role(dst, dst_mesh, dst_placements, "dst", api)
    else:
        (
            src_tensor,
            src_mesh_arg,
            src_mesh_ndim,
            src_placements_arg,
            src_shape,
            src_dtype_value,
        ) = _resolve_buffer_role(
            src, src_mesh, src_placements, src_local_shape, src_dtype, "src", api
        )
        (
            dst_tensor,
            dst_mesh_arg,
            dst_mesh_ndim,
            dst_placements_arg,
            dst_shape,
            dst_dtype_value,
        ) = _resolve_buffer_role(
            dst, dst_mesh, dst_placements, dst_local_shape, dst_dtype, "dst", api
        )
        src_global_shape = None
        dst_global_shape = None

    if (
        src_shape is not None
        and dst_shape is not None
        and len(src_shape) != len(dst_shape)
    ):
        raise ValueError(
            f"src_local_shape rank {len(src_shape)} must match "
            f"dst_local_shape rank {len(dst_shape)}"
        )

    role_shape = src_shape if src_shape is not None else dst_shape
    if role_shape is None:
        raise ValueError(
            f"{api} requires src_local_shape or dst_local_shape when both "
            "src and dst are None"
        )
    tensor_ndim = len(role_shape)
    dtype = _resolve_common_dtype(src_dtype_value, dst_dtype_value, api)

    src_mesh_value = _check_mesh(src_mesh_arg, "src_mesh", api)
    dst_mesh_value = _check_mesh(dst_mesh_arg, "dst_mesh", api)
    src_mesh_ndim = (
        src_mesh_ndim if src_mesh_ndim is not None else _mesh_ndim(src_mesh_value)
    )
    dst_mesh_ndim = (
        dst_mesh_ndim if dst_mesh_ndim is not None else _mesh_ndim(dst_mesh_value)
    )
    src_placement_values = _placements_to_m2n(
        src_placements_arg, src_mesh_ndim, tensor_ndim, "src", api
    )
    dst_placement_values = _placements_to_m2n(
        dst_placements_arg, dst_mesh_ndim, tensor_ndim, "dst", api
    )

    has_dtensor_shape = src_global_shape is not None or dst_global_shape is not None
    global_shape_candidates = []
    if has_dtensor_shape:
        if src_global_shape is not None:
            global_shape_candidates.append(("src DTensor", src_global_shape))
        if dst_global_shape is not None:
            global_shape_candidates.append(("dst DTensor", dst_global_shape))
        if src_shape is not None:
            global_shape_candidates.append(
                (
                    "src local shape/layout",
                    _global_shape_from_local(
                        src_shape, src_mesh_value, src_placement_values, "src"
                    ),
                )
            )
        if dst_shape is not None:
            global_shape_candidates.append(
                (
                    "dst local shape/layout",
                    _global_shape_from_local(
                        dst_shape, dst_mesh_value, dst_placement_values, "dst"
                    ),
                )
            )
    elif src_shape is None:
        global_shape_candidates.append(
            (
                "dst local shape/layout",
                _global_shape_from_local(
                    dst_shape, dst_mesh_value, dst_placement_values, "dst"
                ),
            )
        )
    elif dst_shape is None:
        global_shape_candidates.append(
            (
                "src local shape/layout",
                _global_shape_from_local(
                    src_shape, src_mesh_value, src_placement_values, "src"
                ),
            )
        )
    global_shape = (
        _resolve_global_shape(global_shape_candidates, api)
        if global_shape_candidates
        else None
    )

    src_desc_shape = (
        src_shape
        if src_shape is not None
        else _local_shape_from_global(
            global_shape, src_mesh_value, src_placement_values, "src", api
        )
    )
    dst_desc_shape = (
        dst_shape
        if dst_shape is not None
        else _local_shape_from_global(
            global_shape, dst_mesh_value, dst_placement_values, "dst", api
        )
    )

    src_desc = DistTensor(
        src_tensor,
        local_shape=src_desc_shape,
        dtype=dtype,
        mesh=src_mesh_value,
        placements=_placement_objects(src_placement_values),
    )
    dst_desc = DistTensor(
        dst_tensor,
        local_shape=dst_desc_shape,
        dtype=dtype,
        mesh=dst_mesh_value,
        placements=_placement_objects(dst_placement_values),
    )

    if handle is None:
        if window is None:
            with NATIVE_CALL_LOCK:
                _reshard_with_handle(0, comm, src_desc, dst_desc, stream)
        else:
            _reshard_with_window_with_handle(
                0, comm, window, src_desc, dst_desc, stream
            )
    elif window is None:
        handle.reshard(comm, src_desc, dst_desc, stream=stream)
    else:
        handle.reshard_with_window(comm, window, src_desc, dst_desc, stream=stream)


def reshard(
    src: object | None,
    dst: object | None,
    comm: Any,
    stream: NcclStreamSpec | None = None,
    *,
    src_mesh: object | None = None,
    src_placements: Sequence[object] | None = None,
    src_local_shape: Sequence[int] | None = None,
    src_dtype: object | None = None,
    dst_mesh: object | None = None,
    dst_placements: Sequence[object] | None = None,
    dst_local_shape: Sequence[int] | None = None,
    dst_dtype: object | None = None,
    handle: Handle | None = None,
) -> None:
    """Reshard framework-neutral CUDA buffers using explicit metadata.

    See :meth:`Handle.reshard` for the asynchronous buffer-lifetime contract.
    """

    return _reshard(
        src,
        dst,
        comm,
        stream,
        src_mesh=src_mesh,
        src_placements=src_placements,
        src_local_shape=src_local_shape,
        src_dtype=src_dtype,
        dst_mesh=dst_mesh,
        dst_placements=dst_placements,
        dst_local_shape=dst_local_shape,
        dst_dtype=dst_dtype,
        handle=handle,
        api="reshard",
    )


def reshard_with_window(
    src: object | None,
    dst: object | None,
    comm: Any,
    window: object,
    stream: NcclStreamSpec | None = None,
    *,
    src_mesh: object | None = None,
    src_placements: Sequence[object] | None = None,
    src_local_shape: Sequence[int] | None = None,
    src_dtype: object | None = None,
    dst_mesh: object | None = None,
    dst_placements: Sequence[object] | None = None,
    dst_local_shape: Sequence[int] | None = None,
    dst_dtype: object | None = None,
    handle: Handle | None = None,
) -> None:
    """Reshard through a caller-registered NCCL window.

    See :meth:`Handle.reshard_with_window` for the buffer-lifetime contract.
    """

    if window is None:
        raise TypeError("window must be a registered NCCL window")

    return _reshard(
        src,
        dst,
        comm,
        stream,
        src_mesh=src_mesh,
        src_placements=src_placements,
        src_local_shape=src_local_shape,
        src_dtype=src_dtype,
        dst_mesh=dst_mesh,
        dst_placements=dst_placements,
        dst_local_shape=dst_local_shape,
        dst_dtype=dst_dtype,
        handle=handle,
        window=window,
        api="reshard_with_window",
    )


def xdtensor_reshard(
    src: torch.Tensor | dtensor.DTensor | None,
    dst: torch.Tensor | dtensor.DTensor | None,
    comm: Any,
    stream: torch.cuda.Stream | None = None,
    *,
    src_mesh: object | None = None,
    src_placements: Sequence[object] | None = None,
    dst_mesh: object | None = None,
    dst_placements: Sequence[object] | None = None,
    handle: Handle | None = None,
) -> None:
    """Reshard DTensors, or tensors when explicit mesh metadata is supplied.

    See :meth:`Handle.reshard` for the asynchronous buffer-lifetime contract.
    """

    return _reshard(
        src,
        dst,
        comm,
        stream,
        src_mesh=src_mesh,
        src_placements=src_placements,
        dst_mesh=dst_mesh,
        dst_placements=dst_placements,
        handle=handle,
        allow_dtensor=True,
        api="xdtensor_reshard",
    )


__all__ = [
    "finalize",
    "group",
    "group_abort",
    "group_end",
    "group_start",
    "init",
    "reshard",
    "reshard_with_window",
    "xdtensor_reshard",
]
