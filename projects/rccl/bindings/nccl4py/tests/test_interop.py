# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interop / zero-copy tests for the RCCL fork.

Two tests, both marked ``mpi``:

* PyTorch-ROCm DLPack: ``torch.Tensor`` -> ``Communicator.reduce`` ->
  assert ``data_ptr()`` is unchanged. Additionally verifies that the
  DLPack capsule reports ``device_type == 10`` (kROCM), reusing the
  kROCM helpers shipped with the cuda.core HIP shim.
* CuPy-ROCm CUDA Array Interface: ``cupy.ndarray`` ->
  ``Communicator.reduce`` -> assert ``arr.data.ptr`` is unchanged.

These cover the "zero-copy via pointer identity" guarantee; numerical
correctness of the underlying ``all_reduce`` is exercised by
``test_collectives.py``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch-ROCm required for the DLPack test")

import nccl  # noqa: E402  (also registers the cuda.core shim under sys.modules)
import nccl.core as nccl_core  # noqa: E402

from conftest import require_at_least_n_ranks, xfail_if_fewer_than_8_gpus  # noqa: E402

pytestmark = [
    pytest.mark.mpi,
    xfail_if_fewer_than_8_gpus(),
]

_N = 4096


def test_dlpack_zero_copy_pytorch(comm, rank_info, device):
    """PyTorch tensor: pointer identity preserved across an all_reduce, and
    the DLPack capsule reports kROCM (device_type == 10)."""
    require_at_least_n_ranks(rank_info, 2)
    rank, _ = rank_info

    t = torch.full((_N,), float(rank + 1), dtype=torch.float32, device=device)
    ptr_before = int(t.data_ptr())

    # In-place all_reduce: same buffer for send / recv.
    comm.reduce(t, t, nccl_core.SUM)
    torch.cuda.synchronize()

    ptr_after = int(t.data_ptr())
    assert (
        ptr_after == ptr_before
    ), f"data_ptr mutated by all_reduce: before={ptr_before:#x} after={ptr_after:#x}"

    # __dlpack_device__ is the cheap probe: it must report kROCM (10) on
    # ROCm regardless of which producer (torch / cuda.core) created the
    # capsule.
    dl_dev = t.__dlpack_device__()
    # PyTorch reports (DLDeviceType.ROCM=10, device_id) on a ROCm build.
    assert int(dl_dev[0]) == 10, f"expected device_type=10 (kROCM), got {dl_dev}"

    # Round-trip a capsule and re-parse it via the cuda.core shim's
    # parser to confirm the wire format also says 10.
    from nccl._hip_compat.cuda_core_shim._dlpack import parse_capsule

    capsule = t.__dlpack__()
    dl_tensor, _, _ = parse_capsule(capsule)
    assert int(dl_tensor.device.device_type) == 10


def test_cai_zero_copy_cupy(comm, rank_info, device):
    """CuPy array: pointer identity preserved across an all_reduce.

    The AC asks for ``cupy.ndarray`` consumed via the CUDA Array Interface,
    not for CuPy as the source of the data. We therefore allocate the
    storage with PyTorch (which does not need HIPRTC at runtime) and wrap
    it with ``cupy.asarray``, which is the canonical CAI consumer path:
    ``cupy.asarray`` reads ``__cuda_array_interface__`` and constructs an
    ``ndarray`` that aliases the existing device pointer with no kernel
    launch and no JIT compile. As a side effect this matches the AC's
    "via pointer identity" wording exactly: ``int(arr.data.ptr)`` must
    equal ``int(t.data_ptr())`` both before and after the collective.

    Going through ``cupy.full`` / ``cupy.zeros`` instead would JIT-compile
    a fill / copy kernel via HIPRTC, which fails on dev images that
    don't ship the matching libstdc++ headers (HIPRTC's clang cannot
    locate ``stddef.h``). That failure mode is unrelated to the binding
    and out of scope for this test.
    """
    require_at_least_n_ranks(rank_info, 2)
    cupy = pytest.importorskip("cupy", reason="cupy-ROCm required for the CAI test")

    rank, _ = rank_info
    t = torch.full((_N,), float(rank + 1), dtype=torch.float32, device=device)
    torch_ptr = int(t.data_ptr())

    with cupy.cuda.Device(device.index):
        arr = cupy.asarray(t)
        ptr_before = int(arr.data.ptr)
        assert ptr_before == torch_ptr, (
            "cupy.asarray must alias the source CAI pointer (zero-copy): "
            f"torch={torch_ptr:#x} cupy={ptr_before:#x}"
        )

        comm.reduce(arr, arr, nccl_core.SUM)
        cupy.cuda.runtime.deviceSynchronize()

        ptr_after = int(arr.data.ptr)
    assert (
        ptr_after == ptr_before
    ), f"cupy.data.ptr mutated by all_reduce: before={ptr_before:#x} after={ptr_after:#x}"
