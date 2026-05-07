
import rocshmem4py
from rocshmem4py import (
    rocshmem_get_device_ctx,
    rocshmem_query_thread,
    rocshmem_sync_all,
    rocshmem_sync_all_on_stream,
)
from conftest import requires_torch


def test_sync_all_single_pe():
    """sync_all is callable on a single PE (degenerate but legal)."""
    rocshmem_sync_all()


@requires_torch
def test_sync_all_on_stream():
    import torch

    stream = torch.cuda.current_stream()
    rocshmem_sync_all_on_stream(stream.cuda_stream)
    torch.cuda.synchronize()


def test_context_introspection_bindings():
    assert isinstance(rocshmem_query_thread(), int)
    assert isinstance(rocshmem_get_device_ctx(), int)
