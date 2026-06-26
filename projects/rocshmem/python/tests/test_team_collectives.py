"""Team-scoped collective binding tests."""

from conftest import requires_multi_pe, requires_torch

import rocshmem4py
from rocshmem4py.interop import torch as rshmem_torch

pytestmark = [requires_torch, requires_multi_pe]


def test_world_alltoall_torch_wrapper():
    import torch

    me = rocshmem4py.rocshmem_my_pe()
    n = rocshmem4py.rocshmem_n_pes()
    chunk = 4

    src = rshmem_torch.create_tensor((n, chunk), torch.int32)
    dst = rshmem_torch.create_tensor((n, chunk), torch.int32)
    try:
        for peer in range(n):
            src[peer].fill_(me * 100 + peer)
        dst.fill_(-1)
        torch.cuda.synchronize()
        rshmem_torch.barrier_all()

        rshmem_torch.alltoall(rocshmem4py.ROCSHMEM_TEAM_WORLD, dst, src)
        rshmem_torch.barrier_all()
        torch.cuda.synchronize()

        expected = torch.empty_like(dst)
        for peer in range(n):
            expected[peer].fill_(peer * 100 + me)
        torch.testing.assert_close(dst, expected)
    finally:
        rshmem_torch.free_tensor(src)
        rshmem_torch.free_tensor(dst)


def test_world_broadcast_torch_wrapper():
    import torch

    me = rocshmem4py.rocshmem_my_pe()
    root = 0

    src = rshmem_torch.create_tensor((16,), torch.int32)
    dst = rshmem_torch.create_tensor((16,), torch.int32)
    try:
        src.fill_(me)
        dst.fill_(-1)
        if me == root:
            dst.copy_(src)
        torch.cuda.synchronize()
        rshmem_torch.barrier_all()

        rshmem_torch.broadcast(rocshmem4py.ROCSHMEM_TEAM_WORLD, dst, src, root)
        rshmem_torch.barrier_all()
        torch.cuda.synchronize()

        torch.testing.assert_close(dst, torch.full_like(dst, root))
    finally:
        rshmem_torch.free_tensor(src)
        rshmem_torch.free_tensor(dst)


def test_team_collective_symbols_are_exported():
    for name in (
        "rocshmem_alltoallmem_on_stream",
        "rocshmem_broadcastmem_on_stream",
        "rocshmem_barrier",
        "rocshmem_team_sync",
        "rocshmem_barrier_on_stream",
        "rocshmem_team_sync_on_stream",
        "rocshmem_global_exit",
        "rocshmem_dump_stats",
        "rocshmem_reset_stats",
    ):
        assert hasattr(rocshmem4py, name)
