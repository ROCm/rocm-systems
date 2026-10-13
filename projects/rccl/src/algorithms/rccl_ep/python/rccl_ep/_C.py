# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Supplies the `_C` module name in pure Python. The native code this package
# needs is librccl_ep.so, loaded through ctypes in __init__.py, so there is no
# compiled Python extension to import. This module declares the symbols a caller
# may pull in at module scope; none are reached by the intranode path, and each
# raises rather than returning a plausible value, so a caller that does need one
# fails loudly.
#
# See LICENSE.txt for license information


class EventHandle:
    """Placeholder for the CUDA event handle a compiled `_C` would export.

    Needed only because callers import the symbol eagerly at module scope, for
    forward compatibility, even though nothing on the intranode path ever
    constructs one.

    The async surface is EventOverlap in __init__.py, backed by a real
    torch.cuda.Event; this exists so those eager imports resolve.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "_C.EventHandle is not provided. The intranode "
            "path is synchronous and uses torch.cuda.Event via EventOverlap.")

    def current_stream_wait(self, *args, **kwargs):
        raise NotImplementedError("_C.EventHandle is not provided.")


def _unavailable(name):
    def f(*args, **kwargs):
        raise NotImplementedError(
            f"_C.{name} is not provided. The intranode path "
            f"creates its own communicator from a broadcast unique id, so it "
            f"never needs to adopt PyTorch's.")
    return f


get_logical_domain_size = _unavailable("get_logical_domain_size")
get_physical_domain_size = _unavailable("get_physical_domain_size")
destroy_nccl_comm = _unavailable("destroy_nccl_comm")
get_local_nccl_unique_id = _unavailable("get_local_nccl_unique_id")
create_nccl_comm = _unavailable("create_nccl_comm")
