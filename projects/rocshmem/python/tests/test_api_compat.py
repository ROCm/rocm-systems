"""Public-API surface tests for rocshmem4py.

These tests assert that the *public Python API* of ``rocshmem4py`` (built with
the nanobind backend) is complete and stable: the module name, function names,
argument behavior, and return types must not change.  This is the contract
consumers depend on, independent of the C++ binding framework.

They are torch-free and single-PE safe, so they run in any CI environment.
"""

import pytest

import rocshmem4py
import _rocshmem4py


# Host-facing symbols that must be importable and callable from the compiled
# extension.  Mirrors __init__.py:_HOST_API_BINDINGS.
_REQUIRED_EXT_CALLABLES = (
    "rocshmem_init",
    "rocshmem_finalize",
    "rocshmem_hipmodule_init",
    "rocshmem_my_pe",
    "rocshmem_n_pes",
    "rocshmem_team_my_pe",
    "rocshmem_team_n_pes",
    "rocshmem_malloc",
    "rocshmem_free",
    "rocshmem_calloc",
    "rocshmem_align",
    "rocshmem_buffer_register",
    "rocshmem_buffer_unregister",
    "rocshmem_buffer_unregister_all",
    "rocshmem_ptr",
    "rocshmem_barrier_all",
    "rocshmem_barrier",
    "rocshmem_barrier_all_on_stream",
    "rocshmem_barrier_on_stream",
    "rocshmem_fence",
    "rocshmem_quiet",
    "rocshmem_get_uniqueid",
    "rocshmem_init_attr",
    "rocshmem_putmem",
    "rocshmem_getmem",
    "rocshmem_putmem_nbi",
    "rocshmem_getmem_nbi",
    "rocshmem_putmem_on_stream",
    "rocshmem_getmem_on_stream",
    "rocshmem_putmem_signal_on_stream",
    "rocshmem_signal_wait_until_on_stream",
    "rocshmem_team_split_strided",
    "rocshmem_team_destroy",
    "rocshmem_team_translate_pe",
    "rocshmem_sync_all",
    "rocshmem_team_sync",
    "rocshmem_sync_all_on_stream",
    "rocshmem_team_sync_on_stream",
    "rocshmem_alltoallmem_on_stream",
    "rocshmem_broadcastmem_on_stream",
    "rocshmem_query_thread",
    "rocshmem_global_exit",
    "rocshmem_dump_stats",
    "rocshmem_reset_stats",
    "rocshmem_get_device_ctx",
    "hip_device_synchronize",
)

_REQUIRED_CONSTANTS = (
    "ROCSHMEM_SUCCESS",
    "ROCSHMEM_SIGNAL_SET",
    "ROCSHMEM_SIGNAL_ADD",
    "ROCSHMEM_CMP_EQ",
    "ROCSHMEM_CMP_NE",
    "ROCSHMEM_CMP_GT",
    "ROCSHMEM_CMP_GE",
    "ROCSHMEM_CMP_LT",
    "ROCSHMEM_CMP_LE",
)


def test_module_name_is_stable():
    # The compiled extension must keep the same import name so the wheel
    # import path is unchanged.
    assert _rocshmem4py.__name__ == "_rocshmem4py"
    assert rocshmem4py.__version__ is not None


def test_required_callables_present():
    for name in _REQUIRED_EXT_CALLABLES:
        assert hasattr(_rocshmem4py, name), f"_rocshmem4py missing {name}"
        assert callable(getattr(_rocshmem4py, name)), f"{name} not callable"


def test_required_constants_are_ints():
    for name in _REQUIRED_CONSTANTS:
        val = getattr(_rocshmem4py, name)
        assert isinstance(val, int), f"{name} must be int, got {type(val)}"


def test_constant_values():
    assert rocshmem4py.ROCSHMEM_SUCCESS == 0
    assert rocshmem4py.ROCSHMEM_TEAM_WORLD == 0
    assert rocshmem4py.ROCSHMEM_TEAM_INVALID == -1
    assert rocshmem4py.ROCSHMEM_SIGNAL_SET == 0
    assert rocshmem4py.ROCSHMEM_SIGNAL_ADD == 1


def test_pe_queries_return_ints():
    my_pe = rocshmem4py.rocshmem_my_pe()
    n_pes = rocshmem4py.rocshmem_n_pes()
    assert isinstance(my_pe, int) and isinstance(n_pes, int)
    assert 0 <= my_pe < n_pes


def test_pointer_apis_return_ints():
    ptr = rocshmem4py.rocshmem_malloc(1024)
    try:
        assert isinstance(ptr, int) and ptr > 0
        remote = rocshmem4py.rocshmem_ptr(ptr, rocshmem4py.rocshmem_my_pe())
        assert isinstance(remote, int)
    finally:
        rocshmem4py.rocshmem_free(ptr)


def test_uniqueid_returns_bytes():
    # get_uniqueid round-trips a fixed-size binary blob through nb::bytes;
    # the exact length must be preserved.
    uid = rocshmem4py.rocshmem_get_uniqueid()
    assert isinstance(uid, bytes)
    assert len(uid) > 0
    # Stable length across calls (the blob is a fixed-size struct).
    assert len(rocshmem4py.rocshmem_get_uniqueid()) == len(uid)


def test_team_config_attribute_roundtrip():
    cfg = rocshmem4py.TeamConfig()
    cfg.num_contexts = 4
    assert cfg.num_contexts == 4
    cfg.num_contexts = 0
    assert cfg.num_contexts == 0


def test_capsule_buffer_api():
    # SymmetricBuffer exposes the raw device pointer and the
    # __cuda_array_interface__ protocol.
    buf = rocshmem4py.SymmetricBuffer(512)
    try:
        assert isinstance(buf.ptr, int) and buf.ptr > 0
        cai = buf.__cuda_array_interface__
        assert cai["data"] == (buf.ptr, False)
        assert cai["version"] == 3
    finally:
        buf.free()


def test_public_surface_matches_init_all():
    # The high-level package surface (__all__) is independent of the binding
    # framework; verify the documented names are all resolvable.
    for name in rocshmem4py.__all__:
        assert hasattr(rocshmem4py, name), f"rocshmem4py.__all__ lists missing {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
