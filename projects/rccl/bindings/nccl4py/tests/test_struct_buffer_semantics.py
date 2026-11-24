# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Struct wrapper semantics from the NCCL 2.32.4 cybind regeneration.

Upstream "update cybind for nccl 2.32.4" changed the Python-side behaviour of
the generated struct wrappers in ``nccl.bindings``:

* ``from_buffer()`` passes the buffer's read-only flag through; it used to
  invert it, so a writable ``bytearray`` gave a read-only wrapper and a
  read-only buffer gave a writable one.
* ``obj[0] = buf`` accepts any buffer-protocol object, copies it into the
  wrapper's inline storage, rejects a source smaller than the struct and
  refuses to write into a read-only wrapper.
* ``GinBarrierOptions`` and ``DevCommRequirements.gin_custom_array*`` are
  exposed.

``test_baseline_*`` covers wrapper behaviour that predates the regeneration.
None of this calls into librccl, so it runs on a single rank without a GPU.
The module runs under pytest or standalone (``python -m`` / ``python file``,
optionally with a test-name regex), so it needs no pytest at run time.
"""

import contextlib
import re
import sys


@contextlib.contextmanager
def _raises(exc, match):
    try:
        yield
    except exc as e:
        assert re.search(match, str(e)), f"{exc.__name__} {e!r} does not match {match!r}"
        return
    raise AssertionError(f"expected {exc.__name__} matching {match!r}")


def test_baseline_config_fields_and_ndarray_setitem():
    import numpy as np
    import nccl.bindings as b

    cfg = b.Config()
    cfg.blocking = 1
    assert cfg.blocking == 1

    size = b.unique_id_dtype.itemsize
    src = np.arange(size, dtype=np.uint8)
    uid = b.UniqueId()
    uid[0] = src
    assert bytes(memoryview(uid)) == src.tobytes()


def test_from_buffer_writable_buffer_gives_writable_config():
    import nccl.bindings as b

    buf = bytearray(b.config_dtype.itemsize)
    cfg = b.Config.from_buffer(buf)
    cfg.blocking = 1
    assert cfg.blocking == 1
    # The wrapper aliases the caller's buffer.
    assert b.Config.from_buffer(buf).blocking == 1


def test_from_buffer_readonly_buffer_gives_readonly_config():
    import nccl.bindings as b

    buf = memoryview(bytearray(b.config_dtype.itemsize)).toreadonly()
    cfg = b.Config.from_buffer(buf)
    with _raises(ValueError, "read-only"):
        cfg.blocking = 1


def test_unique_id_setitem_copies_any_buffer():
    import nccl.bindings as b

    size = b.unique_id_dtype.itemsize
    src = bytes(i % 251 for i in range(size))
    uid = b.UniqueId()
    # A writable source leaves the wrapper writable.
    uid[0] = bytearray(src)
    assert bytes(memoryview(uid)) == src
    assert not memoryview(uid).readonly
    # A read-only source (bytes) is copied too; the wrapper takes its
    # read-only flag, as upstream's __setitem__ does.
    uid[0] = bytes(size)
    assert bytes(memoryview(uid)) == bytes(size)
    assert memoryview(uid).readonly
    with _raises(ValueError, "read-only"):
        uid[0] = bytearray(src)


def test_unique_id_setitem_rejects_short_buffer():
    import nccl.bindings as b

    uid = b.UniqueId()
    with _raises(ValueError, "too small"):
        uid[0] = b"\x01"


def test_unique_id_setitem_refuses_readonly_wrapper():
    import nccl.bindings as b

    size = b.unique_id_dtype.itemsize
    uid = b.UniqueId.from_buffer(memoryview(bytearray(size)).toreadonly())
    with _raises(ValueError, "read-only"):
        uid[0] = bytes(size)


def test_gin_barrier_options_and_custom_array():
    import nccl.bindings as b

    assert int(b.GinBarrierOptions.GIN_BARRIER_DEFAULT) == 0
    assert int(b.GinBarrierOptions.GIN_BARRIER_SIGNAL_EFFICIENT) == 1
    assert int(b.GinConnectionType.CUSTOM_ARRAY) == 4

    reqs = b.DevCommRequirements()
    reqs.gin_custom_array_count = 3
    assert reqs.gin_custom_array_count == 3


def _main(pattern):
    tests = [(n, f) for n, f in sorted(globals().items()) if re.match(pattern, n) and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASSED {name}")
        except Exception as e:
            failed += 1
            print(f"FAILED {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    return 1 if failed or not tests else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1] if len(sys.argv) > 1 else "test_"))
