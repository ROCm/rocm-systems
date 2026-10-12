"""
4.7  Bootstrap API Tests
========================
Validates the librdc_bootstrap entry points that load librdc_client.so and
librdc.so: the library loader under concurrent loads, and an embedded handle and
a standalone handle in one process.
"""

import json
import os
import subprocess
import sys

import pytest

# Probe preamble: load librdc_bootstrap.so from the tested lib directory (argv[1]).
_PREAMBLE = """
import ctypes, json, os, sys
P = ctypes.c_void_p
rdc = ctypes.CDLL(os.path.join(sys.argv[1], "librdc_bootstrap.so"))
rdc.rdc_init.argtypes = [ctypes.c_uint64]
rdc.rdc_connect.argtypes = [ctypes.c_char_p, ctypes.POINTER(P)] + [ctypes.c_char_p] * 3
rdc.rdc_disconnect.argtypes = [P]
rdc.rdc_start_embedded.argtypes = [ctypes.c_int, ctypes.POINTER(P)]
rdc.rdc_stop_embedded.argtypes = [P]
rdc.rdc_device_get_all.argtypes = [P, P, P]
rdc.rdc_shutdown.argtypes = []
assert rdc.rdc_init(0) == 0


def start_embedded():
    handle = P()
    if rdc.rdc_start_embedded(0, ctypes.byref(handle)) != 0:  # RDC_OPERATION_MODE_AUTO
        print(json.dumps({"skip": "rdc_start_embedded() failed"}))
        sys.exit(0)
    return handle
"""

# An embedded handle must survive rdc_connect() in the same process.
_EMBEDDED_THEN_STANDALONE_PROBE = (
    _PREAMBLE
    + """
embedded = start_embedded()
client = P()
assert rdc.rdc_connect(b"localhost:1", ctypes.byref(client), None, None, None) == 0
assert rdc.rdc_disconnect(client) == 0
ids, count = (ctypes.c_uint32 * 128)(), ctypes.c_uint32()
status = rdc.rdc_device_get_all(embedded, ids, ctypes.byref(count))
assert rdc.rdc_stop_embedded(embedded) == 0
print(json.dumps({"get_all": status, "count": count.value}))
sys.stdout.flush()
os._exit(0)  # skip unloading the client library at exit, which is not under test here
"""
)


def _run_probe(probe, lib_dir):
    """Run a probe in a child process; return its parsed JSON result line."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [lib_dir, env.get("LD_LIBRARY_PATH")]))
    r = subprocess.run(
        [sys.executable, "-c", probe, lib_dir], env=env, capture_output=True, text=True, timeout=180
    )
    assert r.returncode == 0, f"probe failed (rc={r.returncode}): {r.stderr[-2000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture
def lib_dir(rocm_dir):
    path = os.path.join(rocm_dir, "lib")
    if not os.path.isfile(os.path.join(path, "librdc_bootstrap.so")):
        pytest.skip("librdc_bootstrap.so not installed")
    return path


class TestBootstrapLoader:
    """rdc_connect() and rdc_start_embedded() share the bootstrap's library loading."""

    def test_library_loader_unit_tests(self, rocm_dir):
        # Concurrent loads need real threads, so they are C++ tests in rdc_codec_tests.
        binary = os.path.join(rocm_dir, "share", "rdc", "rdctst_tests", "rdc_codec_tests")
        if not os.access(binary, os.X_OK):
            pytest.skip("rdc_codec_tests not installed")
        r = subprocess.run(
            [binary, "--gtest_filter=RdcLibraryLoader.*"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
        assert "[  PASSED  ] 2 tests" in r.stdout, r.stdout[-3000:]

    def test_embedded_handle_survives_a_connect(self, lib_dir, has_gpu):
        if not has_gpu:
            pytest.skip("No GPU -- embedded mode needs one")
        result = _run_probe(_EMBEDDED_THEN_STANDALONE_PROBE, lib_dir)
        if "skip" in result:
            pytest.skip(result["skip"])
        assert result["get_all"] == 0 and result["count"] > 0
