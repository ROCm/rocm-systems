"""
4.7  Bootstrap API Tests
========================
Validates the librdc_bootstrap entry points that load librdc_client.so and
librdc.so: the library loader under concurrent loads, an embedded handle and a
standalone handle in one process, and device attributes for GPU indexes that do
not exist.
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
    status = rdc.rdc_start_embedded(0, ctypes.byref(handle))  # RDC_OPERATION_MODE_AUTO
    assert status == 0, f"rdc_start_embedded() returned {status}"
    return handle
"""

# An embedded handle must survive rdc_connect() in the same process. On develop this crashed
# in rdc_stop_embedded() with the TheRock toolchain; GCC builds did not crash.
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


# Attributes of GPU indexes that do not exist: an error status and an empty name. On develop
# the embedded name held whatever bytes were on the stack (so it failed only when they were
# non-zero), and the standalone call left the caller's bytes unchanged.
_INVALID_GPU_ATTRIBUTES_PROBE = (
    _PREAMBLE
    + """
class Attributes(ctypes.Structure):  # rdc_device_attributes_t in rdc/rdc.h
    _fields_ = [
        ("device_name", ctypes.c_char * 256),
        ("device_id", ctypes.c_uint64),
        ("num_of_compute_units", ctypes.c_uint32),
        ("target_graphics_version", ctypes.c_uint64),
    ]


rdc.rdc_device_get_attributes.argtypes = [P, ctypes.c_uint32, ctypes.POINTER(Attributes)]
target = sys.argv[2]  # "embedded" or an rdcd address
if target == "embedded":
    handle = start_embedded()
else:
    handle = P()
    assert rdc.rdc_connect(target.encode(), ctypes.byref(handle), None, None, None) == 0
results = []
for index in (9999, 4242, 100000):
    attributes = Attributes()
    attributes.device_name = b"X" * 255
    status = rdc.rdc_device_get_attributes(handle, index, ctypes.byref(attributes))
    name = ctypes.string_at(ctypes.addressof(attributes), 256).split(b"\\0")[0]
    results.append({"index": index, "status": status, "name": name.hex()})
if target == "embedded":
    assert rdc.rdc_stop_embedded(handle) == 0
else:
    assert rdc.rdc_disconnect(handle) == 0
print(json.dumps(results))
sys.stdout.flush()
os._exit(0)  # skip unloading the client library at exit, which is not under test here
"""
)


_RDCD_ADDRESS = "localhost:50051"  # where the rdcd_server fixture starts rdcd


def _run_probe(probe, lib_dir, *args):
    """Run a probe in a child process; return its parsed JSON result line."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [lib_dir, env.get("LD_LIBRARY_PATH")]))
    r = subprocess.run(
        [sys.executable, "-c", probe, lib_dir, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, (
        f"probe failed (rc={r.returncode}): {r.stdout[-1000:]}{r.stderr[-2000:]}"
    )
    results = [line for line in r.stdout.splitlines() if line.startswith(("{", "["))]
    assert results, f"probe printed no result: {r.stdout[-2000:]}"
    return json.loads(results[-1])


@pytest.fixture
def lib_dir(rocm_dir):
    path = os.path.join(rocm_dir, "lib")
    if not os.path.isfile(os.path.join(path, "librdc_bootstrap.so")):
        pytest.skip("librdc_bootstrap.so not installed")
    return path


class TestBootstrapLoader:
    """rdc_connect() and rdc_start_embedded() share the bootstrap's library loading."""

    def test_library_loader_unit_tests(self, rocm_dir, tmp_path):
        # Concurrent loads need real threads, so they are C++ tests in rdc_codec_tests. These
        # fail on develop with every toolchain; the GPU probes below fail only with some.
        tests_dir = os.path.join(rocm_dir, "share", "rdc", "rdctst_tests")
        if not os.path.isdir(tests_dir):
            pytest.skip("RDC tests are not installed")
        binary = os.path.join(tests_dir, "rdc_codec_tests")
        assert os.access(binary, os.X_OK), f"{binary} is missing from the installed RDC tests"
        report = tmp_path / "loader.json"
        r = subprocess.run(
            [binary, "--gtest_filter=RdcLibraryLoader.*", f"--gtest_output=json:{report}"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
        summary = json.loads(report.read_text())
        assert summary["tests"] >= 3, summary
        assert summary["failures"] == 0 and summary["disabled"] == 0, summary

    def test_embedded_handle_survives_a_connect(self, lib_dir, has_gpu):
        if not has_gpu:
            pytest.skip("No GPU -- embedded mode needs one")
        result = _run_probe(_EMBEDDED_THEN_STANDALONE_PROBE, lib_dir)
        assert result["get_all"] == 0 and result["count"] > 0


class TestDeviceAttributes:
    """rdc_device_get_attributes() must never hand back uninitialized bytes."""

    @pytest.mark.parametrize("mode", ["embedded", "standalone"])
    def test_invalid_gpu_has_empty_name(self, lib_dir, has_gpu, request, mode):
        if not has_gpu:
            pytest.skip("No GPU -- embedded mode and rdcd need one")
        target = "embedded"
        if mode == "standalone":
            request.getfixturevalue("rdcd_server")
            target = _RDCD_ADDRESS
        results = _run_probe(_INVALID_GPU_ATTRIBUTES_PROBE, lib_dir, target)
        for result in results:
            assert result["status"] != 0, result
            assert result["name"] == "", result
