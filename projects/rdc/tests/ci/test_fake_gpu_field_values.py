"""Field values from the fake amd-smi, from the tested prefix.

Runs the gtest rdc_field_values_tests (RdcMetricFetcherImpl against the fake),
then reads some of the values it checks through rdcd and rdci. Runs like
test_fake_gpu.py (same environment variables and helpers).
"""

import os
import re
import subprocess
import sys

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import (  # noqa: F401 (tools is a fixture)
    FACTOR,
    PREFIX,
    REQUIRE,
    Rdcd,
    _dmon_rows,
    _first_file,
    _missing,
    golden,
    tools,
)

RDC_FI_GPU_CLOCK = 100
RDC_HEALTH_EEPROM_CONFIG_VALID = 3005


def test_installed_gtest():
    binary = os.path.join(PREFIX, "share", "rdc", "rdctst_tests", "rdc_field_values_tests")
    if not os.path.isfile(binary):
        if REQUIRE:
            pytest.fail(f"{binary} is missing")
        pytest.skip(f"{binary} is missing")
    env = {k: v for k, v in os.environ.items() if not k.startswith("RDC_FAKE_")}
    r = subprocess.run([binary], env=env, capture_output=True, text=True, timeout=120 * FACTOR)
    assert r.returncode == 0, (
        f"rdc_field_values_tests exited with {r.returncode}\n{r.stdout[-3000:]}{r.stderr[-2000:]}"
    )


def _partition_clocks(stdout):
    """{partition: GPU_CLOCK} from `rdci dmon` rows such as `g0.3  1000000000`."""
    return dict(re.findall(r"^(g\d+\.\d+)\s+(\S+)\s*$", stdout, re.M))


def test_eeprom_validity_reads_1(tools, tmp_path):
    d = Rdcd(tools, tmp_path).start()
    try:
        out = d.ok("dmon", "-e", str(RDC_HEALTH_EEPROM_CONFIG_VALID), "-i", "0", "-c", "1")
        assert _dmon_rows(out)[0]["EEPROM_CONFIG_VALID"] == "1", out
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


def test_partition_clocks_skip_unavailable_readings(tools, tmp_path):
    # Four valid clocks for four partitions, after four that are not available.
    d = Rdcd(tools, tmp_path, gpus=1, RDC_FAKE_PARTITIONS=4, RDC_FAKE_GFXCLK_HOLES=4).start()
    try:
        out = d.ok("dmon", "-e", str(RDC_FI_GPU_CLOCK), "-i", "g0.0,g0.1,g0.2,g0.3", "-c", "1")
        want = str(golden.expected(RDC_FI_GPU_CLOCK, 0))
        assert _partition_clocks(out) == {f"g0.{p}": want for p in range(4)}, out
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


def test_partition_past_the_partition_count_has_no_clock(tools, tmp_path):
    # Partition 3 is watched while the GPU has four partitions; then the profile
    # cannot be read, and the GPU counts as one partition. Partition 3 then reads
    # nothing: dmon shows its last value or N/A, not a value read past the clocks.
    profile = "amdsmi_get_gpu_accelerator_partition_profile"
    d = Rdcd(
        tools,
        tmp_path,
        gpus=1,
        RDC_FAKE_PARTITIONS=4,
        RDC_FAKE_FAIL=f"{profile}=AMDSMI_STATUS_NOT_SUPPORTED",
    ).start()
    try:
        cmd = [d.rdci_path, "dmon", "-e", str(RDC_FI_GPU_CLOCK), "-i", "g0.0,g0.3"]
        samples = str(max(8, int(8 * FACTOR)))
        cmd += ["-c", samples, "-d", "500", "-u", "--host", f"127.0.0.1:{d.port}"]
        lines, failed_at = [], None
        with subprocess.Popen(
            cmd, env=d.rdci_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as p:
            try:
                for line in p.stdout:
                    if failed_at is None and d.armed.exists():
                        if any(api == profile and failed for api, _, failed in d.calls()):
                            # The rows after this one come after a failed read.
                            failed_at = len(lines) + 1
                    lines.append(line)
                    if line.startswith("g0.3") and not d.armed.exists():
                        d.arm()  # after the first sample
                assert p.wait(timeout=30 * FACTOR) == 0, "".join(lines)
            finally:
                p.kill()
        out = "".join(lines)
        d.check_alive("rdci dmon")
        assert failed_at is not None, f"no failed {profile} call while dmon ran\n{out}"
        rows = r"^(g0\.[03])\s+(\S+)\s*$"
        after = re.findall(rows, "".join(lines[failed_at:]), re.M)
        assert [part for part, _ in after].count("g0.3") >= 2, (
            f"too few rows after the failure\n{out}"
        )
        want = str(golden.expected(RDC_FI_GPU_CLOCK, 0))
        for partition, clock in re.findall(rows, out, re.M):
            assert clock in ((want,) if partition == "g0.0" else (want, "N/A")), out
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


# rdc_instance_profile_get() through rdcd, from a client that fills the profile with
# 0x5A first: the profile names the resource it describes.
_PROFILE = r"""
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
class Profile(ctypes.Structure):
    _fields_ = [("resource_type", ctypes.c_int), ("partition_resource", ctypes.c_uint32),
                ("num_partitions_share_resource", ctypes.c_uint32)]
def call(name, *args):
    status = getattr(lib, name)(*args)
    if status != 0:
        sys.exit(f"{name} returned {status}")
call("rdc_init", ctypes.c_uint64(0))
handle = ctypes.c_void_p()
call("rdc_connect", sys.argv[2].encode(), ctypes.byref(handle), None, None, None)
for resource in (0, 2):  # RDC_ACCELERATOR_XCC, RDC_ACCELERATOR_DECODER
    profile = Profile(0x5A5A5A5A, 0, 0)
    call("rdc_instance_profile_get", handle, ctypes.c_uint32(0), ctypes.c_int(resource),
         ctypes.byref(profile))
    print(resource, profile.resource_type, profile.partition_resource)
call("rdc_disconnect", handle)
call("rdc_shutdown")
"""


def test_instance_profile_names_its_resource(tools, tmp_path):
    bootstrap = _first_file(os.path.join(PREFIX, "lib", "librdc_bootstrap.so"))
    if bootstrap is None:
        _missing("librdc_bootstrap.so")
    d = Rdcd(tools, tmp_path, gpus=1, RDC_FAKE_PARTITIONS=2).start()
    try:
        r = subprocess.run(
            [sys.executable, "-c", _PROFILE, bootstrap, f"127.0.0.1:{d.port}"],
            env=d.rdci_env,
            capture_output=True,
            text=True,
            timeout=60 * FACTOR,
        )
    finally:
        code = d.stop()
    assert r.returncode == 0, f"client exited {r.returncode}\n{r.stdout}{r.stderr[-3000:]}"
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"
    rows = [list(map(int, line.split())) for line in r.stdout.splitlines()]
    assert [(asked, named) for asked, named, _ in rows] == [(0, 0), (2, 2)], r.stdout
    assert all(resources > 0 for _, _, resources in rows), r.stdout
