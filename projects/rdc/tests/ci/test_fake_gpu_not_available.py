"""Values amd-smi cannot read, against the fake amd-smi: RDC reports N/A, not a number.

Runs like test_fake_gpu.py (same environment variables and helpers). RDC_FAKE_SENTINEL makes
a call succeed with what amd-smi reports for a value it cannot read; see
tests/fake_amdsmi/README.md.
"""

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import Rdcd, _created_id, golden, tools  # noqa: F401 (fixture)

# amd-smi call that reports "not available" on GPU 1 -> RDC fields that must be N/A there.
NOT_AVAILABLE = {
    # No current clock level (a power-gated clock domain): rdcd read past frequency[].
    "amdsmi_get_clk_freq": [100, 101],  # GPU_CLOCK, MEM_CLOCK
    "amdsmi_get_power_info": [300],  # POWER_USAGE
    "amdsmi_get_gpu_activity": [500, 505],  # GPU_UTIL, GPU_MEMORY_ACTIVITY
    "amdsmi_get_clock_info": [104, 105, 106],  # SOC_CLOCK, VCLK0, DCLK0
    "amdsmi_get_gpu_metrics_info": [507],  # GPU_MEMORY_CUR_BANDWIDTH
    "amdsmi_get_violation_status": [
        3006,  # RDC_HEALTH_POWER_THROTTLE_TIME
        3007,  # RDC_HEALTH_THERMAL_THROTTLE_TIME
        3008,  # RDC_HEALTH_VIOLATION_COUNTER
        3009,  # RDC_HEALTH_PROCHOT_RESIDENCY_ACC
        3010,  # RDC_HEALTH_PROCHOT_RESIDENCY_PCT
        3011,  # RDC_HEALTH_PPT_RESIDENCY_PCT
        3012,  # RDC_HEALTH_SOCKET_THRM_ACC
        3013,  # RDC_HEALTH_SOCKET_THRM_PCT
        3014,  # RDC_HEALTH_VR_THRM_ACC
        3015,  # RDC_HEALTH_VR_THRM_PCT
        3016,  # RDC_HEALTH_HBM_THRM_ACC
        3017,  # RDC_HEALTH_HBM_THRM_PCT
        3018,  # RDC_HEALTH_GFX_CLK_LMT_PWR_ACC
        3019,  # RDC_HEALTH_GFX_CLK_LMT_PWR_PCT
        3020,  # RDC_HEALTH_GFX_CLK_LMT_THM_ACC
        3021,  # RDC_HEALTH_GFX_CLK_LMT_THM_PCT
        3022,  # RDC_HEALTH_LOW_UTIL_ACC
        3023,  # RDC_HEALTH_LOW_UTIL_PCT
        3024,  # RDC_HEALTH_GFX_CLK_LMT_TOTAL_ACC
        3025,  # RDC_HEALTH_GFX_CLK_LMT_TOTAL_PCT
    ],
}


@pytest.fixture
def start_rdcd(tools, tmp_path):
    """Start rdcd with fake settings; faults apply from the start of the test."""
    started = []

    def start(**fake_env):
        d = Rdcd(tools, tmp_path, **fake_env).start()
        started.append(d)
        d.arm()
        return d

    yield start
    for d in started:
        code = d.stop()
        assert code == 0, f"rdcd exited with {code} after SIGTERM\n{d.output()}"


def _dmon(d, fields, gpus=(0, 1)):
    """{gpu: {field: value}} from one dmon sample.

    Columns are matched by position: dmon runs header names of 20 or more characters together.
    """
    out = d.ok(
        "dmon",
        "-e",
        ",".join(map(str, fields)),
        "-i",
        ",".join(map(str, gpus)),
        "-c",
        "1",
        "-d",
        "500",
    )
    rows = {}
    for line in out.splitlines():
        cols = line.split()
        if cols and cols[0].isdigit():
            assert len(cols) == len(fields) + 1, f"malformed dmon row {line!r} in:\n{out}"
            rows[int(cols[0])] = dict(zip(fields, cols[1:]))
    assert sorted(rows) == sorted(gpus), out
    return rows


@pytest.mark.parametrize("api", sorted(NOT_AVAILABLE))
def test_value_amd_smi_cannot_read_is_na(start_rdcd, api):
    d = start_rdcd(RDC_FAKE_SENTINEL=f"{api}@1")
    fields = NOT_AVAILABLE[api]
    rows = _dmon(d, fields)
    for field in fields:
        assert rows[1][field] == "N/A", f"GPU 1 field {field}: {rows[1][field]}, not N/A"
        assert rows[0][field] != "N/A", f"GPU 0 field {field} is N/A"
        if field in golden.FIELDS:
            assert rows[0][field] == str(golden.expected(field, 0)), rows[0]


def test_busy_values_the_gpu_does_not_report_are_na(start_rdcd):
    # Like an RDNA GPU's, the fake's gpu_metrics has no per-XCP busy values (so no GPU
    # shows a number here).
    d = start_rdcd()
    rows = _dmon(d, [519, 520, 521])  # GPU_GFX_BUSY_INST, GPU_VCN_BUSY_INST, GPU_JPEG_BUSY_INST
    for gpu, values in rows.items():
        assert set(values.values()) == {"N/A"}, f"GPU {gpu}: {values}"


def test_ecc_totals_are_na_without_ecc(start_rdcd):
    # amd-smi on a GPU without ECC fails every block's ECC status with NOT_SUPPORTED.
    d = start_rdcd(RDC_FAKE_FAIL="amdsmi_get_gpu_ecc_status@1=AMDSMI_STATUS_NOT_SUPPORTED")
    fields = [600, 601, 659]  # ECC_CORRECT_TOTAL, ECC_UNCORRECT_TOTAL, ECC_DEFERRED_TOTAL
    rows = _dmon(d, fields)
    assert set(rows[1].values()) == {"N/A"}, f"GPU 1: {rows[1]}"
    assert rows[0][600] == str(golden.reading("ECC_UMC_CORRECTABLE", 0)), rows[0]
    assert "N/A" not in rows[0].values(), rows[0]


def test_health_checks_pages_of_a_gpu_without_ecc(start_rdcd):
    d = start_rdcd(RDC_FAKE_FAIL="amdsmi_get_gpu_ecc_status@1=AMDSMI_STATUS_NOT_SUPPORTED")
    gid = _created_id(d.ok("group", "-c", "no_ecc"))
    d.ok("group", "-g", gid, "-a", "1")
    d.ok("health", "-g", gid, "-s", "m")
    out = d.ok("health", "-g", gid, "-c")
    # Fake GPU 1 has one bad page, pending retirement.
    assert "Detected 1 pending retired page(s)." in out, out


def test_bulk_fetch_leaves_metrics_the_gpu_does_not_report_to_other_calls(start_rdcd):
    d = start_rdcd(RDC_BULK_FETCH_ENABLED="true", RDC_FAKE_SENTINEL="amdsmi_get_gpu_metrics_info@1")
    fields = [100, 200, 201, 300, 500]  # what RDC reads in bulk from gpu_metrics
    rows = _dmon(d, fields)
    for gpu in (0, 1):
        for field in fields:
            want = str(golden.expected(field, gpu))
            assert rows[gpu][field] == want, (
                f"GPU {gpu} field {field}: {rows[gpu][field]}, not {want}"
            )
