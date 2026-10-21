"""rdci health on fake GPUs: a health check that cannot read its fields says
that it did not complete instead of passing, and clearing the health watches of
a destroyed GPU group drops their field group.

Runs like test_fake_gpu.py (same environment variables and helpers).
"""

import re

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import (  # noqa: F401 (fixture)
    NUM_GPUS,
    Rdcd,
    _all_gpus_group,
    tools,
)

HEALTH_FIELD_GROUP = "health-field-group"  # the field group of a health watch


def test_health_check_reports_a_check_that_cannot_run(tools, tmp_path):
    # After arm(), every read of the PCIe replay count fails.
    d = Rdcd(tools, tmp_path, RDC_FAKE_FAIL="amdsmi_get_pcie_info").start()
    try:
        gid = _all_gpus_group(d)
        d.ok("health", "-g", gid, "-s", "p")
        d.arm()
        out = d.ok("health", "-g", gid, "-c")
    finally:
        d.stop()
    assert "Overall Health: Warning" in out, out
    not_run = re.findall(r"The PCIe health check did not complete \(status \d+\)\.", out)
    assert len(not_run) == NUM_GPUS, out


def test_health_clear_of_a_destroyed_group_drops_its_field_group(tools, tmp_path):
    d = Rdcd(tools, tmp_path).start()
    try:
        gid = _all_gpus_group(d)
        d.ok("health", "-g", gid, "-s", "p")
        assert HEALTH_FIELD_GROUP in d.ok("fieldgroup", "-l")
        d.ok("group", "-d", gid)
        # The field watch cannot stop without its group, so the clear fails, but it
        # drops the field group anyway.
        assert d.rdci("health", "-g", gid, "--clear").returncode != 0
        assert HEALTH_FIELD_GROUP not in d.ok("fieldgroup", "-l")
    finally:
        d.stop()
