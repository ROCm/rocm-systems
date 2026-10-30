#!/usr/bin/env python3

import sys
import pytest


def _find_table_or_view(conn, base_name):
    for typ in ("view", "table"):
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name LIKE ?",
            (typ, f"{base_name}%"),
        ).fetchone()
        if row:
            return row[0]
    return None


def test_device_mode_samples_have_null_event_id(rocpd_data):
    """Device-mode SPM samples must have NULL event_id."""
    sample_table = _find_table_or_view(rocpd_data, "rocpd_sample")
    assert sample_table is not None

    rows = rocpd_data.execute(
        f"SELECT id, event_id, timestamp FROM {sample_table}"
    ).fetchall()

    assert len(rows) > 0, "No samples found in rocpd"
    for row in rows:
        assert row[1] is None, (
            f"sample id={row[0]} has event_id={row[1]}, expected NULL for device mode"
        )
        assert row[2] > 0, f"sample id={row[0]} has timestamp={row[2]}, expected > 0"


def test_device_mode_spm_events_have_null_event_id(rocpd_data):
    """Device-mode pmc_event rows must have NULL event_id."""
    pmc_event_table = _find_table_or_view(rocpd_data, "rocpd_pmc_event")
    assert pmc_event_table is not None

    rows = rocpd_data.execute(
        f"SELECT id, event_id FROM {pmc_event_table}"
    ).fetchall()

    assert len(rows) > 0, "No pmc_event rows found in rocpd"
    for row in rows:
        assert row[1] is None, (
            f"pmc_event id={row[0]} has event_id={row[1]}, expected NULL for device mode"
        )


def test_device_mode_spm_counters_present(rocpd_data):
    """Device-mode output must contain counter data in pmc_event."""
    pmc_table = _find_table_or_view(rocpd_data, "rocpd_info_pmc")
    pmc_event_table = _find_table_or_view(rocpd_data, "rocpd_pmc_event")

    assert pmc_table is not None
    assert pmc_event_table is not None

    counter_names = rocpd_data.execute(
        f"SELECT DISTINCT p.name FROM {pmc_event_table} e "
        f"JOIN {pmc_table} p ON e.pmc_id = p.id "
    ).fetchall()

    assert len(counter_names) > 0, "No counter names found in device-mode pmc_event"

    names = {row[0] for row in counter_names}
    assert "SQ_WAVES" in names, f"SQ_WAVES not found in counters: {sorted(names)}"


def test_device_mode_spm_topology(rocpd_data):
    """Device-mode pmc records must have valid topology metadata."""
    pmc_table = _find_table_or_view(rocpd_data, "rocpd_info_pmc")
    pmc_event_table = _find_table_or_view(rocpd_data, "rocpd_pmc_event")

    assert pmc_table is not None
    assert pmc_event_table is not None

    info_columns = {
        row[1] for row in rocpd_data.execute(f"PRAGMA table_info({pmc_table})")
    }
    assert {"xcc", "shader_engine", "shader_array", "wgp", "instance"}.issubset(
        info_columns
    )

    topology = rocpd_data.execute(
        f"SELECT p.xcc, p.shader_engine, p.shader_array, p.wgp, p.instance "
        f"FROM {pmc_event_table} e "
        f"JOIN {pmc_table} p ON e.pmc_id = p.id AND e.guid = p.guid "
        f"WHERE e.sample_id IS NOT NULL"
    ).fetchall()
    assert topology, "No topology data found in device-mode pmc_event"
    assert all(
        (xcc is None or xcc >= 0)
        and (shader_engine is None or 0 <= shader_engine < 256)
        and (shader_array is None or shader_array >= 0)
        and (wgp is None or wgp >= 0)
        and instance is not None
        and instance >= 0
        for xcc, shader_engine, shader_array, wgp, instance in topology
    )


def test_device_mode_no_dispatch_events(rocpd_data):
    """Device-mode SPM should not generate dispatch events in rocpd_event."""
    event_table = _find_table_or_view(rocpd_data, "rocpd_event")
    if event_table is None:
        return

    count = rocpd_data.execute(f"SELECT COUNT(*) FROM {event_table}").fetchone()[0]
    assert count == 0, (
        f"Device-mode SPM should not produce dispatch events, found {count}"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
