# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the metric-health local driver site defaults."""

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "scripts" / "run_metric_health_test_report_local_driver.sh"
EXAMPLE = REPO_ROOT / "scripts" / ".metric_health_sites.env.example"

GFX942_SITE_BASE = """
GFX942_HOST=host.example
GFX942_REMOTE_ROOT=/remote/root
GFX942_THEROCK=/opt/therock
GFX942_REMOTE_LOG_DIR=/remote/logs
"""


def gfx942_rocr_default(
    site_text: str,
    tmp_path: Path,
    extra_env: Optional[Dict[str, str]] = None,
) -> str:
    """Source the driver and return the gfx942 ROCR default."""
    site = tmp_path / "sites.env"
    site.write_text(site_text, encoding="utf-8")
    env = os.environ.copy()
    env["METRIC_HEALTH_SITES_ENV"] = str(site)
    env["DRIVER"] = str(DRIVER)
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'set -- --arch gfx942; source "$DRIVER"; printf "%s\\n" "$ROCR_DEFAULT"',
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_gfx942_device_id_comes_from_site_config(tmp_path: Path) -> None:
    """gfx942 uses the site-config device id."""
    site_text = GFX942_SITE_BASE + "GFX942_ROCR_VISIBLE_DEVICES=17\n"
    assert gfx942_rocr_default(site_text, tmp_path) == "17"


def test_gfx942_device_id_is_empty_without_site_value(tmp_path: Path) -> None:
    """An unset site device id does not fall back to a baked-in number."""
    assert gfx942_rocr_default(GFX942_SITE_BASE, tmp_path) == ""


def test_exported_gfx942_device_id_wins_over_site_file(tmp_path: Path) -> None:
    """An already-exported site variable wins over the file."""
    site_text = GFX942_SITE_BASE + "GFX942_ROCR_VISIBLE_DEVICES=17\n"
    resolved = gfx942_rocr_default(
        site_text,
        tmp_path,
        {"GFX942_ROCR_VISIBLE_DEVICES": "3"},
    )
    assert resolved == "3"


def test_example_names_gfx942_device_without_a_value() -> None:
    """The committed example names the variable and leaves it empty."""
    text = EXAMPLE.read_text(encoding="utf-8")
    device_lines = [
        line
        for line in text.splitlines()
        if line.startswith("GFX942_ROCR_VISIBLE_DEVICES=")
    ]
    assert device_lines == ["GFX942_ROCR_VISIBLE_DEVICES="]
    assert "17" not in text
    driver = DRIVER.read_text(encoding="utf-8")
    assert "ROCR_DEFAULT=17" not in driver


def test_gfx1201_site_configuration(tmp_path: Path) -> None:
    site = tmp_path / "sites.env"
    site.write_text(
        "GFX1201_HOST=host.example\n"
        "GFX1201_REMOTE_ROOT=/remote/root\n"
        "GFX1201_THEROCK=/opt/rocm\n"
        "GFX1201_REMOTE_LOG_DIR=/remote/logs\n"
        "GFX1201_ROCR_VISIBLE_DEVICES=2\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["METRIC_HEALTH_SITES_ENV"] = str(site)
    env["DRIVER"] = str(DRIVER)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'set -- --arch gfx1201; source "$DRIVER"; '
            'printf "%s %s\\n" "$TRANSPORT" "$ROCR_DEFAULT"',
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ssh-key 2"


def test_gfx1201_worker_requires_rocm_prefix() -> None:
    worker = REPO_ROOT / "scripts" / "run_metric_health_test_report.sh"
    completed = subprocess.run(
        ["bash", str(worker), "--arch", "gfx1201"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--therock is required for --arch gfx1201" in completed.stderr


def test_gfx1201_remote_wait_stops_on_worker_failure(tmp_path: Path) -> None:
    site = tmp_path / "sites.env"
    site.write_text(
        "GFX1201_HOST=host.example\n"
        "GFX1201_REMOTE_ROOT=/remote/root\n"
        "GFX1201_THEROCK=/opt/rocm\n"
        "GFX1201_REMOTE_LOG_DIR=/remote/logs\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["METRIC_HEALTH_SITES_ENV"] = str(site)
    env["DRIVER"] = str(DRIVER)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'set -- --arch gfx1201; source "$DRIVER"; '
            'ssh_cmd() { [[ "$1" == *"^RUN_FAILED "* ]]; }; wait_ssh',
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert completed.returncode != 0
    assert "remote metric health run failed" in completed.stderr


def test_gfx1201_worker_logs_failure(tmp_path: Path) -> None:
    worker = REPO_ROOT / "scripts" / "run_metric_health_test_report.sh"
    log = tmp_path / "worker.log"
    completed = subprocess.run(
        [
            "bash",
            str(worker),
            "--arch",
            "gfx1201",
            "--therock",
            str(tmp_path),
            "--root",
            str(tmp_path),
            "--profiler-root",
            str(tmp_path),
            "--log",
            str(log),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "RUN_FAILED status=" in log.read_text(encoding="utf-8")
