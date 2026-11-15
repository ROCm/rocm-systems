"""CPU-only regression tests for PC-sampling capture status propagation."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from run_capture import run_capture


@pytest.fixture
def capture_fixtures():
    spec = importlib.util.spec_from_file_location(
        "pc_capture_fixtures", Path(__file__).with_name("conftest.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_request(filename):
    return SimpleNamespace(config=SimpleNamespace(getoption=lambda name: filename))


@pytest.mark.parametrize(
    "output,returncode,expected",
    [
        ("capture completed", 0, "passed"),
        ("capture failed", 7, "failed"),
        ("PC sampling unavailable", 0, "skipped"),
        ("PC sampling unavailable", 1, "skipped"),
    ],
)
def test_capture_result(tmp_path, capsys, output, returncode, expected):
    status_file = tmp_path / "capture.json"
    status_file.write_text(json.dumps({"status": "skipped"}))
    command = [
        sys.executable,
        "-c",
        "import sys; print(sys.argv[1]); sys.exit(int(sys.argv[2]))",
        output,
        str(returncode),
    ]
    assert run_capture(command, status_file, ["PC sampling unavailable"]) == returncode
    assert json.loads(status_file.read_text()) == {
        "status": expected,
        "returncode": returncode,
    }
    assert output in capsys.readouterr().out


def test_capture_launch_failure_replaces_stale_skip(tmp_path):
    status_file = tmp_path / "capture.json"
    status_file.write_text(json.dumps({"status": "skipped"}))
    with pytest.raises(OSError):
        run_capture([str(tmp_path / "missing-executable")], status_file, [])
    assert json.loads(status_file.read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "status", [None, "invalid-json", [], {}, {"status": "running"}, {"status": "failed"}]
)
def test_invalid_capture_status_fails(tmp_path, capture_fixtures, status):
    status_file = tmp_path / "capture.json"
    if status is not None:
        status_file.write_text(status if isinstance(status, str) else json.dumps(status))
    with pytest.raises(pytest.fail.Exception):
        capture_fixtures.capture_status.__wrapped__(make_request(str(status_file)))


def test_skipped_capture_does_not_validate_stale_output(tmp_path, capture_fixtures):
    status_file = tmp_path / "capture.json"
    status_file.write_text(json.dumps({"status": "skipped"}))
    (tmp_path / "out_results.db").write_text("stale database")
    (tmp_path / "out_results.json").write_text("stale JSON")
    with pytest.raises(pytest.skip.Exception, match="PC sampling unavailable"):
        capture_fixtures.capture_status.__wrapped__(make_request(str(status_file)))


def test_successful_capture_status(tmp_path, capture_fixtures):
    status_file = tmp_path / "capture.json"
    status_file.write_text(json.dumps({"status": "passed"}))
    capture_fixtures.capture_status.__wrapped__(make_request(str(status_file)))
    capture_fixtures.capture_status.__wrapped__(make_request(None))


@pytest.mark.parametrize(
    "fixture", ["json_data", "rocpd_connection", "rocpd_disasm_connection"]
)
def test_successful_capture_requires_output(tmp_path, capture_fixtures, fixture):
    request = make_request(str(tmp_path / "missing-output"))
    with pytest.raises(pytest.fail.Exception, match="does not exist"):
        result = getattr(capture_fixtures, fixture).__wrapped__(request)
        if fixture != "json_data":
            next(result)


def test_failure_traceback_does_not_match_skip_regex(tmp_path):
    status_file = tmp_path / "capture.json"
    status_file.write_text(json.dumps({"status": "failed"}))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__)) + "::test_successful_capture_status",
            "--capture-status",
            str(status_file),
            "--confcutdir=" + str(Path(__file__).parent),
            "--tb=long",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    assert result.returncode == 1, result.stdout
    assert "Capture fixture did not complete successfully" in result.stdout
    assert "PC sampling unavailable" not in result.stdout
    assert "PC sampling configuration is not supported" not in result.stdout
