# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Generated output for the fake tables, and the --check mode."""

import os
import re
import shutil

from conftest import DATA, EXPECTED, generate_files, run_cli


def test_output_matches_golden_files(fake_config):
    files = generate_files(fake_config)
    if os.environ.get("ROCM_TRACE_GEN_UPDATE_GOLDEN") == "1":
        EXPECTED.mkdir(exist_ok=True)
        for name, text in files.items():
            with (EXPECTED / name).open("w", encoding="utf-8", newline="\n") as f:
                f.write(text)
    assert sorted(files) == ["fake_probes.cpp", "fake_probes.h"]
    for name, text in files.items():
        assert text == (EXPECTED / name).read_text(encoding="utf-8"), (
            f"{name} differs from tests/data/expected; if the change is intended, "
            "rerun with ROCM_TRACE_GEN_UPDATE_GOLDEN=1"
        )


def test_every_entry_has_a_probe_or_a_reason(fake_config):
    source = generate_files(fake_config)["fake_probes.cpp"]
    installed = set(re.findall(r"table->(\w+)_fn = probe_", source))
    skipped = set(re.findall(r"// (\w+)_fn: not traced \(", source))
    assert skipped == {"fakeSkipped"}
    assert installed == {
        "fakeScalars",
        "fakeAlloc",
        "fakeName",
        "fakeLoad",
        "fakeLaunch",
        "fakeOpen",
        "fakeCallback",
        "fakeRawFn",
        "fakeErrorString",
        "fakeNoResult",
        "fakeMakeDim",
        "fakeGetDim",
        "fakeWide",
        "fakeInit",
        "fakeCount",
    }
    assert "fakeRetired" not in source


def test_two_events_per_call(fake_config):
    source = generate_files(fake_config)["fake_probes.cpp"]
    enters = re.findall(r'"(\w+)_enter"', source)
    exits = re.findall(r'"(\w+)_exit"', source)
    assert enters == exits and len(enters) == 15
    assert (
        source.count("WINEVENT_OPCODE_START")
        == source.count("WINEVENT_OPCODE_STOP")
        == 15
    )


def test_outputs_are_recorded_only_after_success(fake_config):
    source = generate_files(fake_config)["fake_probes.cpp"]
    assert (
        "(rocm_trace_result == fakeSuccess && ptr != nullptr ? "
        "reinterpret_cast<const void*>((*ptr)) : static_cast<const void*>(nullptr))"
    ) in source


def test_check_passes_on_current_files_and_fails_on_stale(
    fake_config, tmp_path, capsys
):
    for name in ("fake_probes.h", "fake_probes.cpp"):
        shutil.copy(EXPECTED / name, tmp_path / name)
    args = [
        "--config",
        str(fake_config),
        "--output-dir",
        str(tmp_path),
        "-I",
        str(DATA),
    ]
    assert run_cli(*args, "--check") == 0

    stale = tmp_path / "fake_probes.cpp"
    stale.write_text(
        stale.read_text().replace('"fakeAlloc_enter"', '"fakeAlloc_begin"')
    )
    assert run_cli(*args, "--check") == 1
    out = capsys.readouterr()
    assert "fakeAlloc_begin" in out.out and "out of date" in out.err


def test_generate_writes_files(fake_config, tmp_path):
    out = tmp_path / "out"
    assert (
        run_cli("--config", str(fake_config), "--output-dir", str(out), "-I", str(DATA))
        == 0
    )
    assert (out / "fake_probes.cpp").read_text() == (
        EXPECTED / "fake_probes.cpp"
    ).read_text()
    assert (
        run_cli(
            "--config",
            str(fake_config),
            "--output-dir",
            str(out),
            "-I",
            str(DATA),
            "--check",
        )
        == 0
    )
