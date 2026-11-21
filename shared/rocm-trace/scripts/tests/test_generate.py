# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Generated output for the fake tables, and the --check mode."""

import os
import re
import shutil

import pytest
import yaml

from conftest import DATA, EXPECTED, generate_files, run_cli

PROBED = {
    "fakeScalars", "fakeAlloc", "fakeName", "fakeLoad", "fakeGetFunction",
    "fakeScratch", "fakeLaunch", "fakeLaunchEx", "fakeOpen", "fakeBits",
    "fakeCallback", "fakeRawFn", "fakeErrorString", "fakeNoResult",
    "fakeMakeDim", "fakeGetDim", "fakeWide", "fakeUnnamed", "fakeReserved",
    "fakeThrow", "fakeInit", "fakeCount", "fakeRenamed",
}  # fmt: skip


@pytest.fixture(scope="module")
def source():
    return generate_files(DATA / "fake.yaml")["fake_probes.cpp"]


def _function(source, name):
    """The text of the generated function called name."""
    match = re.search(rf"^\S[^\n]* {name}\(.*?^}}", source, re.M | re.S)
    assert match, name
    return match.group(0)


def test_output_matches_golden_files(fake_config):
    files = generate_files(fake_config)
    if os.environ.get("ROCM_TRACE_GEN_UPDATE_GOLDEN") == "1":
        EXPECTED.mkdir(exist_ok=True)
        for name, text in files.items():
            with (EXPECTED / name).open("w", encoding="utf-8", newline="\n") as f:
                f.write(text)
    assert sorted(files) == ["fake_probes.cpp", "fake_probes.h", "fake_probes.json"]
    for name, text in files.items():
        assert text == (EXPECTED / name).read_text(encoding="utf-8"), (
            f"{name} differs from tests/data/expected; if the change is intended, "
            "rerun with ROCM_TRACE_GEN_UPDATE_GOLDEN=1"
        )


def test_every_entry_has_a_probe_or_a_reason(source):
    installed = set(re.findall(r"table->(\w+)_fn = probe_", source))
    skipped = set(re.findall(r"// (\w+)_fn: not traced \(", source))
    assert skipped == {"fakeSkipped"}
    assert installed == PROBED
    assert "fakeRetired" not in source


def test_one_enter_and_one_exit_write_per_api(source):
    # One call site per event name: LTTng corrupts a trace in which one event
    # name has two field layouts.
    enters = re.findall(r'TraceLoggingWrite\(\w+, "(\w+)_enter"', source)
    exits = re.findall(r'TraceLoggingWrite\(\w+, "(\w+)_exit"', source)
    assert sorted(enters) == sorted(exits) == sorted(PROBED)
    assert source.count("WINEVENT_OPCODE_START") == len(PROBED)
    assert source.count("WINEVENT_OPCODE_STOP") == len(PROBED)


def test_exceptions_write_exit_and_rethrow(source):
    probe = _function(source, "probe_FakeTable_fakeThrow")
    assert (
        "} catch (...) {\n        exit_FakeTable_fakeThrow(true, rocm_trace_result);\n"
        in probe
    )
    assert "        throw;\n" in probe
    assert "exit_FakeTable_fakeThrow(false, rocm_trace_result);" in probe
    exit_write = _function(source, "exit_FakeTable_fakeThrow")
    assert (
        'TraceLoggingBoolean(static_cast<uint8_t>(rocm_trace_exception), "exception")'
        in exit_write
    )


def test_outputs_are_recorded_only_after_success(source):
    exit_write = _function(source, "exit_FakeTable_fakeAlloc")
    assert (
        "(fake_capture_level >= ROCM_TRACE_CAPTURE_FULL && !rocm_trace_exception && "
        "rocm_trace_result == fakeSuccess && ptr != nullptr ? "
        "reinterpret_cast<const void*>((*ptr)) : static_cast<const void*>(nullptr)), "
        '"ptr")'
    ) in exit_write


def test_field_names_follow_the_contract(source):
    # RC-API-1: arg_<N> for unnamed parameters, the arg_ prefix for reserved
    # names, the public declaration's names, and outputs under their own names.
    unnamed = _function(source, "probe_FakeTable_fakeUnnamed")
    assert '"arg_1"' in unnamed and '"arg_2"' in unnamed
    reserved = _function(source, "probe_FakeTable_fakeReserved")
    for name in ("arg_result", "arg_timestamp", "arg_exception"):
        assert f'"{name}"' in reserved
    assert '"arg_exception"' in _function(source, "exit_FakeTable_fakeReserved")
    renamed = _function(source, "probe_FakeRenamedTable_fakeRenamed")
    assert "int public_name" in renamed
    assert "_out" not in source


def test_const_char_inputs_are_capped_strings(source):
    load = _function(source, "probe_FakeTable_fakeLoad")
    assert "TraceLoggingCountedString(" in load and "ROCM_TRACE_STRING_CAP" in load
    get_function = _function(source, "probe_FakeTable_fakeGetFunction")
    assert (
        "TraceLoggingString((fake_capture_level >= ROCM_TRACE_CAPTURE_FULL ? kname"
        in get_function
    )
    scratch = _function(source, "probe_FakeTable_fakeScratch")
    assert "reinterpret_cast<const void*>(buffer)" in scratch


def test_aggregates_record_all_components(source):
    launch = _function(source, "probe_FakeTable_fakeLaunchEx")
    for name in ("config_grid_x", "config_extra_1_z", "config_flags"):
        assert f'"{name}"' in launch
    assert "TraceLoggingPointerFixedArray(" in launch
    # Floating-point arrays are one field per element: LTTng has no float array fields.
    assert '"config_weights_1"' in launch and "Float32FixedArray" not in source
    # A named member of an unnamed struct type keeps its name.
    assert "config.named.b" in launch and '"config_named_b"' in launch
    bits = _function(source, "probe_FakeTable_fakeBits")
    assert "TraceLoggingUInt8FixedArray(" in bits and '"value_union"' in bits
    assert "TraceLoggingInt8FixedArray(" in _function(
        source, "probe_FakeTable_fakeOpen"
    )


def test_enums_keep_their_signedness(source):
    wide = _function(source, "probe_FakeTable_fakeWide")
    assert "TraceLoggingUInt64(" in wide and "TraceLoggingUInt32(" in wide


def test_capture_levels_guard_fields(source):
    load = _function(source, "probe_FakeTable_fakeLoad")
    assert "fake_capture_level >= ROCM_TRACE_CAPTURE_FULL ? path" in load
    assert (
        "fake_capture_level >= ROCM_TRACE_CAPTURE_SCALARS ? "
        "reinterpret_cast<const void*>(stream)"
    ) in load
    exit_write = _function(source, "exit_FakeTable_fakeAlloc")
    assert (
        'TraceLoggingUInt32(static_cast<uint32_t>(rocm_trace_result), "result")'
        in exit_write
    )


def test_check_passes_on_current_files_and_fails_on_stale(
    fake_config, tmp_path, capsys
):
    for name in ("fake_probes.h", "fake_probes.cpp", "fake_probes.json"):
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
    args = ["--config", str(fake_config), "--output-dir", str(out), "-I", str(DATA)]
    assert run_cli(*args) == 0
    expected = (EXPECTED / "fake_probes.cpp").read_text()
    assert (out / "fake_probes.cpp").read_text() == expected
    assert run_cli(*args, "--check") == 0


def test_char_input_listed_in_strings_is_a_string(tmp_path):
    # An API that is not const-correct (__hipRegisterFunction's char* deviceFunction): the
    # annotation records the char* input as text instead of its address.
    data = yaml.safe_load((DATA / "fake.yaml").read_text())
    data["apis"]["fakeScratch"] = {"strings": ["buffer"], "uncapped": ["buffer"]}
    config = tmp_path / "fake.yaml"
    config.write_text(yaml.safe_dump(data))
    files = generate_files(config)
    scratch = _function(files["fake_probes.cpp"], "probe_FakeTable_fakeScratch")
    assert (
        "TraceLoggingString((fake_capture_level >= ROCM_TRACE_CAPTURE_FULL ? buffer"
        in scratch
    )
    assert "reinterpret_cast<const void*>(buffer)" not in scratch
