# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Every configuration mistake is reported, with a message that names it."""

import re

import pytest
import yaml

from conftest import DATA, ROCM_TRACE_INCLUDE, generate_files, run_cli
from rocm_trace_gen.config import EXTRA_KEYWORDS, ConfigError
from rocm_trace_gen.emit import GenerateError
from rocm_trace_gen.parse import ParseError


def _write(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _base():
    return yaml.safe_load((DATA / "fake.yaml").read_text())


def _with_api(name, entry):
    data = _base()
    data["apis"][name] = entry
    return data


GENERATE_ERRORS = [
    (
        _with_api("fakeNotAnEntry", {"level": "verbose"}),
        "apis.fakeNotAnEntry: no table has this entry",
    ),
    (
        _with_api("fakeAlloc", {"out": ["pointer"]}),
        "out names 'pointer', which is not a parameter",
    ),
    (_with_api("fakeName", {"out": ["len"]}), "out parameter 'len' is not a pointer"),
    (
        _with_api("fakeAlloc", {"strings": ["size"]}),
        "strings names 'size', which is not a char pointer",
    ),
    (
        _with_api("fakeOpen", {}),
        "parameter 'handle' has no field mapping: struct 'fakeHandle'",
    ),
    (
        _with_api("fakeWide", {"out": ["props"]}),
        "out parameter 'props' points to a type with no field mapping",
    ),
    (
        _with_api("fakeNoResult", {"strings": ["result"]}),
        "strings names 'result', which is not a char pointer",
    ),
]


@pytest.mark.parametrize("data,message", GENERATE_ERRORS)
def test_generate_errors(tmp_path, data, message):
    with pytest.raises(GenerateError) as err:
        generate_files(_write(tmp_path, data))
    assert any(message in e for e in err.value.errors), err.value.errors


def test_all_errors_are_reported_together(tmp_path):
    data = _base()
    data["apis"]["fakeOpen"] = {}
    data["apis"]["fakeNotAnEntry"] = {"level": "verbose"}
    with pytest.raises(GenerateError) as err:
        generate_files(_write(tmp_path, data))
    assert len(err.value.errors) == 2


CONFIG_ERRORS = [
    (_with_api("fakeSkipped", {"skip": True}), "skip needs a reason"),
    (
        _with_api("fakeSkipped", {"skip": True, "reason": "x", "out": []}),
        "a skipped API takes no out",
    ),
    (_with_api("fakeAlloc", {"reason": "x"}), "reason is only for skip"),
    (_with_api("fakeAlloc", {"keywords": ["BOGUS"]}), "keywords: unknown BOGUS"),
    (_with_api("fakeAlloc", {"keywords": ["API"]}), "keywords: unknown API"),
    (
        _with_api("fakeAlloc", {"level": "debug"}),
        "level must be one of information, verbose",
    ),
    (_with_api("fakeAlloc", {"out": ["result"]}), "the result is always recorded"),
    (
        _with_api("fakeAlloc", {"out": ["ptr"], "omit": ["ptr"]}),
        "ptr both omitted and recorded",
    ),
    (_with_api("fakeAlloc", {"outs": ["ptr"]}), "unknown key(s) outs"),
    ({**_base(), "version": 2}, "only version 1 is supported"),
    ({**_base(), "extra": 1}, "unknown key(s) extra"),
    ({k: v for k, v in _base().items() if k != "provider"}, "missing provider"),
    ({**_base(), "includes": ["fake table.h"]}, "not a header path"),
    (
        {**_base(), "tables": [{"name": "FakeTable", "size_member": "size"}] * 2},
        "listed twice",
    ),
]


@pytest.mark.parametrize("data,message", CONFIG_ERRORS)
def test_config_errors(tmp_path, data, message):
    with pytest.raises(ConfigError) as err:
        generate_files(_write(tmp_path, data))
    assert message in str(err.value)


def test_duplicate_keys_are_rejected(tmp_path):
    text = (DATA / "fake.yaml").read_text() + "  fakeAlloc:\n    level: verbose\n"
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="duplicate key 'fakeAlloc'"):
        generate_files(path)


def test_missing_table(tmp_path):
    data = _base()
    data["tables"].append({"name": "NoSuchTable", "size_member": "size"})
    with pytest.raises(
        ParseError, match="tables not found in the headers: NoSuchTable"
    ):
        generate_files(_write(tmp_path, data))


def test_missing_size_member(tmp_path):
    data = _base()
    data["tables"][0]["size_member"] = "bytes"
    with pytest.raises(ParseError, match="FakeTable has no size member 'bytes'"):
        generate_files(_write(tmp_path, data))


def test_header_errors_fail_the_parse(tmp_path):
    data = _base()
    data["includes"] = ["no_such_header.h"]
    with pytest.raises(ParseError, match="no_such_header.h"):
        generate_files(_write(tmp_path, data))


def test_cli_reports_errors_and_fails(tmp_path, capsys):
    path = _write(tmp_path, _with_api("fakeOpen", {}))
    assert (
        run_cli("--config", str(path), "--output-dir", str(tmp_path), "-I", str(DATA))
        == 1
    )
    err = capsys.readouterr().err
    assert "parameter 'handle' has no field mapping" in err and "1 error(s)" in err
    assert not (tmp_path / "fake_probes.cpp").exists()


def test_keywords_match_values_header():
    text = (ROCM_TRACE_INCLUDE / "rocm" / "trace" / "values.h").read_text()
    names = re.findall(r"#define ROCM_TRACE_KW_(\w+) ", text)
    assert sorted(names) == sorted(("API",) + EXTRA_KEYWORDS)


def test_duplicate_field_names_are_rejected(tmp_path):
    (tmp_path / "dup_table.h").write_text(
        "typedef struct dupDim3 { unsigned x, y, z; } dupDim3;\n"
        "typedef int (*t_dupLaunch)(dupDim3 grid, int grid_x);\n"
        "struct DupTable { unsigned long size; t_dupLaunch dupLaunch_fn; };\n"
    )
    config = {
        "version": 1,
        "provider": "dup_provider",
        "output_name": "dup_probes",
        "includes": ["dup_table.h"],
        "tables": [{"name": "DupTable", "size_member": "size"}],
    }
    path = _write(tmp_path, config)
    with pytest.raises(GenerateError) as err:
        generate_files(path, [f"-I{tmp_path}"])
    assert any("two fields named 'grid_x'" in e for e in err.value.errors)

    config["apis"] = {"dupLaunch": {"omit": ["grid_x"]}}
    files = generate_files(_write(tmp_path, config), [f"-I{tmp_path}"])
    assert '"grid_x"' in files["dup_probes.cpp"]
