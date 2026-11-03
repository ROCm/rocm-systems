# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the shared ML API marker wire format."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from ml_api_tracing.marker_format import (
    cap_args,
    compose_marker,
    decode_marker_args,
    decode_marker_name,
    encode_args,
    encode_marker_name,
    parse_marker_function,
    split_operator_args,
    stacked_marker_function_cell,
)

# Matches MarkerEncoding.NameFields_EncodeOnceAndPreserveEncodedArguments in
# src/lib/torch_trace_collector/tests/test_torch_trace_collector.cpp.
NATIVE_ENCODED_MARKER = (
    "aten::add%2F%25:n/a|seqNr=7|tid=11|ftid=13|ltid=17|scope=FUNCTION"
    "|args=(value=%25%7C%3B%0D%0A)|torch"
)


def test_encode_marker_name_wrap_chars():
    assert encode_marker_name("nn.Module.Linear.forward") == "nn.Module.Linear.forward"
    assert encode_marker_name("pkg/mod%hook") == "pkg%2Fmod%25hook"
    assert encode_marker_name("aten::addmm") == "aten::addmm"


def test_marker_name_encoding_preserves_literal_percent_escapes():
    marker_name = "pkg/%2F/%25/%2f"
    encoded_name = "pkg%2F%252F%2F%2525%2F%252f"
    assert encode_marker_name(marker_name) == encoded_name
    assert decode_marker_name(encoded_name) == marker_name
    assert decode_marker_name("%2f%2F%252F%25") == "%2f/%2F%"


def test_compose_marker_python_wrap_wire():
    expected = (
        "nn.Module.Linear.forward:simple_net.py:51"
        "|seqNr=n/a|tid=n/a|ftid=n/a|ltid=n/a|scope=n/a|args=n/a|torch"
    )
    assert (
        compose_marker("nn.Module.Linear.forward", "simple_net.py:51", backend="torch")
        == expected
    )
    empty_backend = compose_marker(
        "nn.Module.Linear.forward", "simple_net.py:51", backend=""
    )
    assert empty_backend.endswith("args=n/a")
    assert not empty_backend.endswith("args=n/a|")


def test_compose_marker_matches_native_wire_field_order():
    assert (
        compose_marker(
            "aten::add/%",
            "n/a",
            backend="torch",
            args="(value=%25%7C%3B%0D%0A)",
            seqNr="7",
            tid="11",
            ftid="13",
            ltid="17",
            scope="FUNCTION",
        )
        == NATIVE_ENCODED_MARKER
    )


def test_compose_marker_preserves_supplied_arguments_without_capping():
    encoded_arguments = "%7C" * 513
    assert compose_marker("pkg/mod%", "file.py:2", args=encoded_arguments) == (
        "pkg%2Fmod%25:file.py:2"
        "|seqNr=n/a|tid=n/a|ftid=n/a|ltid=n/a|scope=n/a|args=" + encoded_arguments
    )


def test_encode_args_wrap_delimiters():
    assert "%7C" in encode_args("(float32[2x4], bias=float32[2x4]|x)")
    assert encode_args("%|;|\r\n") == "%25%7C%3B%7C%0D%0A"


def test_argument_encoding_preserves_literal_percent_escapes():
    arguments = "%7C|%3B;%0D\r%0A\n%25"
    encoded_arguments = "%257C%7C%253B%3B%250D%0D%250A%0A%2525"
    assert encode_args(arguments) == encoded_arguments
    assert decode_marker_args(encoded_arguments) == arguments
    assert decode_marker_args("%7c%7C%257C%3b%3B%0d%0D%0a%0A") == (
        "%7c|%7C%3b;%0d\r%0a\n"
    )


def test_cap_args_balanced_parens():
    assert cap_args("x" * 512) == "x" * 512
    assert cap_args("x" * 513) == "x" * 512 + "..."
    blob = "(" + "float32[2x4], " * 40 + ")"
    assert cap_args(blob) == blob[:512] + "...)"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        pytest.param("(abc)", "(a...)", id="balanced"),
        pytest.param("(abc", "(a...", id="open-only"),
        pytest.param("abc)", "ab...", id="close-only"),
        pytest.param("ab", "ab", id="at-limit"),
    ],
)
def test_cap_args_appends_suffix_after_requested_prefix(arguments, expected):
    assert cap_args(arguments, max_len=2) == expected


def test_split_operator_args_respects_nested_commas():
    tokens = split_operator_args("(self=float32[2, 2], other=(1, 2), name='a,b')")
    assert tokens == ["self=float32[2, 2]", "other=(1, 2)", "name='a,b'"]


def test_split_operator_args_empty_blob():
    assert split_operator_args("") == []
    assert split_operator_args("()") == []
    assert split_operator_args("  (  )  ") == []


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        pytest.param(
            '(config={"x": [1, 2]}, label="a,b")',
            ['config={"x": [1, 2]}', 'label="a,b"'],
            id="nested-mapping-and-double-quotes",
        ),
        pytest.param(
            r"(label='a\',b', value=2)",
            [r"label='a\',b'", "value=2"],
            id="escaped-quote",
        ),
        pytest.param("a,, ,b,", ["a", "b"], id="empty-tokens"),
        pytest.param("a],b", ["a]", "b"], id="unmatched-close"),
        pytest.param("(a, nested=[1, 2", ["(a, nested=[1, 2"], id="truncated"),
    ],
)
def test_split_operator_args_preserves_permissive_tokens(arguments, expected):
    assert split_operator_args(arguments) == expected


def test_parse_marker_function_aten_addmm_n_a():
    parsed = parse_marker_function(
        "aten::addmm:n/a|seqNr=1|tid=1|ftid=0|scope=FUNCTION|args=()|torch"
    )
    assert parsed["Operator_Name"] == "aten::addmm"
    assert parsed["File_Name"] == ""
    assert parsed["Line_Number"] is None
    assert parsed["Backend"] == "torch"
    assert parsed["launcher_thread_id"] == ""
    assert parsed["args"] == "()"


def test_parse_marker_function_autograd_keeps_inner_colon_and_space():
    parsed = parse_marker_function(
        "autograd::engine::evaluate_function: SumBackward0:n/a"
        "|seqNr=n/a|tid=2|ftid=0|scope=FUNCTION|args=n/a|torch"
    )
    assert (
        parsed["Operator_Name"] == "autograd::engine::evaluate_function: SumBackward0"
    )
    assert parsed["File_Name"] == ""
    assert parsed["Line_Number"] is None


def test_parse_marker_function_linear_forward_file_and_line():
    parsed = parse_marker_function(
        "nn.Module.Linear.forward:simple_torch_code.py:19"
        "|seqNr=n/a|tid=n/a|ftid=n/a|scope=n/a|args=()|torch"
    )
    assert parsed["Operator_Name"] == "nn.Module.Linear.forward"
    assert parsed["File_Name"] == "simple_torch_code.py"
    assert parsed["Line_Number"] == 19


def test_parse_marker_function_triton_backend_and_location():
    parsed = parse_marker_function(
        "triton.JITFunction.rmsnorm_kernel:llama_triton_layer.py:381"
        "|seqNr=n/a|tid=n/a|ftid=n/a|scope=n/a|args=n/a|triton"
    )
    assert parsed["Backend"] == "triton"
    assert parsed["File_Name"] == "llama_triton_layer.py"
    assert parsed["Line_Number"] == 381


def test_parse_marker_function_user_range_without_keys():
    parsed = parse_marker_function("training_loop")
    assert parsed["Operator_Name"] == "training_loop"
    assert parsed["File_Name"] == ""
    assert parsed["Line_Number"] is None
    assert parsed["Backend"] == "user"


def test_parse_marker_function_decodes_name_and_args():
    parsed = parse_marker_function(
        "aten::foo%2Fbar%25baz:n/a"
        "|seqNr=1|tid=1|ftid=0|scope=FUNCTION"
        "|args=a%7Cb%3Bc%0Dd%0Ae%25f|torch"
    )
    assert parsed["Operator_Name"] == "aten::foo/bar%baz"
    assert parsed["args"] == "a|b;c\rd\ne%f"


def test_parse_marker_function_reads_native_wire_and_keeps_return_key_order():
    parsed = parse_marker_function(NATIVE_ENCODED_MARKER)
    assert parsed == {
        "Operator_Name": "aten::add/%",
        "File_Name": "",
        "Line_Number": None,
        "Backend": "torch",
        "launcher_thread_id": "17",
        "args": "(value=%|;\r\n)",
    }
    assert list(parsed) == [
        "Operator_Name",
        "File_Name",
        "Line_Number",
        "Backend",
        "launcher_thread_id",
        "args",
    ]


def test_parse_marker_function_keeps_last_recognized_duplicate():
    parsed = parse_marker_function(
        "operator:n/a|args=first|ltid=1|torch|extra=data|triton"
        "|args=value=a%7Cb|ltid=7|unknown-backend"
    )
    assert parsed["args"] == "value=a|b"
    assert parsed["launcher_thread_id"] == "7"
    assert parsed["Backend"] == "triton"


@pytest.mark.parametrize(
    ("launcher_thread", "expected"),
    [("", ""), ("n/a", ""), ("0", "0")],
)
def test_parse_marker_function_preserves_empty_argument_and_launcher_fields(
    launcher_thread, expected
):
    parsed = parse_marker_function(f"operator|ltid={launcher_thread}|args=")
    assert parsed["launcher_thread_id"] == expected
    assert parsed["args"] == ""


@pytest.mark.parametrize(
    ("function_value", "expected_name"),
    [
        pytest.param(None, "", id="none"),
        pytest.param(float("nan"), "", id="python-float-nan"),
        pytest.param(np.float64("nan"), "", id="numpy-float64-nan"),
        pytest.param(np.float32("nan"), "nan", id="numpy-float32-nan"),
        pytest.param(Decimal("NaN"), "NaN", id="decimal-nan"),
        pytest.param(pd.NA, "<NA>", id="pandas-na"),
        pytest.param(pd.NaT, "NaT", id="pandas-nat"),
        pytest.param(1.5, "1.5", id="finite-float"),
        pytest.param(float("inf"), "inf", id="infinite-float"),
    ],
)
def test_parse_marker_function_only_treats_float_nan_as_missing(
    function_value, expected_name
):
    assert parse_marker_function(function_value) == {
        "Operator_Name": expected_name,
        "File_Name": "",
        "Line_Number": None,
        "Backend": "user",
        "launcher_thread_id": "",
        "args": "n/a",
    }


@pytest.mark.parametrize(
    ("function_value", "expected_name", "expected_file", "expected_line"),
    [
        pytest.param(
            "aten::mm:/tmp/package/model.py:12|torch",
            "aten::mm",
            "model.py",
            12,
            id="basename",
        ),
        pytest.param(
            "operator:file.py:bad", "operator:file.py:bad", "", None, id="invalid-line"
        ),
        pytest.param("operator::12", "operator::12", "", None, id="empty-file"),
    ],
)
def test_parse_marker_function_preserves_permissive_location_rules(
    function_value, expected_name, expected_file, expected_line
):
    parsed = parse_marker_function(function_value)
    assert parsed["Operator_Name"] == expected_name
    assert parsed["File_Name"] == expected_file
    assert parsed["Line_Number"] == expected_line


@pytest.mark.parametrize(
    ("function_value", "expected"),
    [
        pytest.param(None, None, id="none"),
        pytest.param(float("nan"), None, id="float-nan"),
        pytest.param("", None, id="empty"),
        pytest.param(
            "triton.JITFunction.foo:#1@file.py:1|triton",
            "triton.JITFunction.foo:#1@file.py:1|triton",
            id="legacy-stack",
        ),
        pytest.param("operator|args=:#1@file.py", None, id="arguments-only"),
        pytest.param("triton.JITFunction.foo@file.py", None, id="ordinary-at"),
        pytest.param("triton.JITFunction.foo:#file.py", None, id="no-frame-number"),
    ],
)
def test_stacked_marker_function_cell_checks_only_legacy_first_token(
    function_value, expected
):
    assert stacked_marker_function_cell(function_value) == expected


def test_marker_format_import_fresh_with_only_stdlib_under_profile_guard(
    assert_stdlib_import,
):
    assert_stdlib_import("ml_api_tracing.marker_format")
