# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Golden tests for the marker format shared with native producers."""

import pytest

from utils.inject_roctx import core
from utils.inject_roctx.marker_format import cap_args, encode_args


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("plain", "plain", id="unchanged"),
        pytest.param("%|;\r\n", "%25%7C%3B%0D%0A", id="reserved-characters"),
    ],
)
def test_encode_args_escapes_reserved_characters(raw: str, expected: str) -> None:
    assert encode_args(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("x" * 512, "x" * 512, id="exact-limit"),
        pytest.param("x" * 513, ("x" * 512) + "...", id="unbalanced"),
        pytest.param(
            "(" + ("x" * 511) + ")",
            "(" + ("x" * 511) + "...)",
            id="balanced",
        ),
    ],
)
def test_cap_args_preserves_truncation_contract(
    raw: str,
    expected: str,
) -> None:
    assert cap_args(raw) == expected


def test_compose_marker_matches_native_wire_golden() -> None:
    encoded_args = encode_args("(value=%|;\r\n)")

    marker = core.compose_marker(
        "aten::add/%",
        "n/a",
        backend="torch",
        args=encoded_args,
        seqNr="7",
        tid="11",
        ftid="13",
        ltid="17",
        scope="FUNCTION",
    )

    assert marker == (
        "aten::add%2F%25:n/a|seqNr=7|tid=11|ftid=13|ltid=17|scope=FUNCTION|"
        "args=(value=%25%7C%3B%0D%0A)|torch"
    )


def test_compose_marker_defaults_launcher_tid_to_unavailable() -> None:
    assert core.compose_marker("operation", "n/a") == (
        "operation:n/a|seqNr=n/a|tid=n/a|ftid=n/a|ltid=n/a|scope=n/a|args=n/a"
    )
