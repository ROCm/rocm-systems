# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared ML marker wire encoding and parsing."""

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ml_api_tracing.constants import KNOWN_ML_API_BACKENDS, USER_REGION_BACKEND

_MAX_ARGS_LEN = 512
_MAX_TENSOR_ARGUMENT_ITEMS = 32
_STACKED_MARKER_WIRE = re.compile(r":#\d+@")


def encode_marker_name(name: str) -> str:
    """Escape percent signs and slashes in the operator name."""
    return name.replace("%", "%25").replace("/", "%2F")


def compose_marker(
    marker: str,
    context: str,
    backend: str = "",
    args: str = "n/a",
    seqNr: str = "n/a",
    tid: str = "n/a",
    ftid: str = "n/a",
    ltid: str = "n/a",
    scope: str = "n/a",
) -> str:
    """Encode the raw marker name while preserving supplied args verbatim."""
    backend_suffix = f"|{backend}" if backend else ""
    return (
        f"{encode_marker_name(marker)}:{context}"
        f"|seqNr={seqNr}|tid={tid}|ftid={ftid}|ltid={ltid}"
        f"|scope={scope}|args={args}{backend_suffix}"
    )


def encode_args(args: str) -> str:
    """Escape argument text so it cannot introduce marker field separators."""
    return (
        args
        .replace("%", "%25")
        .replace("|", "%7C")
        .replace(";", "%3B")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def cap_args(blob: str, max_len: int = _MAX_ARGS_LEN) -> str:
    """Truncate to max_len characters and add '...', or '...)' for a
    parenthesized blob.
    """
    if len(blob) <= max_len:
        return blob
    balanced = blob.startswith("(") and blob.endswith(")")
    truncated = blob[:max_len]
    return truncated + ("...)" if balanced else "...")


def format_args_blob(parts: List[str]) -> str:
    """Join argument parts into a capped, encoded (a, b) marker blob,
    or n/a if empty."""
    if not parts:
        return "n/a"
    return encode_args(cap_args("(" + ", ".join(parts) + ")"))


def render_dtype_shape(obj: object) -> Optional[str]:
    """Return dtype[dims] for an object with shape and dtype, else None.

    A 0-d value (empty shape) prints repr(obj.item()), for example 42. If
    item() is missing or raises, print the dtype with empty brackets, for
    example float64[].
    """
    try:
        shape = getattr(obj, "shape", None)
        dtype = getattr(obj, "dtype", None)
        if shape is None or dtype is None:
            return None
        dims = "x".join(str(int(d)) for d in shape)
        dt = str(dtype).replace("torch.", "")
        if dims:
            return f"{dt}[{dims}]"
        return _render_empty_shape(obj, dt)
    except Exception:
        return None


def _render_empty_shape(obj: object, dt: str) -> str:
    """Return repr(obj.item()), or the dtype with empty brackets if item() fails."""
    item = getattr(obj, "item", None)
    if not callable(item):
        return f"{dt}[]"
    try:
        return repr(item())
    except Exception:
        return f"{dt}[]"


def decode_marker_name(name: str) -> str:
    """Decode a percent-encoded marker segment ('%2F' -> '/', '%25' -> '%')."""
    return name.replace("%2F", "/").replace("%25", "%")


def decode_marker_args(args: str) -> str:
    """Decode percent-encoded argument text back to '|', ';', CR, LF, and '%'."""
    return (
        args
        .replace("%7C", "|")
        .replace("%3B", ";")
        .replace("%0D", "\r")
        .replace("%0A", "\n")
        .replace("%25", "%")
    )


def split_operator_args(args_blob: str) -> List[str]:
    """Split operator arguments on commas outside brackets and quoted strings."""
    args_text = args_blob.strip()
    if args_text.startswith("(") and args_text.endswith(")"):
        args_text = args_text[1:-1]
    args_text = args_text.strip()
    if not args_text:
        return []

    tokens: List[str] = []
    token_chars: List[str] = []
    depth = 0
    quote_char: Optional[str] = None
    escaped = False
    for char in args_text:
        if quote_char is not None:
            token_chars.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                quote_char = None
        elif char in "'\"":
            quote_char = char
            token_chars.append(char)
        elif char in "([{":
            depth += 1
            token_chars.append(char)
        elif char in ")]}":
            depth = max(depth - 1, 0)
            token_chars.append(char)
        elif char == "," and depth == 0:
            tokens.append("".join(token_chars).strip())
            token_chars = []
        else:
            token_chars.append(char)
    tokens.append("".join(token_chars).strip())
    return [token for token in tokens if token]


def parse_marker_function(function_value: object) -> Dict[str, Any]:
    """Read the operator name, source location and metadata from a Function cell.

    Untagged markers are user ROCTx regions. Missing locations and launcher
    thread IDs become empty values, while missing arguments remain 'n/a'.
    """
    if function_value is None or (
        isinstance(function_value, float) and math.isnan(function_value)
    ):
        raw = ""
    else:
        raw = str(function_value)
    tokens = raw.split("|") if raw else [""]
    first_token = tokens[0]
    operator_name, file_name, line_number = _split_name_and_location(first_token)
    parsed_keys = {
        "ltid": "n/a",
        "args": "n/a",
    }
    backend = USER_REGION_BACKEND
    for token in tokens[1:]:
        if token in KNOWN_ML_API_BACKENDS:
            backend = token
            continue
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key == "args":
            parsed_keys["args"] = decode_marker_args(value)
        elif key in parsed_keys:
            parsed_keys[key] = value
    launcher_thread_id = ""
    if parsed_keys["ltid"] not in ("", "n/a"):
        launcher_thread_id = parsed_keys["ltid"]
    return {
        "Operator_Name": decode_marker_name(operator_name),
        "File_Name": file_name,
        "Line_Number": line_number,
        "Backend": backend,
        "launcher_thread_id": launcher_thread_id,
        "args": parsed_keys["args"],
    }


def stacked_marker_function_cell(function_value: object) -> Optional[str]:
    """Return the raw cell when it uses legacy stacked wire, else None."""
    if function_value is None or (
        isinstance(function_value, float) and math.isnan(function_value)
    ):
        return None
    raw = str(function_value)
    first_token = raw.split("|", 1)[0] if raw else ""
    if _STACKED_MARKER_WIRE.search(first_token):
        return raw
    return None


def _split_name_and_location(first_token: str) -> Tuple[str, str, Optional[int]]:
    """Read a trailing source path and line, keeping colons in operator names."""
    if first_token.endswith(":n/a"):
        return first_token[: -len(":n/a")], "", None
    parts = first_token.rsplit(":", 2)
    if len(parts) == 3:
        operator_name, file_part, line_part = parts
        if file_part and line_part.isdigit():
            return operator_name, Path(file_part).name, int(line_part)
    return first_token, "", None
