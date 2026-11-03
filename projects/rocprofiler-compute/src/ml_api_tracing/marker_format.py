# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared ML marker wire encoding and parsing."""

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ml_api_tracing.constants import KNOWN_ML_API_BACKENDS

_MAX_ARGS_LEN = 512
_STACKED_MARKER_WIRE = re.compile(r":#\d+@")


def encode_marker_name(name: str) -> str:
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
    full = (
        f"{encode_marker_name(marker)}:{context}"
        f"|seqNr={seqNr}|tid={tid}|ftid={ftid}|ltid={ltid}"
        f"|scope={scope}|args={args}"
    )
    if backend:
        full = f"{full}|{backend}"
    return full


def encode_args(args: str) -> str:
    return (
        args
        .replace("%", "%25")
        .replace("|", "%7C")
        .replace(";", "%3B")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def cap_args(blob: str, max_len: int = _MAX_ARGS_LEN) -> str:
    """Cap the raw prefix; the appended truncation suffix can exceed max_len."""
    if len(blob) <= max_len:
        return blob
    balanced = blob.startswith("(") and blob.endswith(")")
    truncated = blob[:max_len]
    return truncated + ("...)" if balanced else "...")


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
    """Split a parenthesized operator-args blob into top-level argument tokens."""
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
    """Parse one Function cell into operator, location, backend, ltid, and args."""
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
    backend = "user"
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
    """Split `{name}:{location}` from the right into name, file, and line."""
    if first_token.endswith(":n/a"):
        return first_token[: -len(":n/a")], "", None
    parts = first_token.rsplit(":", 2)
    if len(parts) == 3:
        operator_name, file_part, line_part = parts
        if file_part and line_part.isdigit():
            return operator_name, Path(file_part).name, int(line_part)
    return first_token, "", None
