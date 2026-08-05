# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Wire-format codec for the marker args blob.

The blob occupies the ``|args=<ENC>`` segment of a marker. ``%``, ``|``, ``;``,
and newlines are percent-encoded so the segment is parsed as a single field.
"""

# Bounds applied when building an args blob.
MAX_ARGS_LEN = 512
MAX_ARG_ITEMS = 32


def encode_args(args: str) -> str:
    """Percent-encode the reserved characters in an args blob."""
    if not args:
        return ""
    return (
        args
        .replace("%", "%25")
        .replace("|", "%7C")
        .replace(";", "%3B")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def decode_args(encoded: str) -> str:
    """Inverse of :func:`encode_args`."""
    if not encoded:
        return ""
    return (
        encoded
        .replace("%0A", "\n")
        .replace("%0D", "\r")
        .replace("%7C", "|")
        .replace("%3B", ";")
        .replace("%25", "%")
    )


def cap_args(blob: str) -> str:
    """Truncate an args blob to ``MAX_ARGS_LEN`` characters, appending ``...``
    when truncated."""
    if len(blob) > MAX_ARGS_LEN:
        return blob[:MAX_ARGS_LEN] + "..."
    return blob
