# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

_MAX_ARGS_LEN = 512
_MAX_ARG_ITEMS = 32
_MAX_NESTED_ARG_ITEMS = 8


def encode_args(args: str) -> str:
    return (
        args.replace("%", "%25")
        .replace("|", "%7C")
        .replace(";", "%3B")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def cap_args(blob: str, max_len: int = _MAX_ARGS_LEN) -> str:
    if len(blob) <= max_len:
        return blob
    balanced = blob.startswith("(") and blob.endswith(")")
    truncated = blob[:max_len]
    return truncated + ("...)" if balanced else "...")
