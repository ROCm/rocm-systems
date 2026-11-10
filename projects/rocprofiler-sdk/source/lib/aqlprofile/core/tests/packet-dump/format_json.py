#!/usr/bin/env python3
"""Format generated packet-dump JSON for human inspection.

Examples:
    python3 format_json.py build/bin/packet-dumps/gfx942/SQ.json
    python3 format_json.py build/bin/packet-dumps
"""

import argparse
import json
import os
import stat
import sys
import tempfile
from decimal import Decimal, DecimalException
from pathlib import Path

INDENT_WIDTH = 2


class FormatError(Exception):
    """An error that can be reported directly to the command-line user."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Format JSON packet dumps for human inspection."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report files that would change without rewriting them",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="JSON file or directory to search recursively for *.json",
    )
    return parser.parse_args(argv)


def collect_files(input_paths):
    files = []
    seen = set()

    for input_path in input_paths:
        path = Path(input_path).expanduser()
        if path.is_file():
            candidates = (path,)
        elif path.is_dir():
            try:
                candidates = sorted(path.rglob("*.json"))
            except OSError as error:
                raise FormatError(f"{path}: cannot search directory: {error}") from error
        else:
            raise FormatError(f"{path}: not a file or directory")

        for candidate in candidates:
            if not candidate.is_file():
                continue
            key = candidate.resolve()
            if key not in seen:
                seen.add(key)
                files.append(candidate)

    if not files:
        raise FormatError("no JSON files found in the supplied paths")
    return files


def reject_non_json_constant(value):
    raise ValueError(f"invalid JSON constant {value!r}")


def encode_scalar(value):
    if isinstance(value, Decimal):
        return str(value)
    return json.dumps(value, ensure_ascii=True, allow_nan=False)


def indent(depth):
    return " " * (INDENT_WIDTH * depth)


def format_value(value, depth=0):
    if isinstance(value, dict):
        if not value:
            return "{}"
        if all(not isinstance(child, (dict, list)) for child in value.values()):
            entries = [
                f"{encode_scalar(key)}: {format_value(child, depth)}"
                for key, child in value.items()
            ]
            return "{" + ", ".join(entries) + "}"
        entries = [
            f"{indent(depth + 1)}{encode_scalar(key)}: "
            f"{format_value(child, depth + 1)}"
            for key, child in value.items()
        ]
        return "{\n" + ",\n".join(entries) + f"\n{indent(depth)}}}"

    if isinstance(value, list):
        if all(not isinstance(child, (dict, list)) for child in value):
            return "[" + ", ".join(encode_scalar(child) for child in value) + "]"
        entries = [
            f"{indent(depth + 1)}{format_value(child, depth + 1)}" for child in value
        ]
        return "[\n" + ",\n".join(entries) + f"\n{indent(depth)}]"

    return encode_scalar(value)


def formatted_contents(path):
    try:
        source = path.read_bytes()
    except OSError as error:
        raise FormatError(f"{path}: cannot read file: {error}") from error

    try:
        document = json.loads(
            source,
            parse_int=Decimal,
            parse_float=Decimal,
            parse_constant=reject_non_json_constant,
        )
    except json.JSONDecodeError as error:
        raise FormatError(
            f"{path}: invalid JSON at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error
    except (DecimalException, UnicodeError, ValueError) as error:
        raise FormatError(f"{path}: invalid JSON: {error}") from error
    except RecursionError as error:
        raise FormatError(f"{path}: JSON nesting is too deep") from error

    try:
        formatted = (format_value(document) + "\n").encode("utf-8")
    except RecursionError as error:
        raise FormatError(f"{path}: JSON nesting is too deep") from error
    return source, formatted


def replace_atomically(path, contents):
    descriptor = None
    temporary_path = None
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        stream = os.fdopen(descriptor, "wb")
        descriptor = None
        with stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as error:
        raise FormatError(f"{path}: cannot rewrite file: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def main(argv=None):
    args = parse_args(argv)
    try:
        files = collect_files(args.paths)
        changes_found = False
        for path in files:
            source, formatted = formatted_contents(path)
            if source == formatted:
                continue
            changes_found = True
            if args.check:
                print(f"would reformat: {path}")
            else:
                replace_atomically(path, formatted)
                print(f"reformatted: {path}")
    except FormatError as error:
        print(f"{Path(sys.argv[0]).name}: error: {error}", file=sys.stderr)
        return 2

    return 1 if args.check and changes_found else 0


if __name__ == "__main__":
    raise SystemExit(main())
