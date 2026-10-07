#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compare the pinned AMDF C headers with the checked-in Rust ABI bindings."""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
RUNTIMES = HERE.parents[3]
HEADERS = RUNTIMES / "api-headers/include"
BINDINGS = HERE.parents[1] / "src/generated/amdf.rs"
IDENT = r"[A-Za-z_]\w*"
TYPE = re.compile(rf"type ({IDENT}) \d+ \d+")
FIELD = re.compile(rf"field ({IDENT})\.({IDENT}) \d+ \d+ \d+")
CONSTANT = re.compile(rf"constant ({IDENT}) \d+")
STRING = re.compile(rf"string ({IDENT}) .*")
RUST_KEYWORDS = {
    "type", "self", "Self", "super", "crate", "async", "await", "move", "ref"
}
EXPECTED_RECORDS = 967  # Pinned C snapshot; update only with the headers.


def run(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"{command[0]} failed:\n{completed.stderr}")
    return completed.stdout


def rust_probe(c_lines: list[str]) -> str:
    source = [
        "#[path = " + json.dumps(str(BINDINGS)) + "] mod abi;",
        "use abi::*;",
        "use std::mem::{align_of, size_of, MaybeUninit};",
        "fn field<T>(name: &str, base: *const u8, pointer: *const T) {",
        '    println!("field {name} {} {} {}", ',
        "        (pointer as usize) - (base as usize), "
        "size_of::<T>(), align_of::<T>());",
        "}",
        "fn main() {",
    ]
    for line in c_lines:
        if match := TYPE.fullmatch(line):
            name = match.group(1)
            source.append(
                f'    println!("type {name} {{}} {{}}", '
                f"size_of::<{name}>(), align_of::<{name}>());"
            )
        elif match := FIELD.fullmatch(line):
            name, member = match.groups()
            rust_member = f"r#{member}" if member in RUST_KEYWORDS else member
            source.extend(
                [
                    "    {",
                    f"        let record = MaybeUninit::<{name}>::uninit();",
                    "        let base = record.as_ptr();",
                    f"        let pointer = unsafe {{ "
                    f"std::ptr::addr_of!((*base).{rust_member}) }};",
                    f'        field("{name}.{member}", base.cast(), pointer);',
                    "    }",
                ]
            )
        elif match := CONSTANT.fullmatch(line):
            name = match.group(1)
            source.append(f'    println!("constant {name} {{}}", {name} as u64);')
        elif match := STRING.fullmatch(line):
            name = match.group(1)
            source.append(
                f'    println!("string {name} {{}}", '
                f"std::ffi::CStr::from_bytes_with_nul({name})"
                ".unwrap().to_str().unwrap());"
            )
        else:
            raise ValueError(f"unrecognized C ABI record: {line}")
    source.append("}")
    return "\n".join(source) + "\n"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rocddi-amdf-layout-") as temporary:
        output = Path(temporary)
        c_binary = output / "c_layout"
        rust_binary = output / "rust_layout"
        rust_source = output / "rust_layout.rs"
        run(
            [
                os.environ.get("CC", "cc"),
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(HEADERS),
                str(HERE / "amdf_layout.c"),
                "-o",
                str(c_binary),
            ]
        )
        c_lines = run([str(c_binary)]).splitlines()
        if len(c_lines) != EXPECTED_RECORDS:
            raise ValueError(
                f"expected {EXPECTED_RECORDS} AMDF ABI records, got {len(c_lines)}"
            )
        rust_source.write_text(rust_probe(c_lines))
        run(
            [
                os.environ.get("RUSTC", "rustc"),
                "--edition=2024",
                str(rust_source),
                "-o",
                str(rust_binary),
            ]
        )
        rust_lines = run([str(rust_binary)]).splitlines()
    if c_lines != rust_lines:
        differences = difflib.unified_diff(
            c_lines, rust_lines, fromfile="C headers",
            tofile="Rust bindings", lineterm=""
        )
        print("\n".join(list(differences)[:80]), file=sys.stderr)
        return 1
    print(f"{len(c_lines)} AMDF C/Rust ABI layout and constant records match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
