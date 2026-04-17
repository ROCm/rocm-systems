#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Extract and decode the .sqtt_funcmap section from an AMDGPU code object.

Usage:
    python3 sqtt_decode_funcmap.py <code_object> [--demangle]

The .sqtt_funcmap section contains:
  - "ID:mangled_name"  for instrumented device functions
  - "K:mangled_name"   for kernel entry points (not instrumented)

This script also resolves ELF vaddrs for all symbols using llvm-nm.
"""

import argparse
import subprocess
import sys
import os


def find_tool(name: str) -> str | None:
    """Find an LLVM tool, preferring ROCm installation."""
    candidates = [
        f"/opt/rocm/llvm/bin/{name}",
        name,
    ]
    for path in candidates:
        if "/" in path:
            if os.path.isfile(path):
                return path
        else:
            import shutil
            if shutil.which(path):
                return path
    return None


def read_funcmap(binary: str) -> str:
    """Read .sqtt_funcmap section content."""
    # Prefer llvm-objcopy --dump-section (gives raw bytes with real newlines)
    objcopy = find_tool("llvm-objcopy")
    if objcopy:
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
                tmp = f.name
            subprocess.run(
                [objcopy, f"--dump-section=.sqtt_funcmap={tmp}", binary],
                capture_output=True, timeout=10, check=True)
            with open(tmp, "r") as f:
                data = f.read()
            os.unlink(tmp)
            return data
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError):
            pass

    # Fallback: llvm-readelf -p (newlines rendered as dots)
    readelf = find_tool("llvm-readelf")
    if readelf:
        try:
            r = subprocess.run(
                [readelf, "-p", ".sqtt_funcmap", binary],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                return r.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return ""


def read_symbol_addrs(binary: str) -> dict[str, int]:
    """Read symbol vaddrs from the ELF using llvm-nm."""
    addrs: dict[str, int] = {}
    nm = find_tool("llvm-nm")
    if not nm:
        return addrs
    try:
        r = subprocess.run(
            [nm, "--defined-only", binary],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return addrs
        for line in r.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    addr = int(parts[0], 16)
                    name = parts[2]
                    addrs[name] = addr
                except ValueError:
                    continue
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return addrs


def demangle(name: str) -> str:
    """Demangle a C++ name using c++filt."""
    try:
        r = subprocess.run(
            ["c++filt", name], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return name


# Entry: (id_or_none, symbol_name, entry_type)
# entry_type: "kernel", "func", "user"
Entry = tuple[int | None, str, str]


def parse_funcmap(output: str) -> list[Entry]:
    """Parse funcmap content into entries."""
    entries: list[Entry] = []

    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("String") or line.startswith("---"):
            continue

        # llvm-readelf -p wraps lines as "[  offset]  content"
        bracket = line.find("]")
        if bracket >= 0:
            line = line[bracket + 1:].strip()

        # Clean up trailing null/period artifacts
        line = line.rstrip("\x00").rstrip(".")

        if not line or ":" not in line:
            continue

        prefix, rest = line.split(":", 1)
        prefix = prefix.strip()
        rest = rest.strip()

        if not rest:
            continue

        if prefix == "K":
            entries.append((None, rest, "kernel"))
        elif prefix in ("F", "U", "P"):
            # Funcmap v2: "F:ID:name", "U:ID:name", "P:ID:name"
            if ":" not in rest:
                continue
            id_str, name = rest.split(":", 1)
            try:
                mid = int(id_str)
            except ValueError:
                continue
            type_map = {"F": "func", "U": "user", "P": "point"}
            entries.append((mid, name, type_map[prefix]))

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Decode .sqtt_funcmap section from AMDGPU code objects")
    parser.add_argument("binary", help="Path to the code object")
    parser.add_argument("--demangle", action="store_true",
                        help="Demangle C++ function names")
    args = parser.parse_args()

    output = read_funcmap(args.binary)
    if not output:
        print(f"Error: could not read .sqtt_funcmap from {args.binary}",
              file=sys.stderr)
        sys.exit(1)

    entries = parse_funcmap(output)
    if not entries:
        print("No function map entries found.", file=sys.stderr)
        sys.exit(1)

    # Resolve ELF vaddrs
    sym_addrs = read_symbol_addrs(args.binary)

    # Print table
    print(f"{'Type':>6}  {'ID':>8}  {'Vaddr':>18}  {'Name'}")
    print(f"{'----':>6}  {'---':>8}  {'-----':>18}  {'----'}")

    # Sort: kernels first, then funcs, then points, then user markers
    type_order = {"kernel": 0, "func": 1, "point": 2, "user": 3}

    def sort_key(e: Entry):
        fid, name, etype = e
        return (type_order.get(etype, 3), fid if fid is not None else -1, name)

    for fid, name, etype in sorted(entries, key=sort_key):
        id_str = "-" if fid is None else str(fid)
        addr = sym_addrs.get(name)
        addr_str = f"0x{addr:016x}" if addr is not None else "-"
        display = demangle(name) if args.demangle else name
        print(f"{etype:>6}  {id_str:>8}  {addr_str:>18}  {display}")


if __name__ == "__main__":
    main()
