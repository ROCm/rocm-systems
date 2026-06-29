#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Generate ``code.json`` / ``snapshots.json`` for the ROCprof Compute Viewer from
GPU code objects, so raw ``.att``/``.out`` traces collected straight from
rocprofiler-sdk get ISA <-> source correlation.

Usage:
    generate_code.py [code_object ...]

The code objects are ELF files (``.hsaco``, ``.out``, ``.co`` or ``.o``). When
no argument is given, every file with one of those suffixes in the current
directory is used. For each file we extract DWARF line ranges (with inlined call
stacks) and disassembly, and snapshot the referenced source files into the
current directory.

Each row is tagged with a code object id parsed from the filename (the part
after the last ``_``, e.g. ``codeobj_12345.out`` -> 12345), matching how the
viewer keys trace instructions by (code object id, virtual address). A single
code object whose filename does not encode an id is tagged as id 0. Multiple
unnamed code objects cannot be inferred from the files and are rejected.

Comment format (mirrors ``code_printing.hpp``):
    "<file>:<line> -> <call_file>:<call_line> -> ..."
where the first segment is the source location for the address and each
subsequent segment is an inlined-call-site, innermost first.

Produces, for all inputs combined:
  - ``code.json``       : disassembly + DWARF line/inline correlation
  - ``snapshots.json``  : tree of snapshotted source file names
  - ``source_<n>_<file>``   : copies of source files referenced by DWARF
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TOOL_VERSION = "snapshot_dwarf-1.1"
HEADER = "ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle"
SEPARATOR = " -> "  # matches Instruction::separator in code_printing.hpp


def _find_tool(name: str) -> str:
    candidates: list[Path] = []
    for root in _rocm_roots():
        candidates.append(root / "llvm" / "bin" / name)
        candidates.append(root / "bin" / name)

    for found in (shutil.which(name),):
        if found:
            candidates.append(Path(found))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Could not locate {name}")


def _rocm_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for var in ("ROCM_HOME", "ROCM_PATH"):
        value = os.environ.get(var)
        if value:
            root = Path(value).expanduser()
            if root not in seen:
                roots.append(root)
                seen.add(root)

    default = Path("/opt/rocm")
    if default not in seen:
        roots.append(default)
    return roots


_LLVM_OBJDUMP: Optional[str] = None


def llvm_objdump() -> str:
    """Resolve llvm-objdump lazily so the module can be imported without it."""
    global _LLVM_OBJDUMP
    if _LLVM_OBJDUMP is None:
        _LLVM_OBJDUMP = _find_tool("llvm-objdump")
    return _LLVM_OBJDUMP


def default_code_object_inputs() -> list[str]:
    suffixes = {".co", ".hsaco", ".o", ".out"}
    return sorted(
        path.name
        for path in Path.cwd().iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def is_elf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def codeobj_id_from_path(path: str) -> int:
    """Return the filename-encoded code object id, or 0 when there is none."""
    parsed = _parse_codeobj_id_from_path(path)
    return parsed if parsed is not None else 0


def _parse_codeobj_id_from_path(path: str) -> int | None:
    stem = Path(path).stem
    pos = stem.rfind("_")
    if pos == -1:
        return None
    try:
        return int(stem[pos + 1:])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# DWARF helpers
# ---------------------------------------------------------------------------

def _decode(b) -> str:
    if isinstance(b, bytes):
        return b.decode("utf-8", errors="replace")
    return b or ""


def _resolve_file(line_program, file_index: int) -> Optional[str]:
    """Resolve a DWARF file index to a full path using the CU line program."""
    if line_program is None:
        return None
    header = line_program.header
    file_entries = header.get("file_entry", [])
    include_dirs = header.get("include_directory", [])

    # In DWARF <=4 file_entry is 1-based; in DWARF 5 it's 0-based.
    version = header.get("version", 4)
    idx = file_index if version >= 5 else file_index - 1
    if idx < 0 or idx >= len(file_entries):
        return None
    fe = file_entries[idx]
    name = _decode(fe.name)
    d_idx = fe.dir_index
    # Same indexing rule for include_directory.
    if version >= 5:
        d = _decode(include_dirs[d_idx]) if 0 <= d_idx < len(include_dirs) else ""
    else:
        if d_idx == 0:
            d = ""
        else:
            j = d_idx - 1
            d = _decode(include_dirs[j]) if 0 <= j < len(include_dirs) else ""
    if d and not os.path.isabs(name):
        return os.path.normpath(os.path.join(d, name))
    return os.path.normpath(name)


def _get_high_pc(die, low_pc: Optional[int]) -> Optional[int]:
    if "DW_AT_high_pc" not in die.attributes:
        return None
    attr = die.attributes["DW_AT_high_pc"]
    form = attr.form
    if form == "DW_FORM_addr":
        return attr.value
    if form.startswith("DW_FORM_data") or form in (
        "DW_FORM_udata", "DW_FORM_sdata",
        "DW_FORM_implicit_const",
    ):
        return (low_pc or 0) + attr.value
    return attr.value


def _get_die_ranges(die, dwarfinfo, cu) -> List[Tuple[int, int]]:
    """Return list of (begin, end) for a DIE, handling DW_AT_ranges and lo/hi."""
    if "DW_AT_ranges" in die.attributes:
        rl_reader = dwarfinfo.range_lists()
        if rl_reader is None:
            return []
        offset = die.attributes["DW_AT_ranges"].value
        try:
            entries = rl_reader.get_range_list_at_offset(offset, cu=cu)
        except TypeError:
            entries = rl_reader.get_range_list_at_offset(offset)
        # CU base address (default 0).
        base = 0
        top = cu.get_top_DIE()
        if "DW_AT_low_pc" in top.attributes:
            base = top.attributes["DW_AT_low_pc"].value

        ranges: List[Tuple[int, int]] = []
        for e in entries or []:
            tname = type(e).__name__
            if tname == "BaseAddressEntry":
                base = e.base_address
            elif tname == "RangeEntry":
                # pyelftools: is_absolute means begin/end are absolute addrs.
                is_abs = getattr(e, "is_absolute", False)
                begin = e.begin_offset
                end = e.end_offset
                if not is_abs:
                    begin += base
                    end += base
                if end > begin:
                    ranges.append((begin, end))
        return ranges

    if "DW_AT_low_pc" in die.attributes:
        low = die.attributes["DW_AT_low_pc"].value
        high = _get_high_pc(die, low)
        if high is not None and high > low:
            return [(low, high)]
    return []


class _Inlined:
    """Mirror of DIEInfo for a single CU's DIE tree."""

    __slots__ = ("ranges", "children", "total_low", "total_high",
                 "children_low", "children_high", "file_and_line", "is_inlined")

    def __init__(self):
        self.ranges: List[Tuple[int, int]] = []
        self.children: List[_Inlined] = []
        self.total_low = 1 << 63
        self.total_high = 0
        self.children_low = 1 << 63
        self.children_high = 0
        self.file_and_line: str = ""
        self.is_inlined = False

    def add_range(self, lo: int, hi: int) -> None:
        self.ranges.append((lo, hi))
        if lo < self.total_low:
            self.total_low = lo
        if hi > self.total_high:
            self.total_high = hi

    def expand_children(self, lo: int, hi: int) -> None:
        if lo < self.children_low:
            self.children_low = lo
        if hi > self.children_high:
            self.children_high = hi

    def contains_total(self, addr: int) -> bool:
        return self.total_low <= addr < self.total_high

    def contains_children(self, addr: int) -> bool:
        return self.children_low <= addr < self.children_high

    def get_call_stack(self, addr: int, out: List[str]) -> bool:
        if not self.contains_children(addr):
            return False
        added = False
        for c in self.children:
            if c.get_call_stack(addr, out):
                added = True
                break
        if self.is_inlined and self.contains_total(addr):
            for (lo, hi) in self.ranges:
                if lo <= addr < hi:
                    out.append(self.file_and_line)
                    return True
        return added


def _build_die_tree(die, dwarfinfo, cu, line_program) -> _Inlined:
    info = _Inlined()
    if die.tag == "DW_TAG_inlined_subroutine":
        info.is_inlined = True
        for lo, hi in _get_die_ranges(die, dwarfinfo, cu):
            info.add_range(lo, hi)
        # Resolve call-site (where this inlined subroutine was called).
        call_file_attr = die.attributes.get("DW_AT_call_file")
        call_line_attr = die.attributes.get("DW_AT_call_line")
        if call_file_attr is not None and call_line_attr is not None:
            fname = _resolve_file(line_program, call_file_attr.value)
            if fname:
                info.file_and_line = f"{fname}:{call_line_attr.value}"
        # Always include this node's own range into children_range so parents
        # can find it via children_range (matches the C++ comment).
        info.children_low = info.total_low
        info.children_high = info.total_high

    for child in die.iter_children():
        sub = _build_die_tree(child, dwarfinfo, cu, line_program)
        info.children.append(sub)
        if sub.children_high > sub.children_low:
            info.expand_children(sub.children_low, sub.children_high)
    return info


def build_address_ranges(
    elf_path: str,
) -> Tuple[List[Tuple[int, int, str]], List[str]]:
    """Return (sorted [(begin, end, comment)], list of source files)."""
    from elftools.elf.elffile import ELFFile

    sources: "OrderedDict[str, None]" = OrderedDict()
    range_map: List[Tuple[int, int, str]] = []

    with open(elf_path, "rb") as f:
        elf = ELFFile(f)
        if not elf.has_dwarf_info():
            return [], []
        dwarfinfo = elf.get_dwarf_info()

        for cu in dwarfinfo.iter_CUs():
            try:
                line_program = dwarfinfo.line_program_for_CU(cu)
            except Exception:
                line_program = None
            if line_program is None:
                continue

            # Collect source files referenced by this CU.
            for fe in line_program.header.get("file_entry", []):
                idx = line_program.header.get("file_entry").index(fe)
                # Easier: build via _resolve_file by walking indices.
                pass
            # Reliable enumeration:
            n_files = len(line_program.header.get("file_entry", []))
            version = line_program.header.get("version", 4)
            start = 0 if version >= 5 else 1
            for fi in range(start, start + n_files):
                p = _resolve_file(line_program, fi)
                if p:
                    sources.setdefault(p, None)

            # Build inlined-subroutine DIE tree once per CU.
            top = cu.get_top_DIE()
            die_root = _build_die_tree(top, dwarfinfo, cu, line_program)

            # Walk the line program; each entry covers [addr, next_addr).
            entries = list(line_program.get_entries())
            states = [(e.state, e) for e in entries if e.state is not None]
            for i in range(len(states) - 1):
                st, _ = states[i]
                nxt, _ = states[i + 1]
                if st.end_sequence:
                    continue
                addr = st.address
                end_addr = nxt.address
                if end_addr <= addr:
                    continue
                src = _resolve_file(line_program, st.file)
                if not src:
                    continue
                line_str = f"{src}:{st.line}" if st.line != 0 else f"{src}:?"
                # Build inlined call stack for this address.
                stack: List[str] = []
                die_root.get_call_stack(addr, stack)
                if stack:
                    line_str = line_str + SEPARATOR + SEPARATOR.join(stack)
                range_map.append((addr, end_addr, line_str))

    # Sort by start address; later we'll binary-search.
    range_map.sort(key=lambda t: t[0])
    return range_map, list(sources.keys())


def lookup_range(ranges: List[Tuple[int, int, str]], addr: int) -> str:
    """Return the comment for the range containing addr, or ''."""
    if not ranges:
        return ""
    starts = [r[0] for r in ranges]
    i = bisect.bisect_right(starts, addr) - 1
    if i < 0:
        return ""
    begin, end, text = ranges[i]
    if begin <= addr < end:
        return text
    return ""


# ---------------------------------------------------------------------------
# Disassembly
# ---------------------------------------------------------------------------

_RE_FUNC_HEADER = re.compile(r"^([0-9a-fA-F]+)\s+<([^>]+)>:\s*$")
# Instruction lines: leading whitespace, mnemonic+operands, "// HEX: HEX..."
_RE_INST = re.compile(r"^\s+(.*?)\s*//\s*([0-9A-Fa-f]+):\s*(.+)\s*$")
_RE_DECIMAL = re.compile(r"^-?\d+$")


def _normalize_branch_operand(inst: str, raw_words: str) -> str:
    """Make branch targets compatible with the decoder stitcher.

    The stitcher expects the last token of branch instructions to be the SOPP
    16-bit branch displacement in decimal form. llvm-objdump may print symbolic
    labels for ELF-local branch targets, e.g. ``s_cbranch_vccz label``. The
    encoded immediate is still present in the raw instruction word, so use that
    as the source of truth. Keep the raw unsigned 16-bit value to match the
    decoder/control CSV disassembly; the stitcher converts values >= 32768 to
    signed internally.
    """
    if "branch" not in inst:
        return inst

    parts = inst.rsplit(None, 1)
    if len(parts) != 2:
        return inst

    if _RE_DECIMAL.match(parts[1]):
        return inst

    first_word = raw_words.split()[0] if raw_words.split() else ""
    if len(first_word) < 4:
        return inst

    try:
        imm = int(first_word[-4:], 16)
    except ValueError:
        return inst
    return f"{parts[0]} {imm}"


def parse_disassembly(elf_path: str) -> Tuple[List[Tuple[int, str]], Dict[int, dict]]:
    """Return ([(vaddr, instruction_text)], {kernel_addr: {name, demangled}})."""
    proc = subprocess.run(
        [llvm_objdump(), "-d", "--no-show-raw-insn", elf_path],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    insts: List[Tuple[int, str]] = []
    kernels: Dict[int, dict] = {}

    for raw in proc.stdout.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        m = _RE_FUNC_HEADER.match(line)
        if m:
            addr = int(m.group(1), 16)
            name = m.group(2)
            kernels[addr] = {"name": name, "demangled": name}
            continue
        if line.lstrip().startswith(";"):
            continue
        im = _RE_INST.match(line)
        if im:
            inst = _normalize_branch_operand(im.group(1).strip(), im.group(3))
            vaddr = int(im.group(2), 16)
            insts.append((vaddr, inst))
    return insts, kernels


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

_RE_PATH_LINE = re.compile(r"((?:/|\.{1,2}/)?[\w./+\-]+\.\w+):(\d+|\?)")


def extract_paths_from_comments(comments: List[str]) -> List[str]:
    found: "OrderedDict[str, None]" = OrderedDict()
    for c in comments:
        for m in _RE_PATH_LINE.finditer(c):
            found.setdefault(os.path.normpath(m.group(1)), None)
    return list(found.keys())


def snapshot_sources(
    sources: List[str],
    extracted: List[str],
    out_dir: Path,
) -> Tuple[dict, int]:
    seen: Dict[str, str] = {}
    tree: dict = {}
    num = 0

    candidates: "OrderedDict[str, None]" = OrderedDict()
    for s in sources:
        candidates.setdefault(s, None)
    for s in extracted:
        candidates.setdefault(s, None)

    for src in candidates:
        if not src or src in seen or not os.path.isfile(src):
            continue
        new_name = f"source_{num}_{os.path.basename(src)}"
        num += 1
        try:
            shutil.copyfile(src, out_dir / new_name)
        except OSError as e:
            print(f"  warn: could not copy {src}: {e}", file=sys.stderr)
            continue
        seen[src] = new_name

        #parts = [p for p in Path(src).parts if p != "/"]
        parts = list(Path(src).parts)
        node = tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                node[part] = new_name
            else:
                child = node.setdefault(part, {})
                if not isinstance(child, dict):
                    break
                node = child

    return tree, num


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    cwd = Path.cwd()

    # Inputs: explicit list of code objects on the command line, otherwise fall
    # back to code-object-looking files in the current directory.
    parser = argparse.ArgumentParser(
        description="Generate code.json and source snapshots from GPU code objects."
    )
    parser.add_argument(
        "code_objects",
        nargs="*",
        help="ELF code objects (.hsaco, .out, .co, .o)",
    )
    args = parser.parse_args(argv)
    inputs = args.code_objects or default_code_object_inputs()
    if not inputs:
        print(
            "No code object given and no .hsaco/.out/.co/.o found in current directory.",
            file=sys.stderr,
        )
        return 1

    # Each ELF is tagged with (path, code object id). If the filename does not
    # carry an id, exactly one such input may use id 0. Multiple unnamed IDs
    # would alias distinct code objects and cannot be inferred from the files.
    input_paths = []
    untagged_paths = []
    parsed_ids: set[int] = set()
    for p in inputs:
        if not os.path.isfile(p):
            print(f"skipping {p}: not a file", file=sys.stderr)
            continue
        if not is_elf(p):
            print(f"skipping {p}: not an ELF file", file=sys.stderr)
            continue
        parsed = _parse_codeobj_id_from_path(p)
        input_paths.append((p, parsed))
        if parsed is None:
            untagged_paths.append(p)
        else:
            parsed_ids.add(parsed)

    if len(untagged_paths) > 1:
        print(
            "Cannot infer code object IDs for multiple unnamed inputs: "
            + ", ".join(untagged_paths),
            file=sys.stderr,
        )
        return 1
    if untagged_paths and 0 in parsed_ids:
        print(
            f"Cannot assign code object id 0 to {untagged_paths[0]}: another input already uses id 0.",
            file=sys.stderr,
        )
        return 1

    elf_files = []
    seen_ids: Dict[int, str] = {}
    for p, parsed_id in input_paths:
        cid = parsed_id if parsed_id is not None else 0
        if cid in seen_ids:
            print(
                f"skipping {p}: code object id {cid} already used by {seen_ids[cid]}",
                file=sys.stderr,
            )
            continue
        seen_ids[cid] = p
        elf_files.append((p, cid))
    if not elf_files:
        print("No ELF code objects to process.", file=sys.stderr)
        return 1

    print(f"Using llvm-objdump: {llvm_objdump()}")

    rows: List[list] = []
    comments: List[str] = []
    kernels_out: List[dict] = []
    line_no = 0
    n_with = 0
    n_insts = 0

    for elf_path, codeobj_id in elf_files:
        print(f"[+] {elf_path} (codeobj id {codeobj_id})")

        range_map, _ = build_address_ranges(elf_path)
        insts, kernels = parse_disassembly(elf_path)
        print(f"    DWARF address ranges: {len(range_map)}  "
              f"instructions: {len(insts)}  kernels: {len(kernels)}")

        kernel_set = set(kernels.keys())
        n_insts += len(insts)
        for vaddr, inst in insts:
            if vaddr in kernel_set:
                line_no += 1
                k = kernels[vaddr]
                rows.append([
                    f"; {k['name']}", 0, line_no - 1, k["demangled"],
                    codeobj_id, vaddr, 0, 0, 0, 0,
                ])
            comment = lookup_range(range_map, vaddr)
            if comment:
                n_with += 1
            comments.append(comment)
            line_no += 1
            rows.append([
                inst, 0, line_no, comment,
                codeobj_id, vaddr, 0, 0, 0, 0,
            ])

        for a, k in sorted(kernels.items()):
            kernels_out.append({
                "address": a, "codeobj": codeobj_id,
                "name": k["name"], "demangled": k["demangled"],
            })

    print(f"    rows with DWARF comment: {n_with}/{n_insts}")

    code_doc = {
        "code": rows,
        "version": TOOL_VERSION,
        "header": HEADER,
        "kernels": kernels_out,
        "hsaco": [p for p, _ in elf_files],
    }
    code_path = cwd / "code.json"
    with open(code_path, "w") as f:
        json.dump(code_doc, f, indent=2)
    print(f"wrote {code_path}")

    # Only snapshot files actually referenced by code.json's comments.
    referenced = extract_paths_from_comments(comments)
    tree, num_snap = snapshot_sources(referenced, [], cwd)
    if num_snap:
        snap_path = cwd / "snapshots.json"
        with open(snap_path, "w") as f:
            json.dump(tree, f, indent=2)
        print(f"snapshotted {num_snap} source files -> {snap_path}")
    else:
        print("no source files snapshotted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
