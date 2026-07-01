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

"""Build in-memory code metadata from explicit code object inputs."""

from __future__ import annotations

import bisect
import logging
import os
import re
import shutil
import subprocess
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .code_index import CodeIndex

TOOL_VERSION = "snapshot_dwarf-1.1"
HEADER = "ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle"
SEPARATOR = " -> "  # matches Instruction::separator in code_printing.hpp
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodeObject:
    path: str | Path
    code_object_id: int


@dataclass(frozen=True)
class CodeArtifacts:
    code_index: CodeIndex
    source_paths: tuple[Path, ...]


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


_LLVM_OBJDUMP: str | None = None


def llvm_objdump() -> str:
    """Resolve llvm-objdump lazily so the module can be imported without it."""
    global _LLVM_OBJDUMP
    if _LLVM_OBJDUMP is None:
        _LLVM_OBJDUMP = _find_tool("llvm-objdump")
    return _LLVM_OBJDUMP


def is_elf(path: str | Path) -> bool:
    try:
        with Path(path).open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


# ---------------------------------------------------------------------------
# DWARF helpers
# ---------------------------------------------------------------------------


def _decode(b) -> str:
    if isinstance(b, bytes):
        return b.decode("utf-8", errors="replace")
    return b or ""


def _resolve_file(line_program, file_index: int) -> str | None:
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


def _get_high_pc(die, low_pc: int | None) -> int | None:
    if "DW_AT_high_pc" not in die.attributes:
        return None
    attr = die.attributes["DW_AT_high_pc"]
    form = attr.form
    if form == "DW_FORM_addr":
        return attr.value
    if form.startswith("DW_FORM_data") or form in (
        "DW_FORM_udata",
        "DW_FORM_sdata",
        "DW_FORM_implicit_const",
    ):
        return (low_pc or 0) + attr.value
    return attr.value


def _get_die_ranges(die, dwarfinfo, cu) -> list[tuple[int, int]]:
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

        ranges: list[tuple[int, int]] = []
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

    __slots__ = (
        "ranges",
        "children",
        "total_low",
        "total_high",
        "children_low",
        "children_high",
        "file_and_line",
        "is_inlined",
    )

    def __init__(self):
        self.ranges: list[tuple[int, int]] = []
        self.children: list[_Inlined] = []
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

    def get_call_stack(self, addr: int, out: list[str]) -> bool:
        if not self.contains_children(addr):
            return False
        added = False
        for c in self.children:
            if c.get_call_stack(addr, out):
                added = True
                break
        if self.is_inlined and self.contains_total(addr):
            for lo, hi in self.ranges:
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
        if sub.children_high > sub.children_low:
            info.children.append(sub)
            info.expand_children(sub.children_low, sub.children_high)
    return info


def build_address_ranges(elf_path: str) -> list[tuple[int, int, str]]:
    """Return sorted ``(begin, end, comment)`` source ranges."""
    from elftools.common import exceptions as elftools_exceptions
    from elftools.elf.elffile import ELFFile

    expected_errors = (
        OSError,
        elftools_exceptions.ELFError,
        getattr(elftools_exceptions, "DWARFError", elftools_exceptions.ELFError),
        AssertionError,
        KeyError,
        IndexError,
        ValueError,
    )
    range_map: list[tuple[int, int, str]] = []

    with Path(elf_path).open("rb") as f:
        try:
            elf = ELFFile(f)
            if not elf.has_dwarf_info():
                return []
            dwarfinfo = elf.get_dwarf_info()
            cu_iter = iter(dwarfinfo.iter_CUs())
        except expected_errors:
            return []
        except Exception:
            LOGGER.debug("Unexpected error reading DWARF metadata from %s", elf_path, exc_info=True)
            return []

        while True:
            try:
                cu = next(cu_iter)
            except StopIteration:
                break
            except expected_errors:
                break
            except Exception:
                LOGGER.debug("Unexpected error reading a DWARF CU from %s", elf_path, exc_info=True)
                break

            try:
                line_program = dwarfinfo.line_program_for_CU(cu)
            except expected_errors:
                line_program = None
            except Exception:
                LOGGER.debug(
                    "Unexpected error reading a DWARF line program from %s", elf_path, exc_info=True
                )
                line_program = None
            if line_program is None:
                continue

            # Build inlined-subroutine DIE tree once per CU.
            try:
                top = cu.get_top_DIE()
                die_root = _build_die_tree(top, dwarfinfo, cu, line_program)
            except expected_errors:
                continue
            except Exception:
                LOGGER.debug("Unexpected error reading DWARF DIEs from %s", elf_path, exc_info=True)
                continue

            # Walk the line program; each entry covers [addr, next_addr).
            try:
                entries = list(line_program.get_entries())
            except expected_errors:
                continue
            except Exception:
                LOGGER.debug(
                    "Unexpected error reading DWARF line entries from %s", elf_path, exc_info=True
                )
                continue
            states = [(e.state, e) for e in entries if e.state is not None]
            line_ranges: dict[int, tuple[int, str]] = {}
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
                line_ranges[addr] = (end_addr, line_str)

            for addr, (end_addr, line_str) in line_ranges.items():
                stack: list[str] = []
                die_root.get_call_stack(addr, stack)
                if stack:
                    line_str = line_str + SEPARATOR + SEPARATOR.join(stack)
                range_map.append((addr, end_addr, line_str))

    # Sort by start address; later we'll binary-search.
    range_map.sort(key=lambda t: t[0])
    return range_map


def lookup_range(ranges: list[tuple[int, int, str]], addr: int) -> str:
    """Return the comment for the range containing addr, or ''."""
    if not ranges:
        return ""
    i = bisect.bisect_right(ranges, addr, key=lambda row: row[0]) - 1
    if i < 0:
        return ""
    begin, end, text = ranges[i]
    if begin <= addr < end:
        return text
    return ""


# ---------------------------------------------------------------------------
# Disassembly
# ---------------------------------------------------------------------------

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


def read_symbol_labels(elf_path: str) -> tuple[dict[int, tuple[str, str]], dict[int, list[str]]]:
    """Return (function labels, debug labels) keyed by virtual address."""
    from elftools.elf.elffile import ELFFile

    functions: dict[int, tuple[str, str]] = {}
    labels: dict[int, list[str]] = {}
    with Path(elf_path).open("rb") as f:
        elf = ELFFile(f)
        sections = [
            section
            for name in (".symtab", ".dynsym")
            if (section := elf.get_section_by_name(name)) is not None
        ]

        for section in sections:
            for symbol in section.iter_symbols():
                name = symbol.name
                if not name:
                    continue
                vaddr = int(symbol["st_value"])
                symbol_type = symbol["st_info"]["type"]
                if symbol_type == "STT_FUNC" and vaddr not in functions:
                    functions[vaddr] = (name, name)
                elif symbol_type == "STT_NOTYPE" and name.startswith("label"):
                    shndx = symbol["st_shndx"]
                    if shndx not in ("SHN_UNDEF", "SHN_ABS"):
                        labels.setdefault(vaddr, [])
                        if name not in labels[vaddr]:
                            labels[vaddr].append(name)
    return functions, labels


def parse_disassembly(elf_path: str) -> list[tuple[int, str]]:
    """Return [(vaddr, instruction_text)] from llvm-objdump output."""
    proc = subprocess.run(
        [llvm_objdump(), "-d", "--no-show-raw-insn", elf_path],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        message = f"llvm-objdump failed for {elf_path} with exit code {proc.returncode}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message)

    insts: list[tuple[int, str]] = []

    for raw in proc.stdout.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.lstrip().startswith(";"):
            continue
        im = _RE_INST.match(line)
        if im:
            inst = _normalize_branch_operand(im.group(1).strip(), im.group(3))
            vaddr = int(im.group(2), 16)
            insts.append((vaddr, inst))
    return insts


def _code_label(name: str) -> str:
    # RCV recognizes semicolon label rows only when the text starts with "; _".
    # Keep the real symbol in kernels[]; this marker affects only the code row.
    return f"; {name}" if name.startswith("_") else f"; _{name}"


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

_RE_PATH_LINE = re.compile(r"((?:/|\.{1,2}/)?[\w./+\-]+\.\w+):(\d+|\?)")


def extract_paths_from_comments(comments: list[str]) -> list[str]:
    found: OrderedDict[str, None] = OrderedDict()
    for c in comments:
        for m in _RE_PATH_LINE.finditer(c):
            found.setdefault(os.path.normpath(m.group(1)), None)
    return list(found.keys())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _normalize_code_objects(code_objects: Iterable[CodeObject]) -> list[CodeObject]:
    out: list[CodeObject] = []
    seen_ids: dict[int, Path] = {}
    for code_object in code_objects:
        path = Path(code_object.path).expanduser().resolve()
        code_object_id = int(code_object.code_object_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        if not is_elf(path):
            raise ValueError(f"Not an ELF code object: {path}")
        if code_object_id in seen_ids:
            raise ValueError(
                f"Code object id {code_object_id} is used by both "
                f"{seen_ids[code_object_id]} and {path}."
            )
        seen_ids[code_object_id] = path
        out.append(CodeObject(path, code_object_id))
    if not out:
        raise ValueError("No code objects were provided.")
    return out


def generate_code_artifacts(code_objects: Iterable[CodeObject]) -> CodeArtifacts:
    """Build RCV code metadata and source snapshot inputs from code objects.

    The caller supplies explicit code object ids. This function disassembles
    each ELF, combines that ISA with DWARF source ranges and symbol labels, and
    returns an in-memory CodeIndex plus the source paths referenced by the ISA
    comments.
    """
    normalized = _normalize_code_objects(code_objects)
    rows: list[list] = []
    comments: list[str] = []
    kernels_out: list[dict] = []
    line_no = 1

    for code_object in normalized:
        elf_path = str(code_object.path)
        codeobj_id = code_object.code_object_id
        range_map = build_address_ranges(elf_path)
        insts = parse_disassembly(elf_path)
        kernels, debug_labels = read_symbol_labels(elf_path)

        for vaddr, inst in insts:
            # Kernel and debug labels are emitted as code rows immediately
            # before the instruction at the same vaddr so branch targets in RCV
            # can land on the following ISA line.
            kernel = kernels.get(vaddr)
            if kernel is not None:
                kernel_name, kernel_demangled = kernel
                rows.append(
                    [
                        _code_label(kernel_name),
                        0,
                        line_no,
                        kernel_demangled,
                        codeobj_id,
                        vaddr,
                        0,
                        0,
                        0,
                        0,
                    ]
                )
                line_no += 1
            for label in debug_labels.get(vaddr, []):
                rows.append(
                    [
                        _code_label(label),
                        0,
                        line_no,
                        "",
                        codeobj_id,
                        vaddr,
                        0,
                        0,
                        0,
                        0,
                    ]
                )
                line_no += 1

            comment = lookup_range(range_map, vaddr)
            comments.append(comment)
            rows.append([inst, 0, line_no, comment, codeobj_id, vaddr, 0, 0, 0, 0])
            line_no += 1

        for address, (name, demangled) in sorted(kernels.items()):
            kernels_out.append(
                {
                    "address": address,
                    "codeobj": codeobj_id,
                    "name": name,
                    "demangled": demangled,
                }
            )

    code_doc = {
        "code": rows,
        "version": TOOL_VERSION,
        "header": HEADER,
        "kernels": kernels_out,
        "hsaco": [str(code_object.path) for code_object in normalized],
    }

    source_paths = tuple(Path(path) for path in extract_paths_from_comments(comments))
    return CodeArtifacts(CodeIndex.from_document(code_doc), source_paths)


def generate_code_index(code_objects: Iterable[CodeObject]) -> CodeIndex:
    return generate_code_artifacts(code_objects).code_index
