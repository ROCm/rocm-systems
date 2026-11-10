#!/usr/bin/env python3
"""Insert SQTT site markers into an already-compiled AMDGPU code object.

Prototype for marking LDS (and later memory) instructions without recompiling.
Each selected instruction is moved into a trampoline stub:

    site:  s_branch stub           ; replaces the original instruction in place
           s_nop 0                 ; pads to the original instruction size
    stub:  s_ttracedata_imm <id>   ; the marker (same encoding as sqtt-marker)
           <original instruction>  ; DS ops are not PC-relative, so they move as-is
           s_branch site+size      ; back to the instruction after the site

Nothing in the original code moves, so branches, s_getpc sequences, kernel
descriptors, symbols and DWARF stay valid. Stubs live in the free address space
between the end of .text and the next PT_LOAD segment (page-alignment slack);
the file is grown there and later file offsets are shifted by whole pages.

A .sqtt_funcmap section with rows P:<id>:site_<kind>@<file>:<line> is added, so
the existing decoder and counter_fit.py read the markers like compile-time
SQTT_TRACE_SITES markers.

Usage:
    sqtt_site_rewrite.py in.co out.co [--sites lds] [--llvm /opt/rocm/lib/llvm/bin]
"""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import sys
from dataclasses import dataclass

# --- gfx10+ scalar encodings (SOPP) -------------------------------------------
S_NOP_0 = 0xBF800000
S_BRANCH = 0xBFA00000  # | simm16 (dwords, relative to the next instruction)
S_TTRACEDATA_IMM = 0xBFBB0000  # | imm; the trace keeps the low 8 bits
S_CODE_END = 0xBF9F0000

# Marker header layout shared with sqtt-marker: bits 31:2 are the ID, bits 1:0
# are scope flags. s_ttracedata_imm keeps 8 bits, so IDs 1..63 fit.
MAX_IMM_ID = 63

# CP instruction prefetch reads ahead of the PC; keep a readable s_code_end
# guard after the stubs so prefetch past the last stub stays inside the segment.
PREFETCH_GUARD = 256

PT_LOAD = 1
SHT_NOBITS = 8
PF_X = 1

# Instruction kinds that can become sites, by mnemonic prefix. Permute ops go
# through the LDS crossbar but are not LDS data accesses, so they are left out.
SITE_KINDS = {
    "lds": [
        (re.compile(r"ds_(load|read)"), "lds_load"),
        (re.compile(r"ds_(store|write)"), "lds_store"),
        (re.compile(r"ds_(?!.*permute)"), "lds_atomic"),
    ],
}


@dataclass
class Insn:
    addr: int
    words: list[int]
    text: str
    function: str = ""

    @property
    def size(self) -> int:
        return 4 * len(self.words)

    @property
    def mnemonic(self) -> str:
        return self.text.split()[0]


def disassemble(llvm: str, path: str) -> tuple[list[Insn], list[str]]:
    """Instructions in .text (address, encoding words, text) and function names."""
    out = subprocess.run(
        [f"{llvm}/llvm-objdump", "-d", "--section=.text", path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    insns = []
    functions = []
    # e.g. "\tds_load_b32 v4, v4   // 000000001D3C: D8D80000 04000004"
    pattern = re.compile(r"^\s+(\S.*?)\s*//\s*([0-9A-Fa-f]+):\s*((?:[0-9A-Fa-f]{8}\s*)+)$")
    # e.g. "0000000000001900 <_Z12lds_conflictPfii>:"
    label = re.compile(r"^[0-9a-f]+ <([^>]+)>:$")
    for line in out.splitlines():
        if m := pattern.match(line):
            words = [int(w, 16) for w in m.group(3).split()]
            current = functions[-1] if functions else ""
            insns.append(Insn(int(m.group(2), 16), words, m.group(1), current))
        elif m := label.match(line):
            functions.append(m.group(1))
    return insns, functions


def source_lines(llvm: str, path: str, addrs: list[int]) -> dict[int, str]:
    """Map instruction addresses to file:line through the DWARF line table."""
    if not addrs:
        return {}
    out = subprocess.run(
        [f"{llvm}/llvm-addr2line", "-e", path] + [hex(a) for a in addrs],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    result = {}
    for addr, loc in zip(addrs, out):
        # Drop the column / discriminator suffix that addr2line may append.
        loc = re.sub(r" \(discriminator \d+\)$", "", loc.strip())
        result[addr] = "" if loc.startswith("??") else loc
    return result


def classify(insn: Insn, categories: list[str]) -> str | None:
    for category in categories:
        for regex, kind in SITE_KINDS[category]:
            if regex.match(insn.mnemonic):
                return kind
    return None


def clause_covered(insns: list[Insn]) -> set[int]:
    """Indices of instructions inside an s_clause window (left in place)."""
    covered = set()
    for i, insn in enumerate(insns):
        if insn.mnemonic == "s_clause":
            # s_clause N groups the next N + 1 instructions.
            length = (insn.words[0] & 0x3F) + 1
            covered.update(range(i + 1, i + 1 + length))
    return covered


def branch(from_addr: int, to_addr: int) -> int:
    """s_branch at from_addr jumping to to_addr (both byte addresses)."""
    delta = (to_addr - (from_addr + 4)) // 4
    if not -0x8000 <= delta <= 0x7FFF:
        raise ValueError(f"s_branch out of range: {from_addr:#x} -> {to_addr:#x}")
    return S_BRANCH | (delta & 0xFFFF)


class Elf64:
    """Just enough little-endian ELF64 editing for this tool."""

    def __init__(self, data: bytes):
        self.data = bytearray(data)
        h = struct.unpack_from("<16sHHIQQQIHHHHHH", self.data, 0)
        (
            _,
            _,
            _,
            _,
            _,
            self.phoff,
            self.shoff,
            _,
            _,
            self.phentsize,
            self.phnum,
            self.shentsize,
            self.shnum,
            self.shstrndx,
        ) = h

    def phdrs(self):
        for i in range(self.phnum):
            off = self.phoff + i * self.phentsize
            yield off, list(struct.unpack_from("<IIQQQQQQ", self.data, off))

    def shdrs(self):
        for i in range(self.shnum):
            off = self.shoff + i * self.shentsize
            yield off, list(struct.unpack_from("<IIQQQQIIQQ", self.data, off))

    def section_name(self, sh) -> str:
        strtab = list(self.shdrs())[self.shstrndx][1]
        start = strtab[4] + sh[0]
        return self.data[start : self.data.index(b"\0", start)].decode()

    def section(self, name: str):
        for off, sh in self.shdrs():
            if self.section_name(sh) == name:
                return off, sh
        raise KeyError(name)


def rewrite(args) -> int:
    with open(args.input, "rb") as f:
        data = f.read()
    elf = Elf64(data)
    _, text = elf.section(".text")
    text_addr, text_fileoff, text_size = text[3], text[4], text[5]
    text_end = text_addr + text_size

    # The executable PT_LOAD that holds .text, and the next segment's start.
    loads = [(off, ph) for off, ph in elf.phdrs() if ph[0] == PT_LOAD]
    exec_off, exec_ph = next((o, p) for o, p in loads if p[1] & PF_X)
    later = [p[3] for _, p in loads if p[3] > exec_ph[3]]
    limit = min(later) if later else text_end + 0x10000
    if exec_ph[3] + exec_ph[5] != text_end:
        sys.exit(".text is not at the end of its segment; not supported yet")

    insns, kernels = disassemble(args.llvm, args.input)
    covered = clause_covered(insns)
    sites = []
    for i, insn in enumerate(insns):
        kind = classify(insn, args.sites)
        if kind is None:
            continue
        if args.kernel and not args.kernel.search(insn.function):
            continue
        if i in covered:
            print(f"skip {insn.addr:#x} {insn.text}: inside s_clause", file=sys.stderr)
            continue
        if insn.size < 8:
            # 4-byte sites would need the s_branch alone; not needed for DS ops.
            print(f"skip {insn.addr:#x} {insn.text}: 4-byte instruction", file=sys.stderr)
            continue
        sites.append((insn, kind))
    if len(sites) > MAX_IMM_ID:
        print(
            f"only the first {MAX_IMM_ID} of {len(sites)} sites get markers "
            "(s_ttracedata_imm IDs)",
            file=sys.stderr,
        )
        sites = sites[:MAX_IMM_ID]
    if not sites:
        # Exit code 3 tells callers (e.g. sqtt_site_injector) to keep the original.
        print("no sites to mark; code object left unchanged")
        return 3

    lines = source_lines(args.llvm, args.input, [insn.addr for insn, _ in sites])

    # Build stubs after .text (16-byte aligned), then the prefetch guard.
    stub_base = (text_end + 15) & ~15
    stubs = bytearray(b"\0" * (stub_base - text_end))
    patches = []  # (address, words) written over the original instruction
    funcmap = []
    for marker_id, (insn, kind) in enumerate(sites, start=1):
        stub_addr = stub_base + len(stubs) - (stub_base - text_end)
        back_addr = stub_addr + 4 + insn.size
        words = [S_TTRACEDATA_IMM | (marker_id << 2)] + insn.words
        words.append(branch(back_addr, insn.addr + insn.size))
        stubs += struct.pack(f"<{len(words)}I", *words)
        site_words = [branch(insn.addr, stub_addr)] + [S_NOP_0] * (len(insn.words) - 1)
        patches.append((insn.addr, site_words))
        loc = lines.get(insn.addr) or f"pc{insn.addr:#x}"
        funcmap.append(f"P:{marker_id}:site_{kind}@{loc}")
    stubs += struct.pack("<I", S_CODE_END) * (PREFETCH_GUARD // 4)
    added = len(stubs)
    if text_end + added > limit:
        sys.exit(f"stubs need {added} bytes but only {limit - text_end} are free after .text")

    # Patch the sites in the original text bytes.
    out = bytearray(data)
    for addr, words in patches:
        fileoff = text_fileoff + (addr - text_addr)
        struct.pack_into(f"<{len(words)}I", out, fileoff, *words)

    # Insert the stubs after .text and shift the rest of the file by whole pages,
    # so later PT_LOADs keep p_offset == p_vaddr (mod page).
    insert_at = text_fileoff + text_size
    shift = (added + 0xFFF) & ~0xFFF
    out[insert_at:insert_at] = stubs + b"\0" * (shift - added)
    elf2 = Elf64(bytes(out))
    for off, ph in elf2.phdrs():
        if off == exec_off:
            ph[5] += added  # p_filesz
            ph[6] += added  # p_memsz
        elif ph[2] >= insert_at:
            ph[2] += shift  # p_offset
        struct.pack_into("<IIQQQQQQ", out, off, *ph)
    new_shoff = elf2.shoff + (shift if elf2.shoff >= insert_at else 0)
    struct.pack_into("<Q", out, 0x28, new_shoff)

    # Section names come from the original file: once bytes have moved, the
    # .shstrtab offset in the headers is stale until its own header is updated.
    names_by_index = [elf.section_name(sh) for _, sh in elf.shdrs()]
    elf3 = Elf64(bytes(out))
    for (off, sh), name in zip(elf3.shdrs(), names_by_index):
        if name == ".text":
            sh[5] += added
        elif sh[4] >= insert_at and sh[1] != SHT_NOBITS:
            sh[4] += shift
        struct.pack_into("<IIQQQQIIQQ", out, off, *sh)

    # Append .sqtt_funcmap: new data, a new .shstrtab copy, and a new section
    # header table with one extra entry, all at the end of the file.
    elf4 = Elf64(bytes(out))
    rows = [f"K:{k}" for k in kernels] + funcmap
    fm = ("\n".join(rows) + "\n").encode()
    _, shstr = list(elf4.shdrs())[elf4.shstrndx]
    names = bytes(out[shstr[4] : shstr[4] + shstr[5]])
    name_off = len(names)
    names += b".sqtt_funcmap\0"

    fm_off = len(out)
    out += fm
    names_off = len(out)
    out += names
    while len(out) % 8:
        out += b"\0"
    headers = bytearray(out[elf4.shoff : elf4.shoff + elf4.shnum * elf4.shentsize])
    struct.pack_into("<QQ", headers, elf4.shstrndx * elf4.shentsize + 24, names_off, len(names))
    headers += struct.pack("<IIQQQQIIQQ", name_off, 1, 0, 0, fm_off, len(fm), 0, 0, 1, 0)
    new_shoff = len(out)
    out += headers
    struct.pack_into("<Q", out, 0x28, new_shoff)
    struct.pack_into("<H", out, 0x3C, elf4.shnum + 1)

    with open(args.output, "wb") as f:
        f.write(out)
    print(
        f"{len(sites)} sites marked, {added} bytes of stubs at {stub_base:#x}; "
        f"funcmap rows: {len(rows)}"
    )
    for row in funcmap:
        print("  " + row)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--sites",
        action="append",
        choices=sorted(SITE_KINDS),
        help="Instruction categories to mark (default: lds).",
    )
    parser.add_argument(
        "--kernel",
        type=re.compile,
        help="Only mark sites in functions whose (mangled) name matches this regex.",
    )
    parser.add_argument("--llvm", default="/opt/rocm/lib/llvm/bin")
    args = parser.parse_args()
    args.sites = args.sites or ["lds"]
    return rewrite(args)


if __name__ == "__main__":
    raise SystemExit(main())
