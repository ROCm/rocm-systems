#!/usr/bin/env python3
"""Insert SQTT site markers into an already-compiled AMDGPU code object.

Prototype for marking LDS (and later memory) instructions without recompiling.
Each selected instruction is moved into a trampoline stub:

    site:  s_branch stub           ; replaces the original instruction in place
           s_nop 0                 ; pads to the original instruction size
    stub:  s_ttracedata_imm <id>   ; the marker (same encoding as sqtt-marker)
           <original instruction>  ; DS ops are not PC-relative, so they move as-is
           s_branch site+size      ; back to the instruction after the site

s_ttracedata_imm only carries IDs 1..63. Later sites use s_ttracedata through
M0, like sqtt-marker does, saving M0 in an SGPR the code never touches:

    stub:  s_mov_b32 sF, m0 ; s_mov_b32 m0, <id << 2> ; s_nop 3 ; s_ttracedata
           s_nop 3 ; s_mov_b32 m0, sF ; s_nop 3 ; <original> ; s_branch back

Nothing in the original code moves, so branches, s_getpc sequences, kernel
descriptors, symbols and DWARF stay valid. Stubs live right after .text; the
file is grown there and later file offsets are shifted by whole pages. If the
page slack before the next segment is too small, later segments move up by
whole pages too, which is only done when code cannot refer to them (they hold
just .dynamic, or the code has no s_getpc).

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

# Highest SGPR a wave can address on gfx10+ (vcc is s106:s107).
MAX_SGPR = 105

PT_LOAD = 1
PF_X = 1
SHT_SYMTAB, SHT_RELA, SHT_DYNAMIC, SHT_NOBITS, SHT_DYNSYM = 2, 4, 6, 8, 11
SHF_ALLOC = 2
SHN_UNDEF, SHN_ABS = 0, 0xFFF1
PAGE = 0x1000

# Sections that code never addresses, so the segments holding them may move.
MOVABLE_SECTIONS = {".dynamic", ".relro_padding"}

# .dynamic tags whose values are addresses.
DT_ADDRESS_TAGS = {3, 4, 5, 6, 7, 12, 13, 17, 23, 25, 26, 32, 0x6FFFFEF5, 0x6FFFFFF0}

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


def clause_owner(insns: list[Insn]) -> dict[int, int]:
    """Map each instruction index inside an s_clause window to the s_clause index."""
    owner = {}
    for i, insn in enumerate(insns):
        if insn.mnemonic == "s_clause":
            # s_clause N groups the next N + 1 instructions.
            length = (insn.words[0] & 0x3F) + 1
            owner.update({j: i for j in range(i + 1, i + 1 + length)})
    return owner


def target_cpu(llvm: str, path: str) -> str:
    """GPU target of a code object, e.g. gfx1201, from the ELF header flags."""
    out = subprocess.run(
        [f"{llvm}/llvm-readelf", "-h", path], check=True, capture_output=True, text=True
    ).stdout
    m = re.search(r"Flags:.*\b(gfx[0-9a-f]+)", out)
    if not m:
        sys.exit("cannot tell the GPU target from the ELF header")
    return m.group(1)


def encode(llvm: str, cpu: str, lines: list[str]) -> list[list[int]]:
    """Encode assembly lines with llvm-mc; one word list per line."""
    out = subprocess.run(
        [f"{llvm}/llvm-mc", "-arch=amdgcn", f"-mcpu={cpu}", "-show-encoding"],
        input="\n".join(lines) + "\n",
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    encodings = []
    for m in re.finditer(r"encoding: \[([^\]]*)\]", out):
        raw = bytes(int(b, 16) for b in m.group(1).split(","))
        encodings.append(list(struct.unpack(f"<{len(raw) // 4}I", raw)))
    if len(encodings) != len(lines):
        sys.exit(f"llvm-mc encoded {len(encodings)} of {len(lines)} lines")
    return encodings


def free_sgpr(insns: list[Insn]) -> int | None:
    """An SGPR above every SGPR the code names, or None if there is none.

    Kernel metadata (.sgpr_count) is not reliable for hand-written kernels, so
    the operands are scanned instead. Relative SGPR addressing (s_movrel*) can
    reach any SGPR, so it rules the free register out.
    """
    highest = -1
    for insn in insns:
        if insn.mnemonic.startswith("s_movrel"):
            return None
        operands = insn.text[len(insn.mnemonic) :]
        for m in re.finditer(r"\bs\[(\d+):(\d+)\]|\bs(\d+)\b", operands):
            highest = max(highest, int(m.group(2) or m.group(3)))
    return highest + 1 if highest + 1 <= MAX_SGPR else None


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


def move_addresses(elf: Elf64, out: bytearray, limit: int, vshift: int) -> None:
    """Add vshift to every address at or above limit in symbols, .dynamic and relocations.

    Called after later segments have moved up by vshift; section headers and
    program headers are already updated.
    """

    def moved(value: int) -> int:
        return value + vshift if value >= limit else value

    for _, sh in elf.shdrs():
        kind, start, size, entsize = sh[1], sh[4], sh[5], sh[9]
        if kind in (SHT_SYMTAB, SHT_DYNSYM) and entsize:
            for off in range(start, start + size, entsize):
                name, info, other, shndx, value, sym_size = struct.unpack_from("<IBBHQQ", out, off)
                if shndx not in (SHN_UNDEF, SHN_ABS):
                    struct.pack_into(
                        "<IBBHQQ", out, off, name, info, other, shndx, moved(value), sym_size
                    )
        elif kind == SHT_DYNAMIC:
            for off in range(start, start + size, 16):
                tag, value = struct.unpack_from("<qQ", out, off)
                if tag in DT_ADDRESS_TAGS:
                    struct.pack_into("<qQ", out, off, tag, moved(value))
        elif kind == SHT_RELA and entsize:
            for off in range(start, start + size, entsize):
                r_offset, r_info, r_addend = struct.unpack_from("<QQq", out, off)
                addend = r_addend + vshift if r_addend >= limit else r_addend
                struct.pack_into("<QQq", out, off, moved(r_offset), r_info, addend)


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
    names_by_index = [elf.section_name(sh) for _, sh in elf.shdrs()]
    later_sections = {
        name
        for (_, sh), name in zip(elf.shdrs(), names_by_index)
        if sh[2] & SHF_ALLOC and sh[3] >= limit
    }

    insns, kernels = disassemble(args.llvm, args.input)
    owner = clause_owner(insns)
    sites = []
    clauses = set()  # s_clause instructions that cover a site
    for i, insn in enumerate(insns):
        kind = classify(insn, args.sites)
        if kind is None:
            continue
        if args.kernel and not args.kernel.search(insn.function):
            continue
        if i in owner:
            if args.keep_clauses:
                print(f"skip {insn.addr:#x} {insn.text}: inside s_clause", file=sys.stderr)
                continue
            # A branch out of a clause would break it, so the clause is turned
            # into s_nop 0 below. On gfx10+ clauses only group memory
            # instructions for issue; dropping one costs performance, not results.
            clauses.add(owner[i])
        if insn.size < 8:
            # 4-byte sites would need the s_branch alone; not needed for DS ops.
            print(f"skip {insn.addr:#x} {insn.text}: 4-byte instruction", file=sys.stderr)
            continue
        sites.append((insn, kind))

    # Marker templates for this target. IDs past MAX_IMM_ID need a free SGPR to
    # save M0 in; without one, only the first MAX_IMM_ID sites are marked.
    cpu = target_cpu(args.llvm, args.input)
    spare = free_sgpr(insns)
    if len(sites) > MAX_IMM_ID and spare is None:
        print(
            f"no free SGPR for M0 markers: only the first {MAX_IMM_ID} of {len(sites)} "
            "sites get markers",
            file=sys.stderr,
        )
        sites = sites[:MAX_IMM_ID]
    if not sites:
        # Exit code 3 tells callers (e.g. sqtt_site_injector) to keep the original.
        print("no sites to mark; code object left unchanged")
        return 3
    m0_template = None
    if len(sites) > MAX_IMM_ID:
        m0_template = encode(
            args.llvm,
            cpu,
            [
                f"s_mov_b32 s{spare}, m0",
                "s_mov_b32 m0, 0x12345678",  # the literal word is replaced per site
                "s_nop 3",  # gfx10+ needs 4 wait states between an M0 write and its use
                "s_ttracedata",
                f"s_mov_b32 m0, s{spare}",
            ],
        )

    def marker_words(marker_id: int) -> list[int]:
        encoded = marker_id << 2
        if marker_id <= MAX_IMM_ID:
            return [S_TTRACEDATA_IMM | encoded]
        save, set_m0, nop3, ttracedata, restore = m0_template
        # Waits around the M0 writes keep s_ttracedata and later M0 readers safe.
        return save + [set_m0[0], encoded] + nop3 + ttracedata + nop3 + restore + nop3

    def stub_size(marker_id: int, insn: Insn) -> int:
        return 4 * len(marker_words(marker_id)) + insn.size + 4

    # Room for the stubs: the page slack before the next segment, plus whole
    # pages gained by moving later segments up when nothing in them is
    # addressed from code.
    stub_base = (text_end + 15) & ~15
    needed = (stub_base - text_end) + PREFETCH_GUARD
    needed += sum(stub_size(i, insn) for i, (insn, _) in enumerate(sites, start=1))
    vshift = 0
    if text_end + needed > limit:
        has_getpc = any(insn.mnemonic.startswith("s_getpc") for insn in insns)
        if later_sections <= MOVABLE_SECTIONS or not has_getpc:
            vshift = (text_end + needed - limit + PAGE - 1) & ~(PAGE - 1)
        else:
            # Keep the sites that fit in the slack.
            room = limit - text_end - (stub_base - text_end) - PREFETCH_GUARD
            kept = []
            for marker_id, (insn, kind) in enumerate(sites, start=1):
                room -= stub_size(marker_id, insn)
                if room < 0:
                    break
                kept.append((insn, kind))
            print(
                f"stub space: only {len(kept)} of {len(sites)} sites fit; later segments "
                f"({', '.join(sorted(later_sections))}) may be addressed from code",
                file=sys.stderr,
            )
            sites = kept
            if not sites:
                print("no room for stubs; code object left unchanged")
                return 3

    lines = source_lines(args.llvm, args.input, [insn.addr for insn, _ in sites])

    # Build stubs after .text (16-byte aligned), then the prefetch guard.
    stubs = bytearray(b"\0" * (stub_base - text_end))
    patches = []  # (address, words) written over the original instruction
    funcmap = []
    for marker_id, (insn, kind) in enumerate(sites, start=1):
        stub_addr = text_end + len(stubs)
        words = marker_words(marker_id) + insn.words
        back_addr = stub_addr + 4 * len(words)
        words.append(branch(back_addr, insn.addr + insn.size))
        stubs += struct.pack(f"<{len(words)}I", *words)
        site_words = [branch(insn.addr, stub_addr)] + [S_NOP_0] * (len(insn.words) - 1)
        patches.append((insn.addr, site_words))
        loc = lines.get(insn.addr) or f"pc{insn.addr:#x}"
        funcmap.append(f"P:{marker_id}:site_{kind}@{loc}")
    stubs += struct.pack("<I", S_CODE_END) * (PREFETCH_GUARD // 4)
    added = len(stubs)

    # Patch the sites, and the clauses that cover them, in the original text bytes.
    # A clause whose sites were all dropped above stays an s_nop; harmless.
    patches += [(insns[i].addr, [S_NOP_0]) for i in sorted(clauses)]
    out = bytearray(data)
    for addr, words in patches:
        fileoff = text_fileoff + (addr - text_addr)
        struct.pack_into(f"<{len(words)}I", out, fileoff, *words)

    # Insert the stubs after .text and shift the rest of the file by whole pages,
    # so later PT_LOADs keep p_offset == p_vaddr (mod page); their addresses
    # move by vshift (also whole pages).
    insert_at = text_fileoff + text_size
    shift = (added + PAGE - 1) & ~(PAGE - 1)
    out[insert_at:insert_at] = stubs + b"\0" * (shift - added)
    elf2 = Elf64(bytes(out))
    for off, ph in elf2.phdrs():
        if off == exec_off:
            ph[5] += added  # p_filesz
            ph[6] += added  # p_memsz
        else:
            if ph[2] >= insert_at:
                ph[2] += shift  # p_offset
            if ph[3] >= limit:
                ph[3] += vshift  # p_vaddr
                ph[4] += vshift  # p_paddr
        struct.pack_into("<IIQQQQQQ", out, off, *ph)
    new_shoff = elf2.shoff + (shift if elf2.shoff >= insert_at else 0)
    struct.pack_into("<Q", out, 0x28, new_shoff)

    # Section names come from the original file: once bytes have moved, the
    # .shstrtab offset in the headers is stale until its own header is updated.
    elf3 = Elf64(bytes(out))
    for (off, sh), name in zip(elf3.shdrs(), names_by_index):
        if name == ".text":
            sh[5] += added
        elif sh[4] >= insert_at and sh[1] != SHT_NOBITS:
            sh[4] += shift
        if sh[2] & SHF_ALLOC and sh[3] >= limit:
            sh[3] += vshift
        struct.pack_into("<IIQQQQIIQQ", out, off, *sh)
    if vshift:
        move_addresses(Elf64(bytes(out)), out, limit, vshift)

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
    m0_sites = max(0, len(sites) - MAX_IMM_ID)
    print(
        f"{len(sites)} sites marked ({len(sites) - m0_sites} imm, {m0_sites} via M0"
        + (f" saved in s{spare}" if m0_sites else "")
        + f"), {added} bytes of stubs at {stub_base:#x}"
        + (f", later segments moved up {vshift:#x}" if vshift else "")
        + f"; funcmap rows: {len(rows)}"
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
    parser.add_argument(
        "--keep-clauses",
        action="store_true",
        help="Skip sites inside s_clause groups instead of dissolving the clause.",
    )
    parser.add_argument("--llvm", default="/opt/rocm/lib/llvm/bin")
    args = parser.parse_args()
    args.sites = args.sites or ["lds"]
    return rewrite(args)


if __name__ == "__main__":
    raise SystemExit(main())
