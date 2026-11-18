#!/usr/bin/env python3
"""Insert SQTT site markers into an already-compiled AMDGPU code object.

Marks LDS instructions without recompiling. Each selected instruction is
replaced in place by a branch to a stub that emits the marker, runs the
original instruction and branches back:

    site:  s_branch stub           ; replaces the original instruction in place
           s_nop 0                 ; pads to the original instruction size
    stub:  <marker>                ; same encoding as the sqtt-marker plugin
           <original instruction>  ; DS ops are not PC-relative, so they move as-is
           s_branch site+size      ; back to the instruction after the site

Markers: s_ttracedata_imm <id << 2> for IDs 1..63 where the target has it
(gfx10+). Otherwise s_ttracedata through M0, with M0 saved in an SGPR that the
code never names:

    s_mov_b32 sF, m0 ; s_mov_b32 m0, <id << 2> ; s_nop N ; s_ttracedata
    s_nop N ; s_mov_b32 m0, sF ; s_nop N       (N = 3 on gfx10+, 0 on gfx9)

On gfx9 the SGPR allocation comes from the kernel descriptor, so it is raised
when the spare SGPR (plus VCC, FLAT_SCRATCH, XNACK_MASK) would not fit.

Where stubs go. Nothing in the original code moves, so branches, s_getpc
sequences, kernel descriptors, symbols and DWARF stay valid. A stub goes in
the nearest place s_branch (+-128 KiB) can reach:
  * an island: alignment padding (s_nop 0 / s_code_end) between functions
    that follows an unconditional end of code (s_endpgm, s_branch, s_setpc)
    and that no branch targets, so it never runs;
  * the pool after .text (the file grows there; later file offsets move by
    whole pages, and later segments move up only when code cannot refer to
    them: they hold just .dynamic, or the code has no s_getpc).
If the stub is out of s_branch range, the path to it and back goes through
hops: single s_branch instructions placed in islands. A site with no
reachable place is skipped and reported; other sites are still marked.

Safety. Before the output is written, it is checked: every changed byte of
the original .text is a site, a dissolved clause or island padding; every
site's branch chain reaches its stub and returns to the next instruction; the
stub holds the marker and the original instruction; segments and sections are
consistent; llvm-readelf and llvm-objdump read it without new warnings. If any
check fails, nothing is written and the exit status is 4, so the caller (e.g.
sqtt_site_injector) loads the original code object.

A .sqtt_funcmap section with rows P:<id>:site_<kind>@<file>:<line> is added, so
the existing decoder and counter_fit.py read the markers like compile-time
SQTT_TRACE_SITES markers.

Exit status: 0 rewritten, 3 nothing to mark (no output), 4 verification failed
(no output), anything else an error (no output).

Usage:
    sqtt_site_rewrite.py in.co out.co [--sites lds] [--kernel REGEX]
                         [--keep-clauses] [--no-islands] [--llvm DIR]
"""

from __future__ import annotations

import argparse
import bisect
import os
import re
import struct
import subprocess
import tempfile
import sys
from dataclasses import dataclass, field

# Marker header layout shared with sqtt-marker: bits 31:2 are the ID, bits 1:0
# are scope flags. s_ttracedata_imm keeps 8 bits, so IDs 1..63 fit.
MAX_IMM_ID = 63

# CP instruction prefetch reads ahead of the PC; keep a readable guard after
# the pool so prefetch past the last stub stays inside the segment.
PREFETCH_GUARD = 256

# Longest chain of hops tried for one direction of one site.
MAX_HOPS = 16
# Stub places (islands or the pool) tried per site, nearest first.
MAX_CANDIDATES = 8

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

# Instructions after which execution never falls through.
TERMINATORS = re.compile(r"^(s_endpgm\w*|s_branch|s_setpc_b64|s_code_end)$")
# PC-relative branches with a simm16 dword offset in the low 16 bits.
DIRECT_BRANCH = re.compile(
    r"^(s_branch|s_cbranch_(?!g_fork|join)\w+|s_call_b64|s_subvector_loop_(begin|end))$"
)
# Instructions that read or write the program counter, and those of them a
# relaid-out kernel can keep (the others make a kernel stay where it is).
PC_TOUCHING = re.compile(r"(^|_)(get|set|swap)?pc(_|$)")
PC_RELAYOUT_OK = {"s_getpc_b64", "s_setpc_b64", "s_swappc_b64", "s_prefetch_inst_pc_rel"}
# Code that can compute addresses inside .text (data in .text, indirect jumps).
PC_RELATIVE = re.compile(r"^s_(getpc|setpc|swappc)_b64$")


class Unchanged(Exception):
    """The code object cannot be marked; it is left as it is (exit status 3)."""


class VerifyError(Exception):
    """The rewritten code object failed a check (exit status 4)."""


@dataclass(slots=True)
class Insn:
    addr: int
    words: list[int]
    text: str
    function: str = ""
    mnemonic: str = ""  # filled in from text when not given

    def __post_init__(self):
        if not self.mnemonic:
            self.mnemonic = self.text.split(None, 1)[0]

    @property
    def size(self) -> int:
        return 4 * len(self.words)

    @property
    def end(self) -> int:
        return self.addr + self.size


@dataclass
class Target:
    """Encodings and rules for one GPU target, probed with llvm-mc."""

    cpu: str
    gfx9: bool
    s_branch: int  # s_branch 0; the low 16 bits take the offset
    s_nop0: int
    s_ttracedata: int
    s_ttracedata_imm: int | None  # None where only the M0 form exists
    guard: int  # s_code_end, or s_nop 0 where there is none
    padding: set[int]  # words the compiler uses for alignment padding
    has_clause: bool
    m0_wait: int  # s_nop argument after an M0 write
    max_sgpr: int  # highest SGPR a wave can name

    def is_branch(self, word: int) -> bool:
        return word & 0xFFFF0000 == self.s_branch

    def branch(self, from_addr: int, to_addr: int) -> int:
        delta = to_addr - (from_addr + 4)
        if not reach(from_addr, to_addr):
            raise ValueError(f"s_branch out of range: {from_addr:#x} -> {to_addr:#x}")
        return self.s_branch | ((delta // 4) & 0xFFFF)


def reach(from_addr: int, to_addr: int) -> bool:
    """Can an s_branch at from_addr jump to to_addr?"""
    delta = to_addr - (from_addr + 4)
    return delta % 4 == 0 and -0x8000 <= delta // 4 <= 0x7FFF


def branch_target(addr: int, word: int) -> int:
    simm = word & 0xFFFF
    if simm & 0x8000:
        simm -= 0x10000
    return addr + 4 + 4 * simm


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def mc_encode(llvm: str, cpu: str, lines: list[str]) -> dict[str, list[int]]:
    """Encode assembly lines with llvm-mc; unsupported lines are left out."""
    out = run(
        [f"{llvm}/llvm-mc", "-arch=amdgcn", f"-mcpu={cpu}", "-show-encoding"],
        input="\n".join(lines) + "\n",
    ).stdout
    result = {}
    for m in re.finditer(r"^\s*(.*?)\s*;\s*encoding: \[([^\]]*)\]", out, re.M):
        raw = bytes(int(b, 16) for b in m.group(2).split(","))
        result[" ".join(m.group(1).split())] = list(struct.unpack(f"<{len(raw) // 4}I", raw))
    return result


def probe_target(llvm: str, cpu: str) -> Target:
    enc = mc_encode(
        llvm,
        cpu,
        ["s_branch 0", "s_nop 0", "s_ttracedata", "s_ttracedata_imm 0x0", "s_code_end", "s_clause 0x0"],
    )
    for need in ("s_branch 0", "s_nop 0", "s_ttracedata"):
        if need not in enc:
            raise Unchanged(f"llvm-mc cannot encode {need} for {cpu}")
    gfx9 = re.fullmatch(r"gfx9[0-9a-f]+", cpu) is not None
    if not gfx9 and not re.fullmatch(r"gfx1[0-9]{3}", cpu):
        raise Unchanged(f"target {cpu} is not supported (gfx9 or gfx10+ only)")
    imm = enc.get("s_ttracedata_imm 0x0")
    code_end = enc.get("s_code_end")
    nop0 = enc["s_nop 0"][0]
    return Target(
        cpu=cpu,
        gfx9=gfx9,
        s_branch=enc["s_branch 0"][0],
        s_nop0=nop0,
        s_ttracedata=enc["s_ttracedata"][0],
        s_ttracedata_imm=imm[0] if imm else None,
        guard=code_end[0] if code_end else nop0,
        padding={nop0} | ({code_end[0]} if code_end else set()),
        has_clause="s_clause 0x0" in enc,
        m0_wait=0 if gfx9 else 3,
        # gfx9: s0..s101 (VCC and friends sit above). gfx10+: s0..s105.
        max_sgpr=101 if gfx9 else 105,
    )


def disassemble(llvm: str, path: str, start: int | None = None) -> tuple[list[Insn], dict[int, str]]:
    """Instructions in .text (address, encoding words, text) and labels by address,
    optionally only from address start onwards."""
    cmd = [f"{llvm}/llvm-objdump", "-d", "-z", "--section=.text", path]
    if start is not None:
        cmd.append(f"--start-address={start:#x}")
    # Through a temporary file, read line by line: faster than a large pipe,
    # and the whole listing is never in memory at once.
    with tempfile.TemporaryFile("w+") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
        if res.returncode:
            raise subprocess.CalledProcessError(res.returncode, cmd, stderr=res.stderr)
        f.seek(0)
        return _parse_listing(line.rstrip("\n") for line in f)


def _parse_listing(lines) -> tuple[list[Insn], dict[int, str]]:
    """Parse llvm-objdump lines.

    Instruction lines, e.g. "\\tds_load_b32 v4, v4   // 000000001D3C: D8D80000 04000004";
    branches end with the target, e.g. "BF88007B <Cijk_H_GA+0x230>".
    Label lines, e.g. "0000000000001900 <_Z12lds_conflictPfii>:".
    """
    hexdigits = set("0123456789abcdefABCDEF")
    insns: list[Insn] = []
    labels: dict[int, str] = {}
    current = ""
    for line in lines:
        if line[:1] in ("\t", " "):
            c = line.rfind("//")
            if c < 0:
                continue
            text = line[:c].strip()
            rest = line[c + 2 :]
            colon = rest.find(":")
            if not text or colon < 0:
                continue
            tail = rest[colon + 1 :]
            lt = tail.find("<")
            if lt >= 0:
                tail = tail[:lt]
            fields = tail.split()
            if not fields or any(len(w) != 8 or not set(w) <= hexdigits for w in fields):
                continue
            insn = Insn(int(rest[:colon], 16), [int(w, 16) for w in fields], text, current)
            insns.append(insn)
        elif line.endswith(">:") and " <" in line:
            addr, _, name = line.partition(" <")
            if addr and set(addr) <= hexdigits:
                current = name[:-2]
                labels[int(addr, 16)] = current
    return insns, labels


def source_lines(llvm: str, path: str, addrs: list[int]) -> dict[int, str]:
    """Map instruction addresses to file:line through the DWARF line table."""
    if not addrs:
        return {}
    # Addresses go through stdin: there can be too many for a command line.
    out = run(
        [f"{llvm}/llvm-addr2line", "-e", path], input="\n".join(hex(a) for a in addrs) + "\n",
        check=True,
    ).stdout.splitlines()
    if len(out) != len(addrs):
        raise Unchanged(f"llvm-addr2line returned {len(out)} lines for {len(addrs)} addresses")
    result = {}
    for addr, loc in zip(addrs, out):
        # Drop the discriminator suffix that addr2line may append.
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
    out = run([f"{llvm}/llvm-readelf", "-h", path], check=True).stdout
    m = re.search(r"Flags:.*\b(gfx[0-9a-f]+)", out)
    if not m:
        raise Unchanged("cannot tell the GPU target from the ELF header")
    return m.group(1)


def free_sgpr(insns: list[Insn], max_sgpr: int) -> int | None:
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
    return highest + 1 if highest + 1 <= max_sgpr else None


@dataclass
class PcRef:
    """A recognised PC-relative address: s_getpc_b64 plus a constant offset."""

    getpc: int  # address of the s_getpc_b64
    offset: int
    # LLVM form: addresses of the low and high literal words.
    lits: tuple[int, int] | None = None
    # Tensile long branch: (address of the s_add_i32 literal word, the other
    # constant operand, backward?). The offset is literal + constant.
    lit32: tuple[int, int, bool] | None = None

    @property
    def base(self) -> int:
        return self.getpc + 4  # s_getpc_b64 returns the address of the next instruction

    @property
    def target(self) -> int:
        return self.base + self.offset

    @property
    def adjustable(self) -> bool:
        return self.lits is not None or self.lit32 is not None

    def encode(self, offset: int) -> list[tuple[int, int]] | None:
        """(address, word) writes that make this reference use a new offset, or
        None if this form cannot express it."""
        if self.lits is not None:
            value = offset & 0xFFFFFFFFFFFFFFFF
            return [(self.lits[0], value & 0xFFFFFFFF), (self.lits[1], value >> 32)]
        if self.lit32 is not None:
            addr, other, backward = self.lit32
            # Forward adds the (non-negative) sum; backward subtracts its absolute value.
            if (offset > 0 if backward else offset < 0) or not -(1 << 31) <= offset < (1 << 31):
                return None
            return [(addr, (offset - other) & 0xFFFFFFFF)]
        return None


def operands(insn: Insn) -> list[str]:
    return [o.strip() for o in insn.text[len(insn.mnemonic) :].split(",") if o.strip()]


def constant(op: str) -> int | None:
    """Value of an immediate operand (inline or literal), or None for a register."""
    m = re.fullmatch(r"(?:lit\()?(-?(?:0x[0-9a-fA-F]+|\d+))\)?", op)
    return int(m.group(1), 0) if m else None


def sext32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >> 31 else value


def pc_references(insns: list[Insn]) -> tuple[list[PcRef], list[int]]:
    """Recognised s_getpc_b64 uses, and the addresses of any not recognised.

    Recognised forms:
      LLVM PC-relative address (the literals are adjustable):
        s_getpc_b64 s[a:b] ; [s_sext_i32_i16 sb, sb] ; s_add(_co)_u32 sa, sa, lit ;
        s_addc_u32 / s_add_co_ci_u32 sb, sb, lit
      Tensile long branch, forward and backward:
        s_getpc_b64 s[a:b] ; s_add_i32 sT, x, y ; s_add_u32 sa, sa, sT ; s_addc_u32 sb, sb, 0
        s_getpc_b64 s[a:b] ; s_add_i32 sT, x, y ; s_abs_i32 sT, sT ;
        s_sub_u32 sa, sa, sT ; s_subb_u32 sb, sb, 0
    """
    refs, unknown = [], []

    def at(k: int) -> Insn | None:
        return insns[k] if k < len(insns) else None

    def shaped(insn: Insn | None, mnemonics: tuple[str, ...], ops: list[str | None]) -> bool:
        if insn is None or insn.mnemonic not in mnemonics:
            return False
        got = operands(insn)
        return len(got) == len(ops) and all(o is None or o == g for o, g in zip(ops, got))

    for i, insn in enumerate(insns):
        if insn.mnemonic != "s_getpc_b64":
            continue
        m = re.fullmatch(r"s_getpc_b64\s+s\[(\d+):(\d+)\]", insn.text.strip())
        if not m:
            unknown.append(insn.addr)
            continue
        a, b = f"s{m.group(1)}", f"s{m.group(2)}"
        j = i + 1
        if shaped(at(j), ("s_sext_i32_i16",), [b, b]):
            j += 1
        lo, hi = at(j), at(j + 1)
        if (
            shaped(lo, ("s_add_u32", "s_add_co_u32"), [a, a, None])
            and shaped(hi, ("s_addc_u32", "s_add_co_ci_u32"), [b, b, None])
            and len(lo.words) == 2
            and len(hi.words) == 2
        ):
            value = hi.words[1] << 32 | lo.words[1]
            offset = value - (1 << 64) if value >> 63 else value
            refs.append(PcRef(insn.addr, offset, (lo.addr + 4, hi.addr + 4)))
            continue
        add = at(i + 1)
        if add is not None and add.mnemonic == "s_add_i32" and len(operands(add)) == 3:
            t, x, y = operands(add)
            cx, cy = constant(x), constant(y)
            if cx is not None and cy is not None:
                delta = sext32(cx + cy)
                # Which operand is the 32-bit literal (adjustable), if any.
                lit = None
                if len(add.words) == 2:
                    if cx & 0xFFFFFFFF == add.words[1]:
                        lit = (add.addr + 4, cy)
                    elif cy & 0xFFFFFFFF == add.words[1]:
                        lit = (add.addr + 4, cx)
                if (
                    shaped(at(i + 2), ("s_add_u32",), [a, a, t])
                    and shaped(at(i + 3), ("s_addc_u32",), [b, b, "0"])
                    and delta >= 0
                ):
                    refs.append(PcRef(insn.addr, delta, lit32=lit and (*lit, False)))
                    continue
                if (
                    shaped(at(i + 2), ("s_abs_i32",), [t, t])
                    and shaped(at(i + 3), ("s_sub_u32",), [a, a, t])
                    and shaped(at(i + 4), ("s_subb_u32",), [b, b, "0"])
                ):
                    refs.append(PcRef(insn.addr, -abs(delta), lit32=lit and (*lit, True)))
                    continue
        unknown.append(insn.addr)
    return refs, unknown


SGPR_OPERAND = re.compile(r"\bs\[(\d+):(\d+)\]|\bs(\d+)\b")
LIT_ADD_LO = ("s_add_u32", "s_add_co_u32")
LIT_ADD_HI = ("s_addc_u32", "s_add_co_ci_u32")
TERMINATOR_MNEMONICS = re.compile(r"^(s_endpgm\w*|s_branch|s_setpc_b64|s_code_end)$")


def sgprs_of(op: str) -> set[int]:
    regs = set()
    for m in SGPR_OPERAND.finditer(op):
        lo = int(m.group(1) or m.group(3))
        hi = int(m.group(2) or m.group(3))
        regs.update(range(lo, hi + 1))
    return regs


def pair_of(op: str) -> int | None:
    """Low register of an SGPR pair operand s[n:n+1], else None."""
    m = re.fullmatch(r"s\[(\d+):(\d+)\]", op)
    return int(m.group(1)) if m and int(m.group(2)) == int(m.group(1)) + 1 else None


@dataclass
class LitAdd:
    """sD = sS + literal over a register pair (lo add + hi add-with-carry),
    where sS holds the value of one s_getpc_b64: a PC-relative address."""

    lo: int  # index of the low add (its literal is the second word)
    hi: int  # index of the high add
    getpc: int  # address of the s_getpc_b64 whose value is the input
    offset: int  # target = getpc + 4 + offset
    hi_literal: bool  # the high add has its own literal word (else inline 0 / -1)


@dataclass
class PcFlow:
    """What a scan of every path from the code entry points found."""

    reachable: set[int]  # indices of instructions some entry reaches
    jumps: dict[int, set[int]]  # s_setpc / s_swappc index -> resolved target addresses
    lit_adds: dict[int, LitAdd]  # by index of the low add
    escapes: set[int]  # s_getpc addresses whose value is used in a way not followed


def pc_flow(insns: list[Insn], roots: set[int], pc_refs: list["PcRef"]) -> PcFlow:
    """Follow every path from the roots, tracking SGPR pairs that hold
    PC-derived values (an s_getpc result plus constants), to find which code
    is reachable and where computed jumps (s_setpc / s_swappc) go.

    Values are tracked as (getpc address, offset). A pair stops being tracked
    when anything else writes to it; a use the scan does not understand marks
    the s_getpc as escaped (its code cannot be moved). Recognised Tensile and
    LLVM patterns (pc_refs) are applied as a whole.
    """
    index = {i.addr: k for k, i in enumerate(insns)}
    n = len(insns)
    # Recognised patterns: from the s_getpc to the last instruction that
    # forms the address, applied as one step.
    pattern_end: dict[int, tuple[int, "PcRef", int]] = {}  # getpc index -> (last index, ref, pair)
    for r in pc_refs:
        k = index[r.getpc]
        pair = pair_of(operands(insns[k])[0])
        last = k
        for j in range(k + 1, min(k + 6, n)):
            if insns[j].mnemonic in ("s_setpc_b64", "s_swappc_b64"):
                break
            if pair is not None and sgprs_of(operands(insns[j])[0] if operands(insns[j]) else "") & {pair, pair + 1}:
                last = j
        pattern_end[k] = (last, r, pair)

    state: list[dict | None] = [None] * n
    jumps: dict[int, set[int]] = {}
    lit_adds: dict[int, LitAdd] = {}
    escapes: set[int] = set()
    reachable: set[int] = set()
    empty: dict = {}  # shared: most instructions track nothing
    work = [(index[a], empty) for a in sorted(roots) if a in index]

    def merge(old: dict | None, new: dict) -> dict:
        if old is None:
            return new
        if not old or not new:
            return empty
        out = {}
        for reg in old.keys() & new.keys():
            union = old[reg] | new[reg]
            if len(union) <= 16:
                out[reg] = union
        return out or empty

    while work:
        k, incoming = work.pop()
        merged = merge(state[k], incoming)
        if k in reachable and merged == state[k]:
            continue
        state[k] = merged
        reachable.add(k)
        if not merged and k not in pattern_end and insns[k].mnemonic != "s_getpc_b64":
            # Nothing tracked here: only follow the control flow.
            insn = insns[k]
            mn = insn.mnemonic
            fall = not TERMINATOR_MNEMONICS.match(mn) or mn == "s_swappc_b64"
            if DIRECT_BRANCH.match(mn):
                t = branch_target(insn.addr, insn.words[0])
                if t in index:
                    work.append((index[t], empty))
                fall = fall or mn == "s_call_b64"
            if mn == "s_swappc_b64":
                fall = True
            if fall and k + 1 < n:
                work.append((k + 1, empty))
            continue
        # Recognised pattern: jump over it with the formed address.
        if k in pattern_end and pattern_end[k][2] is not None:
            last, ref, pair = pattern_end[k]
            st = {r: v for r, v in merged.items() if not {r, r + 1} & {pair, pair + 1}}
            for j in range(k + 1, last + 1):
                reachable.add(j)
                written = sgprs_of(operands(insns[j])[0]) if operands(insns[j]) else set()
                st = {r: v for r, v in st.items() if not {r, r + 1} & written}
            st[pair] = frozenset({(ref.getpc, ref.offset)})
            work.append((last + 1, st)) if last + 1 < n else None
            continue
        insn = insns[k]
        mn, ops = insn.mnemonic, operands(insn)
        out = dict(merged)
        succ: list[int] = []
        fall = not TERMINATOR_MNEMONICS.match(mn)

        def kill(regs: set[int]) -> None:
            for r in [r for r in out if {r, r + 1} & regs]:
                del out[r]

        def escape_reads(regs: set[int]) -> None:
            for r in [r for r in merged if {r, r + 1} & regs]:
                for g, off in merged[r]:
                    if off == 0:
                        escapes.add(g)  # the raw PC value is used directly

        if mn == "s_getpc_b64" and ops and pair_of(ops[0]) is not None:
            a = pair_of(ops[0])
            kill({a, a + 1})
            out[a] = frozenset({(insn.addr, 0)})
        elif (
            mn in LIT_ADD_LO and len(ops) == 3 and len(insn.words) == 2 and k + 1 < n
            and insns[k + 1].mnemonic in LIT_ADD_HI
            and re.fullmatch(r"s\d+", ops[0] or "") and re.fullmatch(r"s\d+", ops[1] or "")
            and int(ops[1][1:]) in merged
        ):
            d, s = int(ops[0][1:]), int(ops[1][1:])
            hops = operands(insns[k + 1])
            hk = constant(hops[2]) if len(hops) == 3 else None
            if hops[:2] == [f"s{d + 1}", f"s{s + 1}"] and hk is not None:
                hi_lit = len(insns[k + 1].words) == 2
                value = (hk & 0xFFFFFFFF) << 32 | insn.words[1]
                off = value - (1 << 64) if value >> 63 else value
                src = merged[s]
                if len(src) == 1 and next(iter(src))[1] == 0:
                    g = next(iter(src))[0]
                    lit_adds[k] = LitAdd(k, k + 1, g, off, hi_lit)
                else:
                    escape_reads({s})  # several or chained origins: cannot re-aim one literal
                    for g, _ in src:
                        escapes.add(g)
                kill({d, d + 1})
                out[d] = frozenset((g, o + off) for g, o in src)
                # The high add is part of this step.
                reachable.add(k + 1)
                state[k + 1] = out
                work.append((k + 2, out)) if k + 2 < n else None
                continue
            escape_reads(sgprs_of(insn.text[len(mn):]))
            kill(sgprs_of(ops[0]) if ops else set())
        elif mn == "s_mov_b64" and len(ops) == 2 and pair_of(ops[0]) is not None:
            c, a = pair_of(ops[0]), pair_of(ops[1])
            kill({c, c + 1})
            if a is not None and a in merged:
                out[c] = merged[a]
        elif mn == "s_sext_i32_i16" and len(ops) == 2 and ops[0] == ops[1]:
            pass  # sign-extends the high PC half in place; the address is unchanged
        elif mn in ("s_setpc_b64", "s_swappc_b64"):
            target_op = ops[-1] if ops else ""
            a = pair_of(target_op)
            if a is not None and a in merged:
                jumps[k] = {g + 4 + off for g, off in merged[a]}
                succ += [index[t] for t in jumps[k] if t in index]
            if mn == "s_swappc_b64" and ops:
                kill(sgprs_of(ops[0]))
                fall = True  # the call returns to the next instruction
        else:
            regs = sgprs_of(insn.text[len(mn):])
            if regs & {r for p in merged for r in (p, p + 1)}:
                escape_reads(regs)
                kill(regs)
        if DIRECT_BRANCH.match(mn):
            t = branch_target(insn.addr, insn.words[0])
            if t in index:
                succ.append(index[t])
            if mn == "s_call_b64":
                fall = True
        if fall and k + 1 < n:
            succ.append(k + 1)
        for j in succ:
            work.append((j, out))
    return PcFlow(reachable, jumps, lit_adds, escapes)


def flow_successors(insns: list[Insn], flow: PcFlow, k: int, index: dict[int, int]):
    """(successor indices, leaves to unknown code?) of a reachable instruction."""
    insn = insns[k]
    mn = insn.mnemonic
    succ, unknown = [], False
    fall = not TERMINATOR_MNEMONICS.match(mn)
    if DIRECT_BRANCH.match(mn):
        t = branch_target(insn.addr, insn.words[0])
        if t in index:
            succ.append(index[t])
        if mn == "s_call_b64":
            fall = True
    if mn == "s_setpc_b64":
        if k in flow.jumps:
            succ += [index[t] for t in flow.jumps[k] if t in index]
        else:
            unknown = True  # a return or a jump the scan could not resolve
    if mn == "s_swappc_b64":
        fall = True
    if fall and k + 1 < len(insns):
        succ.append(k + 1)
    return succ, unknown


def loop_depths(insns: list[Insn], flow: PcFlow, sites: list[int]) -> list[int]:
    """How many loops (backward branches or resolved backward jumps in
    reachable code) contain each of the given instruction indices."""
    index = {i.addr: k for k, i in enumerate(insns)}
    events = []
    for k in flow.reachable:
        for j in flow_successors(insns, flow, k, index)[0]:
            if j <= k:  # back edge: the loop is [j, k]
                events += [(j, 1), (k + 1, -1)]
    events.sort()
    starts = [e[0] for e in events]
    prefix, total = [], 0
    for _, d in events:
        total += d
        prefix.append(total)
    return [prefix[bisect.bisect_right(starts, s) - 1] if bisect.bisect_right(starts, s) else 0 for s in sites]


ALL_SGPRS = (1 << 106) - 1
# Instructions that write their first operand (an SGPR or SGPRs). Anything not
# listed is treated as only reading its SGPR operands, which can only make a
# register look live, never dead.
SGPR_WRITERS = re.compile(
    r"^(s_(mov|cmov|add|sub|mul|and|or|xor|nand|nor|xnor|andn2|orn2|lshl|lshr|ashr|bfe|bfm|"
    r"bitset|brev|bcnt|ff0|ff1|flbit|sext|not|wqm|quadmask|abs|min|max|cselect|getpc|load|"
    r"buffer_load|getreg|absdiff|pack|cvt|ctz|clz|cls|bitreplicate|movk|cmovk|addk|mulk|"
    r"memtime|memrealtime|sendmsg_rtn)\w*|v_readlane\w*|v_readfirstlane\w*|v_cmp_\w*_e64)$"
)
# Writers that also read their destination (conditional or read-modify-write).
SGPR_RMW = re.compile(r"^s_(cmov|cmovk|addk|mulk|bitset)\w*$")


def sgpr_mask(regs: set[int]) -> int:
    mask = 0
    for r in regs:
        mask |= 1 << r
    return mask


def sgpr_def_use(insn: Insn) -> tuple[int, int]:
    """(SGPRs certainly written, SGPRs possibly read) as bit masks."""
    mn = insn.mnemonic
    ops = operands(insn)
    if "movrel" in mn:
        return 0, ALL_SGPRS  # indexed access: may read any SGPR
    if mn in ("s_swappc_b64", "s_call_b64"):
        return (sgpr_mask(sgprs_of(ops[0])) if ops else 0), ALL_SGPRS  # the callee may read any
    if not ops:
        return 0, 0
    first = sgpr_mask(sgprs_of(ops[0]))
    rest = 0
    for op in ops[1:]:
        rest |= sgpr_mask(sgprs_of(op))
    if SGPR_WRITERS.match(mn) and not mn.startswith("s_movrel"):
        return first, rest | (first if SGPR_RMW.match(mn) else 0)
    return 0, first | rest


def sgpr_liveness(insns: list[Insn], flow: PcFlow) -> list[int]:
    """SGPRs live on entry to each reachable instruction (bit masks).

    Backward data flow over the reachable code. Returns, jumps the scan could
    not resolve and calls count as reading every SGPR, so a register is only
    reported dead when every path writes it before reading it (or ends).
    """
    n = len(insns)
    index = {i.addr: k for k, i in enumerate(insns)}
    succ: dict[int, list[int]] = {}
    preds: dict[int, list[int]] = {}
    unknown: set[int] = set()
    du: dict[int, tuple[int, int]] = {}
    for k in flow.reachable:
        s, u = flow_successors(insns, flow, k, index)
        succ[k] = s
        if u:
            unknown.add(k)
        for j in s:
            preds.setdefault(j, []).append(k)
        du[k] = sgpr_def_use(insns[k])
    live = [0] * n
    work = sorted(flow.reachable)
    pending = set(work)
    while work:
        k = work.pop()
        pending.discard(k)
        out = ALL_SGPRS if k in unknown else 0
        for j in succ[k]:
            out |= live[j]
        d, u = du[k]
        new = u | (out & ~d)
        if new != live[k]:
            live[k] = new
            for p in preds.get(k, []):
                if p not in pending:
                    pending.add(p)
                    work.append(p)
    return live


def gfx9_sgpr_granules(sgprs: int) -> int:
    """GRANULATED_WAVEFRONT_SGPR_COUNT for gfx9 (LLVM AMDGPUUsage formula)."""
    return 2 * max(0, (sgprs + 15) // 16 - 1)


class Elf64:
    """Just enough little-endian ELF64 editing for this tool."""

    def __init__(self, data: bytes):
        self.data = bytearray(data)
        if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
            raise Unchanged("not a little-endian ELF64 file")
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
        raise Unchanged(f"no {name} section")

    def file_offset(self, addr: int) -> int | None:
        for _, ph in self.phdrs():
            if ph[0] == PT_LOAD and ph[3] <= addr < ph[3] + ph[5]:
                return ph[2] + (addr - ph[3])
        return None

    def symbols(self):
        """(name, value, size, info, shndx) of every symbol in .symtab and .dynsym."""
        shdrs = [sh for _, sh in self.shdrs()]
        for sh in shdrs:
            if sh[1] not in (SHT_SYMTAB, SHT_DYNSYM) or not sh[9]:
                continue
            strtab = shdrs[sh[6]]
            for off in range(sh[4], sh[4] + sh[5], sh[9]):
                name, info, _, shndx, value, size = struct.unpack_from("<IBBHQQ", self.data, off)
                start = strtab[4] + name
                text = self.data[start : self.data.index(b"\0", start)].decode(errors="replace")
                yield text, value, size, info, shndx

    def kernel_descriptors(self) -> dict[str, int]:
        """Kernel name -> address of its 64-byte kernel descriptor."""
        kds = {}
        for name, value, size, _, shndx in self.symbols():
            if name.endswith(".kd") and shndx not in (SHN_UNDEF, SHN_ABS) and size == 64:
                kds[name[:-3]] = value
        return kds


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


# --- stub placement -------------------------------------------------------------


@dataclass
class Region:
    """Free space for stubs and hops: an island in .text, or the pool after it."""

    start: int
    end: int  # exclusive; may be huge for an unbounded pool
    pool: bool = False
    cursor: int = field(init=False)

    def __post_init__(self):
        self.cursor = self.start

    @property
    def free(self) -> int:
        return self.end - self.cursor


@dataclass
class Placement:
    insn: Insn
    kind: str
    marker_id: int
    stub: int  # stub address
    stub_words: list[int]  # marker + original instruction (branch back excluded)
    forward: list[int]  # hop addresses from the site to the stub
    back: list[int]  # hop addresses from the stub back to site.end
    orig: int = -1  # the site's address in the original code (differs in a kernel copy)

    @property
    def back_branch(self) -> int:
        return self.stub + 4 * len(self.stub_words)


@dataclass
class Relocation:
    """A kernel's code [start, end) copied unchanged to new (in the pool)."""

    start: int
    end: int
    new: int

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def delta(self) -> int:
        return self.new - self.start


@dataclass
class Relayout:
    """A kernel's code [start, end) laid out again at new, with markers inline.

    Every instruction keeps its encoding except branch offsets and the
    literals of PC-relative references, which are re-aimed at the new layout.
    """

    start: int
    end: int
    new: int
    words: list[int]
    group: dict[int, int]  # original address -> new address of its markers (or of itself)
    new_at: dict[int, int]  # original address -> new address of the instruction
    markers: list[tuple[int, str, int, int, list[int]]]  # (id, kind, site, marker address, words)
    refs: list[tuple["PcRef", int, int | None]]  # (reference, new base, new target if inside)
    branches: set[int]  # original addresses of re-aimed direct branches
    literals: set[int]  # new addresses of re-aimed literal words
    lit_refs: list = field(default_factory=list)  # (LitAdd, new base, new target if inside)
    # Pinned s_getpc: original address -> (words inserted after it, literal index).
    pins: dict = field(default_factory=dict)
    island_words: dict = field(default_factory=dict)  # island start -> words
    island_branch: dict = field(default_factory=dict)  # island branch address -> its target

    @property
    def size(self) -> int:
        return 4 * len(self.words)


def find_islands(target: Target, insns: list[Insn], labels: dict[int, str]) -> list[Region]:
    """Alignment padding that never runs, usable for stubs and hops.

    A run of padding words qualifies when it follows an unconditional end of
    code, ends where a function (label) starts, and no direct branch lands in
    it. When the code can compute addresses inside .text (s_getpc, s_setpc),
    the run must also end on a 256-byte boundary and be shorter than 256 bytes,
    i.e. be pure function alignment, so data tables in .text are never touched.
    The padding after the last function is the prefetch guard and is kept.
    """
    targets = {
        branch_target(i.addr, i.words[0]) for i in insns if DIRECT_BRANCH.match(i.mnemonic)
    }
    pc_relative = any(PC_RELATIVE.match(i.mnemonic) for i in insns)
    islands = []
    i = 0
    while i < len(insns):
        insn = insns[i]
        if not TERMINATORS.match(insn.mnemonic):
            i += 1
            continue
        j = i + 1
        while j < len(insns) and len(insns[j].words) == 1 and insns[j].words[0] in target.padding:
            j += 1
        if j > i + 1 and j < len(insns) and insns[j].addr in labels:
            start, end = insns[i + 1].addr, insns[j].addr
            ok = not any(start <= t < end for t in targets)
            if pc_relative:
                ok = ok and end % 256 == 0 and end - start < 256
            if ok:
                islands.append(Region(start, end))
        i = max(j, i + 1)
    return islands


class Placer:
    def __init__(self, islands: list[Region], pool: Region):
        self.islands = islands
        self.starts = [r.start for r in islands]
        self.pool = pool
        self.journal: list[tuple[Region, int]] = []

    def add_island(self, region: Region) -> None:
        """Extra stub space (in the pool, above every existing island)."""
        assert not self.starts or region.start >= self.starts[-1]
        self.islands.append(region)
        self.starts.append(region.start)

    def take(self, region: Region, size: int) -> int:
        addr = region.cursor
        self.journal.append((region, addr))
        region.cursor += size
        return addr

    def rollback(self, mark: int) -> None:
        while len(self.journal) > mark:
            region, cursor = self.journal.pop()
            region.cursor = cursor

    def near(self, addr: int, span: int = 4 * 0x8000 + 8):
        """Islands that may hold an address within s_branch reach of addr."""
        # Islands are short (alignment padding), so a 64 KiB margin is plenty.
        lo = bisect.bisect_left(self.starts, addr - span - 0x10000)
        hi = bisect.bisect_right(self.starts, addr + span)
        return self.islands[lo:hi]

    def route(self, frm: int, dest: int) -> list[int] | None:
        """Hops (in islands) for a branch chain from frm to dest, or None."""
        hops = []
        cur = frm
        for _ in range(MAX_HOPS + 1):
            if reach(cur, dest):
                return hops
            best = None
            for region in self.near(cur):
                if region.free < 4:
                    continue
                a = region.cursor
                if reach(cur, a) and abs(dest - a) < abs(dest - cur):
                    if best is None or abs(dest - a) < abs(dest - best.cursor):
                        best = region
            if best is None:
                return None
            cur = self.take(best, 4)
            hops.append(cur)
        return None

    def place(self, insn: Insn, kind: str, marker_id: int, words: list[int]) -> Placement | None:
        size = 4 * len(words) + 4
        candidates = [r for r in self.islands if r.free >= size]
        if self.pool.free >= size:
            candidates.append(self.pool)
        candidates.sort(key=lambda r: abs(r.cursor - insn.addr))
        for region in candidates[:MAX_CANDIDATES]:
            mark = len(self.journal)
            stub = self.take(region, size)
            forward = self.route(insn.addr, stub)
            back = self.route(stub + size - 4, insn.end) if forward is not None else None
            if back is not None:
                return Placement(insn, kind, marker_id, stub, words, forward, back)
            self.rollback(mark)
        return None


# --- verification ----------------------------------------------------------------


def verify(
    target: Target,
    original: bytes,
    out: bytes,
    placements: list[Placement],
    clauses: list[Insn],
    islands: list[Region],
    pool: Region,
    bumped: dict[str, int],
    pc_fixes: list[tuple[int, int]],
    relocations: list[Relocation],
    moved_entries: dict[str, int],
    planned: set[int],
    relayouts: list[Relayout],
    lit_checks: list[tuple[int, int | None, int | None, int, int]],
    ref_checks: list[tuple["PcRef", int]],
) -> None:
    """Check the rewritten bytes against the plan; raise VerifyError on any mismatch."""

    def fail(msg: str):
        raise VerifyError(msg)

    old, new = Elf64(original), Elf64(out)
    _, old_text = old.section(".text")
    _, new_text = new.section(".text")
    t_addr, t_off, t_size = old_text[3], old_text[4], old_text[5]
    if new_text[3] != t_addr or new_text[4] != t_off:
        fail(".text moved")
    new_size = new_text[5]
    if new_size < t_size:
        fail(".text shrank")

    def word(blob: bytes, addr: int) -> int:
        if not t_addr <= addr < t_addr + new_size - 3:
            fail(f"address {addr:#x} outside .text")
        return struct.unpack_from("<I", blob, t_off + addr - t_addr)[0]

    # 1. Only planned words of the original .text changed, and island words were padding.
    allowed = {}
    for p in placements:
        for k in range(len(p.insn.words)):
            allowed[p.insn.addr + 4 * k] = "site"
    for c in clauses:
        allowed[c.addr] = "clause"
    for a, w in pc_fixes:
        allowed[a] = "getpc"
        if word(out, a) != w:
            fail(f"PC-relative literal at {a:#x} not adjusted as planned")
    island_ranges = [(r.start, r.cursor) for r in islands if r.cursor > r.start]
    old_bytes = original[t_off : t_off + t_size]
    new_bytes = out[t_off : t_off + t_size]
    for blk in range(0, t_size, 4096):
        if old_bytes[blk : blk + 4096] == new_bytes[blk : blk + 4096]:
            continue  # unchanged block
        for a in range(t_addr + blk, min(t_addr + blk + 4096, t_addr + t_size), 4):
            before, after = word(original, a), word(out, a)
            if before == after:
                continue
            why = allowed.get(a)
            if why is None:
                if not any(s <= a < e for s, e in island_ranges):
                    fail(f"unplanned change at {a:#x}")
                if before not in target.padding:
                    fail(f"island word at {a:#x} was not padding")
    for s, e in island_ranges:
        if s >= t_addr + t_size:
            continue  # stub space carved from the pool, not original padding
        for a in range(s, e, 4):
            if word(original, a) not in target.padding:
                fail(f"island {s:#x} covers non-padding at {a:#x}")

    # 2. Each site's branch chain reaches its stub and comes back.
    scratch = set(range(pool.start, pool.cursor, 4))
    for s, e in island_ranges:
        scratch.update(range(s, e, 4))

    def follow(pc: int, stop: int, what: str) -> None:
        for _ in range(MAX_HOPS + 2):
            w = word(out, pc)
            if not target.is_branch(w):
                fail(f"{what}: no branch at {pc:#x}")
            pc = branch_target(pc, w)
            if pc == stop:
                return
            if pc not in scratch:
                fail(f"{what}: hop {pc:#x} outside islands and pool")
        fail(f"{what}: chain too long")

    seen_ids = set()
    for p in placements:
        what = f"site {p.insn.addr:#x}"
        if p.marker_id in seen_ids:
            fail(f"{what}: duplicate marker id {p.marker_id}")
        seen_ids.add(p.marker_id)
        follow(p.insn.addr, p.stub, what)
        for k in range(1, len(p.insn.words)):
            if word(out, p.insn.addr + 4 * k) != target.s_nop0:
                fail(f"{what}: padding after the branch is not s_nop 0")
        got = [word(out, p.stub + 4 * k) for k in range(len(p.stub_words))]
        if got != p.stub_words or got[-len(p.insn.words) :] != p.insn.words:
            fail(f"{what}: stub at {p.stub:#x} does not hold marker + original instruction")
        if p.stub not in scratch:
            fail(f"{what}: stub outside islands and pool")
        follow(p.back_branch, p.insn.end, what + " (return)")
    for c in clauses:
        if word(out, c.addr) != target.s_nop0:
            fail(f"clause at {c.addr:#x} not dissolved")

    # 2b. Kernel copies equal the original code except at planned words.
    for r in relocations:
        if r.new % 256 or not (pool.start <= r.new and r.new + r.size <= pool.cursor):
            fail(f"kernel copy at {r.new:#x} misplaced")
        for off in range(0, r.size, 4):
            if r.new + off not in planned and word(out, r.new + off) != word(original, r.start + off):
                fail(f"kernel copy differs from the original at {r.new + off:#x}")

    # 2c. Relaid-out kernels: the original instructions in their original order
    # with only the planned markers between them; every branch lands on the new
    # place of its original target; all other words are unchanged except the
    # re-aimed PC-relative literals (whose targets tool_check decodes).
    for rl in relayouts:
        if rl.new % 256 or not (pool.start <= rl.new and rl.new + rl.size <= pool.cursor):
            fail(f"relaid-out kernel at {rl.new:#x} misplaced")
        olds = sorted(rl.new_at)
        if not olds or olds[0] != rl.start:
            fail(f"relaid-out kernel at {rl.new:#x} does not start at its entry")
        markers_at = {m[3]: m for m in rl.markers}
        pos, group, seen = rl.new, {}, 0
        islands_left = dict(rl.island_words)
        for k, a in enumerate(olds):
            size = (olds[k + 1] if k + 1 < len(olds) else rl.end) - a
            while pos in islands_left:
                iw = islands_left.pop(pos)
                if [word(out, pos + 4 * j) for j in range(len(iw))] != iw:
                    fail(f"branch island at {pos:#x} has the wrong words")
                if len(iw) == 2 and branch_target(pos, iw[0]) != pos + 8:
                    fail(f"branch island at {pos:#x} does not skip itself")
                pos += 4 * len(iw)
            group[a] = pos
            while pos in markers_at:
                m = markers_at.pop(pos)
                if [word(out, pos + 4 * j) for j in range(len(m[4]))] != m[4]:
                    fail(f"marker {m[0]} at {pos:#x} has the wrong words")
                if not 0 <= m[2] - a <= 1024:
                    fail(f"marker {m[0]} is not just before its site {m[2]:#x}")
                pos += 4 * len(m[4])
                seen += 1
            if rl.new_at[a] != pos:
                fail(f"relaid-out instruction {a:#x} is not where planned")
            for off in range(0, size, 4):
                o, n = word(original, a + off), word(out, pos + off)
                if off == 0 and a in rl.branches:
                    if (o ^ n) & 0xFFFF0000:
                        fail(f"branch at {pos:#x} changed opcode")
                elif o != n and pos + off not in rl.literals and pos + off not in planned:
                    fail(f"relaid-out word at {pos + off:#x} differs from the original at {a + off:#x}")
            pos += size
            if a in rl.pins:
                # A pinned s_getpc: the words after it subtract exactly the distance moved.
                pin, li = rl.pins[a]
                if [word(out, pos + 4 * j) for j in range(len(pin))] != pin:
                    fail(f"pin after s_getpc at {a:#x} has the wrong words")
                if pin[li] != (rl.new_at[a] - a) & 0xFFFFFFFF:
                    fail(f"pin after s_getpc at {a:#x} subtracts the wrong distance")
                pos += 4 * len(pin)
        if pos != rl.new + rl.size or markers_at or islands_left or seen != len(rl.markers):
            fail(f"relaid-out kernel at {rl.new:#x}: layout, markers and islands disagree")

        def landing(at: int) -> int:
            """Where a branch at this address ends up, following branch islands."""
            dest = branch_target(at, word(out, at))
            for _ in range(64):
                if dest not in rl.island_branch:
                    return dest
                dest = branch_target(dest, word(out, dest))
            fail(f"branch island chain from {at:#x} too long")

        for at, dest in rl.island_branch.items():
            if not target.is_branch(word(out, at)) or branch_target(at, word(out, at)) != dest:
                fail(f"branch island at {at:#x} is not the planned s_branch")
        for a in rl.branches:
            t = branch_target(a, word(original, a))
            want = group.get(t) if rl.start <= t < rl.end else t
            if landing(rl.new_at[a]) != want:
                fail(f"relaid-out branch from {a:#x} lands at the wrong place")

    # 2d. Every PC-relative add (original code, copies, relayouts), read back
    # from the output bytes, points where it should.
    # Recognised s_getpc patterns in the original code, decoded from the output bytes.
    for ref, want in ref_checks:
        if ref.lits is not None:
            value = word(out, ref.lits[1]) << 32 | word(out, ref.lits[0])
            offset = value - (1 << 64) if value >> 63 else value
        elif ref.lit32 is not None:
            addr, other, backward = ref.lit32
            v = sext32(word(out, addr) + other)
            offset = -abs(v) if backward else v
        else:
            continue  # not adjustable: its words are unchanged (checked in step 1)
        if ref.base + offset != want:
            fail(f"s_getpc at {ref.getpc:#x} points at {ref.base + offset:#x}, planned {want:#x}")
    for lo_a, hi_a, hi_inline, base, want in lit_checks:
        hi = word(out, hi_a) if hi_a is not None else (hi_inline or 0) & 0xFFFFFFFF
        value = hi << 32 | word(out, lo_a)
        offset = value - (1 << 64) if value >> 63 else value
        if base + offset != want:
            fail(f"PC-relative add with literal at {lo_a:#x} points at {base + offset:#x}, planned {want:#x}")

    # 3. Pool guard and segment/section consistency.
    if pool.cursor > pool.start:
        guard_start = pool.cursor
        guard_end = t_addr + new_size
        if guard_end - guard_start < PREFETCH_GUARD:
            fail("prefetch guard too small")
        for a in range(guard_start, guard_end, 4):
            if word(out, a) != target.guard:
                fail(f"prefetch guard word at {a:#x}")
    loads = sorted((ph for _, ph in new.phdrs() if ph[0] == PT_LOAD), key=lambda ph: ph[3])
    for ph in loads:
        if ph[2] + ph[5] > len(out):
            fail("segment past end of file")
        if ph[7] >= PAGE and (ph[2] - ph[3]) % PAGE:
            fail("segment offset and address disagree modulo the page size")
        if ph[5] > ph[6]:
            fail("segment file size above memory size")
    for a, b in zip(loads, loads[1:]):
        if a[3] + a[6] > b[3]:
            fail("segments overlap")
    for _, sh in new.shdrs():
        if not sh[2] & SHF_ALLOC or sh[1] == SHT_NOBITS or not sh[5]:
            continue
        if not any(ph[3] <= sh[3] and sh[3] + sh[5] <= ph[3] + ph[5] and
                   ph[2] + sh[3] - ph[3] == sh[4] for ph in loads):
            fail(f"section {new.section_name(sh)} not mapped by its segment")

    # 4. Kernel descriptors: same entry points (or exactly the planned copy);
    # only the entry offset and the SGPR field may change.
    old_kds, new_kds = old.kernel_descriptors(), new.kernel_descriptors()
    if set(old_kds) != set(new_kds):
        fail("kernel descriptor set changed")
    for name, kd in old_kds.items():
        ob = original[old.file_offset(kd) : old.file_offset(kd) + 64]
        nb = out[new.file_offset(new_kds[name]) : new.file_offset(new_kds[name]) + 64]
        o_entry = kd + struct.unpack_from("<q", ob, 16)[0]
        n_entry = new_kds[name] + struct.unpack_from("<q", nb, 16)[0]
        if n_entry != moved_entries.get(name, o_entry):
            fail(f"kernel {name}: entry point is {n_entry:#x}, planned {moved_entries.get(name, o_entry):#x}")
        if n_entry % 256:
            fail(f"kernel {name}: entry point not 256-byte aligned")
        o_rsrc1, n_rsrc1 = struct.unpack_from("<I", ob, 48)[0], struct.unpack_from("<I", nb, 48)[0]
        rsrc1 = o_rsrc1
        if name in bumped:
            rsrc1 = (o_rsrc1 & ~(0xF << 6)) | (bumped[name] << 6)
            if n_rsrc1 != rsrc1:
                fail(f"kernel {name}: SGPR granules not raised as planned")
        expected = (
            ob[:16] + struct.pack("<q", n_entry - new_kds[name]) + ob[24:48]
            + struct.pack("<I", rsrc1) + ob[52:]
        )
        if nb != expected:
            fail(f"kernel {name}: descriptor changed")


def tool_check(
    llvm: str,
    original: str,
    rewritten: str,
    placements: list[Placement],
    expected_pc: dict[int, int],
    expected_mnemonics: dict[int, str],
    new_code: int,
) -> None:
    """llvm-readelf and llvm-objdump must read the output without new complaints.

    Also decodes the new code (from new_code, the end of the original .text,
    onwards) and checks that every recognised s_getpc in it (kernel copies and
    relaid-out kernels) points where it should, that relaid-out code decodes to
    the planned instructions, and that stubs hold their original instruction.
    The original code's patched words are checked byte by byte in verify().
    """

    def complaints(path: str) -> set[str]:
        res = run([f"{llvm}/llvm-readelf", "-a", "--dyn-syms", path])
        lines = res.stderr.replace(path, "<file>").splitlines()
        return {l for l in lines if "warning" in l.lower() or "error" in l.lower()}

    extra = complaints(rewritten) - complaints(original)
    if extra:
        raise VerifyError("llvm-readelf: " + "; ".join(sorted(extra))[:500])
    try:
        out_insns, _ = disassemble(llvm, rewritten, start=new_code)
    except subprocess.CalledProcessError as e:
        raise VerifyError("llvm-objdump failed: " + (e.stderr or "")[:300])
    by_addr = {i.addr: i.mnemonic for i in out_insns}
    out_refs, _ = pc_references(out_insns)
    by_base = {r.base: r.target for r in out_refs}
    for base, want in expected_pc.items():
        if base >= new_code and by_base.get(base) != want:
            raise VerifyError(f"objdump: s_getpc at {base - 4:#x} points at the wrong address")
    for addr, mnemonic in expected_mnemonics.items():
        if by_addr.get(addr) != mnemonic:
            raise VerifyError(f"objdump: {by_addr.get(addr)} at {addr:#x}, planned {mnemonic}")
    for p in placements:
        if p.stub < new_code:
            continue  # a stub in padding inside the original .text: checked in verify()
        if by_addr.get(p.back_branch) != "s_branch":
            raise VerifyError(f"objdump: stub at {p.stub:#x} does not end in s_branch")
        if by_addr.get(p.stub + 4 * (len(p.stub_words) - len(p.insn.words))) != p.insn.mnemonic:
            raise VerifyError(f"objdump: stub at {p.stub:#x} lost {p.insn.mnemonic}")


# --- the rewrite ---------------------------------------------------------------


def rewrite(args) -> int:
    with open(args.input, "rb") as f:
        data = f.read()
    elf = Elf64(data)
    if struct.unpack_from("<H", data, 18)[0] != 224:
        raise Unchanged("not an AMDGPU code object")
    _, text = elf.section(".text")
    text_addr, text_fileoff, text_size = text[3], text[4], text[5]
    text_end = text_addr + text_size

    # The executable PT_LOAD that holds .text, and the next segment's start.
    loads = [(off, ph) for off, ph in elf.phdrs() if ph[0] == PT_LOAD]
    execs = [(o, p) for o, p in loads if p[1] & PF_X]
    if len(execs) != 1:
        raise Unchanged(f"{len(execs)} executable segments; expected 1")
    exec_off, exec_ph = execs[0]
    if exec_ph[3] + exec_ph[5] != text_end or exec_ph[5] != exec_ph[6]:
        raise Unchanged(".text is not at the end of its segment; not supported")
    later = [p[3] for _, p in loads if p[3] > exec_ph[3]]
    limit = min(later) if later else text_end + 0x10000
    names_by_index = [elf.section_name(sh) for _, sh in elf.shdrs()]
    later_sections = {
        name
        for (_, sh), name in zip(elf.shdrs(), names_by_index)
        if sh[2] & SHF_ALLOC and sh[3] >= limit
    }

    # Kernels, and the code each one owns: from its entry to the next entry.
    kds = elf.kernel_descriptors()
    entries = {}  # kernel name -> (descriptor file offset, entry address)
    for name, kd in kds.items():
        off = elf.file_offset(kd)
        if off is None:
            raise Unchanged(f"kernel descriptor {name}.kd is not in a segment")
        entries[name] = (off, kd + struct.unpack_from("<q", data, off + 16)[0])
    entry_list = sorted({e for _, e in entries.values() if text_addr <= e < text_end})
    kernel_at = {e: name for name, (_, e) in entries.items()}

    def kernel_range(addr: int) -> tuple[int, int] | None:
        k = bisect.bisect_right(entry_list, addr) - 1
        if k < 0:
            return None
        return entry_list[k], entry_list[k + 1] if k + 1 < len(entry_list) else text_end

    def kernel_name(addr: int) -> str:
        rng = kernel_range(addr)
        return kernel_at.get(rng[0], "") if rng else ""

    # With --kernel, skip objects with no matching kernel or function before
    # the (slow) disassembly of large libraries.
    if args.kernel and not any(args.kernel.search(n) for n, *_ in elf.symbols()):
        raise Unchanged("no kernel matches --kernel")

    target = probe_target(args.llvm, target_cpu(args.llvm, args.input))
    insns, labels = disassemble(args.llvm, args.input)
    # The decoded instructions must tile .text exactly; otherwise there is data
    # or something objdump cannot decode, and a site could be misread.
    pos = text_addr
    for insn in insns:
        if insn.addr != pos or insn.mnemonic.startswith("<unknown>"):
            raise Unchanged(f"cannot decode .text cleanly at {pos:#x}")
        pos = insn.end
    if pos != text_end:
        raise Unchanged(f"cannot decode .text cleanly at {pos:#x}")
    kernels = list(dict.fromkeys(labels[a] for a in sorted(labels)))

    # Which code actually runs: every path from kernel entries and function
    # symbols, following computed jumps where their targets can be worked out.
    # Only reachable instructions become sites, so data stored in .text that
    # happens to decode as an LDS instruction is never patched.
    pc_refs, pc_unknown = pc_references(insns)
    roots = set(entry_list) | {
        value for _, value, _, info, shndx in elf.symbols()
        if info & 0xF == 2 and shndx not in (SHN_UNDEF, SHN_ABS) and text_addr <= value < text_end
    }
    flow = pc_flow(insns, roots, pc_refs)
    unreached = 0

    owner = clause_owner(insns) if target.has_clause else {}
    sites = []
    clause_of = {}  # site address -> index of the s_clause covering it
    for i, insn in enumerate(insns):
        kind = classify(insn, args.sites)
        if kind is None:
            continue
        if i not in flow.reachable:
            unreached += 1
            print(f"skip {insn.addr:#x} {insn.text}: not reached from any entry", file=sys.stderr)
            continue
        # Match the kernel that owns the code (from the descriptors), or the
        # function label; hand-written code labels internal blocks too.
        if args.kernel and not (
            args.kernel.search(kernel_name(insn.addr)) or args.kernel.search(insn.function)
        ):
            continue
        if i in owner:
            if args.keep_clauses:
                print(f"skip {insn.addr:#x} {insn.text}: inside s_clause", file=sys.stderr)
                continue
            # A branch out of a clause would break it, so the clause is turned
            # into s_nop 0. On gfx10+ clauses only group memory instructions
            # for issue; dropping one costs performance, not results.
            clause_of[insn.addr] = owner[i]
        if insn.size < 8:
            # 4-byte sites would need the s_branch alone; not needed for DS ops.
            print(f"skip {insn.addr:#x} {insn.text}: 4-byte instruction", file=sys.stderr)
            continue
        sites.append((insn, kind))
    if not sites:
        raise Unchanged("no sites to mark")

    index_of = {i.addr: k for k, i in enumerate(insns)}

    # Marker IDs: one-instruction markers (IDs 1..MAX_IMM_ID, gfx10+) go to the
    # sites in the most deeply nested loops, where marker cost matters most.
    depth = loop_depths(insns, flow, [index_of[s[0].addr] for s in sites])
    order = sorted(range(len(sites)), key=lambda j: (-depth[j], sites[j][0].addr))
    site_id = {sites[j][0].addr: rank + 1 for rank, j in enumerate(order)}

    # The SGPR that saves M0 (and SCC for pinned s_getpc): one above every SGPR
    # the code names if there is one; otherwise, per site, an SGPR the kernel
    # names but that is dead at the site (register liveness over all paths).
    spare = free_sgpr(insns, target.max_sgpr)
    live_in: list[int] | None = None
    kernel_named: dict = {}

    def free_at(k: int, exclude: int = 0) -> int | None:
        """A spare SGPR usable just before instruction k, or None."""
        nonlocal live_in
        if spare is not None:
            return spare
        if live_in is None:
            live_in = sgpr_liveness(insns, flow)
        rng = kernel_range(insns[k].addr)
        if rng not in kernel_named:
            lo, hi = (index_of[rng[0]], bisect.bisect_left(insn_addrs, rng[1])) if rng else (k, k + 1)
            named = 0
            for j in range(lo, hi):
                d, u = sgpr_def_use(insns[j])
                named |= d | u
            kernel_named[rng] = named & ((1 << (target.max_sgpr + 1)) - 1)
        dead = kernel_named[rng] & ~live_in[k] & ~exclude
        return dead.bit_length() - 1 if dead else None

    m0_templates: dict[int, list[list[int]]] = {}

    def m0_template(reg: int) -> list[list[int]] | None:
        if reg not in m0_templates:
            enc = mc_encode(
                args.llvm,
                target.cpu,
                [f"s_mov_b32 s{reg}, m0", "s_mov_b32 m0, 0x12345678", f"s_nop {target.m0_wait}",
                 f"s_mov_b32 m0, s{reg}"],
            )
            if len(enc) != 4:
                raise Unchanged("llvm-mc could not encode the M0 marker")
            m0_templates[reg] = list(enc.values())
        return m0_templates[reg]

    def uses_m0(marker_id: int) -> bool:
        return target.s_ttracedata_imm is None or marker_id > MAX_IMM_ID

    def marker_words(marker_id: int, site: int) -> list[int] | None:
        """The marker for a site (original address), or None if it needs M0
        and no SGPR is free there."""
        encoded = marker_id << 2
        if not uses_m0(marker_id):
            return [target.s_ttracedata_imm | encoded]
        reg = free_at(index_of[site])
        if reg is None:
            return None
        save, set_m0, nop, restore = m0_template(reg)
        # Waits after each M0 write keep s_ttracedata and later M0 readers safe.
        return save + [set_m0[0], encoded] + nop + [target.s_ttracedata] + nop + restore + nop

    # Where stubs can go: islands, then the pool after .text. The pool is
    # bounded by the page slack unless later segments may move: when code
    # cannot refer to them, or every s_getpc is recognised and the ones that
    # point into later segments have literals that are adjusted below.
    stub_base = (text_end + 15) & ~15
    insn_addrs = [i.addr for i in insns]

    # PC-relative addresses the flow scan found outside the recognised
    # patterns: s_getpc + a literal add (e.g. hand-written jump tables and
    # tables of constants stored in .text).
    def la_target(la: LitAdd) -> int:
        return la.getpc + 4 + la.offset

    def la_encode(la: LitAdd, offset: int) -> list[tuple[int, int]] | None:
        """(address, word) writes giving this add a new offset, or None."""
        lo_i, hi_i = insns[la.lo], insns[la.hi]
        lo_v, hi_v = offset & 0xFFFFFFFF, (offset >> 32) & 0xFFFFFFFF
        if la.hi_literal:
            return [(lo_i.addr + 4, lo_v), (hi_i.addr + 4, hi_v)]
        # Inline high part (0 or -1): only offsets of the same sign fit.
        return [(lo_i.addr + 4, lo_v)] if constant(operands(hi_i)[2]) & 0xFFFFFFFF == hi_v else None

    pc_adjustable = (
        not flow.escapes
        and all(r.adjustable or r.target < limit for r in pc_refs)
        and all(la_target(la) < limit or la.hi_literal or la.offset >= 0 for la in flow.lit_adds.values())
    )
    movable = later_sections <= MOVABLE_SECTIONS or pc_adjustable
    if not movable:
        print("PC-derived values used in ways not followed: later segments stay put", file=sys.stderr)
    capacity = 1 << 40 if movable else max(0, limit - stub_base - PREFETCH_GUARD)

    island_spans = [] if args.no_islands else [(r.start, r.end) for r in find_islands(target, insns, labels)]

    def new_placer() -> Placer:
        return Placer([Region(s, e) for s, e in island_spans], Region(stub_base, stub_base + capacity, pool=True))

    def relocation_blocker(start: int, end: int) -> str | None:
        """Why a kernel's code cannot run from new code, or None if it can."""
        lo, hi = bisect.bisect_left(insn_addrs, start), bisect.bisect_left(insn_addrs, end)
        for k in range(lo, hi):
            insn = insns[k]
            if k in flow.reachable and DIRECT_BRANCH.match(insn.mnemonic):
                t = branch_target(insn.addr, insn.words[0])
                if not start <= t < end:
                    return f"branch at {insn.addr:#x} leaves the kernel"
        for r in pc_refs:
            if start <= r.getpc < end and not start <= r.target < end and not r.adjustable:
                return f"PC-relative reference at {r.getpc:#x} leaves the kernel"
        for la in flow.lit_adds.values():
            inside_add = start <= insns[la.lo].addr < end
            if inside_add != (start <= la.getpc < end):
                return f"PC-relative add at {insns[la.lo].addr:#x} crosses the kernel boundary"
            if inside_add and not start <= la_target(la) < end and not la.hi_literal:
                return f"PC-relative add at {insns[la.lo].addr:#x} leaves the kernel"
        return None

    pin_template: dict[tuple[int, int], tuple[list[int], int]] = {}

    def pin_words(getpc: int) -> tuple[list[int], int] | None:
        """Words that turn the PC written by the s_getpc at this address back
        into the original code's PC (subtract a literal, filled in per site),
        keeping SCC, and the index of the literal word. A free SGPR saves SCC."""
        k = index_of[getpc]
        a = pair_of(operands(insns[k])[0])
        if a is None or k + 1 >= len(insns):
            return None
        reg = free_at(k + 1, exclude=sgpr_mask({a, a + 1}))
        if reg is None:
            return None
        if (a, reg) not in pin_template:
            wait = re.fullmatch(r"gfx1[12]\d\w*", target.cpu) is not None
            lines = [
                f"s_cselect_b32 s{reg}, 1, 0",
                f"s_sub_u32 s{a}, s{a}, 0x12345678",
                f"s_subb_u32 s{a + 1}, s{a + 1}, 0",
                f"s_cmp_lg_u32 s{reg}, 0",
            ] + (["s_waitcnt_depctr depctr_sa_sdst(0)"] if wait else [])
            enc = mc_encode(args.llvm, target.cpu, lines)
            if len(enc) != len(lines):
                return None
            cselect, sub, subb, cmp_ = (list(enc.values())[j] for j in range(4))
            waits = list(enc.values())[4] if wait else []
            # SGPR writes are waited for before anything reads them (gfx11+).
            words = cselect + waits + sub + subb + cmp_ + waits
            pin_template[(a, reg)] = (words, len(cselect + waits) + 1)
        return pin_template[(a, reg)]

    def build_relayout(start: int, end: int, base: int, todo):
        """The kernel [start, end) laid out at base with each site's marker
        inline, or (None, reason). Sites inside an s_clause get their marker
        before the clause, so the clause stays intact.

        An s_getpc whose value is used in ways the flow scan does not follow
        (e.g. as the base of a table indexed at run time) is pinned: words after
        it subtract the distance the code moved, so everything derived from it
        still points into the original code, which stays intact."""
        lo, hi = bisect.bisect_left(insn_addrs, start), bisect.bisect_left(insn_addrs, end)
        body = insns[lo:hi]
        live = {insns[k].addr for k in range(lo, hi) if k in flow.reachable}
        for insn in body:
            if insn.addr not in live:
                continue  # never runs (data, padding): copied as it is
            if PC_TOUCHING.search(insn.mnemonic) and insn.mnemonic not in PC_RELAYOUT_OK:
                return None, f"{insn.mnemonic} at {insn.addr:#x}"
        pins: dict[int, tuple[list[int], int]] = {}
        for g in sorted(g for g in flow.escapes if start <= g < end):
            template = pin_words(g)
            if template is None:
                return None, f"s_getpc at {g:#x} cannot be pinned (no free SGPR)"
            pins[g] = template
        before: dict[int, list] = {}
        for insn, kind in todo:
            marker_id = site_id[insn.addr]
            marker = marker_words(marker_id, insn.addr)
            if marker is None:
                continue  # no free SGPR here: the caller reports it as skipped
            anchor = insns[clause_of[insn.addr]].addr if insn.addr in clause_of else insn.addr
            before.setdefault(anchor, []).append((marker_id, kind, insn.addr, marker))
        # Layout. A branch whose target ends up out of s_branch reach goes
        # through an island: one s_branch placed between live instructions part
        # of the way there (plus an s_branch over it if code can fall into it).
        # Islands are added until every branch reaches; they need no registers
        # and leave SCC alone.
        live_branches = [i for i in body if i.addr in live and DIRECT_BRANCH.match(i.mnemonic)]
        body_pos = {insn.addr: j for j, insn in enumerate(body)}
        anchors_ok = [
            insn.addr for j, insn in enumerate(body)
            if insn.addr in live and (lo + j) not in owner  # never inside an s_clause group
        ]
        islands: list[dict] = []  # {"anchor": addr, "dest": ("insn", addr) | ("isl", i), "skip": bool}
        at_anchor: dict[int, list[int]] = {}
        redirect: dict[int, tuple[str, int]] = {}  # branch address -> ("isl", i)
        for _ in range(64):
            group, new_at, isl_at, pos = {}, {}, {}, base
            for insn in body:
                for i in at_anchor.get(insn.addr, []):
                    isl_at[i] = pos + (4 if islands[i]["skip"] else 0)  # its branch to dest
                    pos += 8 if islands[i]["skip"] else 4
                group[insn.addr] = pos
                pos += sum(4 * len(m[3]) for m in before.get(insn.addr, []))
                new_at[insn.addr] = pos
                pos += insn.size + 4 * len(pins[insn.addr][0] if insn.addr in pins else [])

            def resolve(dest: tuple[str, int]) -> int | None:
                return group.get(dest[1]) if dest[0] == "insn" else isl_at[dest[1]]

            far = []
            for insn in live_branches:
                t = branch_target(insn.addr, insn.words[0])
                if not start <= t < end:
                    if not reach(new_at[insn.addr], t):
                        return None, f"branch at {insn.addr:#x} to {t:#x} out of reach"
                    continue
                if group.get(t) is None:
                    return None, f"branch at {insn.addr:#x} lands inside an instruction"
                dest = redirect.get(insn.addr, ("insn", t))
                if not reach(new_at[insn.addr], resolve(dest)):
                    far.append((("br", insn.addr), new_at[insn.addr], dest))
            for i, isl in enumerate(islands):
                if not reach(isl_at[i], resolve(isl["dest"])):
                    far.append((("isl", i), isl_at[i], isl["dest"]))
            if not far:
                break
            by_pos = sorted((group[a], a) for a in anchors_ok)
            positions = [p for p, _ in by_pos]
            for who, frm, dest in far:
                to = resolve(dest)
                goal = frm + max(-0x18000, min(0x18000, to - frm))  # within reach, toward the target
                j = bisect.bisect_left(positions, goal)
                choices = [by_pos[x] for x in range(max(0, j - 64), min(len(by_pos), j + 64))]
                choices = [c for c in choices if reach(frm, c[0]) and abs(c[0] - goal) < 0x4000]
                if not choices:
                    return None, f"no place for a branch island near {goal:#x}"

                def falls_in(anchor: int) -> bool:
                    k = body_pos[anchor]
                    prev = body[k - 1] if k else None
                    return prev is not None and prev.addr in live and not TERMINATOR_MNEMONICS.match(prev.mnemonic)

                # Prefer a place nothing falls into (after s_branch, s_endpgm, ...).
                choices.sort(key=lambda c: (falls_in(c[1]), abs(c[0] - goal)))
                anchor = choices[0][1]
                i = len(islands)
                islands.append({"anchor": anchor, "dest": dest, "skip": falls_in(anchor)})
                at_anchor.setdefault(anchor, []).append(i)
                if who[0] == "br":
                    redirect[who[1]] = ("isl", i)
                else:
                    islands[who[1]]["dest"] = ("isl", i)
        else:
            return None, "branches still out of reach after adding islands"

        words: list[int] = []
        pinned: dict[int, tuple[list[int], int]] = {}
        island_words: dict[int, list[int]] = {}  # island start -> its words
        island_branch: dict[int, int] = {}  # island branch address -> where it jumps
        markers, branches = [], set()
        for insn in body:
            for i in at_anchor.get(insn.addr, []):
                here = base + 4 * len(words)
                to = resolve(islands[i]["dest"])
                iw = ([target.branch(here, here + 8)] if islands[i]["skip"] else [])
                iw.append(target.branch(isl_at[i], to))
                island_words[here] = iw
                island_branch[isl_at[i]] = to
                words += iw
            for m_id, kind, site, marker in before.get(insn.addr, []):
                markers.append((m_id, kind, site, base + 4 * len(words), marker))
                words += marker
            w = list(insn.words)
            if insn.addr in live and DIRECT_BRANCH.match(insn.mnemonic):
                t = branch_target(insn.addr, w[0])
                at = new_at[insn.addr]
                nt = resolve(redirect.get(insn.addr, ("insn", t))) if start <= t < end else t
                w[0] = (w[0] & 0xFFFF0000) | (((nt - at - 4) // 4) & 0xFFFF)
                branches.add(insn.addr)
            words += w
            if insn.addr in pins:
                pin, li = pins[insn.addr]
                pin = list(pin)
                pin[li] = (new_at[insn.addr] - insn.addr) & 0xFFFFFFFF  # distance moved
                pinned[insn.addr] = (pin, li)
                words += pin
        # PC-relative references: inside the kernel they are re-aimed now; those
        # leaving it are fixed once the final segment shift is known.
        refs, lit_refs, literals = [], [], set()

        def apply(writes) -> None:
            for a, value in writes:
                na = new_at[a - 4] + 4  # a literal is the second word of its instruction
                words[(na - base) // 4] = value
                literals.add(na)

        for r in pc_refs:
            if not start <= r.getpc < end or r.getpc not in live:
                continue
            nbase = new_at[r.getpc] + 4
            if not start <= r.target < end:
                if not r.adjustable:
                    return None, f"s_getpc at {r.getpc:#x} leaves the kernel"
                refs.append((r, nbase, None))
                continue
            nt = group.get(r.target)
            writes = None if nt is None else r.encode(nt - nbase)
            if writes is None:
                return None, f"s_getpc at {r.getpc:#x} cannot be re-aimed"
            apply(writes)
            refs.append((r, nbase, nt))
        for la in flow.lit_adds.values():
            if not start <= insns[la.lo].addr < end:
                continue
            if not start <= la.getpc < end:
                return None, f"PC-relative add at {insns[la.lo].addr:#x} uses another kernel's PC"
            if la.getpc in pins:
                # Pinned: the base is the original PC, so the add keeps pointing
                # into the original code (adjusted only if its segment moves).
                lit_refs.append((la, la.getpc + 4, None))
                continue
            nbase = new_at[la.getpc] + 4
            t = la_target(la)
            if not start <= t < end:
                lit_refs.append((la, nbase, None))  # re-aimed once the segment shift is known
                continue
            nt = group.get(t)
            writes = None if nt is None else la_encode(la, nt - nbase)
            if writes is None:
                return None, f"PC-relative add at {insns[la.lo].addr:#x} cannot be re-aimed"
            apply(writes)
            lit_refs.append((la, nbase, nt))
        rl = Relayout(start, end, base, words, group, new_at, markers, refs, branches, literals)
        rl.lit_refs = lit_refs
        rl.pins = pinned
        rl.island_words = island_words
        rl.island_branch = island_branch
        return rl, None

    def stub_bytes(insn: Insn) -> int:
        marker = marker_words(site_id[insn.addr], insn.addr)
        return 0 if marker is None else 4 * len(marker) + insn.size + 4

    relayouts: list[Relayout] = []
    inline_count = [0]  # sites marked inline in relayouts

    def try_sites(placer: Placer, todo, placements: list, skipped: list, delta: int = 0) -> None:
        for insn, kind in todo:
            marker_id = site_id[insn.addr]
            marker = marker_words(marker_id, insn.addr)
            if marker is None:
                skipped.append((insn, "no free SGPR for its M0 marker"))
                continue
            moved = Insn(insn.addr + delta, insn.words, insn.text, insn.function)
            p = placer.place(moved, kind, marker_id, marker + insn.words)
            if p is None:
                skipped.append((insn, "no reachable room for a stub"))
                continue
            p.orig = insn.addr
            placements.append(p)

    # Pass 1: every site in place (stubs in islands or the pool, hops if needed).
    placer = new_placer()
    placements: list[Placement] = []
    skipped: list[tuple[Insn, str]] = []
    try_sites(placer, sites, placements, skipped)

    # Pass 2 (if pass 1 left sites out, or --relayout / --relocate): such a
    # kernel runs from new code in the pool, and its descriptor points there.
    # Preferred: a relayout with the markers inline (no detours). Otherwise an
    # unchanged copy marked with stubs. The original code stays intact either
    # way, so anything that still reaches it runs correct, unmarked code.
    relocations: list[Relocation] = []
    to_move = []
    if not args.no_relocate:
        if args.relocate or args.relayout:  # every kernel with a site
            wanted = {kernel_range(i.addr) for i, _ in sites}
        else:
            wanted = {kernel_range(i.addr) for i, why in skipped if why.startswith("no reachable")}
        for rng in sorted(wanted - {None}):
            reason = relocation_blocker(*rng)
            if reason:
                print(f"kernel at {rng[0]:#x} stays in place: {reason}", file=sys.stderr)
            else:
                to_move.append(rng)
    if to_move:
        placer = new_placer()
        placements, skipped = [], []
        moving = set(to_move)
        try_sites(placer, [s for s in sites if kernel_range(s[0].addr) not in moving], placements, skipped)
        for start, end in to_move:
            todo = [s for s in sites if start <= s[0].addr < end]
            pool = placer.pool
            if not args.relocate:
                pad = (-pool.cursor) % 256  # kernel entries are 256-byte aligned
                rl, reason = build_relayout(start, end, pool.cursor + pad, todo)
                if rl is not None and pool.free >= pad + rl.size + PREFETCH_GUARD:
                    placer.take(pool, pad)
                    placer.take(pool, rl.size)
                    relayouts.append(rl)
                    inline_count[0] += len(rl.markers)
                    marked = {m[2] for m in rl.markers}
                    skipped += [(i, "no free SGPR for its M0 marker") for i, _ in todo if i.addr not in marked]
                    continue
                print(f"kernel at {start:#x} not relaid out: {reason or 'no room'}", file=sys.stderr)
            # An unchanged copy cannot pin an s_getpc, so values used in ways not
            # followed would point into the copy instead of the original.
            if any(start <= g < end for g in flow.escapes):
                print(f"kernel at {start:#x} not copied: a PC-derived value is used in a way "
                      "not followed", file=sys.stderr)
                try_sites(placer, todo, placements, skipped)  # in place, as far as reachable
                continue
            # Stubs for sites in the first half go just before the copy, the
            # rest just after it, so both halves stay within s_branch reach.
            size = end - start
            first = sum(stub_bytes(insn) for insn, _ in todo if insn.addr - start < size // 2)
            pool = placer.pool
            pad = (-(pool.cursor + first)) % 256  # kernel entries are 256-byte aligned
            if pool.free < first + pad + size + PREFETCH_GUARD:
                skipped += [(insn, "no room to relocate its kernel") for insn, _ in todo]
                continue
            before = Region(pool.cursor, pool.cursor + first + pad)
            placer.take(pool, first + pad)
            new = placer.take(pool, size)
            placer.add_island(before)
            relocations.append(Relocation(start, end, new))
            try_sites(placer, todo, placements, skipped, delta=new - start)
    pool = placer.pool
    islands = placer.islands
    placements.sort(key=lambda p: p.marker_id)
    for insn, why in skipped:
        print(f"skip {insn.addr:#x} {insn.text}: {why}", file=sys.stderr)
    # Every marked site: (marker id, kind, original address).
    marked = sorted(
        [(p.marker_id, p.kind, p.orig) for p in placements]
        + [(m[0], m[1], m[2]) for rl in relayouts for m in rl.markers]
    )
    if not marked:
        raise Unchanged(f"no site could be placed ({len(sites)} candidates)")
    if len({m[0] for m in marked}) != len(marked):
        raise VerifyError("duplicate marker IDs")

    # gfx9: make sure every kernel that can run a stub with an M0 marker has
    # the spare SGPR (plus VCC, FLAT_SCRATCH, XNACK_MASK) allocated.
    bumped: dict[str, int] = {}
    if target.gfx9 and spare is not None and any(uses_m0(m[0]) for m in marked):
        want = gfx9_sgpr_granules(spare + 1 + 6)
        # Only kernels that own a site, unless code can call across kernels
        # (then a site in one kernel's range may run under another's descriptor).
        site_ranges = {kernel_range(m[2]) for m in marked}
        calls = any(i.mnemonic in ("s_swappc_b64", "s_call_b64") for i in insns)
        all_kernels = calls or None in site_ranges
        for name, (off, entry) in entries.items():
            if not all_kernels and kernel_range(entry) not in site_ranges:
                continue
            rsrc1 = struct.unpack_from("<I", data, off + 48)[0]
            if (rsrc1 >> 6) & 0xF < want:
                bumped[name] = want

    # Kernels whose descriptor will point at their copy.
    moved_entries: dict[str, int] = {}
    for r in [*relocations, *relayouts]:
        for name, (_, entry) in entries.items():
            if entry == r.start:
                moved_entries[name] = r.new

    lines = source_lines(args.llvm, args.input, [m[2] for m in marked])

    # Lay the words down: original .text patches and the pool.
    words_at: dict[int, int] = {}

    def put(addr: int, words: list[int]) -> None:
        for k, w in enumerate(words):
            if addr + 4 * k in words_at:
                raise VerifyError(f"two writes to {addr + 4 * k:#x}")
            words_at[addr + 4 * k] = w

    def chain(frm: int, hops: list[int], dest: int) -> int:
        """Write the hop branches; return the branch word to place at frm."""
        path = [frm] + hops + [dest]
        for a, b in zip(path[1:-1], path[2:]):
            put(a, [target.branch(a, b)])
        return target.branch(frm, path[1])

    for p in placements:
        first = chain(p.insn.addr, p.forward, p.stub)
        put(p.insn.addr, [first] + [target.s_nop0] * (len(p.insn.words) - 1))
        put(p.stub, p.stub_words)
        back_first = chain(p.back_branch, p.back, p.insn.end)
        put(p.back_branch, [back_first])
    funcmap = []
    for marker_id, kind, orig in marked:
        loc = lines.get(orig) or f"pc{orig:#x}"
        funcmap.append(f"P:{marker_id}:site_{kind}@{loc}")
    # Dissolve the clauses that cover a site (in a kernel copy: the copy's clause).
    clause_addrs = sorted(
        {insns[clause_of[p.orig]].addr + p.insn.addr - p.orig for p in placements if p.orig in clause_of}
    )
    clause_insns = [Insn(a, [target.s_nop0], "s_clause") for a in clause_addrs]
    for c in clause_insns:
        put(c.addr, [target.s_nop0])

    # Pool size, and how far later segments move (whole pages).
    pool_used = pool.cursor - pool.start
    pool_bytes = bytearray()
    if pool_used:
        pool_bytes = bytearray(stub_base - text_end + pool_used)
        pool_bytes += struct.pack("<I", target.guard) * (PREFETCH_GUARD // 4)
    added = len(pool_bytes)
    vshift = 0
    if added and text_end + added > limit:
        if not movable:
            raise VerifyError("pool overflowed its page slack")
        vshift = (text_end + added - limit + PAGE - 1) & ~(PAGE - 1)

    # PC-relative addresses: those into moved segments get the shift added, in
    # the original code and in kernel copies; in a copy, those leaving the
    # kernel are re-aimed at their original target. expected_pc records where
    # every recognised s_getpc must point afterwards (checked on the output).
    def final(addr: int) -> int:
        return addr + vshift if addr >= limit else addr

    pc_fixes: list[tuple[int, int]] = []  # (literal address, new literal)
    expected_pc: dict[int, int] = {}  # s_getpc base -> target

    def encode(ref: PcRef, offset: int) -> list[tuple[int, int]]:
        writes = ref.encode(offset)
        if writes is None:
            raise VerifyError(f"s_getpc at {ref.getpc:#x} cannot be re-aimed")
        return writes

    for ref in pc_refs:
        want = final(ref.target)
        if want != ref.target:
            pc_fixes += encode(ref, want - ref.base)
        expected_pc[ref.base] = want
        for r in relocations:
            if not r.start <= ref.getpc < r.end:
                continue
            inside = r.start <= ref.target < r.end
            want_copy = ref.target + r.delta if inside else final(ref.target)
            if want_copy - (ref.base + r.delta) != ref.offset:
                pc_fixes += [(a + r.delta, w) for a, w in encode(ref, want_copy - ref.base - r.delta)]
            expected_pc[ref.base + r.delta] = want_copy
    for rl in relayouts:
        for ref, nbase, inside_target in rl.refs:
            if inside_target is not None:
                expected_pc[nbase] = inside_target  # already re-aimed in rl.words
                continue
            want = final(ref.target)
            for a, w in encode(ref, want - nbase):
                pc_fixes.append((rl.new_at[a - 4] + 4, w))
            expected_pc[nbase] = want

    # The same for PC-relative adds found by the flow scan. lit_checks records,
    # for verify(), where each one must point afterwards:
    # (low literal address, high literal address or None, inline high value, base, target).
    lit_checks: list[tuple[int, int | None, int | None, int, int]] = []

    def la_where(la: LitAdd, moved) -> tuple[int, int | None, int | None]:
        lo_a = moved(insns[la.lo].addr) + 4
        if la.hi_literal:
            return lo_a, moved(insns[la.hi].addr) + 4, None
        return lo_a, None, constant(operands(insns[la.hi])[2])

    def la_aim(la: LitAdd, base: int, want: int, moved) -> None:
        if want - base != la.offset:
            writes = la_encode(la, want - base)
            if writes is None:
                raise VerifyError(f"PC-relative add at {insns[la.lo].addr:#x} cannot be re-aimed")
            pc_fixes.extend((moved(a - 4) + 4, w) for a, w in writes)
        lit_checks.append((*la_where(la, moved), base, want))

    for la in flow.lit_adds.values():
        t = la_target(la)
        la_aim(la, la.getpc + 4, final(t), lambda a: a)
        for r in relocations:
            if r.start <= la.getpc < r.end:
                want = t + r.delta if r.start <= t < r.end else final(t)
                la_aim(la, la.getpc + 4 + r.delta, want, lambda a, d=r.delta: a + d)
    for rl in relayouts:
        for la, nbase, inside_target in rl.lit_refs:
            moved = lambda a, rl=rl: rl.new_at[a]
            if inside_target is not None:  # already re-aimed in rl.words
                lit_checks.append((*la_where(la, moved), nbase, inside_target))
            else:
                la_aim(la, nbase, final(la_target(la)), moved)
    for addr, w in pc_fixes:
        put(addr, [w])

    out = bytearray(data)
    for r in relocations:
        # The copy itself; sites and literals inside it are patched below.
        src = text_fileoff + r.start - text_addr
        pool_bytes[r.new - text_end : r.new - text_end + r.size] = data[src : src + r.size]
    for rl in relayouts:
        pool_bytes[rl.new - text_end : rl.new - text_end + rl.size] = struct.pack(
            f"<{len(rl.words)}I", *rl.words
        )
    for name, new_entry in moved_entries.items():
        off, _ = entries[name]
        kd = kds[name]
        struct.pack_into("<q", out, off + 16, new_entry - kd)
    for addr, w in words_at.items():
        if text_addr <= addr < text_end:
            struct.pack_into("<I", out, text_fileoff + addr - text_addr, w)
        elif stub_base <= addr < pool.cursor:
            struct.pack_into("<I", pool_bytes, addr - text_end, w)
        else:
            raise VerifyError(f"write outside .text and pool at {addr:#x}")
    for name, granules in bumped.items():
        off = elf.file_offset(kds[name]) + 48
        rsrc1 = struct.unpack_from("<I", out, off)[0]
        struct.pack_into("<I", out, off, (rsrc1 & ~(0xF << 6)) | (granules << 6))

    # Insert the pool after .text and shift the rest of the file by whole pages,
    # so later PT_LOADs keep p_offset == p_vaddr (mod page); their addresses
    # move by vshift (also whole pages).
    if added:
        insert_at = text_fileoff + text_size
        shift = (added + PAGE - 1) & ~(PAGE - 1)
        out[insert_at:insert_at] = pool_bytes + b"\0" * (shift - added)
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

    # Check everything before the output exists under its final name.
    verify(
        target, data, bytes(out), placements, clause_insns, islands, pool, bumped, pc_fixes,
        relocations, moved_entries, set(words_at), relayouts, lit_checks,
        [(r, expected_pc[r.base]) for r in pc_refs if r.base in expected_pc],
    )
    # What relaid-out code must decode to: the original instruction at its new
    # address, and the first instruction of each marker.
    insn_at = {i.addr: i for i in insns}
    expected_mnemonics = {}
    for rl in relayouts:
        for a, na in rl.new_at.items():
            expected_mnemonics[na] = insn_at[a].mnemonic
        for m_id, _, _, maddr, _ in rl.markers:
            expected_mnemonics[maddr] = "s_mov_b32" if uses_m0(m_id) else "s_ttracedata_imm"
        for g in rl.pins:
            expected_mnemonics[rl.new_at[g] + 4] = "s_cselect_b32"
        for start_at, iw in rl.island_words.items():
            for j in range(len(iw)):
                expected_mnemonics[start_at + 4 * j] = "s_branch"
    tmp = f"{args.output}.tmp{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            f.write(out)
        tool_check(args.llvm, args.input, tmp, placements, expected_pc, expected_mnemonics, text_end)
        os.replace(tmp, args.output)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    m0_sites = sum(uses_m0(m[0]) for m in marked)
    in_islands = sum(p.stub < text_end for p in placements)
    in_copies = sum(p.insn.addr != p.orig for p in placements)
    hops = sum(len(p.forward) + len(p.back) for p in placements)
    print(
        f"{len(marked)} of {len(sites)} sites marked"
        + (f" ({len(skipped)} skipped)" if skipped else "")
        + (f"; {unreached} unreachable LDS instructions left alone" if unreached else "")
        + f"; {len(marked) - m0_sites} imm, {m0_sites} via M0"
        + ((f" saved in s{spare}" if spare is not None else " saved in per-site dead SGPRs") if m0_sites else "")
        + f"; stubs: {in_islands} in islands, {len(placements) - in_islands} in pool"
        + (
            f"; {len(relayouts)} kernels relaid out ({inline_count[0]} markers inline"
            + (f", {sum(len(r.island_branch) for r in relayouts)} branch islands" if any(r.island_branch for r in relayouts) else "")
            + ")"
            if relayouts
            else ""
        )
        + (f" ({added} bytes at {stub_base:#x})" if added else "")
        + (f"; {hops} hops" if hops else "")
        + (f"; later segments moved up {vshift:#x}" if vshift else "")
        + (f" ({len(pc_fixes) // 2} PC-relative addresses adjusted)" if pc_fixes else "")
        + (
            f"; {len(relocations)} kernels run from a marked copy ({in_copies} sites)"
            if relocations
            else ""
        )
        + (f"; SGPR allocation raised in {len(bumped)} kernels" if bumped else "")
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
    parser.add_argument(
        "--no-islands",
        action="store_true",
        help="Put every stub in the pool after .text (no stubs or hops in padding).",
    )
    parser.add_argument(
        "--no-relocate",
        action="store_true",
        help="Never run a kernel from a marked copy; skip sites that cannot be marked in place.",
    )
    parser.add_argument(
        "--relayout",
        action="store_true",
        help="Lay out every kernel that has sites again with the markers inline (no detours, "
        "lowest overhead); kernels that cannot be relaid out are marked in place.",
    )
    parser.add_argument(
        "--relocate",
        action="store_true",
        help="Testing: run every kernel that has sites from an unchanged copy marked with stubs.",
    )
    parser.add_argument("--llvm", default="/opt/rocm/lib/llvm/bin")
    args = parser.parse_args()
    args.sites = args.sites or ["lds"]
    try:
        return rewrite(args)
    except Unchanged as e:
        # Exit code 3 tells callers (e.g. sqtt_site_injector) to keep the original.
        print(f"code object left unchanged: {e}")
        return 3
    except VerifyError as e:
        print(f"verification failed, code object left unchanged: {e}")
        return 4
    except Exception as e:  # any bug: report it on the first line, leave the code object alone
        print(f"rewriter error, code object left unchanged: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
