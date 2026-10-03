# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Helpers to find the LL protocol's flag-poll loads in RCCL device code.

The LL receiver (readLL / readLLBeginAll / readLLFinish in
src/device/prims_ll.h) spins on a single 16-byte load of an ncclLLFifoLine
{data1, flag1, data2, flag2} until both flags match the expected step. The
line is written by a peer GPU or a NIC, so the load must bypass any stale
cached copy -- otherwise the poll never observes the new flag and the
collective hangs. The cache-policy bits that make this true are chosen by the
compiler, so they are only visible in the final ISA.

A poll load is recognised by its shape, not by its cache bits, so a load that
lost its bits is still found (and then reported):

  * a 128-bit vector load into v[N:N+3], followed (before the next 128-bit
    load) by
  * v_cmp_ne_u32 of BOTH v[N+1] (flag1) and v[N+3] (flag2) against the same
    register (the expected flag).
"""

import collections
import os
import re
import subprocess

# "0000000007981f40 <_Z43ncclDevFunc_...>:"
FUNC_RE = re.compile(r"^[0-9a-fA-F]+ <(.+)>:\s*$")
# GFX9 spells the 128-bit load dwordx4; GFX11+ spells it b128.
LOAD128_RE = re.compile(
    r"^\s*(?P<op>(?:global|flat|buffer)_load_(?:dwordx4|b128))\s+"
    r"v\[(?P<lo>\d+):(?P<hi>\d+)\]"
)
CMP_NE_RE = re.compile(
    r"^\s*v_cmp_ne_u32\w*\s+[^,]+,\s*(?P<a>[vs]\d+),\s*(?P<b>[vs]\d+)"
)
# Instructions to scan after a load for its flag compares.
POLL_WINDOW = 16

# Mangled template argument for the LL protocol; LL128 mangles as
# "10ProtoLL128", so the length prefix keeps the two apart.
PROTO_LL_RE = re.compile(r"7ProtoLL(?!128)")
# Where LL poll loads can live: Primitives<ProtoLL> instantiations, kernels
# with LL in their name, and RunWorkBatch bodies that inline them.
CANDIDATE_RE = re.compile(r"7ProtoLL(?!128)|_LL_|RunWorkBatch")

Poll = collections.namedtuple("Poll", "func insn modifiers")


def _strip_comment(line):
    return line.split("//", 1)[0].rstrip()


def modifiers(insn):
    """Return the trailing modifier tokens of a load, e.g. ['nt'] or
    ['th:TH_LOAD_NT', 'scope:SCOPE_SYS']."""
    operands = insn.split(None, 1)[1] if " " in insn.strip() else ""
    last = operands.split(",")[-1].split()
    return last[1:]


def find_polls(lines):
    """Yield a Poll for every LL flag-poll load in disassembly `lines`."""
    func = None
    lines = [_strip_comment(l) for l in lines]
    for i, line in enumerate(lines):
        m = FUNC_RE.match(line)
        if m:
            func = m.group(1)
            continue
        m = LOAD128_RE.match(line)
        if not m:
            continue
        lo = int(m.group("lo"))
        flags = {"v%d" % (lo + 1), "v%d" % (lo + 3)}
        # expected-flag register -> which of our two flags were compared to it
        compared = collections.defaultdict(set)
        for nxt in lines[i + 1:i + 1 + POLL_WINDOW]:
            if FUNC_RE.match(nxt) or LOAD128_RE.match(nxt):
                break
            c = CMP_NE_RE.match(nxt)
            if not c:
                continue
            a, b = c.group("a"), c.group("b")
            if a in flags:
                compared[b].add(a)
            if b in flags:
                compared[a].add(b)
        if any(regs == flags for regs in compared.values()):
            insn = line.strip()
            yield Poll(func, insn, modifiers(insn))


def functions(lines):
    """Return the names of all functions in disassembly `lines`."""
    return [m.group(1) for m in map(FUNC_RE.match, lines) if m]


class Toolchain:
    """Thin wrapper over the ROCm LLVM tools used to reach the device ISA."""

    TOOLS = ("llvm-objcopy", "clang-offload-bundler", "llvm-nm", "llvm-objdump")

    def __init__(self, rocm_path):
        self.bindir = os.path.join(rocm_path, "llvm", "bin")

    def missing(self):
        return [t for t in self.TOOLS
                if not os.access(os.path.join(self.bindir, t), os.X_OK)]

    def run(self, tool, *args):
        return subprocess.run(
            [os.path.join(self.bindir, tool), *args], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        ).stdout

    def extract_fatbin(self, lib, out):
        """Copy the .hip_fatbin section of `lib` into file `out`."""
        self.run("llvm-objcopy", "--dump-section", ".hip_fatbin=" + out,
                 lib, os.devnull)

    def targets(self, fatbin):
        """Return {arch: bundle target id} for the device code in `fatbin`.

        Target ids look like "hip-amdgcn-amd-amdhsa--gfx942[:xnack-]"."""
        out = {}
        for tid in self.run("clang-offload-bundler", "--list", "--type=o",
                            "--input=" + fatbin).split():
            if "amdgcn" in tid:
                out[tid.split("--", 1)[1].split(":", 1)[0]] = tid
        return out

    def unbundle(self, fatbin, target, out):
        self.run("clang-offload-bundler", "--unbundle", "--type=o",
                 "--input=" + fatbin, "--targets=" + target, "--output=" + out)

    def candidate_disassembly(self, code_object, workdir):
        """Disassemble only the functions that can hold LL poll loads.

        A full disassembly of a code object is millions of lines; restricting
        it to the candidates makes this seconds instead of minutes."""
        syms = [parts[2] for parts in
                (l.split() for l in self.run("llvm-nm", "--defined-only",
                                             code_object).splitlines())
                if len(parts) == 3 and parts[1] in "Tt"
                and CANDIDATE_RE.search(parts[2])]
        if not syms:
            return []
        # The symbol list overflows the command line; pass it in a response file.
        rsp = os.path.join(workdir, "symbols.rsp")
        with open(rsp, "w") as f:
            f.write("--disassemble-symbols=" + ",".join(syms))
        return self.run("llvm-objdump", "-d", "--no-show-raw-insn", "@" + rsp,
                        code_object).splitlines()
