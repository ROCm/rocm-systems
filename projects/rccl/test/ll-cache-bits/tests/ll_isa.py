# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Helpers to find LL flag-poll loads in RCCL device code.

An LL receiver spins on a single 16-byte load of a line
{data1, flag1, data2, flag2} until both flags match the expected value: the
LL primitives (src/device/prims_ll.h), the DDA LL kernels (ddaLLWaitLineFlags)
and LLA2A in the symmetric kernels. A peer GPU or a NIC writes the line, so
the load must bypass any stale cached copy -- otherwise the poll never
observes the new flag and the collective hangs. The cache-policy bits that
make this true are chosen by the compiler, so they are only visible in the
final ISA.

A poll load is recognised by its shape, not by its cache bits, so a load that
lost its bits is still found (and then reported):

  * a 128-bit vector load into v[N:N+3], followed by
  * v_cmp_ne_u32 or v_cmp_eq_u32 of v[N+1] (flag1) and v[N+3] (flag2), or
    v_mov copies of them, against the same register (the expected flag), or
    of one flag against a register and flag1 against flag2,
  * before either flag register is overwritten.
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
CMP_RE = re.compile(
    r"^\s*v_cmp_(?:ne|eq)_u32\w*\s+[^,]+,\s*(?P<a>[vs]\d+),\s*(?P<b>[vs]\d+)"
)
MOV_RE = re.compile(r"^\s*v_mov_b32\w*\s+(v\d+),\s*(v\d+)\s*$")
# Instructions whose first operand is not a VGPR they write.
NO_VDST_RE = re.compile(r"^\s*(?:s_|v_cmp|\w+_store|ds_write)")
VDST_RE = re.compile(r"^\s*\w+\s+v(?:(\d+)|\[(\d+):(\d+)\])(?=[\s,])")
# Disassembly lines to scan after a load for its flag compares, past any
# inlined abort check.
POLL_WINDOW = 48

# Mangled template argument for the LL protocol; LL128 mangles as
# "10ProtoLL128", so the length prefix keeps the two apart.
PROTO_LL_RE = re.compile(r"7ProtoLL(?!128)")
# Functions that must hold an LL poll: Primitives<ProtoLL> instantiations,
# DDA LL kernels (ddaAllReduceFlatLL<...>, not LL128) and symmetric LL kernels.
MUST_POLL_RE = re.compile(
    PROTO_LL_RE.pattern + r"|dda\w+LLI|ncclSymkDevKernel_\w*LL")
# Where LL poll loads can live: the above, kernels with LL in their name, and
# RunWorkBatch bodies that inline them.
CANDIDATE_RE = re.compile(MUST_POLL_RE.pattern + r"|_LL_|RunWorkBatch")
# Each fat object adds its own 4 KiB-aligned offload bundle to .hip_fatbin.
BUNDLE_MAGICS = (b"__CLANG_OFFLOAD_BUNDLE__", b"CCOB")
BUNDLE_ALIGN = 0x1000

Poll = collections.namedtuple("Poll", "func insn modifiers")


def _strip_comment(line):
    return line.split("//", 1)[0].rstrip()


def modifiers(insn):
    """Return the trailing modifier tokens of a load, e.g. ['nt'] or
    ['th:TH_LOAD_NT', 'scope:SCOPE_SYS']."""
    operands = insn.split(None, 1)[1] if " " in insn.strip() else ""
    last = operands.split(",")[-1].split()
    return last[1:]


def _written(line):
    """Return the VGPRs `line` writes, as far as the poll check needs."""
    if NO_VDST_RE.match(line):
        return set()
    m = VDST_RE.match(line)
    if not m:
        return set()
    if m.group(1):
        return {"v" + m.group(1)}
    return {"v%d" % r for r in range(int(m.group(2)), int(m.group(3)) + 1)}


def _is_poll(lines, lo):
    """Whether the flags of a load into v[lo:lo+3] are checked in `lines`."""
    alias = [{"v%d" % (lo + 1)}, {"v%d" % (lo + 3)}]  # regs holding each flag
    expect = [set(), set()]  # regs each flag was compared against
    linked = False  # flag1 compared against flag2
    for line in lines:
        if FUNC_RE.match(line):
            return False
        c = CMP_RE.match(line)
        if c:
            a, b = c.group("a"), c.group("b")
            if ((a in alias[0] and b in alias[1])
                    or (a in alias[1] and b in alias[0])):
                linked = True
            for k in (0, 1):
                if a in alias[k] and b not in alias[1 - k]:
                    expect[k].add(b)
                elif b in alias[k] and a not in alias[1 - k]:
                    expect[k].add(a)
            if expect[0] & expect[1] or (linked and (expect[0] or expect[1])):
                return True
            continue
        mov = MOV_RE.match(line)
        for k in (0, 1):
            if mov and mov.group(2) in alias[k]:
                alias[k].add(mov.group(1))
            else:
                alias[k] -= _written(line)
        if not alias[0] or not alias[1]:
            return False
    return False


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
        if m and _is_poll(lines[i + 1:i + 1 + POLL_WINDOW],
                          int(m.group("lo"))):
            insn = line.strip()
            yield Poll(func, insn, modifiers(insn))


def functions(lines):
    """Return the names of all functions in disassembly `lines`."""
    return [m.group(1) for m in map(FUNC_RE.match, lines) if m]


def unpolled(lines, polls):
    """Return the MUST_POLL_RE functions in `lines` that hold none of `polls`.

    Each of them receives through an LL poll loop, so any hit means the
    detector no longer matches what the compiler emits."""
    polled = {p.func for p in polls}
    return [f for f in functions(lines)
            if MUST_POLL_RE.search(f) and f not in polled]


def split_bundles(fatbin, outdir):
    """Write each offload bundle in `fatbin` to its own file; return paths.

    clang-offload-bundler reads only the first bundle of a file."""
    with open(fatbin, "rb") as f:
        data = f.read()
    starts = [o for o in range(0, len(data), BUNDLE_ALIGN)
              if data.startswith(BUNDLE_MAGICS, o)]
    paths = []
    for n, (a, b) in enumerate(zip(starts, starts[1:] + [len(data)])):
        path = os.path.join(outdir, "bundle%d" % n)
        with open(path, "wb") as f:
            f.write(data[a:b])
        paths.append(path)
    return paths


class Toolchain:
    """Thin wrapper over the ROCm LLVM tools used to reach the device ISA."""

    TOOLS = ("llvm-objcopy", "clang-offload-bundler", "llvm-nm", "llvm-objdump")

    def __init__(self, rocm_path):
        # /opt/rocm has an llvm -> lib/llvm symlink; a TheRock dist may not.
        dirs = [os.path.join(rocm_path, d, "bin") for d in ("llvm", "lib/llvm")]
        self.bindir = next((d for d in dirs if os.path.isdir(d)), dirs[0])

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
        """Return {arch: [bundle target ids]} for the device code in `fatbin`.

        Target ids look like "hip-amdgcn-amd-amdhsa--gfx942[:xnack-]"; one arch
        can have several feature variants."""
        out = collections.defaultdict(list)
        for tid in self.run("clang-offload-bundler", "--list", "--type=o",
                            "--input=" + fatbin).split():
            if "amdgcn" in tid:
                out[tid.split("--", 1)[1].split(":", 1)[0]].append(tid)
        return dict(out)

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
        # The symbol list overflows the command line; use a response file.
        rsp = os.path.join(workdir, "symbols.rsp")
        with open(rsp, "w") as f:
            f.write("--disassemble-symbols=" + ",".join(syms))
        return self.run("llvm-objdump", "-d", "--no-show-raw-insn", "@" + rsp,
                        code_object).splitlines()
