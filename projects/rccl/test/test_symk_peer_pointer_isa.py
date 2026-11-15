#!/usr/bin/env python3
"""Kernel ISA guard: the LL peer-pointer lookup must not stall the LL store stream.

ncclLLA2ASession::bcast() resolves one peer pointer per LL store through
ncclGetResourceBufferPeerPointer (nccl_device/impl/core__funcs.h). On AMD that
lookup must stay in kernarg/scalar data. If it reads memory through a vector
load (as the NCCL 2.32.6 sync did with ncclWindow_vidmem::lsaPeerDelta4G), the
wave waits for that load once per peer. On gfx9 the wait counter (vmcnt)
retires loads and stores in issue order, so each wait also drains the LL store
issued just before it. That serialized AllGather_LL and AllReduce_AGxLL_R.

This guard builds a small probe kernel with the shape of allgather_LL_body: the
communicator in the kernarg segment and a loop of lla2a.bcast() calls. It
compiles device code only (-S) for each arch in ARCHES with the Release device
build flags, then counts instructions inside the probe's loops:

  1. vector memory loads: must be 0;
  2. waits that hold back the LL stores: must be 0. On gfx9 that is an
     s_waitcnt vmcnt(N) that retires a store. On gfx12, which has separate
     load and store counters, it is an s_wait_loadcnt that retires a load
     issued inside the loop.

Both limits are absolute, so there is no baseline file. The production
symmetric kernels are not scanned: their loops also hold legitimate loads of
user data, so a limit there needs a tolerance that unrelated changes keep
moving. The probe isolates the lookup, so its limit is exactly 0. The verdict
holds at -O1, -O2 and -O3, with and without kernarg preloading.

Not covered: bcast() inlined into an out-of-line callee that gets the
communicator by reference (the gfx1250 fp8 symmetric kernels). There the comm
fields are reloaded through a generic pointer after every store, even on code
that predates the sync, so no absolute limit applies.

Needs a HIP compiler and the hipified tree (the hipify_all target, i.e. the
`hipify` phase of test/host/run_host_tests.sh). No GPU and no librccl.so.
Nothing in RCCL CI runs ctest, so the copy that gates is
test/host/run_host_tests.sh `guards` (folded into `run`).

Knobs (environment variables, all optional):
  RCCL_BUILD_DIR              RCCL build dir with hipify/ and include/nccl.h
                              (default: <rccl>/build)
  HIPCXX                      HIP compiler (default: $ROCM_PATH/bin/amdclang++)
  RCCL_ISA_TEST_KEEP_DIR      write the probe source and per-arch .s files here
  RCCL_ISA_TEST_ALLOW_SKIP=1  skip instead of fail without a compiler or tree
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest

RCCL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.environ.get("RCCL_BUILD_DIR") or os.path.join(RCCL_ROOT, "build")
HIPIFY_INCLUDE = os.path.join(BUILD_DIR, "hipify", "src", "include")
BUILD_INCLUDE = os.path.join(BUILD_DIR, "include")  # configured nccl.h
POISON_H = os.path.join(RCCL_ROOT, "src", "include", "poison_hip_atomics.h")
KEEP_DIR = os.environ.get("RCCL_ISA_TEST_KEEP_DIR")
ALLOW_SKIP = os.environ.get("RCCL_ISA_TEST_ALLOW_SKIP") == "1"

# gfx942/gfx950 carried the regression. gfx1250 is in DEFAULT_GPUS and has its
# own load/store counters, so it gets the gfx12 form of check 2.
ARCHES = ("gfx942", "gfx950", "gfx1250")

# Release flags of dl_add_fat_object (cmake/DeviceLinker.cmake), which builds
# the symmetric kernels.
DEVICE_FLAGS = (
    "-O3",
    "-mllvm",
    "--amdgpu-kernarg-preload-count=16",
    "-fvisibility=hidden",
    "-std=c++17",
    "-fPIC",
    "-DRCCL_DEVICE_LINKER",
)

PROBE_KERNEL = "SymkPeerPointerProbe"

# Same shape as allgather_LL_body (src/device/symmetric/all_gather.cuh): the
# communicator sits in the kernarg segment, like ncclSymkDevWorkArgs, and each
# iteration broadcasts one LL packet to every LSA peer.
PROBE_SRC = """\
#include "nccl_device.h"

struct ProbeArgs {
  ncclDevComm comm;
  ncclLLA2AHandle handle;
  uint64_t const* input;
  int n_iters;
};

extern "C" __global__ void SymkPeerPointerProbe(ProbeArgs const args) {
  ncclLLA2ASession<ncclCoopCta> lla2a(ncclCoopCta(), args.comm, ncclTeamLsa(args.comm), args.handle, blockIdx.x,
                                      /*maxElts=*/512, /*multimem=*/false, args.comm.lsaMultimem);
  uint64_t x = args.input[threadIdx.x];
  for (int i = 0; i < args.n_iters; i++) {
    lla2a.bcast(i * blockDim.x + threadIdx.x, x + i);
  }
}
"""

# bcast() unrolls its peer loop by 8. Fewer in-loop stores means the probe
# loop is not what this guard thinks it is, and the counts prove nothing.
MIN_LOOP_STORES = 8

_INSN_RE = re.compile(r"^\s+([a-z_][a-z0-9_]*)(.*)$")
_LABEL_RE = re.compile(r"^(?:\.LBB\d+_\d+|; %bb\.\d+):")
_COMMENT_RE = re.compile(r"^\s*;")
# LLVM verbose-asm block comments: "Loop Header", "Parent Loop", "in Loop:".
_LOOP_MARK_RE = re.compile(r"Loop Header|Parent Loop|in Loop:")
_VMEM_RE = re.compile(r"^(global|flat|buffer|scratch)_(load|store|atomic)")
_ATOMIC_RETURN_RE = re.compile(r"\b(glc|sc0)\b|TH_ATOMIC_RETURN")
_VMCNT_RE = re.compile(r"vmcnt\((\d+)\)")
_CALL_RE = re.compile(r"^s_(swappc|call)_b64$")


def _vmem_kind(op, operands):
    """'load', 'store' or None. A returning atomic counts as a load."""
    m = _VMEM_RE.match(op)
    if not m:
        return None
    if m.group(2) == "atomic":
        return "load" if _ATOMIC_RETURN_RE.search(operands) else "store"
    return m.group(2)


def _function_body(asm, symbol):
    """Lines of `symbol`'s body in the .s text, or None if it is absent."""
    body = None
    for line in asm.splitlines():
        if body is None:
            if line.startswith(symbol + ":"):
                body = []
        elif line.startswith(".Lfunc_end"):
            return body
        else:
            body.append(line)
    return body


def _retire(queue, keep):
    """Pop and return what a wait for `keep` outstanding events retires.

    The counters retire in issue order, so that is all but the newest `keep`.
    """
    n = max(0, len(queue) - keep)
    retired = queue[:n]
    del queue[:n]
    return retired


def analyze(asm, symbol):
    """Count what the loops of `symbol` do, or return None if it is absent.

    Walks the asm in text order. At each loop header the previous iteration's
    LL stores are assumed in flight (they arrive over the back edge).

    Returns a dict:
      loops        loop headers
      loop_stores  vector stores inside loops
      loop_loads   vector loads inside loops (lines)
      stall_waits  waits inside loops that hold back the LL stores (lines): on
                   gfx9 an s_waitcnt vmcnt(N) that retires a store, on gfx12 an
                   s_wait_loadcnt* that retires a load issued inside a loop
      calls        call instructions (lines); the counts need full inlining
    """
    body = _function_body(asm, symbol)
    if body is None:
        return None
    res = {
        "loops": 0,
        "loop_stores": 0,
        "loop_loads": [],
        "stall_waits": [],
        "calls": [],
    }
    in_loop = False
    vm_queue = []  # gfx9 vmcnt events, oldest first: True for a store
    load_queue = []  # gfx12 loadcnt events, oldest first: True if in a loop
    for line in body:
        label = _LABEL_RE.match(line)
        if label or _COMMENT_RE.match(line):
            if label:
                in_loop = False
            if _LOOP_MARK_RE.search(line):
                in_loop = True
            if "Loop Header" in line:
                res["loops"] += 1
                vm_queue.append(True)
            continue
        m = _INSN_RE.match(line)
        if not m:
            continue
        op, operands = m.group(1), m.group(2)
        text = line.strip()
        if _CALL_RE.match(op):
            res["calls"].append(text)
        kind = _vmem_kind(op, operands)
        if kind == "store":
            vm_queue.append(True)
            if in_loop:
                res["loop_stores"] += 1
        elif kind == "load":
            vm_queue.append(False)
            load_queue.append(in_loop)
            if in_loop:
                res["loop_loads"].append(text)
        elif op == "s_waitcnt":
            mv = _VMCNT_RE.search(operands)
            if mv and any(_retire(vm_queue, int(mv.group(1)))) and in_loop:
                res["stall_waits"].append(text)
        elif op.startswith("s_wait_loadcnt"):
            keep = int(operands.split(",")[0], 0)
            if any(_retire(load_queue, keep)) and in_loop:
                res["stall_waits"].append(text)
    return res


def _find_hipcxx():
    """A clang driver for -x hip. CXX is ignored on purpose: it is often g++."""
    val = os.environ.get("HIPCXX")
    if val:
        return shutil.which(val)
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    for cand in (
        os.path.join(rocm, "bin", "amdclang++"),
        os.path.join(rocm, "llvm", "bin", "clang++"),
    ):
        if os.access(cand, os.X_OK):
            return cand
    return shutil.which("amdclang++")


HIPCXX = _find_hipcxx()
HAVE_TREE = os.path.isfile(
    os.path.join(HIPIFY_INCLUDE, "nccl_device.h")
) and os.path.isfile(os.path.join(BUILD_INCLUDE, "nccl.h"))


def _compile_probe(arch, workdir):
    """Return (returncode, asm text, command + compiler output) for `arch`."""
    src = os.path.join(workdir, "symk_peer_pointer_probe.cpp")
    with open(src, "w") as f:
        f.write(PROBE_SRC)
    out = os.path.join(workdir, f"symk_peer_pointer_probe.{arch}.s")
    # hipify/src/include first: build/include/nccl_device/ is a copy that only
    # copy_nccl_device_headers refreshes.
    cmd = [
        HIPCXX,
        "-x",
        "hip",
        f"--offload-arch={arch}",
        "--offload-device-only",
        "-S",
        *DEVICE_FLAGS,
        f"--include={POISON_H}",
        f"-I{HIPIFY_INCLUDE}",
        f"-I{BUILD_INCLUDE}",
        "-o",
        out,
        src,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    asm = ""
    if proc.returncode == 0:
        with open(out) as f:
            asm = f.read()
    return proc.returncode, asm, " ".join(cmd) + "\n" + proc.stdout + proc.stderr


def _summary(r):
    return (
        f"loops={r['loops']} loop_stores={r['loop_stores']} "
        f"loop_loads={len(r['loop_loads'])} stall_waits={len(r['stall_waits'])} "
        f"calls={len(r['calls'])}"
    )


def _sample(lines, limit=6):
    more = "\n    ..." if len(lines) > limit else ""
    return "\n    ".join(lines[:limit]) + more


class AnalyzerTest(unittest.TestCase):
    """Pins analyze() on hand-written asm. Needs no compiler."""

    GFX9 = textwrap.dedent("""\
        SymkPeerPointerProbe:           ; @SymkPeerPointerProbe
        \tglobal_load_dwordx2 v[0:1], v[2:3], off
        \ts_waitcnt vmcnt(0)
        .LBB0_1:                        ; =>This Inner Loop Header: Depth=1
        \tglobal_load_dwordx2 v[4:5], v6, s[0:1] offset:96
        \ts_waitcnt vmcnt(1)
        \tglobal_store_dwordx4 v[0:1], v[8:11], off sc0 sc1
        \ts_waitcnt vmcnt(0)
        \tflat_load_dword v7, v[4:5]
        \ts_cbranch_scc1 .LBB0_1
        ; %bb.2:
        \ts_waitcnt vmcnt(0)
        \ts_endpgm
        .Lfunc_end0:
        """)

    def test_gfx9_counts_in_loop_waits_that_retire_a_store(self):
        r = analyze(self.GFX9, PROBE_KERNEL)
        self.assertEqual(r["loops"], 1)
        self.assertEqual(r["loop_stores"], 1)
        self.assertEqual(len(r["loop_loads"]), 2)
        # vmcnt(1) retires the back-edge store, vmcnt(0) the new one. The
        # waits before and after the loop are not counted.
        self.assertEqual(r["stall_waits"], ["s_waitcnt vmcnt(1)", "s_waitcnt vmcnt(0)"])
        self.assertEqual(r["calls"], [])

    def test_gfx9_wait_that_leaves_the_store_in_flight_is_not_a_stall(self):
        asm = self.GFX9.replace("s_waitcnt vmcnt(1)", "s_waitcnt vmcnt(2)")
        r = analyze(asm, PROBE_KERNEL)
        self.assertEqual(r["stall_waits"], ["s_waitcnt vmcnt(0)"])

    def test_gfx12_counts_in_loop_waits_for_in_loop_loads(self):
        asm = textwrap.dedent("""\
            SymkPeerPointerProbe:       ; @SymkPeerPointerProbe
            \tglobal_load_b64 v[8:9], v0, s[20:21]
            .LBB0_1:                    ; =>This Loop Header: Depth=1
            \ts_wait_loadcnt 0x0
            \tglobal_store_b128 v[2:3], v[42:45], off th:TH_STORE_NT
            .LBB0_2:                    ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
            \tglobal_load_b64 v[2:3], v57, s[80:81] offset:96
            \ts_wait_loadcnt_dscnt 0x0
            \ts_swappc_b64 s[30:31], s[16:17]
            \ts_cbranch_scc1 .LBB0_2
            .Lfunc_end0:
            """)
        r = analyze(asm, PROBE_KERNEL)
        self.assertEqual(r["loops"], 2)
        self.assertEqual(r["loop_stores"], 1)
        self.assertEqual(len(r["loop_loads"]), 1)
        # The header wait only retires the load issued before the loop, so it
        # costs the first iteration alone.
        self.assertEqual(r["stall_waits"], ["s_wait_loadcnt_dscnt 0x0"])
        self.assertEqual(len(r["calls"]), 1)

    def test_missing_symbol(self):
        self.assertIsNone(analyze(self.GFX9, "SomethingElse"))


class ToolchainTest(unittest.TestCase):
    """Makes a run without a compiler or hipified tree red.

    The ISA checks can only skip then, and a fully skipped suite exits 0.
    """

    @unittest.skipIf(ALLOW_SKIP, "RCCL_ISA_TEST_ALLOW_SKIP=1")
    def test_hip_compiler_available(self):
        self.assertIsNotNone(
            HIPCXX,
            "no HIP compiler found: set ROCM_PATH or HIPCXX, or set "
            "RCCL_ISA_TEST_ALLOW_SKIP=1 to skip on a box without ROCm",
        )

    @unittest.skipIf(ALLOW_SKIP, "RCCL_ISA_TEST_ALLOW_SKIP=1")
    def test_hipified_tree_available(self):
        self.assertTrue(
            HAVE_TREE,
            f"no hipified tree under {BUILD_DIR}: configure RCCL and build the "
            "hipify_all target, or point RCCL_BUILD_DIR at a build that has one",
        )


@unittest.skipUnless(HIPCXX and HAVE_TREE, "no HIP compiler or hipified tree")
class SymkPeerPointerIsaTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.results = {}
        cls.errors = {}
        with tempfile.TemporaryDirectory() as tmp:
            workdir = KEEP_DIR or tmp
            os.makedirs(workdir, exist_ok=True)
            for arch in ARCHES:
                rc, asm, log = _compile_probe(arch, workdir)
                if rc != 0:
                    cls.errors[arch] = log
                    continue
                r = analyze(asm, PROBE_KERNEL)
                cls.results[arch] = r
                print(f"{arch}: {_summary(r) if r else 'probe symbol not found'}")

    def _result(self, arch):
        if arch in self.errors:
            self.fail(f"probe failed to compile for {arch}:\n{self.errors[arch]}")
        r = self.results[arch]
        if r is None:
            self.fail(f"{PROBE_KERNEL} not found in the {arch} asm")
        return r

    def test_probe_is_not_vacuous(self):
        """The limits below are 0: a probe that lost its loop passes them free."""
        for arch in ARCHES:
            with self.subTest(arch=arch):
                r = self._result(arch)
                if r["loops"] < 1 or r["loop_stores"] < MIN_LOOP_STORES:
                    self.fail(
                        f"{arch}: expected a loop with >= {MIN_LOOP_STORES} "
                        f"LL stores in the probe; {_summary(r)}"
                    )
                if r["calls"]:
                    self.fail(
                        f"{arch}: probe is not fully inlined:\n    "
                        + _sample(r["calls"])
                    )

    def test_no_vector_loads_in_peer_loop(self):
        for arch in ARCHES:
            with self.subTest(arch=arch):
                loads = self._result(arch)["loop_loads"]
                if loads:
                    self.fail(
                        f"{arch}: {len(loads)} vector loads inside the LL peer "
                        f"loop:\n    {_sample(loads)}\n"
                        "ncclGetResourceBufferPeerPointer (or another bcast() "
                        "helper) reads memory per peer. Keep the lookup in "
                        "kernarg/scalar data (lsaFlatBase + peer * stride4G for "
                        "LSA peers), see core__funcs.h."
                    )

    def test_no_store_stalling_waits_in_peer_loop(self):
        for arch in ARCHES:
            with self.subTest(arch=arch):
                waits = self._result(arch)["stall_waits"]
                if waits:
                    self.fail(
                        f"{arch}: {len(waits)} waits inside the LL peer loop "
                        f"hold back the LL stores:\n    {_sample(waits)}\n"
                        "Each one serializes a peer store behind a memory read."
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
