#!/usr/bin/env python3
"""Kernel ISA guards for the symmetric kernels' peer-pointer lookups.

LL peer-pointer guard (SymkPeerPointerIsaTest)
----------------------------------------------
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

TDM deep-loop guard (SymkTdmDeepLoopIsaTest)
--------------------------------------------
reduceDeep() (src/device/symmetric/reduce_scatter.cuh) with EnableTma stages
each peer tile into LDS with the Tensor Data Mover (tensor_load_to_lds), then
every lane reads it back with ds_load. The reads overlap only if the compiler
has registers to keep several in flight. When the NCCL 2.32.6 sync cached
lsaFlatBase/stride4G in ncclSymkWorldPointerGetter across the loop,
ReduceScatter_TmaLD avg f32 read a whole peer tile one LDS read at a time.

That register pressure comes from the whole inlined kernel, so this guard
compiles the real ReduceScatter_TmaLD and _LD kernels (f32/f16, sum/avg) for
gfx1250, the only TDM target. The deep loop is every loop that reads LDS and
has a tensor_load_to_lds in its loop nest, with its child loops. Checks:

  1. the TDM path is compiled: tensor_load_to_lds and LDS reads in the TmaLD
     deep loop, none in LD. Otherwise TmaLD is LD and check 2 proves nothing;
  2. at most MAX_SERIALIZED_READS LDS reads there are issued alone and waited
     for before the next one. The regression had 33, develop has 0, at -O2 and
     -O3 with or without kernarg preloading.

Waits, reads in flight, spills, VGPRs and occupancy are printed, not asserted:
unrelated codegen moves them. bf16 and fp8 serialize on develop already.

TDM needs hip/amd_detail/amd_gfx1250_TDM.h and the tensor_load_to_lds builtin
(TDM_SUPPORTED, src/device/tdm/tdmCopy.h). Without them the guard prints why
and skips, unenforced. RCCL_ISA_TDM_ROCM_PATH points it at a TDM-capable ROCm.

Needs a HIP compiler and the hipified tree (the hipify_all target, i.e. the
`hipify` phase of test/host/run_host_tests.sh). No GPU and no librccl.so.
Nothing in RCCL CI runs ctest, so the copy that gates is
test/host/run_host_tests.sh `guards` (folded into `run`).

Knobs (environment variables, all optional):
  RCCL_BUILD_DIR              RCCL build dir with hipify/ and include/nccl.h
                              (default: <rccl>/build)
  HIPCXX                      HIP compiler (default: $ROCM_PATH/bin/amdclang++)
  RCCL_ISA_TDM_ROCM_PATH      ROCm for the TDM guard; no TDM there fails it
  RCCL_ISA_TDM_REQUIRED=1     fail instead of skip without a TDM toolchain
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
HIPIFY_SRC = os.path.join(BUILD_DIR, "hipify", "src")
HIPIFY_INCLUDE = os.path.join(HIPIFY_SRC, "include")
HIPIFY_GENSRC = os.path.join(BUILD_DIR, "hipify", "gensrc")  # device_table.h
BUILD_INCLUDE = os.path.join(BUILD_DIR, "include")  # configured nccl.h
POISON_H = os.path.join(RCCL_ROOT, "src", "include", "poison_hip_atomics.h")
KEEP_DIR = os.environ.get("RCCL_ISA_TEST_KEEP_DIR")
ALLOW_SKIP = os.environ.get("RCCL_ISA_TEST_ALLOW_SKIP") == "1"
TDM_ROCM_PATH = os.environ.get("RCCL_ISA_TDM_ROCM_PATH")
TDM_REQUIRED = bool(TDM_ROCM_PATH) or os.environ.get("RCCL_ISA_TDM_REQUIRED") == "1"

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
_LABEL_RE = re.compile(r"^(?:\.L(BB\d+_\d+)|; %bb\.\d+):")
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


def _find_hipcxx(rocm=None):
    """A clang driver for -x hip, from `rocm` only if given.

    CXX is ignored on purpose: it is often g++.
    """
    val = os.environ.get("HIPCXX")
    if val and not rocm:
        return shutil.which(val)
    root = rocm or os.environ.get("ROCM_PATH", "/opt/rocm")
    for cand in (
        os.path.join(root, "bin", "amdclang++"),
        os.path.join(root, "llvm", "bin", "clang++"),
    ):
        if os.access(cand, os.X_OK):
            return cand
    return None if rocm else shutil.which("amdclang++")


HIPCXX = _find_hipcxx()
# The symmetric headers and device_table.h are for the TDM guard's real kernels.
HAVE_TREE = all(
    os.path.isfile(p)
    for p in (
        os.path.join(HIPIFY_INCLUDE, "nccl_device.h"),
        os.path.join(HIPIFY_SRC, "device", "symmetric", "reduce_scatter.h"),
        os.path.join(HIPIFY_GENSRC, "device_table.h"),
        os.path.join(BUILD_INCLUDE, "nccl.h"),
    )
)


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


# ---- TDM deep-loop guard (see the module docstring) ----

TDM_ARCH = "gfx1250"
# ENABLE_FAULT_INJECTION comes with the FAULT_INJECTION option (ON by default),
# and device/common.h needs it.
TDM_DEVICE_FLAGS = DEVICE_FLAGS + ("-DENABLE_FAULT_INJECTION",)
# (red, type) pairs checked; names and template arguments as symmetric/generate.py.
TDM_CHECKED = (("sum", "f32"), ("avg", "f32"), ("sum", "f16"), ("avg", "f16"))
TDM_RED_FUNC = {"sum": "FuncSum", "avg": "FuncSumPostDiv"}
TDM_TYPE_CXX = {"f32": "float", "f16": "half"}
# A stray lone read (a tail, say) is not the regression: that serialized a tile.
MAX_SERIALIZED_READS = 4

# The TDM_SUPPORTED test of src/device/tdm/tdmCopy.h, in a gfx1250 device pass.
TDM_TOOLCHAIN_PROBE_SRC = """\
#if !__has_include(<hip/amd_detail/amd_gfx1250_TDM.h>)
#error "no hip/amd_detail/amd_gfx1250_TDM.h"
#endif
#if !__has_builtin(__builtin_amdgcn_tensor_load_to_lds)
#error "no __builtin_amdgcn_tensor_load_to_lds"
#endif
"""

_HEADER_RE = re.compile(r"This (?:Inner )?Loop Header: Depth=\d+")
_IN_LOOP_RE = re.compile(r"in Loop: Header=(BB\d+_\d+) Depth=\d+")
_PARENT_RE = re.compile(r"Parent Loop (BB\d+_\d+) Depth=\d+")
_LDS_READ_RE = re.compile(r"^ds_(load|read)")


def tdm_probe_symbol(algo, red, ty):
    return f"SymkTdmProbe_{algo}_{red}_{ty}"


def _tdm_probe_source():
    lines = [
        f'#include "{h}"'
        for h in ("sym_kernels.h", "symmetric/kernel.h", "symmetric/reduce_scatter.h")
    ]
    for algo in ("TmaLD", "LD"):
        for red, ty in TDM_CHECKED:
            lines += [
                f'extern "C" __global__ void {tdm_probe_symbol(algo, red, ty)}('
                "ncclSymkDevWorkArgs4K NCCL_GRID_CONSTANT const args4K) {",
                f"  ncclSymkRun_ReduceScatter_{algo}<false, {TDM_RED_FUNC[red]}, "
                f"{TDM_TYPE_CXX[ty]}>(&args4K.args);",
                "}",
            ]
    return "\n".join(lines) + "\n"


def _resources(asm, symbol):
    """VGPRs, SGPRs, occupancy and scratch bytes from the comments after `symbol`."""
    start = re.compile(r"^%s:" % re.escape(symbol), re.M).search(asm)
    end = asm.find("\n.Lfunc_end", start.end()) if start else -1
    names = {"vgpr": "NumVgprs", "sgpr": "TotalNumSgprs", "occupancy": "Occupancy"}
    names["scratch"] = "ScratchSize"
    res = {}
    for key, name in names.items():
        m = re.compile(r"^; %s: (\d+)" % name, re.M).search(asm, max(end, 0))
        if m and end >= 0:
            res[key] = int(m.group(1))
    return res


def _loop_instructions(body):
    """(innermost loop or None, op, operands) per instruction, and {loop: parent}."""
    insns, parent, parents = [], {}, []
    loop = block = None
    for line in body:
        label = _LABEL_RE.match(line)
        if label:
            block, loop, parents = label.group(1), None, []
        elif not _COMMENT_RE.match(line):
            m = _INSN_RE.match(line)
            if m:
                insns.append((loop, m.group(1), m.group(2).strip()))
            continue
        # The block's label line and the comment lines that follow it.
        mp = _PARENT_RE.search(line)
        if mp:
            parents.append(mp.group(1))
        mi = _IN_LOOP_RE.search(line)
        if _HEADER_RE.search(line) and block:
            loop = block
            parent[block] = parents[-1] if parents else None
        elif mi:
            loop = mi.group(1)
    return insns, parent


def _loop_nest(parent, root):
    nest = {root}
    while True:
        more = {c for c, p in parent.items() if p in nest} - nest
        if not more:
            return nest
        nest |= more


def _dscnt(op, operands):
    """The dscnt a wait leaves outstanding, or None if it does not wait on dscnt."""
    if not op.startswith("s_wait") or "dscnt" not in op:
        return None
    imm = int(operands.split(";")[0].split(",")[0].strip(), 0)
    # s_wait_{load,store}cnt_dscnt pack dscnt into bits [5:0].
    return imm if op == "s_wait_dscnt" else imm & 0x3F


def analyze_tdm_deep_loop(asm, symbol):
    """Measure the TDM deep loop of `symbol`, or return None if it is absent.

    The asm is walked in text order and DS events retire in issue order, as in
    analyze(). Returns a dict:
      region_loops    loops in the deep loop (0 when the TDM path is absent)
      tdm_loads       tensor_load_to_lds in the deep loop (_all: in the kernel)
      lds_reads       LDS reads in the deep loop
      dscnt_waits     waits on dscnt in the deep loop
      drain_waits     of those, waits for dscnt 0 that retire an LDS read
      serialized      LDS reads issued alone and waited for before the next one
      in_flight       mean LDS reads outstanding at the waits that retire one
      loop_scratch    scratch (spill) accesses in the deep loop
      vgpr, sgpr, occupancy, scratch (bytes) of the kernel
    """
    body = _function_body(asm, symbol)
    if body is None:
        return None
    insns, parent = _loop_instructions(body)
    reads_lds = {loop for loop, op, _ in insns if loop and _LDS_READ_RE.match(op)}
    has_tdm = {loop for loop, op, _ in insns if loop and op == "tensor_load_to_lds"}
    region = set()
    for loop in reads_lds:
        nest = _loop_nest(parent, loop)
        if nest & has_tdm:
            region |= nest
    keys = "tdm_loads tdm_loads_all lds_reads dscnt_waits drain_waits serialized"
    res = dict.fromkeys(keys.split() + ["loop_scratch"], 0)
    res["region_loops"] = len(region)
    queue = []  # DS events, oldest first: True for an LDS read
    issued = 0  # deep-loop LDS reads since the last wait that retired one
    in_flight = []
    for loop, op, operands in insns:
        inside = loop in region
        if op == "tensor_load_to_lds":
            res["tdm_loads_all"] += 1
            res["tdm_loads"] += inside
        elif op.startswith("scratch_"):
            res["loop_scratch"] += inside
        elif op.startswith("ds_"):
            queue.append(bool(_LDS_READ_RE.match(op)))
            if inside and queue[-1]:
                res["lds_reads"] += 1
                issued += 1
        keep = _dscnt(op, operands)
        if keep is None:
            continue
        before = sum(queue)
        retired = sum(_retire(queue, keep))
        res["dscnt_waits"] += inside
        if inside and retired:
            in_flight.append(before)
            res["drain_waits"] += keep == 0
            res["serialized"] += before == 1 and issued == 1
            issued = 0
    res["in_flight"] = sum(in_flight) / len(in_flight) if in_flight else 0.0
    res.update(_resources(asm, symbol))
    return res


# RCCL_ISA_TDM_ROCM_PATH's compiler, or the one the LL guard uses.
TDM_ROCM = TDM_ROCM_PATH or os.environ.get("ROCM_PATH", "/opt/rocm")
TDM_HIPCXX = _find_hipcxx(TDM_ROCM_PATH) if TDM_ROCM_PATH else HIPCXX


def _tdm_hip_cmd(*args):
    cmd = [TDM_HIPCXX, "-x", "hip", f"--offload-arch={TDM_ARCH}"]
    if os.path.isdir(os.path.join(TDM_ROCM, "include", "hip")):
        cmd.append(f"--rocm-path={TDM_ROCM}")
    return cmd + ["--offload-device-only", *args]


def _tdm_toolchain_missing(workdir):
    """None if the compiler can build the gfx1250 TDM path, else the reason."""
    if not TDM_HIPCXX:
        return f"no HIP compiler under {TDM_ROCM}"
    src = os.path.join(workdir, "tdm_toolchain_probe.cpp")
    with open(src, "w") as f:
        f.write(TDM_TOOLCHAIN_PROBE_SRC)
    cmd = _tdm_hip_cmd("-fsyntax-only", src)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return None
    out = (proc.stdout + proc.stderr).splitlines()
    errors = [ln.split("error: ", 1)[1] for ln in out if "error: " in ln]
    return f"{TDM_HIPCXX}: " + ("; ".join(errors[:2]) or "TDM probe failed")


def _compile_tdm_probe(workdir):
    """Return (returncode, asm text, command + compiler output)."""
    src = os.path.join(workdir, "symk_tdm_deep_loop_probe.cpp")
    with open(src, "w") as f:
        f.write(_tdm_probe_source())
    out = os.path.join(workdir, f"symk_tdm_deep_loop_probe.{TDM_ARCH}.s")
    # hipify/src/include first, as in _compile_probe().
    sub = ("include/nccl_device", "include/plugin", "", "device")
    includes = [HIPIFY_INCLUDE, *(os.path.join(HIPIFY_SRC, d) for d in sub)]
    includes += [HIPIFY_GENSRC, BUILD_INCLUDE]
    flags = [*TDM_DEVICE_FLAGS, f"--include={POISON_H}", *(f"-I{d}" for d in includes)]
    cmd = _tdm_hip_cmd("-S", *flags, "-o", out, src)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    asm = ""
    if proc.returncode == 0:
        with open(out) as f:
            asm = f.read()
    return proc.returncode, asm, " ".join(cmd) + "\n" + proc.stdout + proc.stderr


def _tdm_summary(r):
    keys = "lds_reads serialized dscnt_waits drain_waits tdm_loads loop_scratch"
    text = " ".join(f"{k}={r.get(k)}" for k in keys.split() + ["vgpr", "occupancy"])
    return f"{text} scratch={r.get('scratch')} in_flight={r['in_flight']:.2f}"


class TdmAnalyzerTest(unittest.TestCase):
    """Pins analyze_tdm_deep_loop() on hand-written gfx12 asm. Needs no compiler."""

    # Work loop BB0_1 > deep loop BB0_2 (reads LDS) > tile-load loop BB0_3
    # (TDM). BB0_4 reads LDS without TDM and stays out of the deep loop.
    ASM = textwrap.dedent("""\
        SymkTdmProbe_TmaLD_avg_f32:     ; @SymkTdmProbe_TmaLD_avg_f32
        \tds_load_b128 v[0:3], v9
        \ts_wait_dscnt 0x0
        .LBB0_1:                        ; =>This Loop Header: Depth=1
        \ts_wait_kmcnt 0x0
        .LBB0_2:                        ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Loop Header: Depth=2
        \tds_load_b128 v[4:7], v9
        \tds_load_b128 v[8:11], v9 offset:512
        \ts_wait_dscnt 0x1
        \ts_wait_dscnt 0x0
        .LBB0_3:                        ;   Parent Loop BB0_1 Depth=1
                                        ;     Parent Loop BB0_2 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
        \ttensor_load_to_lds s[12:15], s[16:23]
        \ts_cbranch_execnz .LBB0_3
        ; %bb.4:                        ;   in Loop: Header=BB0_2 Depth=2
        \tds_load_b128 v[4:7], v9
        \ts_wait_dscnt 0x0
        \tds_load_b128 v[4:7], v9 offset:512
        \ts_wait_loadcnt_dscnt 0x100
        \tscratch_load_b32 v1, off, off offset:4
        \tds_store_b128 v9, v[4:7]
        \tds_load_b128 v[4:7], v9 offset:1024
        \ts_wait_dscnt 0x1
        \ts_wait_dscnt 0x0
        \ts_cbranch_scc1 .LBB0_2
        ; %bb.5:                        ;   in Loop: Header=BB0_1 Depth=1
        \ts_cbranch_scc1 .LBB0_1
        .LBB0_4:                        ; =>This Inner Loop Header: Depth=1
        \tds_load_b128 v[4:7], v9
        \ts_wait_dscnt 0x0
        \ts_cbranch_scc1 .LBB0_4
        .Lfunc_end0:
        ; TotalNumSgprs: 107
        ; NumVgprs: 123
        ; ScratchSize: 20
        ; Occupancy: 8
        """)
    KERNEL = "SymkTdmProbe_TmaLD_avg_f32"

    def test_deep_loop_counts(self):
        r = analyze_tdm_deep_loop(self.ASM, self.KERNEL)
        # BB0_2 and BB0_3: not the work loop, not the non-TDM loop BB0_4.
        self.assertEqual(r["region_loops"], 2)
        self.assertEqual((r["tdm_loads"], r["tdm_loads_all"]), (1, 1))
        self.assertEqual((r["lds_reads"], r["dscnt_waits"]), (5, 6))
        # The first pair overlaps. Each later read is waited for alone; a store
        # ahead of the last one is not a read in flight.
        self.assertEqual((r["serialized"], r["drain_waits"]), (3, 4))
        self.assertEqual(r["loop_scratch"], 1)
        resources = (r["vgpr"], r["sgpr"], r["occupancy"], r["scratch"])
        self.assertEqual(resources, (123, 107, 8, 20))
        self.assertIsNone(analyze_tdm_deep_loop(self.ASM, "SomethingElse"))

    def test_combined_waits_carry_dscnt_in_low_bits(self):
        self.assertEqual(_dscnt("s_wait_loadcnt_dscnt", "0x303"), 3)
        self.assertEqual(_dscnt("s_wait_storecnt_dscnt", "0x100"), 0)
        self.assertIsNone(_dscnt("s_wait_loadcnt", "0x0"))

    def test_no_tdm_means_no_deep_loop(self):
        asm = self.ASM.replace("tensor_load_to_lds", "s_nop 0 ;")
        r = analyze_tdm_deep_loop(asm, self.KERNEL)
        self.assertEqual((r["region_loops"], r["lds_reads"]), (0, 0))


@unittest.skipUnless(HIPCXX and HAVE_TREE, "no HIP compiler or hipified tree")
class SymkTdmDeepLoopIsaTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.error = None
        cls.results = {}
        with tempfile.TemporaryDirectory() as tmp:
            workdir = KEEP_DIR or tmp
            os.makedirs(workdir, exist_ok=True)
            cls.no_tdm = _tdm_toolchain_missing(workdir)
            if cls.no_tdm:
                verdict = "FAIL" if TDM_REQUIRED else "SKIP, not enforced"
                print(f"{TDM_ARCH}: no TDM toolchain ({verdict}): {cls.no_tdm}")
                return
            rc, asm, log = _compile_tdm_probe(workdir)
            if rc != 0:
                cls.error = log
                return
            for algo in ("TmaLD", "LD"):
                for red, ty in TDM_CHECKED:
                    sym = tdm_probe_symbol(algo, red, ty)
                    r = cls.results[sym] = analyze_tdm_deep_loop(asm, sym)
                    if algo == "TmaLD":
                        print(f"{TDM_ARCH} {sym}: {_tdm_summary(r) if r else 'absent'}")

    def setUp(self):
        if not self.no_tdm:
            return
        msg = (
            f"no TDM toolchain for {TDM_ARCH} ({self.no_tdm}), so TmaLD compiles "
            "to LD. Set RCCL_ISA_TDM_ROCM_PATH to a ROCm that ships "
            "hip/amd_detail/amd_gfx1250_TDM.h to run this check"
        )
        if TDM_REQUIRED:
            self.fail(msg)
        self.skipTest(msg)

    def _result(self, algo, red, ty):
        if self.error:
            self.fail(f"TDM probe failed to compile for {TDM_ARCH}:\n{self.error}")
        sym = tdm_probe_symbol(algo, red, ty)
        r = self.results.get(sym)
        if r is None:
            self.fail(f"{sym} not found in the {TDM_ARCH} asm")
        return r

    def test_tdm_path_is_compiled(self):
        """Without TDM in TmaLD the overlap check below proves nothing."""
        for red, ty in TDM_CHECKED:
            with self.subTest(red=red, ty=ty):
                tma = self._result("TmaLD", red, ty)
                if not (tma["region_loops"] and tma["tdm_loads"] and tma["lds_reads"]):
                    self.fail(
                        f"ReduceScatter_TmaLD {red} {ty}: no deep loop with "
                        f"tensor_load_to_lds and LDS reads; {_tdm_summary(tma)}\n"
                        "The toolchain supports TDM, so the TDM path was compiled "
                        "out (NCCL_SYMK_TILE_TDM, TDM_SUPPORTED)."
                    )
                if self._result("LD", red, ty)["tdm_loads_all"]:
                    self.fail(f"ReduceScatter_LD {red} {ty} uses TDM: probe is wrong")

    def test_deep_loop_lds_reads_overlap(self):
        for red, ty in TDM_CHECKED:
            with self.subTest(red=red, ty=ty):
                r = self._result("TmaLD", red, ty)
                if r["serialized"] > MAX_SERIALIZED_READS:
                    self.fail(
                        f"ReduceScatter_TmaLD {red} {ty}: {r['serialized']} LDS "
                        "reads in the TDM deep loop are issued and waited for one "
                        f"at a time (limit {MAX_SERIALIZED_READS}); "
                        f"{_tdm_summary(r)}\nreduceDeep() holds registers the "
                        "reads need, e.g. a value cached across the loop. On the "
                        "EnableTma path, recompute it per use instead."
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
