#!/usr/bin/env python3
"""Kernel ISA guards for the symmetric kernels' peer-pointer lookups.

SymkPeerPointerIsaTest compiles a probe shaped like allgather_LL_body for
gfx942, gfx950 and gfx1250: its LL peer loop must issue no vector loads and no
waits that hold back the LL stores. SymkTdmDeepLoopIsaTest compiles the real
ReduceScatter_TmaLD and _LD kernels for gfx1250: the TDM deep loop of
reduceDeep() must keep its LDS reads in flight together.

Device code only (-S), so no GPU and no librccl.so. Needs a HIP compiler and
the hipified tree (the `hipify` phase of test/host/run_host_tests.sh). Nothing
in RCCL CI runs ctest; the copy that gates is that script's `guards` phase.

Environment (all optional):
  RCCL_BUILD_DIR              build dir with hipify/ (default: <rccl>/build)
  HIPCXX, ROCM_PATH           HIP compiler (default: $ROCM_PATH/bin/amdclang++)
  RCCL_ISA_TEST_ALLOW_SKIP=1  skip instead of fail without a compiler or tree
  RCCL_ISA_TDM_ROCM_PATH      TDM-capable ROCm for the TDM guard
  RCCL_ISA_TDM_REQUIRED=1     fail instead of skip without a TDM toolchain
  RCCL_ISA_TEST_KEEP_DIR      keep the probe sources and .s files here
"""

from __future__ import annotations

import collections
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
# A ROCm named for TDM that lacks it is a setup error, not a skip.
TDM_REQUIRED = bool(TDM_ROCM_PATH) or os.environ.get("RCCL_ISA_TDM_REQUIRED") == "1"

# gfx9 retires loads and stores through one counter (vmcnt), in issue order, so
# a wait for a per-peer load also drains the LL store issued before it. gfx1250
# has separate load and store counters and gets the gfx12 form of that check.
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
# iteration broadcasts one LL packet to every LSA peer. The real kernels' loops
# also load user data, so only a probe that isolates the peer-pointer lookup
# can hold an exact limit of 0. Not covered: bcast() inlined into a callee that
# takes the comm by reference (gfx1250 fp8), which reloads it after each store.
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
# LLVM verbose-asm loop comments on and after a block's label.
_HEADER_RE = re.compile(r"This (?:Inner )?Loop Header: Depth=\d+")
_IN_LOOP_RE = re.compile(r"in Loop: Header=(BB\d+_\d+) Depth=\d+")
_PARENT_RE = re.compile(r"Parent Loop (BB\d+_\d+) Depth=\d+")
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


def _loop_instructions(body):
    """Parse the loop comments of a function body.

    Returns [(innermost loop or None, op, operands)] per instruction,
    {loop: parent loop or None}, and per loop header, in text order, the index
    of the first instruction after its label.
    """
    insns, parent, parents, headers = [], {}, [], []
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
            headers.append(len(insns))
        elif mi:
            loop = mi.group(1)
    return insns, parent, headers


def _wait_keep(op, operands, counter):
    """What an s_wait_* leaves outstanding on `counter`, or None if it is not waited on.

    s_wait_{load,store}cnt_dscnt pack load/storecnt in SIMM16[13:8], dscnt in [5:0].
    """
    counters = op.split("_")[2:] if op.startswith("s_wait_") else []
    if counter not in counters:
        return None
    imm = int(operands.split(";")[0].split(",")[0].strip(), 0)
    shift = 8 if len(counters) == 2 and counter != "dscnt" else 0
    return (imm >> shift) & 0x3F


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
    insns, _, headers = _loop_instructions(body)
    res = {
        "loops": len(headers),
        "loop_stores": 0,
        "loop_loads": [],
        "stall_waits": [],
        "calls": [],
    }
    back_edges = collections.Counter(headers)
    vm_queue = []  # gfx9 vmcnt events, oldest first: True for a store
    load_queue = []  # gfx12 loadcnt events, oldest first: True if in a loop
    for i, (loop, op, operands) in enumerate(insns):
        vm_queue += [True] * back_edges[i]
        in_loop = loop is not None
        text = f"{op} {operands}".rstrip()
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
        else:
            keep = _wait_keep(op, operands, "loadcnt")
            if keep is not None and any(_retire(load_queue, keep)) and in_loop:
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


def _compile(cmd, out):
    """Run `cmd`, which writes asm to `out`: (returncode, asm, command + output)."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    asm = ""
    if proc.returncode == 0:
        with open(out) as f:
            asm = f.read()
    return proc.returncode, asm, " ".join(cmd) + "\n" + proc.stdout + proc.stderr


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
    return _compile(cmd, out)


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

    def test_gfx12_combined_wait_reads_loadcnt_from_bits_13_8(self):
        asm = textwrap.dedent("""\
            SymkPeerPointerProbe:       ; @SymkPeerPointerProbe
            \tglobal_load_b64 v[8:9], v0, s[20:21]
            .LBB0_1:                    ; =>This Inner Loop Header: Depth=1
            \tglobal_load_b64 v[2:3], v57, s[80:81]
            \tglobal_load_b64 v[4:5], v57, s[80:81] offset:8
            \ts_wait_loadcnt_dscnt 0x201
            \ts_wait_loadcnt_dscnt 0x100
            \ts_wait_storecnt_dscnt 0x0
            \tglobal_store_b128 v[2:3], v[42:45], off th:TH_STORE_NT
            \ts_cbranch_scc1 .LBB0_1
            .Lfunc_end0:
            """)
        r = analyze(asm, PROBE_KERNEL)
        # loadcnt 2 retires only the load from before the loop, loadcnt 1 the
        # first in-loop one. s_wait_storecnt_dscnt does not wait on loads.
        self.assertEqual(r["stall_waits"], ["s_wait_loadcnt_dscnt 0x100"])

    def test_wait_keep_decodes_simm16(self):
        self.assertEqual(_wait_keep("s_wait_loadcnt_dscnt", "0x201", "loadcnt"), 2)
        self.assertEqual(_wait_keep("s_wait_loadcnt_dscnt", "0x201", "dscnt"), 1)
        self.assertEqual(_wait_keep("s_wait_storecnt_dscnt", "0x100", "dscnt"), 0)
        self.assertIsNone(_wait_keep("s_wait_storecnt_dscnt", "0x100", "loadcnt"))
        self.assertEqual(_wait_keep("s_wait_loadcnt", "0x3", "loadcnt"), 3)
        self.assertIsNone(_wait_keep("s_wait_loadcnt", "0x0", "dscnt"))
        self.assertIsNone(_wait_keep("s_waitcnt", "vmcnt(0)", "loadcnt"))

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


# ---- TDM deep-loop guard ----

# reduceDeep() with EnableTma stages each peer tile into LDS with
# tensor_load_to_lds, then every lane reads it back with ds_load. The reads
# overlap only if the whole inlined kernel leaves registers to keep several in
# flight, so this compiles the real kernels rather than a probe.
TDM_ARCH = "gfx1250"
# ENABLE_FAULT_INJECTION comes with the FAULT_INJECTION option (ON by default),
# and device/common.h needs it.
TDM_DEVICE_FLAGS = DEVICE_FLAGS + ("-DENABLE_FAULT_INJECTION",)
# (red, type) pairs checked; names and template arguments as symmetric/generate.py.
# bf16 and fp8 already read one LDS value at a time, so they are not checked.
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
    names = {
        "vgpr": "NumVgprs",
        "sgpr": "TotalNumSgprs",
        "occupancy": "Occupancy",
        "scratch": "ScratchSize",
    }
    res = {}
    for key, name in names.items():
        m = re.compile(r"^; %s: (\d+)" % name, re.M).search(asm, max(end, 0))
        if m and end >= 0:
            res[key] = int(m.group(1))
    return res


def _loop_nest(parent, root):
    nest = {root}
    while True:
        more = {c for c, p in parent.items() if p in nest} - nest
        if not more:
            return nest
        nest |= more


def analyze_tdm_deep_loop(asm, symbol):
    """Measure the TDM deep loop of `symbol`, or return None if it is absent.

    The deep loop is every loop that reads LDS and has a tensor_load_to_lds in
    its loop nest, with its child loops. The asm is walked in text order and DS
    events retire in issue order, as in analyze(). Only `serialized` and the TDM
    path are asserted; unrelated codegen moves the rest. Returns a dict:
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
    insns, parent, _ = _loop_instructions(body)
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
        keep = _wait_keep(op, operands, "dscnt")
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
    return _compile(_tdm_hip_cmd("-S", *flags, "-o", out, src), out)


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
