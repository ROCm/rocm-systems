#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Automated test suite for the SQTT instrumentation pass.

Compiles test/markers/kernels/auto.cpp with various env var configurations and verifies:
- IR patterns (readlane loops, ptrtoint, exec mask)
- Assembly patterns (v_readlane_b32, s_ttracedata)
- Funcmap entries in the .sqtt_funcmap ELF section
- Error messages for invalid configurations
- Regression: existing features still work after changes

Usage:
    python3 test/markers/run_tests.py           # run from project root
    python3 test/markers/run_tests.py -v        # verbose (show compiler output on failure)
    python3 test/markers/run_tests.py -k lds    # run only tests matching "lds"
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Resolve project paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BUILD_DIR = os.environ.get("SQTT_BUILD_DIR", os.path.join(PROJECT_ROOT, "build"))
PASS_PLUGIN = os.environ.get(
    "SQTT_PASS_PLUGIN", os.path.join(BUILD_DIR, "lib", "libsqttinstrumentpass.so"))
INCLUDE_DIR = os.path.join(PROJECT_ROOT, "include", "rocprof_trace_decoder", "cxx")
TEST_SOURCE = os.path.join(SCRIPT_DIR, "kernels", "auto.cpp")
ADDR_TRACE_SOURCE = os.path.join(SCRIPT_DIR, "kernels", "addr_trace.cpp")
MARKER_SOURCE = os.path.join(SCRIPT_DIR, "kernels", "marker.cpp")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# Add scripts/ to path for find_llvm_tool
sys.path.insert(0, SCRIPTS_DIR)
from sqtt_data import find_llvm_tool


def find_hipcc():
    """Find hipcc compiler."""
    if os.environ.get("HIPCC"):
        return os.environ["HIPCC"]
    found = shutil.which("hipcc")
    if found:
        return found
    rocm_hipcc = "/opt/rocm/bin/hipcc"
    return rocm_hipcc if os.path.isfile(rocm_hipcc) else None


def build_pass():
    """Build the pass plugin if needed."""
    if os.environ.get("SQTT_PASS_PLUGIN"):
        return os.path.isfile(PASS_PLUGIN), f"missing SQTT_PASS_PLUGIN={PASS_PLUGIN}"
    r = subprocess.run(
        ["cmake", "--build", BUILD_DIR],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120)
    return r.returncode == 0, r.stderr


def compile_test(env_vars: dict, flags: list[str], output: str,
                 hipcc: str, timeout: int = 120,
                 source: str = None) -> subprocess.CompletedProcess:
    """Compile test source with given env vars and flags."""
    env = os.environ.copy()
    env.update(env_vars)

    src = source or TEST_SOURCE
    cmd = [hipcc]
    if env_vars.get("_SQTT_ENABLED", "1") != "0":
        cmd.append("-DSQTT_ENABLED=1")
    cmd += [
        f"-fpass-plugin={PASS_PLUGIN}",
        f"-I{INCLUDE_DIR}",
    ] + flags + [src, "-o", output]

    return subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=timeout,
        cwd=PROJECT_ROOT)


def extract_funcmap(binary: str, tmpdir: str) -> str:
    """Extract .sqtt_funcmap from a compiled binary."""
    # Step 1: extract code object
    objdump = find_llvm_tool("llvm-objdump")
    if not objdump:
        return ""
    r = subprocess.run(
        [objdump, "--offloading", binary],
        capture_output=True, text=True, cwd=tmpdir, timeout=30)
    if r.returncode != 0:
        return ""

    # Find extracted code object
    import glob
    cos = glob.glob(os.path.join(tmpdir, "*.hip*-amdgcn-amd-amdhsa--*"))
    if not cos:
        # llvm-objdump extracts relative to cwd or next to the binary
        base = os.path.basename(binary)
        cos = glob.glob(os.path.join(tmpdir, f"{base}*.hip*-amdgcn-amd-amdhsa--*"))
    if not cos:
        # Try in the binary's directory
        bdir = os.path.dirname(binary)
        cos = glob.glob(os.path.join(bdir, f"{os.path.basename(binary)}*.hip*-amdgcn-amd-amdhsa--*"))
    if not cos:
        return ""

    # Step 2: dump funcmap section
    objcopy = find_llvm_tool("llvm-objcopy")
    if not objcopy:
        return ""
    funcmap_path = os.path.join(tmpdir, "funcmap.txt")
    r = subprocess.run(
        [objcopy, f"--dump-section=.sqtt_funcmap={funcmap_path}", cos[0]],
        capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        return ""
    try:
        with open(funcmap_path) as f:
            return f.read()
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

TESTS = [
    # --- SQTT_MEM_BARRIER tri-state tests (run first so a regression here
    #     is the first failure you see) ---
    {
        "name": "mem_barrier_default_is_fence",
        "desc": "Default (unset SQTT_MEM_BARRIER) emits workgroup acq_rel fence",
        "env": {"SQTT_INSTRUMENT_BARRIERS": "1"},
        "mode": "ir",
        "expect_ir": [
            r'fence syncscope\("workgroup"\) acq_rel',
            r"amdgpu-synchronize-as",
            r"!\"local\"",
            r"ttracedata",
        ],
        "reject_ir": [
            r'call void asm sideeffect "", "~\{memory\}"',
        ],
        "expect_funcmap": [],
        "reject_funcmap": [],
    },
    {
        "name": "mem_barrier_explicit_fence",
        "desc": "SQTT_MEM_BARRIER=fence emits workgroup acq_rel fence",
        "env": {
            "SQTT_INSTRUMENT_BARRIERS": "1",
            "SQTT_MEM_BARRIER": "fence",
        },
        "mode": "ir",
        "expect_ir": [
            r'fence syncscope\("workgroup"\) acq_rel',
            r"amdgpu-synchronize-as",
            r"!\"local\"",
        ],
        "reject_ir": [
            r'call void asm sideeffect "", "~\{memory\}"',
        ],
        "expect_funcmap": [],
        "reject_funcmap": [],
    },
    {
        "name": "mem_barrier_asm_clobber",
        "desc": "SQTT_MEM_BARRIER=asm emits inline-asm memory clobber, no acq_rel fence",
        "env": {
            "SQTT_INSTRUMENT_BARRIERS": "1",
            "SQTT_MEM_BARRIER": "asm",
        },
        "mode": "ir",
        "expect_ir": [
            r'call void asm sideeffect "", "~\{memory\}"',
            r"ttracedata",
        ],
        # Only reject acq_rel workgroup fences — __syncthreads() emits its
        # own release/acquire workgroup fences and those must be allowed.
        "reject_ir": [
            r'fence syncscope\("workgroup"\) acq_rel',
        ],
        "expect_funcmap": [],
        "reject_funcmap": [],
    },
    {
        "name": "mem_barrier_none",
        "desc": "SQTT_MEM_BARRIER=none emits neither acq_rel fence nor asm clobber",
        "env": {
            "SQTT_INSTRUMENT_BARRIERS": "1",
            "SQTT_MEM_BARRIER": "none",
        },
        "mode": "ir",
        "expect_ir": [
            r"ttracedata",
        ],
        # See note above: __syncthreads emits release/acquire workgroup
        # fences of its own; only acq_rel is the SQTT-emitted form.
        "reject_ir": [
            r'fence syncscope\("workgroup"\) acq_rel',
            r'call void asm sideeffect "", "~\{memory\}"',
        ],
        "expect_funcmap": [],
        "reject_funcmap": [],
    },
    {
        "name": "mem_barrier_numeric_2_is_fence",
        "desc": "SQTT_MEM_BARRIER=2 (numeric) is equivalent to fence",
        "env": {
            "SQTT_INSTRUMENT_BARRIERS": "1",
            "SQTT_MEM_BARRIER": "2",
        },
        "mode": "ir",
        "expect_ir": [
            r'fence syncscope\("workgroup"\) acq_rel',
            r"amdgpu-synchronize-as",
            r"!\"local\"",
        ],
        "reject_ir": [],
        "expect_funcmap": [],
        "reject_funcmap": [],
    },
    {
        "name": "addr_trace_memory",
        "desc": "Address tracing for global/flat memory ops",
        "env": {"SQTT_TRACE_ADDRESSES": "memory"},
        "mode": "ir+obj",
        "expect_ir": [
            r"ptrtoint.*i64",
            r"exec_lo",
            r"exec_hi",
            r"readlane",
        ],
        "reject_ir": [],
        "expect_funcmap": ["addr_trace_load", "addr_trace_store"],
        "reject_funcmap": ["addr_trace_lds"],
    },
    {
        "name": "addr_trace_lds",
        "desc": "Address tracing for LDS memory ops",
        "env": {"SQTT_TRACE_ADDRESSES": "lds"},
        "mode": "ir+obj",
        "expect_ir": [
            r"ptrtoint ptr addrspace\(3\).*to i32",
            r"readlane",
        ],
        "reject_ir": [],
        "expect_funcmap": ["addr_trace_lds_load", "addr_trace_lds_store"],
        "reject_funcmap": ["P:1:addr_trace_load", "P:2:addr_trace_store"],
    },
    {
        "name": "addr_trace_both",
        "desc": "Address tracing for both memory and LDS",
        "env": {"SQTT_TRACE_ADDRESSES": "memory,lds"},
        "mode": "obj",
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": [
            "addr_trace_load", "addr_trace_store",
            "addr_trace_lds_load", "addr_trace_lds_store",
        ],
        "reject_funcmap": [],
    },
    {
        "name": "addr_trace_no_scope",
        "desc": "Address tracing without scope checks (all CUs/SIMDs)",
        "env": {
            "SQTT_TRACE_ADDRESSES": "memory",
            "SQTT_SCOPE_CU": "-1",
            "SQTT_SCOPE_SIMD": "-1",
        },
        "mode": "ir",
        "expect_ir": [
            r"readlane",
        ],
        "reject_ir": [
            r"sqtt\.addr\.trace",  # scope-check BB name
        ],
        "expect_funcmap": [],
        "reject_funcmap": [],
    },
    {
        "name": "mutual_exclusion",
        "desc": "SQTT_TRACE_ADDRESSES and SQTT_INSTRUMENT_MEMORY conflict",
        "env": {
            "SQTT_TRACE_ADDRESSES": "memory",
            "SQTT_INSTRUMENT_MEMORY": "2:5",
        },
        "mode": "stderr",
        "expect_ir": [],
        "reject_ir": [],
        "expect_stderr": [r"mutually exclusive"],
        "expect_funcmap": [],
        "reject_funcmap": [],
    },
    {
        "name": "func_barriers_regression",
        "desc": "Function + barrier instrumentation still works",
        "env": {
            "SQTT_INSTRUMENT_FUNCTIONS": "10",
            "SQTT_INSTRUMENT_BARRIERS": "1",
        },
        "mode": "obj",
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": ["F:", "barrier_signal", "barrier_wait", "barrier"],
        "reject_funcmap": ["addr_trace"],
    },
    {
        "name": "memory_ops_regression",
        "desc": "Memory op markers still work",
        "env": {"SQTT_INSTRUMENT_MEMORY": "2:5"},
        "mode": "obj",
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": ["vmem_load", "vmem_store"],
        "reject_funcmap": ["addr_trace"],
    },
    {
        "name": "disabled",
        "desc": "No instrumentation when SQTT_ENABLED is not set",
        "env": {"_SQTT_ENABLED": "0"},  # special flag: don't pass -DSQTT_ENABLED=1
        "mode": "ir_no_plugin",
        "expect_ir": [],
        "reject_ir": [r"ttracedata"],
        "expect_funcmap": [],
        "reject_funcmap": [],
    },
    {
        "name": "asm_check",
        "desc": "Assembly output has expected ISA instructions",
        "env": {"SQTT_TRACE_ADDRESSES": "memory"},
        "mode": "asm",
        "expect_ir": [  # reuse field for asm patterns
            r"v_readlane_b32",
            r"s_ttracedata\b",
        ],
        "reject_ir": [],
        "expect_funcmap": [],
        "reject_funcmap": [],
    },
    {
        "name": "gfx9_exec_m0_trace_spacing",
        "desc": "gfx9: address EXEC traces use the explicit mov/nop/trace sequence",
        "env": {"SQTT_TRACE_ADDRESSES": "memory"},
        "mode": "asm",
        "flags": ["--offload-arch=gfx90a"],
        "expect_ir": [
            r"s_ttracedata\b",
        ],
        "reject_ir": [],
        "expect_funcmap": [],
        "reject_funcmap": [],
        "custom_asm_check": "gfx9_m0_trace_spacing",
    },
    {
        "name": "gfx9_full_m0_trace_nop",
        "desc": "gfx9: every full M0 trace has s_nop 0 before s_ttracedata",
        "env": {},
        "mode": "asm",
        "source": "marker",
        "flags": ["--offload-arch=gfx90a"],
        "expect_ir": [
            r"s_ttracedata\b",
        ],
        "reject_ir": [],
        "expect_funcmap": [],
        "reject_funcmap": [],
        "custom_asm_check": "m0_nop_before_ttracedata",
    },
    {
        "name": "gfx10_full_m0_trace_nop",
        "desc": "gfx10: every full M0 trace has s_nop 0 before s_ttracedata",
        "env": {"SQTT_TRACE_ADDRESSES": "memory"},
        "mode": "asm",
        "flags": ["--offload-arch=gfx1030"],
        "expect_ir": [
            r"s_ttracedata\b",
        ],
        "reject_ir": [],
        "expect_funcmap": [],
        "reject_funcmap": [],
        "custom_asm_check": "m0_nop_before_ttracedata",
    },
    {
        "name": "gfx12_full_m0_trace_nop",
        "desc": "gfx12: every full M0 trace has s_nop 0 before s_ttracedata",
        "env": {"SQTT_TRACE_ADDRESSES": "memory"},
        "mode": "asm",
        "flags": ["--offload-arch=gfx1200"],
        "expect_ir": [
            r"s_ttracedata\b",
        ],
        "reject_ir": [],
        "expect_funcmap": [],
        "reject_funcmap": [],
        "custom_asm_check": "m0_nop_before_ttracedata",
    },
    {
        "name": "addr_trace_unique_ids",
        "desc": "Each memory op gets a unique address trace ID",
        "env": {"SQTT_TRACE_ADDRESSES": "memory"},
        "mode": "obj",
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": ["addr_trace_load", "addr_trace_store"],
        "reject_funcmap": [],
        # Custom check: all P: IDs for addr_trace entries must be unique
        "custom_funcmap_check": "unique_addr_ids",
    },
    # --- Atomic address tracing tests ---
    {
        "name": "addr_trace_atomic_memory",
        "desc": "Address tracing for global atomics",
        "env": {"SQTT_TRACE_ADDRESSES": "memory"},
        "mode": "obj",
        "source": "addr_trace",
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": ["addr_trace_atomic"],
        "reject_funcmap": [],
    },
    {
        "name": "addr_trace_atomic_lds",
        "desc": "Address tracing for LDS atomics",
        "env": {"SQTT_TRACE_ADDRESSES": "lds"},
        "mode": "obj",
        "source": "addr_trace",
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": ["addr_trace_lds_atomic"],
        "reject_funcmap": [],
    },
    {
        "name": "addr_trace_atomic_both",
        "desc": "Address tracing for both memory and LDS atomics",
        "env": {"SQTT_TRACE_ADDRESSES": "memory,lds"},
        "mode": "obj",
        "source": "addr_trace",
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": [
            "addr_trace_atomic", "addr_trace_lds_atomic",
            "addr_trace_load", "addr_trace_store",
            "addr_trace_lds_load", "addr_trace_lds_store",
        ],
        "reject_funcmap": [],
    },
    # --- gfx12 tests ---
    {
        "name": "gfx12_func_instrumentation",
        "desc": "Function instrumentation on gfx12",
        "env": {"SQTT_INSTRUMENT_FUNCTIONS": "10", "SQTT_SHADER_CLOCK_BITS": "12"},
        "mode": "obj",
        "flags": ["--offload-arch=gfx1200"],
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": ["M:shader_clock_bits=12;shader_clock_shift=4", "F:"],
        "reject_funcmap": [],
    },
    {
        "name": "gfx12_barriers",
        "desc": "Barrier instrumentation on gfx12 (split barriers)",
        "env": {
            "SQTT_INSTRUMENT_FUNCTIONS": "10",
            "SQTT_INSTRUMENT_BARRIERS": "1",
        },
        "mode": "obj",
        "flags": ["--offload-arch=gfx1200"],
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": ["barrier"],
        "reject_funcmap": [],
    },
    {
        "name": "gfx12_addr_trace_memory",
        "desc": "Address tracing for memory ops on gfx12",
        "env": {"SQTT_TRACE_ADDRESSES": "memory"},
        "mode": "obj",
        "flags": ["--offload-arch=gfx1200"],
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": ["addr_trace_load", "addr_trace_store"],
        "reject_funcmap": [],
    },
    {
        "name": "gfx12_addr_trace_lds",
        "desc": "Address tracing for LDS ops on gfx12",
        "env": {"SQTT_TRACE_ADDRESSES": "lds"},
        "mode": "obj",
        "flags": ["--offload-arch=gfx1200"],
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": ["addr_trace_lds_load", "addr_trace_lds_store"],
        "reject_funcmap": [],
    },
    {
        "name": "gfx12_memory_ops",
        "desc": "Memory op markers on gfx12",
        "env": {"SQTT_INSTRUMENT_MEMORY": "2:5"},
        "mode": "obj",
        "flags": ["--offload-arch=gfx1200"],
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": ["vmem_load", "vmem_store"],
        "reject_funcmap": [],
    },
    {
        "name": "gfx12_asm_shader_clock",
        "desc": "gfx12 packs shader clock bits into full s_ttracedata markers",
        "env": {"SQTT_INSTRUMENT_FUNCTIONS": "10", "SQTT_SHADER_CLOCK_BITS": "12"},
        "mode": "asm",
        "flags": ["--offload-arch=gfx1200"],
        "expect_ir": [
            r"s_getreg_b32\b",
            r"s_ttracedata\b",
        ],
        "reject_ir": [r"s_ttracedata_imm\b"],
        "expect_funcmap": [],
        "reject_funcmap": [],
    },
    {
        "name": "gfx1250_asm_shader_clock",
        "desc": "gfx1250 uses s_get_shader_cycles_u64 for packed shader clocks",
        "env": {"SQTT_INSTRUMENT_FUNCTIONS": "10", "SQTT_SHADER_CLOCK_BITS": "12"},
        "mode": "asm",
        "flags": ["--offload-arch=gfx1250"],
        "expect_ir": [
            r"s_get_shader_cycles_u64\b",
            r"s_ttracedata\b",
        ],
        "reject_ir": [
            r"HW_REG_SHADER_CYCLES_LO",
            r"s_ttracedata_imm\b",
        ],
        "expect_funcmap": [],
        "reject_funcmap": [],
        "custom_asm_check": "m0_nop_before_ttracedata",
    },
    {
        "name": "gfx12_default_no_clock_asm",
        "desc": "gfx12: default layout keeps immediate headers and dynamic payload traces",
        "env": {},
        "mode": "asm",
        "source": "marker",
        "flags": ["--offload-arch=gfx1200"],
        "expect_ir": [
            r"s_ttracedata_imm\b",
            r"v_readfirstlane_b32\b",
        ],
        "reject_ir": [r"HW_REG_SHADER_CYCLES_LO"],
        "expect_funcmap": [],
        "reject_funcmap": [],
        "custom_asm_check": "m0_nop_before_ttracedata",
    },
    {
        "name": "gfx12_default_no_clock_funcmap",
        "desc": "gfx12: default layout emits no funcmap layout row",
        "env": {},
        "mode": "obj",
        "source": "marker",
        "flags": ["--offload-arch=gfx1200"],
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": ["payload_value"],
        "reject_funcmap": ["M:shader_clock_bits"],
        "custom_funcmap_check": "data_marker_payload",
    },
    {
        "name": "gfx12_markers_no_global_inv",
        "desc": "gfx12 marker fences do not generate global cache invalidates",
        "env": {},
        "mode": "asm",
        "source": "marker",
        "flags": ["--offload-arch=gfx1200"],
        "expect_ir": [
            r"s_ttracedata",
        ],
        "reject_ir": [
            r"global_inv\b",
        ],
        "expect_funcmap": [],
        "reject_funcmap": [],
    },
    # --- User marker tests ---
    {
        "name": "user_markers",
        "desc": "Named user markers produce funcmap entries",
        "env": {},
        "mode": "obj",
        "source": "marker",
        "expect_ir": [],
        "reject_ir": [],
        "expect_funcmap": [
            "load_phase", "compute_phase", "inner_step", "checkpoint", "payload_value",
        ],
        "reject_funcmap": [],
        "custom_funcmap_check": "data_marker_payload",
    },
    {
        "name": "user_markers_ir",
        "desc": "Named user markers emit ttracedata in IR",
        "env": {},
        "mode": "ir",
        "source": "marker",
        "expect_ir": [
            r"ttracedata",
        ],
        "reject_ir": [],
        "expect_funcmap": [],
        "reject_funcmap": [],
    },
    # --- Python tooling self-tests (no compilation required) ---
    {
        "name": "perfetto_exporter_self_test",
        "desc": "sqtt_perfetto.py emits well-formed B/E/i/M events on synthetic input",
        "env": {},
        "mode": "tool_self_test",
        "tool": "sqtt_perfetto.py",
    },
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.errors: list[str] = []

    def fail(self, msg: str):
        self.passed = False
        self.errors.append(msg)


def run_test(test: dict, hipcc: str, verbose: bool) -> TestResult:
    result = TestResult(test["name"])
    mode = test["mode"]
    env_vars = {k: v for k, v in test["env"].items() if not k.startswith("_")}
    extra_flags = test.get("flags", [])

    # Resolve test source file
    source_key = test.get("source")
    source_map = {
        "addr_trace": ADDR_TRACE_SOURCE,
        "marker": MARKER_SOURCE,
    }
    source = source_map.get(source_key, TEST_SOURCE)

    with tempfile.TemporaryDirectory(prefix=f"sqtt_test_{test['name']}_") as tmpdir:
        try:
            if mode == "ir" or mode == "ir+obj":
                ir_path = os.path.join(tmpdir, f"{test['name']}.ll")
                r = compile_test(env_vars, ["-S", "-emit-llvm", "--offload-device-only"] + extra_flags,
                                 ir_path, hipcc, source=source)
                if r.returncode != 0:
                    result.fail(f"Compilation failed (exit {r.returncode})")
                    if verbose:
                        result.fail(f"stderr: {r.stderr[-500:]}")
                    return result
                with open(ir_path) as f:
                    ir_content = f.read()
                check_patterns(result, ir_content, test.get("expect_ir", []),
                               test.get("reject_ir", []), "IR", ir_path)

                if mode == "ir+obj":
                    bin_path = os.path.join(tmpdir, test["name"])
                    r = compile_test(env_vars, extra_flags, bin_path, hipcc,
                                     source=source)
                    if r.returncode != 0:
                        result.fail(f"Binary compilation failed (exit {r.returncode})")
                        if verbose:
                            result.fail(f"stderr: {r.stderr[-500:]}")
                        return result
                    funcmap = extract_funcmap(bin_path, tmpdir)
                    check_funcmap(result, funcmap, test.get("expect_funcmap", []),
                                  test.get("reject_funcmap", []))

            elif mode == "obj":
                bin_path = os.path.join(tmpdir, test["name"])
                r = compile_test(env_vars, extra_flags, bin_path, hipcc,
                                 source=source)
                if r.returncode != 0:
                    result.fail(f"Compilation failed (exit {r.returncode})")
                    if verbose:
                        result.fail(f"stderr: {r.stderr[-500:]}")
                    return result
                funcmap = extract_funcmap(bin_path, tmpdir)
                check_funcmap(result, funcmap, test.get("expect_funcmap", []),
                              test.get("reject_funcmap", []))
                if "custom_funcmap_check" in test:
                    check_custom_funcmap(result, funcmap,
                                         test["custom_funcmap_check"])

            elif mode == "asm":
                asm_path = os.path.join(tmpdir, f"{test['name']}.s")
                r = compile_test(env_vars, ["-S"] + extra_flags, asm_path,
                                 hipcc, source=source)
                if r.returncode != 0:
                    result.fail(f"Compilation failed (exit {r.returncode})")
                    if verbose:
                        result.fail(f"stderr: {r.stderr[-500:]}")
                    return result
                with open(asm_path) as f:
                    asm_content = f.read()
                check_patterns(result, asm_content, test.get("expect_ir", []),
                               test.get("reject_ir", []), "ASM", asm_path)
                if "custom_asm_check" in test:
                    check_custom_asm(result, asm_content,
                                     test["custom_asm_check"])

            elif mode == "stderr":
                bin_path = os.path.join(tmpdir, test["name"])
                r = compile_test(env_vars, ["-S", "-emit-llvm"], bin_path, hipcc)
                # May or may not fail — we just check stderr
                for pat in test.get("expect_stderr", []):
                    if not re.search(pat, r.stderr):
                        result.fail(f"Missing stderr pattern: {pat}")

            elif mode == "tool_self_test":
                # Run a scripts/*.py self-test (no compilation, no rocprofv3).
                # Used to validate Python helpers in isolation.
                tool = os.path.join(SCRIPTS_DIR, test["tool"])
                r = subprocess.run(
                    [sys.executable, tool, "--self-test"],
                    capture_output=True, text=True, timeout=30)
                if r.returncode != 0:
                    result.fail(f"Self-test failed (exit {r.returncode})")
                    if verbose:
                        result.fail(f"stderr: {r.stderr[-500:]}")

            elif mode == "ir_no_plugin":
                ir_path = os.path.join(tmpdir, f"{test['name']}.ll")
                # Compile without pass plugin or SQTT_ENABLED
                env = os.environ.copy()
                cmd = [hipcc, f"-I{INCLUDE_DIR}", "-S", "-emit-llvm", "--offload-device-only",
                       TEST_SOURCE, "-o", ir_path]
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   env=env, timeout=120, cwd=PROJECT_ROOT)
                if r.returncode != 0:
                    result.fail(f"Compilation failed (exit {r.returncode})")
                    if verbose:
                        result.fail(f"stderr: {r.stderr[-500:]}")
                    return result
                with open(ir_path) as f:
                    ir_content = f.read()
                check_patterns(result, ir_content, test.get("expect_ir", []),
                               test.get("reject_ir", []), "IR", ir_path)

        except subprocess.TimeoutExpired:
            result.fail("Compilation timed out")
        except Exception as e:
            result.fail(f"Exception: {e}")

    return result


def check_patterns(result: TestResult, content: str,
                   expect: list[str], reject: list[str],
                   kind: str, path: str):
    for pat in expect:
        if not re.search(pat, content):
            result.fail(f"Missing {kind} pattern: {pat}  (in {path})")
    for pat in reject:
        if re.search(pat, content):
            result.fail(f"Unexpected {kind} pattern found: {pat}  (in {path})")


def check_funcmap(result: TestResult, funcmap: str,
                  expect: list[str], reject: list[str]):
    if not funcmap and (expect or reject):
        result.fail("Failed to extract funcmap from binary")
        return
    for entry in expect:
        if entry not in funcmap:
            result.fail(f"Missing funcmap entry: {entry}")
    for entry in reject:
        if entry in funcmap:
            result.fail(f"Unexpected funcmap entry: {entry}")


def check_custom_asm(result: TestResult, asm: str, check: str):
    """Run custom assembly validation checks."""
    if check not in {"m0_nop_before_ttracedata", "gfx9_m0_trace_spacing"}:
        return

    # The pass emits direct M0 traces as mov/nop/trace. On gfx9, the backend
    # may instead schedule an unrelated scalar instruction in that hazard
    # slot, which is safe but means the final sequence is not contiguous.
    strict_nop = check == "m0_nop_before_ttracedata"
    insn_lines = []
    for line in asm.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";") or stripped.startswith("."):
            continue
        if stripped.endswith(":"):
            continue
        insn_lines.append(stripped)

    if strict_nop:
        trace_count = 0
        for i, line in enumerate(insn_lines):
            if not re.match(r"s_ttracedata\b", line):
                continue
            trace_count += 1
            if (i < 2 or not re.match(r"s_nop\s+0\b", insn_lines[i - 1]) or
                    not re.match(r"s_mov_b32\s+m0\b", insn_lines[i - 2])):
                before = insn_lines[max(0, i - 2):i + 1]
                result.fail(f"full M0 trace does not use mov/nop 0/trace: {before!r}")
        if trace_count == 0:
            result.fail("No full M0 trace found")
        return

    valid_sequences = 0
    for i in range(len(insn_lines) - 1):
        if not re.match(r"s_mov_b32\s+m0\b", insn_lines[i]):
            continue
        nxt = insn_lines[i + 1]
        if re.match(r"s_ttracedata\b", nxt):
            result.fail(
                f"s_mov_b32 m0 immediately followed by s_ttracedata "
                f"(no spacing): {insn_lines[i]!r} -> {nxt!r}")
            continue
        if i + 2 < len(insn_lines) and re.match(r"s_nop\s+0\b", nxt) and re.match(
                r"s_ttracedata\b", insn_lines[i + 2]):
            valid_sequences += 1

    if valid_sequences == 0:
        result.fail("No s_mov_b32 m0 -> s_nop 0 -> s_ttracedata sequence found")


def check_custom_funcmap(result: TestResult, funcmap: str, check: str):
    """Run custom funcmap validation checks."""
    if check == "unique_addr_ids":
        # Parse all P:id:addr_trace_* entries and verify IDs are unique.
        # Address trace headers must also carry nonzero extra payload metadata.
        ids = []
        payload_counts = {}
        for line in funcmap.splitlines():
            line = line.strip().rstrip("\x00")
            if line.startswith("P:") and "addr_trace_" in line:
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    ids.append(int(parts[1]))
            elif line.startswith("R:"):
                parts = line.split(":", 2)
                if len(parts) != 3:
                    continue
                try:
                    mid = int(parts[1])
                except ValueError:
                    continue
                for item in parts[2].split(";"):
                    key, sep, value = item.partition("=")
                    if sep and key == "extra_payload_count":
                        try:
                            payload_counts[mid] = int(value)
                        except ValueError:
                            pass
        if len(ids) < 2:
            result.fail(f"Expected at least 2 addr_trace entries, got {len(ids)}")
        elif len(ids) != len(set(ids)):
            result.fail(f"Addr trace IDs are not unique: {ids}")
        missing_payloads = [mid for mid in ids if payload_counts.get(mid, 0) <= 0]
        if missing_payloads:
            result.fail(
                f"Addr trace IDs missing nonzero extra_payload_count: {missing_payloads}")
    elif check == "data_marker_payload":
        payload_id = None
        payload_counts = {}
        for line in funcmap.splitlines():
            line = line.strip().rstrip("\x00")
            if line.startswith("P:") and line.endswith(":payload_value"):
                parts = line.split(":", 2)
                if len(parts) == 3:
                    payload_id = int(parts[1])
            elif line.startswith("R:"):
                parts = line.split(":", 2)
                if len(parts) != 3:
                    continue
                try:
                    mid = int(parts[1])
                except ValueError:
                    continue
                for item in parts[2].split(";"):
                    key, sep, value = item.partition("=")
                    if sep and key == "extra_payload_count":
                        try:
                            payload_counts[mid] = int(value)
                        except ValueError:
                            pass
        if payload_id is None:
            result.fail("Missing P: entry for payload_value")
        elif payload_counts.get(payload_id) != 1:
            result.fail(
                f"payload_value marker missing extra_payload_count=1 (id {payload_id})")


def main():
    parser = argparse.ArgumentParser(description="SQTT pass automated tests")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show compiler output on failure")
    parser.add_argument("-k", "--filter", default=None,
                        help="Run only tests whose name contains this string")
    args = parser.parse_args()

    # Filter tests first — some modes don't need hipcc/the pass plugin.
    tests = TESTS
    if args.filter:
        tests = [t for t in tests if args.filter in t["name"]]
        if not tests:
            print(f"No tests match filter '{args.filter}'")
            sys.exit(1)

    needs_hipcc = any(t["mode"] != "tool_self_test" for t in tests)

    hipcc = find_hipcc() if needs_hipcc else None
    if needs_hipcc and not hipcc:
        print("ERROR: hipcc not found", file=sys.stderr)
        sys.exit(1)

    if needs_hipcc and not os.path.isfile(TEST_SOURCE):
        print(f"ERROR: test source not found: {TEST_SOURCE}", file=sys.stderr)
        sys.exit(1)

    if needs_hipcc:
        # Build pass plugin
        print("Building pass plugin...", end="  ", flush=True)
        ok, err = build_pass()
        if not ok:
            print("FAIL")
            print(err, file=sys.stderr)
            sys.exit(1)
        print("OK")

    # Run tests
    results: list[TestResult] = []
    for i, test in enumerate(tests):
        tag = f"[{i+1}/{len(tests)}]"
        name = test["name"]
        dots = "." * max(1, 40 - len(name))
        print(f"{tag} {name} {dots}", end=" ", flush=True)

        r = run_test(test, hipcc, args.verbose)
        results.append(r)

        if r.passed:
            print("PASS")
        else:
            print("FAIL")
            for err in r.errors:
                print(f"  {err}")

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{passed}/{total} passed")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
