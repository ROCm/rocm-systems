# *************************************************************************
#  * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Regression guards for librccl_device.ltoir (NCCL_EMIT_LTO_IR).

The LTO IR artifact is a *variant* of the bitcode library: the same
translation unit, the same public C surface, the same defines plus one, but
compiled with

    -O3 -Xclang -disable-llvm-passes

so the optimizer has not run and inlining is deferred to the consumer's
device link. That is the AMDGPU analogue of what `nvcc -dlto --ltoir` gives
on the NVIDIA side: unoptimized, but still fully *optimizable*. Every test
here exists because some plausible-looking change to the recipe silently
destroys one of those properties and nothing else would notice.

Specifically:

  * -O0 would stamp `optnone` on every function, which permanently forbids
    the downstream link-time inliner. That is strictly worse than the
    unoptimized IR we want, and it cannot be undone by the consumer.
    -fno-inline and -finline-hint-functions stamp `noinline` for the same
    net effect.                                            -> test_no_optnone
  * Adding `inline` or `globaldce` to the opt invocation turns the artifact
    back into something equivalent to librccl_device.bc, with no error and
    no change in symbol surface.                           -> test_inlining_is_deferred
  * Dropping `internalize` leaks every non-static C++ header function as an
    external mangled symbol.                               -> test_exports_are_unmangled
  * Dropping -D__clang_llvm_bitcode_lib__ removes NCCL_IR_EXPORT's
    `used, retain`, so there is no @llvm.used root set and internalize
    strips the entire public surface; it also re-enables the
    caller-owned-LDS path in nccl_device/coop.h. A #error in
    bindings/ir/nccl_device_wrapper.h now rejects that combination at
    compile time.                                          -> test_ltoir_macro_requires_bitcode_lib_macro

Measured baselines, gfx950 / ROCm 7.2.4, develop @ de35f5278b. These are
what the thresholds below are calibrated against; they are recorded so a
future reader can tell a real regression from drift in the ROCm toolchain.

    metric              .bc       .ltoir raw   .ltoir (shipped)
    size bytes          2145004   12564356     12455332
    define              99        4742         4742
    define internal     17        81           4660
    defined T symbols   82        213          82
    mangled ^_Z in T    0         50           0
    addrspace(3)        0         4            4
    optnone             0         0            0
    noinline            1         2            2
    alwaysinline        82        151          151
    thunks alwaysinline 82/82     --           0/82

Two of those numbers deserve a note.

`noinline` is 1 in the .bc and 2 in the .ltoir, and neither is RCCL code:
ROCm's own __assert_fail is noinline in its own source, and the .ltoir also
retains __assertfail. So the noinline check is *differential* against the
.bc, never absolute. `optnone` stays absolute -- zero is zero.

`addrspace(3)` is 4 in the .ltoir and 0 in the .bc. The four are two
internal-linkage function-local LDS stashes belonging to
ncclCoopBcast_{WarpSpan,Cta}_stash() in nccl_device/coop.h. They are dead --
no exported thunk reaches them -- and they survive only because this recipe
deliberately runs no DCE. A consumer's LTO link removes them. So the LDS
check is a *reachability* assertion, not a count. See test_no_reachable_lds.
"""

import os
import re
import subprocess

import pytest

pytestmark = pytest.mark.lto_ir


# ---------------------------------------------------------------------------
# LLVM IR inspection helpers
# ---------------------------------------------------------------------------

# Function attributes this module reasons about. All three are function
# attributes only, never parameter or return attributes, so matching them as
# bare words on a `define` line cannot pick up a false positive.
_INLINE_ATTRS = ("optnone", "noinline", "alwaysinline")

_ATTR_GROUP_RE = re.compile(r"^attributes\s+(#\d+)\s*=\s*\{(.*)\}\s*$")
_DEFINE_NAME_RE = re.compile(r"@(\"[^\"]+\"|[\w.$]+)\s*\(")
_ATTR_GROUP_REF_RE = re.compile(r"#\d+")
_WORD_RE = re.compile(r"[A-Za-z_][\w.-]*")


def _run(argv, **kwargs):
    """Run argv, returning CompletedProcess with text output captured."""
    return subprocess.run(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, **kwargs
    )


def _tool(llvm_bin, name):
    return os.path.join(llvm_bin, name)


def _defined_text_symbols(llvm_bin, artifact):
    """Return the set of defined code symbols (nm type 'T') in artifact."""
    proc = _run([_tool(llvm_bin, "llvm-nm"), "--defined-only", artifact])
    assert proc.returncode == 0, f"llvm-nm failed on {artifact}:\n{proc.stdout}"
    syms = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        # "<addr> <type> <name>" or "<type> <name>" when there is no address.
        if len(parts) >= 2 and parts[-2] == "T":
            syms.add(parts[-1])
    return syms


def _disassemble(llvm_bin, artifact, out_ll):
    proc = _run([_tool(llvm_bin, "llvm-dis"), artifact, "-o", out_ll])
    assert proc.returncode == 0, (
        f"llvm-dis failed on {artifact}:\n{proc.stdout}"
    )
    return out_ll


class ModuleFacts:
    """Per-function attribute census of a textual LLVM module.

    Attribute *groups* must be resolved before counting: clang emits
    `define ... #0 {` plus a single `attributes #0 = { alwaysinline ... }`
    line, so grepping the raw text for "alwaysinline" counts group
    definitions, not functions. Two passes: collect the groups, then expand
    each define's `#N` references.
    """

    def __init__(self, ll_path):
        self.path = ll_path
        self.groups = {}
        self.define_lines = []
        self.n_defines = 0
        self.n_internal_defines = 0
        self.n_addrspace3_globals = 0

        with open(ll_path, "r", errors="replace") as fh:
            for line in fh:
                if line.startswith("define "):
                    self.n_defines += 1
                    self.define_lines.append(line)
                    if " internal " in line:
                        self.n_internal_defines += 1
                elif line.startswith("attributes "):
                    m = _ATTR_GROUP_RE.match(line.rstrip("\n"))
                    if m:
                        self.groups[m.group(1)] = set(
                            _WORD_RE.findall(m.group(2)))
                elif line.startswith("@") and "addrspace(3)" in line:
                    self.n_addrspace3_globals += 1

        self.attrs = {}
        for line in self.define_lines:
            name_m = _DEFINE_NAME_RE.search(line)
            if not name_m:
                continue
            name = name_m.group(1).strip('"')
            found = {w for w in _WORD_RE.findall(line) if w in _INLINE_ATTRS}
            for ref in _ATTR_GROUP_REF_RE.findall(line):
                found |= self.groups.get(ref, set()) & set(_INLINE_ATTRS)
            self.attrs[name] = found

    def with_attr(self, attr):
        return {n for n, a in self.attrs.items() if attr in a}

    def body_of(self, func):
        """Return the instruction lines of `func`'s body, or None."""
        want = f"@{func}("
        collecting = False
        body = []
        with open(self.path, "r", errors="replace") as fh:
            for line in fh:
                if not collecting:
                    if line.startswith("define ") and want in line:
                        collecting = True
                    continue
                if line.startswith("}"):
                    return body
                stripped = line.strip()
                if stripped and not stripped.startswith(";"):
                    body.append(stripped)
        return None


@pytest.fixture(scope="session")
def facts(ir_artifacts, tmp_path_factory):
    """Disassemble both artifacts once and build a census of each."""
    out = tmp_path_factory.mktemp("ltoir_ll")
    bc_ll = _disassemble(
        ir_artifacts.LLVM_BIN, ir_artifacts.BITCODE, str(out / "bc.ll"))
    lto_ll = _disassemble(
        ir_artifacts.LLVM_BIN, ir_artifacts.LTO_IR, str(out / "ltoir.ll"))
    return {
        "env": ir_artifacts,
        "tmp": out,
        "bc": ModuleFacts(bc_ll),
        "ltoir": ModuleFacts(lto_ll),
        "bc_syms": _defined_text_symbols(
            ir_artifacts.LLVM_BIN, ir_artifacts.BITCODE),
        "lto_syms": _defined_text_symbols(
            ir_artifacts.LLVM_BIN, ir_artifacts.LTO_IR),
    }


# ---------------------------------------------------------------------------
# 1. Both artifacts are well-formed IR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["bc", "ltoir"])
def test_artifact_verifies(ir_artifacts, which):
    """opt -passes=verify accepts both artifacts.

    Pristine, never-optimized IR is a code path the frontend's output does
    not normally travel, so verify it explicitly rather than assume it.
    """
    artifact = ir_artifacts.BITCODE if which == "bc" else ir_artifacts.LTO_IR
    proc = _run([
        _tool(ir_artifacts.LLVM_BIN, "opt"), "-passes=verify",
        "-disable-output", artifact,
    ])
    assert proc.returncode == 0, (
        f"{os.path.basename(artifact)} failed opt -passes=verify:\n{proc.stdout}"
    )


# ---------------------------------------------------------------------------
# 2. Export surface is unmangled and RCCL-owned
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["bc_syms", "lto_syms"])
def test_exports_are_unmangled(facts, which):
    """Every exported symbol is an extern "C" nccl* thunk.

    This is what the internalize pass exists to guarantee on the LTO IR
    path. Without it the raw compile exports 213 symbols, 50 of them
    mangled C++ header functions that are no part of the published surface.
    """
    syms = facts[which]
    assert syms, f"{which}: no defined code symbols at all"
    mangled = sorted(s for s in syms if s.startswith("_Z"))
    assert not mangled, (
        f"{which}: {len(mangled)} mangled C++ symbol(s) leaked into the "
        f"export surface, e.g. {mangled[:5]}"
    )
    foreign = sorted(s for s in syms if not s.startswith("nccl"))
    assert not foreign, (
        f"{which}: {len(foreign)} exported symbol(s) outside the nccl* "
        f"namespace, e.g. {foreign[:5]}"
    )


# ---------------------------------------------------------------------------
# 3. The two artifacts publish the same surface
# ---------------------------------------------------------------------------

def test_export_surfaces_match(facts):
    """The .bc and .ltoir export exactly the same symbols.

    They are two encodings of one library. A consumer must be able to swap
    one for the other without relinking against a different surface. Note
    that the two recipes arrive here by different routes: the .bc names its
    public API explicitly via -internalize-public-api-list, the .ltoir
    relies on @llvm.used (NCCL_IR_EXPORT's `used, retain`). That they agree
    is a real property, not a tautology.
    """
    bc, lto = facts["bc_syms"], facts["lto_syms"]
    assert lto == bc, (
        "export surfaces diverged\n"
        f"  only in .bc    ({len(bc - lto)}): {sorted(bc - lto)[:10]}\n"
        f"  only in .ltoir ({len(lto - bc)}): {sorted(lto - bc)[:10]}"
    )
    # Baseline was 82. Guard the order of magnitude so a recipe that keeps
    # both surfaces identical while collapsing both to near-nothing fails.
    assert len(bc) >= 64, f"export surface shrank to {len(bc)} symbols (was 82)"


# ---------------------------------------------------------------------------
# 4. The -O0 trap
# ---------------------------------------------------------------------------

def test_no_optnone(facts):
    """No function in the .ltoir carries optnone.

    optnone is the one attribute the consumer cannot undo: it permanently
    forbids the link-time inliner. Any recipe that reaches for -O0 to get
    "unoptimized" IR lands here. Absolute, not differential -- the .bc has
    none either, and neither should.
    """
    offenders = sorted(facts["ltoir"].with_attr("optnone"))
    assert not offenders, (
        f"{len(offenders)} function(s) in librccl_device.ltoir carry optnone, "
        "which permanently forbids link-time inlining. Did the recipe switch "
        f"to -O0? e.g. {offenders[:5]}"
    )


def test_no_new_noinline(facts):
    """The .ltoir adds no noinline beyond what ROCm's own headers bring in.

    Differential on purpose. ROCm marks __assert_fail (and, in the
    unoptimized IR, __assertfail) noinline in its own source; those appear
    whatever RCCL does. What must not appear is a *new* noinline on RCCL
    code, which is what -fno-inline or -finline-hint-functions would add.
    """
    baseline = facts["bc"].with_attr("noinline")
    observed = facts["ltoir"].with_attr("noinline")
    # Allow ROCm's assert helpers, which exist only in the unoptimized IR.
    allowed = baseline | {"__assertfail", "__assert_fail"}
    new = sorted(observed - allowed)
    assert not new, (
        f"{len(new)} function(s) gained noinline in the .ltoir relative to "
        f"the .bc: {new[:10]}"
    )


# ---------------------------------------------------------------------------
# 5. Inlining really is deferred
# ---------------------------------------------------------------------------

# A thunk that is substantial after inlining and nearly empty before it.
# ncclLsaReduceSumCopy_F32 fans out into the ReduceCopy machinery, so in the
# .bc it is a large inlined body and in the .ltoir it is a short sequence
# ending in a call.
_REPRESENTATIVE_THUNK = "ncclLsaReduceSumCopy_F32"


def test_inlining_is_deferred(facts):
    """The .ltoir keeps callees as callees instead of inlining them.

    Two independent signals, because either alone has a plausible innocent
    explanation:

      * define count. Inlining plus globaldce collapses the module to the
        public surface and its survivors (99 defines measured). Deferring
        leaves every callee standing (4742 measured). A ~48x gap is not
        subtle, so the threshold is deliberately loose.
      * body size of a representative thunk, in the opposite direction.
        Deferral means the work stays *behind* calls, so the deferred body
        is SHORT and the module is LARGE. Checking only the define count
        would pass for a module that inlined everything and then kept the
        dead originals around.
    """
    n_bc = facts["bc"].n_defines
    n_lto = facts["ltoir"].n_defines
    assert n_bc > 0 and n_lto > 0
    assert n_lto > 10 * n_bc, (
        f"librccl_device.ltoir defines {n_lto} functions vs the .bc's {n_bc} "
        f"(measured baseline: 4742 vs 99). A ratio this low means the "
        "optimizer ran -- check for -O3 without -Xclang -disable-llvm-passes, "
        "or for `inline`/`globaldce` added to the opt invocation."
    )

    bc_body = facts["bc"].body_of(_REPRESENTATIVE_THUNK)
    lto_body = facts["ltoir"].body_of(_REPRESENTATIVE_THUNK)
    assert bc_body is not None, f"{_REPRESENTATIVE_THUNK} missing from the .bc"
    assert lto_body is not None, f"{_REPRESENTATIVE_THUNK} missing from the .ltoir"
    assert len(lto_body) < len(bc_body), (
        f"{_REPRESENTATIVE_THUNK} is {len(lto_body)} lines in the .ltoir but "
        f"{len(bc_body)} in the .bc. Deferred inlining should make it "
        "SHORTER, not longer -- the callees should still be callees."
    )
    assert any(" call " in ln or ln.startswith("call ") or "tail call " in ln
               for ln in lto_body), (
        f"{_REPRESENTATIVE_THUNK} has no call instruction in the .ltoir; its "
        "callees were inlined after all."
    )


def test_thunks_are_not_pre_inlined(facts):
    """The exported thunks themselves lost always_inline in the .ltoir.

    NCCL_DEVICE_INLINE expands to a bare __device__ under
    __NCCL_DEVICE_LTOIR_LIB__ rather than to __device__
    __attribute__((always_inline)). Measured: 82/82 thunks alwaysinline in
    the .bc, 0/82 in the .ltoir. If this flips back, the macro gate in
    bindings/ir/nccl_device_wrapper.h regressed.
    """
    exported = facts["lto_syms"]
    always = facts["ltoir"].with_attr("alwaysinline")
    offenders = sorted(exported & always)
    assert not offenders, (
        f"{len(offenders)} of the {len(exported)} exported thunks are still "
        f"alwaysinline in the .ltoir, e.g. {offenders[:5]}. The "
        "__NCCL_DEVICE_LTOIR_LIB__ branch of NCCL_DEVICE_INLINE is not in "
        "effect."
    )
    # Sanity: the same thunks ARE alwaysinline in the .bc, so the check above
    # is distinguishing the two recipes rather than passing vacuously.
    bc_always = facts["bc"].with_attr("alwaysinline")
    assert facts["bc_syms"] & bc_always, (
        "no exported thunk is alwaysinline in the .bc either -- the "
        "alwaysinline census is not measuring what it claims to"
    )


# ---------------------------------------------------------------------------
# 6. Group-segment (LDS) memory is unreachable from the public surface
# ---------------------------------------------------------------------------

def test_no_reachable_lds(facts):
    """No addrspace(3) global is reachable from an exported thunk.

    Background: nccl_device/coop.h gates a caller-owned-LDS path on
    __clang_llvm_bitcode_lib__. With that macro absent, the exported thunks
    own their own group segment, lld rejects "local memory global used by
    non-kernel function", and a consumer that does link it hits HSA
    exception 0x1016 at dispatch.

    The .bc has zero addrspace(3) globals. The .ltoir has four, and that is
    expected: two internal-linkage function-local stashes in
    ncclCoopBcast_{WarpSpan,Cta}_stash() that no exported thunk reaches, kept
    alive only because this recipe deliberately runs no DCE. So an absolute
    count would be wrong. Instead, use globaldce as a reachability oracle:
    run internalize+globaldce as a DIAGNOSTIC (never the shipped recipe) and
    require that every addrspace(3) global disappears while the full export
    surface survives. If one survives globaldce, something exported reaches
    it, and that is the real defect.
    """
    env, tmp = facts["env"], facts["tmp"]

    assert facts["bc"].n_addrspace3_globals == 0, (
        f"librccl_device.bc has {facts['bc'].n_addrspace3_globals} "
        "addrspace(3) global(s); it should have none"
    )

    dce = str(tmp / "ltoir.dce.bc")
    proc = _run([
        _tool(env.LLVM_BIN, "opt"), "--passes=internalize,globaldce",
        env.LTO_IR, "-o", dce,
    ])
    assert proc.returncode == 0, f"diagnostic opt run failed:\n{proc.stdout}"

    dce_ll = _disassemble(env.LLVM_BIN, dce, str(tmp / "ltoir.dce.ll"))
    dce_facts = ModuleFacts(dce_ll)

    # The oracle is only meaningful if globaldce kept the public surface.
    surviving = _defined_text_symbols(env.LLVM_BIN, dce)
    assert surviving == facts["lto_syms"], (
        "the globaldce reachability probe removed exported symbols, so it "
        "cannot be used to judge reachability: "
        f"lost {sorted(facts['lto_syms'] - surviving)[:10]}"
    )

    assert dce_facts.n_addrspace3_globals == 0, (
        f"{dce_facts.n_addrspace3_globals} addrspace(3) global(s) in "
        "librccl_device.ltoir survive globaldce, meaning an exported thunk "
        "reaches group-segment memory. A consumer linking this will hit "
        "lld's 'local memory global used by non-kernel function' or HSA "
        "exception 0x1016. Check that -D__clang_llvm_bitcode_lib__ is still "
        "on the LTO IR compile line."
    )


def test_ltoir_macro_requires_bitcode_lib_macro(ir_artifacts, paths, tmp_path):
    """__NCCL_DEVICE_LTOIR_LIB__ alone is rejected at compile time.

    The LTO IR build is a variant of the bitcode-library build, never a
    replacement for it. Defining only __NCCL_DEVICE_LTOIR_LIB__ breaks two
    independent things -- the caller-owned-LDS path above, and
    NCCL_IR_EXPORT's `used, retain`, without which internalize has no root
    set and strips the whole public surface. bindings/ir/nccl_device_wrapper.h
    carries an #error for exactly this; this test is what keeps the #error
    honest.
    """
    out = str(tmp_path / "defective.bc")
    argv = [
        os.path.join(paths.LLVM_BIN, "clang++"),
        "-c", "-emit-llvm", "-O3", "-Xclang", "-disable-llvm-passes",
        "-std=gnu++17", "-x", "hip",
        f"--rocm-path={paths.ROCM_PATH}",
        "--offload-device-only",
        f"--offload-arch={paths.ARCH}",
        f"-I{paths.GENERATED_INC}",
        f"-I{paths.HIPIFY_INC}",
        f"-I{os.path.join(paths.HIPIFY_INC, 'nccl_device')}",
        f"-I{os.path.join(os.path.dirname(paths.HIPIFY_INC), 'device')}",
        f"-I{os.path.join(paths.ROCM_PATH, 'include')}",
        # Deliberately WITHOUT -D__clang_llvm_bitcode_lib__.
        "-D__NCCL_DEVICE_LTOIR_LIB__",
        "-D__CUDACC_EXTENDED_LAMBDA__",
        "-DNCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1",
        "-D__HIP_PLATFORM_AMD__=1",
        os.path.join(paths.IR_DIR, "nccl_device_wrapper__impl.h"),
        "-o", out,
    ]
    proc = _run(argv, cwd=paths.IR_DIR)
    assert proc.returncode != 0, (
        "compiling with __NCCL_DEVICE_LTOIR_LIB__ but without "
        "__clang_llvm_bitcode_lib__ succeeded; the #error guard in "
        "bindings/ir/nccl_device_wrapper.h is gone"
    )
    assert "__NCCL_DEVICE_LTOIR_LIB__ must be defined together with" in proc.stdout, (
        "the compile failed, but not on the #error guard. Output:\n"
        + proc.stdout[-4000:]
    )


# ---------------------------------------------------------------------------
# 7. The artifact is actually consumable
# ---------------------------------------------------------------------------

@pytest.mark.ir_device
def test_lto_ir_links_and_runs(run_lto_ir_gtest, summarize_gtest):
    """Link the .ltoir into the IR_test consumer and run it on a GPU.

    Tests 1-6 are static. This one is the only end-to-end evidence that
    pristine, never-optimized IR survives the AMDGPU device link at all --
    lld has to do the inlining the .bc had already done, against real call
    sites. The link recipe is byte-for-byte the one used for the .bc
    (-Xoffload-linker <lib> plus -amdgpu-internalize-symbols=false); only
    the library differs.

    Runtime tests needing a live ncclDevComm GTEST_SKIP themselves, exactly
    as they do on the .bc path. summarize_gtest fails on any failure and
    reports SKIPPED, not PASSED, if every case skipped.
    """
    proc, log = run_lto_ir_gtest("*", "ir_test_ltoir_run.log")
    summarize_gtest(proc, log, "IR_test linked against librccl_device.ltoir")
