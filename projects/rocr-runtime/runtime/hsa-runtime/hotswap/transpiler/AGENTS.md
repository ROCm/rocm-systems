# AGENTS.md — Salmon Transpiler

Read this before making any change in this directory. It is the minimum
context an agent needs to work on Salmon without regressing the project.

## What this is

**Salmon** is an AMDGPU binary transpiler. It translates native AMDGPU machine
code to LLVM IR, then re-lowers it through the stock LLVM AMDGPU backend to
a different ISA (cross-family, e.g. gfx1250→gfx942, or same-family, e.g.
gfx950→gfx942). It is the `HSA_HOTSWAP_IR_RAISER=1` path inside the ROCR
runtime's hotswap subsystem. `README.md` has the full build and run
instructions; this file is the design-and-hygiene brief.

Salmon is a standalone CMake project living under `hotswap/transpiler/`.
It links against a pre-built LLVM **build tree** (not an install tree) to
reach the AMDGPU target-private headers and TableGen-generated `.inc`
files.

## Scope of this directory

Only code under `hotswap/transpiler/` (plus the integration hooks in
`loader/` that the README describes) is in scope for Salmon work.

Everything else at the top level of `hotswap/` is **legacy** and slated for
removal — do **not** extend, refactor, or add references to it:

- `hotswap.cpp` / `hotswap.hpp`
- `hotswap_rules.cpp` / `hotswap_rules.hpp`
- `trampoline.cpp` / `trampoline.hpp`
- `transpiler.cpp` / `transpiler.hpp` (the legacy byte-level transpiler)
- `waveasm/`
- `llvm_mir_proto/`
- `aster_proto/`

The only legacy surfaces Salmon keeps reusing are documented in the
`README.md` "What Salmon reuses from the legacy hotswap" table — those are
integration-layer plumbing (HIP fat-binary intercept, ELF ISA patching, env
var gating). They live outside this directory.

## Project tracking

Salmon's roadmap, sub-issues, and open questions live in the "Project
Salmon" GitHub project at
https://github.com/users/martin-luecke/projects/1. Issues live in the
`martin-luecke/rocm-systems` fork.

If `gh` is available, agents may use it **read-only** — to look up
issue bodies and comments for context — and nothing else. Do not
create, edit, comment on, label, assign, close, or reopen issues,
project items, or PRs, and do not touch any other GitHub state unless
explicitly told to and with additional confirmation.

### Draft issues vs. repo issues

A "draft issue" is a **project-board draft** on Project Salmon — no
repo number, no permalink, board-only. A repo issue (anything via
`gh issue create`, `POST /repos/.../issues`, or `createIssue`) is
live, indexed, permalinked, and cannot be downgraded — only deleted.
A `[Draft]` title prefix changes nothing.

- "Draft issue" / "file a draft" → create a project draft item
  (`gh project item-create` / `addProjectV2DraftIssue`). Never hit
  the repo-issues surface.
- Convert to a repo issue only on an explicit instruction ("file
  it", "publish it", "open an actual issue"). "Looks good" is not
  enough.
- If unsure, ask before touching the GitHub API.

## Architecture in one page

```
  code object bytes
        │
        ▼
  code_object_utils.{hpp,cpp}   ELF parsing, kernel metadata, kernarg layout
        │
        ▼
  raiser.{hpp,cpp}              orchestrator: disassemble → decode → dispatch
        │   (will be renamed to translation.{hpp,cpp} — see below)
        │
        │   per-instruction:
        │     MCDisassembler → MCInst
        │     OpcodeMap::lookup(MCOpcode) → SemOp
        │     dispatch by format to a handler
        │
        ▼
  handle_<format>.cpp           one file per AMDGPU encoding format:
                                  sop1, sop2, sopc, sopk, sopp,
                                  smem, valu, flat, ds, mubuf,
                                  mfma, vopd
        │
        ▼
  LLVM IR module (allocas for the whole register file; one BB per branch
  target; flag writebacks to SCC/VCC/EXEC allocas)
        │
        ▼
  PromoteMemToReg               allocas → proper SSA with phis
        │
        ▼
  pipeline.{hpp,cpp}            IR → `llc` → `llvm-mc` → `ld.lld` → HSACO
        │
        ▼
  merged HSACO for the target ISA
```

### Key types

- `SemOp` (`semop.hpp`) — architecture-neutral instruction identity.
  The dispatch key. Intentionally small; do not grow it to cover encoding
  variants (e32/e64/DPP/SDWA/subtarget suffixes) — those are folded onto
  their canonical pseudo by `OpcodeMap` via LLVM's `getMCOpcode` /
  `getVOPe64` / `getDPPOp*` / `getSDWAOp` / `getGlobalVaddrOp` helpers.
- `OpcodeMap` (`opcode_map.{hpp,cpp}`) — MC opcode → SemOp. Built once at
  raiser init from `MCInstrInfo`. V_CMP_* / V_CMPX_* are collapsed to two
  SemOps with a `VCmpMeta` side table so we do not have ~100 near-identical
  enumerators.
- `ISAProfile` (`isa_profile.hpp`) — pure snapshot of the subtarget
  capability bits the raiser branches on (`waveSize`, `hasMFMA`, `hasVOPD`,
  `hasWMMA12`, …). Constructed only via `fromSubtarget(MCSubtargetInfo)`;
  every field maps to a `FeatureFoo` bit so new subtargets flow in for
  free.
- `DecodedInst` (`decoded_inst.hpp`) — MCInst plus `semOp`, `tsFlags`,
  `srcMap` / `modMap` (logical-source-index → MCInst-operand-index),
  branch/def bits.
- `RaiseContext` / `OpResolver` (`raise_context.{hpp,cpp}`) — shared state
  threaded through every handler. Handlers read sources via `op.src(i)` /
  `op.srcF(i)` / `op.src64(i)` / `op.dst(i)` — not by poking MCInst
  operands directly.
- `AllocaRegFile` (`reg_file.hpp`) — the alloca-per-register register
  file. SSA is delegated to `PromoteMemToReg`.
- `KernargLayout` (`kernarg_layout.hpp`) — kernel argument metadata
  extracted from `.amdgpu_metadata`.
- `MCState` (`mc_state.{hpp,cpp}`) — owns LLVM's MC layer
  (`MCSubtargetInfo`, `MCInstrInfo`, `MCRegisterInfo`, `MCDisassembler`,
  …) for a given ISA.
- `pipeline.{hpp,cpp}` — post-IR: `runPipeline` (single kernel) and
  `runPipelineAllKernels` (all kernels in a code object → one merged
  HSACO).
- `wmma_lowering.{hpp,cpp}` — pre-backend lowering for gfx1250 WMMA
  intrinsics not yet selectable by stock LLVM.

### Why GPU binary lifting is tractable here

Binary lifting (raising machine code to compiler IR) is notoriously hard on
general-purpose CPUs. GPU compute kernels sidestep nearly all of those
problems:

1. **CFG recovery is trivial.** AMDGPU branches use PC-relative immediate
   offsets — no indirect jumps, no jump tables. A single linear scan
   recovers the full CFG.
2. **SSA construction is delegated.** The entire physical register file is
   modelled as LLVM allocas; `PromoteMemToReg` yields proper SSA with phis.
   Works cleanly because GPU registers have no aliasing (unlike x86
   rax/eax/ax/al).
3. **Type recovery is free.** Types are encoded in instruction mnemonics
   (`v_add_f32` = float, `v_add_u32` = unsigned int). Kernel argument types
   come from `.amdgpu_metadata`.
4. **Flag forwarding is structural.** SCC/VCC are modelled as `i1` allocas
   that participate in normal SSA promotion. Auto-SCC writeback uses
   hardware `implicit_defs()` metadata.
5. **No stack frame recovery.** GPU kernels have no traditional stack.
   Memory accesses use well-defined address spaces encoded in the opcode.
   Kernel arguments arrive via a known ABI with ELF metadata declaring
   every argument's offset, size, and type.
6. **No function boundary recovery.** Kernels are self-contained:
   single-entry, terminated by `s_endpgm`, no calls. The boundary comes
   from ELF symbol metadata.
7. **Information loss is a non-issue.** The information destroyed during
   original compilation (instruction selection, register allocation,
   scheduling, wait counters) is exactly what the LLVM backend re-derives
   from first principles for the new target.

The raised IR is a **waypoint**, not a destination. The one open problem
is EXEC-mask divergence; that is the SPE model's domain (see SPE design
docs for the details).

## Naming note

`raiser.{hpp,cpp}` will be renamed to `translation.{hpp,cpp}` (and the
`raiseToIR` entry point renamed accordingly). The current name leans too
hard on the academic "raising" literature, which addresses a much harder
problem than what we do. Do not introduce new symbols or docs that
entrench the `raiser` name; if you touch these files, prefer the new
name.

## Tests

All tests live in a single GoogleTest binary (`transpiler_tests`) orchestrated
by CTest.  Do not add new test executables; add `TEST()` or `TEST_F()` cases
to the appropriate `tests/*.cpp` file (or create a new `*_test.cpp` and add it
to the `TRANSPILER_TEST_SOURCES` list in `CMakeLists.txt`).

### Running

```bash
# Via CTest (process isolation, timeouts, xfail).
# --output-on-failure prints GoogleTest output only for failing tests.
ctest --test-dir build --output-on-failure

# Direct binary:
./build/transpiler_tests

# Subset:
./build/transpiler_tests --gtest_filter='BatchRaise.*'

# Extended corpus (slow):
./build/transpiler_tests --test-all --gtest_filter='Corpus.*'
```

### Test structure

| File | Suite | GPU? | Purpose |
|------|-------|:----:|---------|
| `test_main.cpp` | — | — | GoogleTest `main()`, `--test-all` flag |
| `test_common.hpp` | — | — | `GpuTest` fixture (with `hipDeviceReset` teardown), `HIP_ASSERT`, helpers |
| `batch_raise_test.cpp` | `BatchRaise` | No | Raise rate on code objects / directories |
| `corpus_test.cpp` | `Corpus` | No | System HSACO corpus, fork-isolated per ISA |
| `ir_gpu_test.cpp` | `IrGpu` | Yes | Same-ISA vecadd roundtrip |
| `mfma_gpu_test.cpp` | `MfmaGpu` | Yes | Same-ISA MFMA GEMM |
| `cross_arch_gpu_test.cpp` | `CrossArchGpu` | Yes | Cross-ISA raise + execute (rocBLAS HSACOs) |
| `gfx1250_gpu_test.cpp` | `Gfx1250Gpu` | Yes | gfx1250 Triton kernels → gfx942 |
| `integration_test.cpp` | `Integration` | Yes | Multi-kernel raise + merge + load |

### Expected failures

Known-failing tests are tracked in `tests/xfail.cmake` using CTest's
`WILL_FAIL` property.  The tests still run; CTest passes them when they fail
as expected and **fails** them if they unexpectedly pass (so you know to
update the xfail list).  Each xfailed test has an `// XFAIL:` comment in its
source pointing to `xfail.cmake` with the reason.

### Conventions

- GPU tests inherit from `GpuTest` (calls `hipDeviceReset()` in teardown).
- Missing build artifacts or test data → `GTEST_SKIP()` with a message, never
  silent pass or silent skip.
- `HIP_ASSERT(call)` for HIP calls whose failure should abort the test.
  Use `(void)hipFree(...)` / `(void)hipModuleUnload(...)` in cleanup where
  failure is non-fatal.
- `batch_raise_test` is the no-GPU smoke test.  Any change must keep its
  raise rate on the AITER corpus stable or improve it.

### Missing targeted tests (follow-up)

The current suite exercises the APIs below end-to-end via BatchRaise /
corpus / GPU tests. Unit tests that hit them directly do not yet exist;
an agent landing changes to these surfaces should consider adding
targeted tests alongside. Not a blocker — the existing end-to-end
coverage catches regressions — but closing these gaps would let a
reviewer trust the APIs in isolation.

- **`WaveProjection` / `ModuloReplicationProjection`** — no unit test
  for `ballotI1ToWidth` wave32→wave64 truncation, or the
  wave64→wave32 `report_fatal_error` path. Cover by synthesising a
  tiny IR function, running each projection primitive, and asserting
  the emitted IR shape.
- **`verifyExecAttrCoverage`** — no negative test (add a manufactured
  MC opcode that declares EXEC as an implicit def but whose SemOp
  isn't in any handler's attribute registration, assert the
  `report_fatal_error`).
- **`decodeGlobalLoadAddr` / `decodeGlobalStoreAddr`** — no targeted
  test for the SADDR-vs-plain discriminator or the `scale_offset` mul.
- **`decodeMubufAddr` / `decodeMubufAtomicAddr`** — same; the SRSRC
  dword routing through `readfirstlane` is only exercised indirectly.
- **`decodeKernel`** — only exercised via `raiseToIR`. Targeted tests
  against a synthetic byte buffer would isolate the drift-check paths
  from IR emission.
- **`RaiseFailure` routing** — no test asserting that a specific
  `reason` value bubbles up to the caller as expected.

## Coding standards

### LLVM style (mandatory)

This code links against LLVM and uses LLVM data structures. Follow
[LLVM's Programmer's Manual](https://llvm.org/docs/ProgrammersManual.html).
The full document is long; these are the points that come up constantly in
this codebase:

- **Always use `SmallVector` without an explicit inline-element count.**
  Prefer `SmallVector<T>`; let LLVM pick the inline size.
- **Use `DenseMap` / `DenseSet` for associative maps and sets**, not
  `std::map` / `std::unordered_map`. Define `DenseMapInfo` to add support
  for custom key classes. Reach for the specialized containers
  (`SmallPtrSet`, `MapVector`, `SetVector`, `StringMap`, …) where they
  fit.
- **Never pass `std::string` around. Use `StringRef`.** Return owned
  strings only when ownership genuinely transfers.
- **Prefer `function_ref` over `std::function`** when the callable is not
  stored past the call.
- **Never use anything from `<iostream>`.** Use `llvm::raw_ostream`,
  `errs()`, `outs()`, `dbgs()`.
- **Avoid string-based dispatch.** Introduce or reuse an enum.

Beyond these, `const`-correctness, RAII, and the no-exceptions /
no-RTTI default that LLVM itself uses are expected.

### Reuse LLVM, do not reimplement it

This project is a long-lived consumer of LLVM's AMDGPU target. Every time
we hand-roll a table that LLVM already generates from TableGen, it drifts
and breaks when a new subtarget, instruction, or register lands upstream.

Concrete rules for any new code:

- **Dispatch on authoritative flags**, not on names:
  `MCInstrDesc::TSFlags`, `SIInstrFlags::*`, feature bits from
  `MCSubtargetInfo`. Not `mnemonic.starts_with(...)`, not register-name
  substring checks.
- **Resolve opcode / operand / register identity through generated
  helpers**:
  - `AMDGPU::getNamedOperandIdx(opc, OpName::X)` for named operands.
  - `MCRegisterInfo::getEncodingValue(reg)` for register indices.
  - `MCRegisterClass::contains(reg)` for register-class membership.
  - `AMDGPU::getMCOpcode` / `getVOPe64` / `getDPPOp*` / `getSDWAOp` /
    `getGlobalVaddrOp` to canonicalize encoding variants.
  - `MCInstrDesc::getOperandConstraint(i, MCOI::TIED_TO)` for tied
    operands.
- **Never rely on disassembler / printer string output.** Mnemonic strings
  and operand strings are formatting choices that LLVM can and does change
  between versions. Any code that does `if (mnemonic == "v_add_u32")` or
  scrapes `printInst` output is a regression waiting to happen. The one
  narrow, deliberate exception is `OpcodeMap`'s build-time parse of
  `v_cmp_*` / `v_cmpx_*` pseudo names into `VCmpMeta`, done once, at
  init, against the canonical pseudo — not on a hot path and not on
  disassembly text.
- **Where we keep our own enum (like `SemOp`), keep it small and make
  drift loud.** A missing mapping must be a `report_fatal_error`, never a
  silent default that lets the handler run with garbage.

### Fail loudly

This is a project-wide rule, not a style preference:

- No `try { … } catch (...) {}`, no `if (err) return ok;`, no silent
  fallbacks. If something unexpected happens — missing named operand,
  unknown register class, `kMaxSrcs` overflow, unsupported opcode —
  `report_fatal_error` (or, in `batch_raise_test`, return the failing
  mnemonic so the batch summary surfaces it).
- No "best effort" string parsing that degrades to `-1` / empty on
  failure. If you cannot answer the question, the build or the test must
  stop.
- Errors carry the MC opcode name, mnemonic, and (where relevant) the
  source ISA and kernel name, so a failure in `batch_raise_test`'s summary
  points straight at the offending case.

## Known limitations

Items that are not bugs but that an incoming agent should be aware of:

- **M0 register handling is ad-hoc across handler files.** M0 is used as
  an implicit address operand by several instruction families and the
  current handling has gaps:
  - `handle_mubuf.cpp`: `buffer_load_*_lds` stores to LDS at M0, but
    does not advance M0 afterwards. If multiple `buffer_load_*_lds` fire
    in sequence the raiser relies on the kernel having explicit
    `s_mov_b32 m0` instructions between them. Verify against real
    kernels before relying on this.
  - `handle_sopc.cpp`: `s_set_gpr_idx_on` writes M0 but the GPR dynamic
    indexing effect is not modelled. Known limitation; revisit if AITER
    kernels start using GPR indexing.
  - `reg_file.cpp`: `LDS_DIRECT` reads from LDS at M0. GFX9 does not
    auto-increment M0 (unlike GFX11+ DSDIR `lds_direct_load`), so the
    current implementation is correct for GFX9. Raising GFX11+ kernels
    that use DSDIR will require explicit modelling of the increment.
  - `handle_ds.cpp`: `ds_bpermute` uses M0 for byte-lane control and
    passes it through correctly. No known issue.
  - General concern: M0 is a single 32-bit alloca shared by all these
    uses. Interleaved M0 uses (e.g. `buffer_load_lds` followed by
    `ds_bpermute`) rely on the alloca preserving M0's value across the
    entire instruction stream; there is no dedicated test for this.

## Before you commit

- Build `hotswap-transpiler` and `transpiler_tests` cleanly with the
  default CMake flow from `README.md`.
- Run `ctest --output-on-failure` and confirm all tests pass (XFAIL tests
  report "Passed" when they fail as expected).
- Run `batch_raise_test` against the AITER corpus (or whatever corpus you
  have locally) and confirm the raise rate does not drop.
- If you renamed or added files, update `CMakeLists.txt`'s
  `hotswap-transpiler` source list (or `TRANSPILER_TEST_SOURCES` for test
  files). Do not add new source files to top-level `hotswap/`; everything
  new belongs under `transpiler/`.
