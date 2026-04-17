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

See `WHY_GPU_RAISING_IS_EASIER.md` for the full argument. Short version:
CFG is recoverable by linear scan (no indirect branches), SSA is delegated
to `PromoteMemToReg`, types come from mnemonics and kernarg metadata, there
is no stack frame and no cross-function calls. The raised IR is a
**waypoint**, not a destination — everything lost during original
compilation (instruction selection, register allocation, scheduling, wait
counters) is re-derived by the backend for the new target.

## Naming note

`raiser.{hpp,cpp}` will be renamed to `translation.{hpp,cpp}` (and the
`raiseToIR` entry point renamed accordingly). The current name leans too
hard on the academic "raising" literature, which addresses a much harder
problem than what we do (see `WHY_GPU_RAISING_IS_EASIER.md`). Do not
introduce new symbols or docs that entrench the `raiser` name; if you touch
these files, prefer the new name.

## Tests

An ongoing effort consolidates the per-feature test executables listed in
`CMakeLists.txt` (`batch_raise_test`, `ir_gpu_test`, `mfma_gpu_test`,
`cross_arch_gpu_test`, `gfx1250_gpu_test`, `integration_test`) into a
**single test binary**. Do not add new top-level test executables; add test
cases inside the consolidated binary (ask if the new home is not obvious
yet).

`batch_raise_test` remains the no-GPU smoke test. Any change in this
directory must keep its raise rate on the AITER corpus stable or improve
it.

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
The `REFACTOR_PLAN.md` in this directory is the running inventory of those
surfaces and how we are replacing them with LLVM-native lookups.

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

## Before you commit

- Build `hotswap-transpiler` and `batch_raise_test` cleanly with the
  default CMake flow from `README.md`.
- Run `batch_raise_test` against the AITER corpus (or whatever corpus you
  have locally) and confirm the raise rate does not drop.
- If the change touches opcode mapping, register classification, or any
  TableGen-adjacent code, read `REFACTOR_PLAN.md` first — the plan has
  already decided how the next step of that refactor should look.
- If you renamed or added files, update `CMakeLists.txt`'s
  `hotswap-transpiler` source list. Do not add new source files to
  top-level `hotswap/`; everything new belongs under `transpiler/`.
