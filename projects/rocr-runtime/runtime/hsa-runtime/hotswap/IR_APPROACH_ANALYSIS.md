# IR Approach Analysis: waveasm vs. Aster vs. SPIR-V vs. TileIR

An analysis of intermediate representation choices for the HotSwap binary
translation pipeline, considering the current waveasm-based approach, Aster
(iree-org's AMDGCN MLIR dialect), and higher-level alternatives.

## The Problem

We receive a **compiled GPU binary** (ELF with AMDGCN machine code) for one
ISA and need to produce a working binary for a different ISA. This is
fundamentally a decompilation/recompilation problem. The choice of IR
determines what transformations are possible, how much information we can
recover, and what targets we can reach.

## Abstraction Levels

There are four natural levels at which a GPU IR can sit:

```
Level 3 (Tensor/Tile)   TileIR, Triton IR, Wave IR
                         Operations: mma, load_tile, reduce
                         Lost from binary: everything

Level 2 (Thread/CFG)    SPIR-V, LLVM IR, NVVM, AMDGPU MIR
                         Operations: load, add, branch, phi
                         Lost from binary: types, CFG structure, variable names

Level 1.5 (Structured   Aster amdgcn dialect
 ISA)                    Operations: amdgcn.vop2.vop2, amdgcn.ds.read
                         SSA, typed registers, instruction-class grouping
                         Lost from binary: labels (recoverable)

Level 1 (ISA)           waveasm, PTX (partially), raw MCInst
                         Operations: v_add_f32_e32, global_load_dword
                         Lost from binary: labels (recoverable)
```

The key constraint: **we start at Level 0** (raw bytes). Each level we want to
reach requires more analysis and makes more assumptions. The choice is not
which level is "best" in the abstract — it's which level we can **reliably
reach** from a binary and which level provides the **transformations we need**.

## Option A: waveasm only (current approach)

**What it is.** A 1:1 representation of hardware instructions in MLIR, based on
the `waveasm` dialect from [iree-org/wave](https://github.com/iree-org/wave).
Each `waveasm.*` op corresponds to one machine instruction. Physical registers
are explicit. Control flow is represented as branches + labels (not structured).

**What works.**
- Lifting is mechanical: disassemble → create op. No analysis required.
- Round-trip fidelity: the emitter can always produce valid assembly because
  the IR never abstracts away from the ISA.
- Cross-ISA translation works for the class of instructions that have 1:1
  mappings (renames, format changes, addressing mode lowering).
- Already demonstrated: gfx1250 → gfx942 with GPU-verified correctness.

**What doesn't work.**
- No high-level semantics: can't reason about "this is a matrix multiply" or
  "these loads are a tiled access pattern". Limits optimization.
- ISA-specific: retargeting to NVIDIA would require a completely different
  set of instruction mappings and ABI handling.
- Transforms are low-level and fragile: scale_offset lowering, wave width
  translation, etc. are hand-written pattern matches.
- No structured control flow: SSA construction, register allocation, and
  scheduling would need to reconstruct CFG from branches.

**Best for.** Fast, correctness-focused retargeting between closely related AMD
ISAs (RDNA ↔ CDNA) where most instructions have direct equivalents.

## Option B: SPIR-V dialect (Level 2)

**What it is.** SPIR-V is a portable binary IR standardized by Khronos. MLIR
has a mature SPIR-V dialect with serialization/deserialization, type system,
structured control flow, and lowerings to/from GPU dialect.

**Lifting feasibility.**
Reaching SPIR-V from a GPU binary requires recovering:
1. **Structured control flow** — SPIR-V mandates structured merge/continue
   constructs. Recovering these from arbitrary branch-based machine code
   requires dominance analysis, loop detection, and "structurization" — a
   solved but non-trivial problem (LLVM's StructurizeCFG, AMD's
   SIAnnotateControlFlow).
2. **Types** — SPIR-V operations are typed (f32, i32, vector types). Machine
   code is untyped (everything is 32-bit dwords). Type recovery requires data
   flow analysis.
3. **Memory model** — SPIR-V distinguishes address spaces (Function, Workgroup,
   CrossWorkgroup, UniformConstant). Machine code uses flat addressing with
   different instruction classes hinting at the space.

None of these are impossible — they're the standard problems of decompilation.
But each introduces a point where the lifter must make assumptions that could
be wrong, and wrong assumptions produce miscompilation.

**What it would enable.**
- Vendor-neutral IR: could lower to AMD (via GPU dialect → ROCDL) or to
  NVIDIA (via GPU dialect → NVVM) or to CPU (via LLVM).
- Standard optimization passes: SPIR-V/SCF canonicalization, loop unrolling,
  vectorization.
- Interop with Vulkan/OpenCL ecosystem.

**What it costs.**
- Significant engineering to build a reliable lifter from AMDGCN → SPIR-V.
  The type recovery and control flow structurization alone are multi-month
  efforts with ongoing correctness risk.
- Lossy round-trip: lowering SPIR-V back to AMDGCN goes through the full LLVM
  backend, which makes its own scheduling/register allocation decisions. The
  output will look nothing like the input — acceptable for correctness, but
  makes debugging much harder.
- Doesn't help with ISA-specific concerns (wait counters, wave width, exec
  mask management) — those still need custom passes.

## Option C: TileIR (Level 3)

**What it is.** NVIDIA's MLIR-based tile-centric IR. Operations are on tiles
(multi-dimensional tensor fragments): `mma`, `load_view`, `broadcast`, etc.
Abstracts away threads, registers, and memory hierarchy.

**Lifting feasibility.**
Reaching Level 3 from a binary requires recovering:
1. Everything from Level 2 (structured CFG, types, memory model).
2. **Tile semantics** — recognizing that a sequence of loads + shared memory
   + synchronization + MMA instructions constitutes a "tiled matrix multiply".
   This is pattern matching on compiled output, essentially the inverse of a
   compiler backend.
3. **Data layout** — recovering the logical tensor layout from physical memory
   access patterns (swizzling, padding, bank-conflict avoidance).

This is extremely hard from arbitrary binaries. It's roughly equivalent to
decompiling optimized assembly back to the original Triton/CUDA kernel.

**What it would enable.**
- Full retargeting to any vendor (AMD, NVIDIA, Intel, CPU).
- Access to NVIDIA's TileIR optimizer for NVIDIA targets.
- Highest optimization potential: the compiler sees tensor operations and can
  make global scheduling/tiling decisions.

**What it costs.**
- Building a reliable binary → tile-level lifter is a research problem, not
  an engineering problem. No existing tool does this.
- Only useful for ML workloads (matrix ops, reductions). Scalar kernels,
  graphics shaders, or custom compute kernels have no tile-level semantics.
- Dependency on NVIDIA's proprietary TileIR specification for NVIDIA targets.

## Option D: Aster (Level 1.5 — structured ISA-level)

**What it is.** [Aster](https://github.com/iree-org/aster) ("Assembly Tooling
and Representations") is an MLIR-based C++ tool from `iree-org` (AMD) for
programmable, highly-controllable assembly production on AMD GPUs. It was
pre-released in late 2025 and is under active development (551 commits, 40
open issues as of April 2026).

Aster defines an `amdgcn` MLIR dialect that sits *between* raw MCInst (Level 1)
and a typed thread-level IR (Level 2). Key characteristics:

- **Instruction-class ops, not 1:1 instructions.** One MLIR op per instruction
  *family* (e.g., `amdgcn.ds.read`, `amdgcn.vop2.vop2`, `amdgcn.flat.global_store`)
  with the specific instruction variant selected by an `#amdgcn.inst<...>`
  attribute. This gives semantic grouping while preserving exact ISA control.
- **Rich register type system.** `!amdgcn.vgpr`, `!amdgcn.sgpr`,
  `!amdgcn.vgpr_range<[? + N]>`, `!amdgcn.sgpr_range<[? + N]>` with both
  allocated (numbered) and unallocated (abstract) variants. Register ranges
  can be split/merged. The `?` denotes "not yet allocated".
- **True SSA + incremental register allocation.** The IR is SSA from the start.
  Users can pin specific registers where needed and let the allocator handle
  the rest. Interference constraints are explicit (`amdgcn.reg_interference`).
- **Kernel-level structure.** `amdgcn.module` → `amdgcn.kernel` with typed
  arguments, kernel descriptor metadata, target/ISA attributes, and LDS
  allocation as first-class ops. No need for text-based splicing.
- **Wait counters as structured ops.** `amdgcn.sopp.s_waitcnt` with named
  counter fields (`lgkmcnt`, `vmcnt`, `expcnt`) — not destroyed during
  translation.
- **Python-first metaprogramming + C++ MLIR passes.** A `KernelBuilder` Python
  API for constructing kernels, plus standard MLIR pass infrastructure.
- **Direct HSACO generation.** Translates MLIR → assembly text → `llvm-mc` →
  `ld.lld` → HSACO, entirely in-process. Supports gfx940, gfx942, gfx950,
  gfx1201.
- **Upcoming "raising" direction.** The README explicitly lists "Raising from
  existing ASM to MLIR ASTER to edit and fine-tune existing ASM using familiar
  MLIR tooling" as a key upcoming feature — this is exactly the binary → IR
  path we need.

**Why it matters for HotSwap.**

Aster solves many of the problems we've identified in `SHORTCUTS_AND_LIMITATIONS.md`:

| Our limitation                           | Aster's answer                                      |
|------------------------------------------|-----------------------------------------------------|
| Wait counter operands destroyed (#1)     | `s_waitcnt` has named counter fields in the IR      |
| Implicit defs/uses not modeled (#2)      | TableGen-defined ops with instruction format info    |
| No real SSA (#4, #8, #13)                | SSA from the ground up + real register allocator     |
| Emitter uses mnemonic heuristics (#10)   | Assembly emission from TableGen instruction metadata |
| Hardcoded scratch regs (#5, #6)          | Incremental register allocator picks free regs      |
| `system()` + `/tmp/` for rebuild (#14)   | In-process HSACO generation pipeline                |
| No kernel descriptor validation (#9)     | `amdgcn.kernel` carries metadata as structured attrs|
| Text-based splicing (#11, #12)           | Full kernel structure in MLIR, no text manipulation  |

**Lifting feasibility (binary → Aster).**

Lifting machine code to Aster IR is harder than lifting to waveasm but much
easier than lifting to SPIR-V:

- **Instruction classification** — We already classify instructions by
  mnemonic. Mapping to Aster's instruction-class ops (`amdgcn.vop2.vop2`,
  `amdgcn.ds.read`, etc.) is a table lookup from `MCInstrDesc` format flags,
  which LLVM already provides. This is mechanical.
- **Register types** — Aster's register types (`!amdgcn.vgpr`, `!amdgcn.sgpr`,
  ranges) are directly derivable from the MC register info we already extract.
  Unallocated registers aren't needed for lifting — we lift to allocated form
  and can optionally "deallocate" later for re-allocation.
- **Structured control flow** — Aster uses MLIR `cf.br`/`cf.cond_br` within
  a graph region. Branches + labels (which our lifter already recovers)
  translate directly — no structurization required.
- **Wait counters** — The lifter currently discards `s_waitcnt` operands.
  With Aster's named counter fields, we'd preserve them. This is a change to
  the lifter, not a fundamental difficulty.
- **Kernel metadata** — Aster's `amdgcn.kernel` op carries all metadata as
  structured attributes. We can populate these from the ELF's kernel
  descriptor and `.amdhsa_kernel` directives.

**What it would enable (beyond what we have today).**

- **Principled cross-ISA translation** — Aster's instruction-class ops abstract
  over encoding differences. A `amdgcn.flat.global_load` with `scale_offset`
  attribute is cleaner than heuristic operand scanning.
- **Real register allocation** — Instead of hardcoding v3/v[4:5] for
  `scale_offset` lowering and s16 for VCC saves, use Aster's SSA-based
  allocator. This fixes items #5, #6 from the shortcuts list.
- **Instruction scheduling** — Aster has modular scheduling expressions. We
  could optionally reschedule translated code for the target microarchitecture.
- **Nanokernel composition** — Aster's "nanokernel" concept (reusable,
  fine-tuned kernel fragments) could let us inline optimized routines into
  translated code where we detect patterns.
- **Upstream alignment** — Aster is from `iree-org` (AMD's ML compiler team).
  Adopting it aligns us with the direction AMD is investing in for low-level
  GPU programming. Both waveasm (from iree-org/wave) and Aster (from
  iree-org/aster) are AMD-backed projects, and converging on Aster avoids
  maintaining two parallel ISA-level dialects.

**What it doesn't solve.**

- **Cross-vendor (AMD → NVIDIA)** — Aster is AMD-specific. Getting to NVIDIA
  still requires lifting to SPIR-V or a vendor-neutral IR.
- **Tile-level recovery** — Aster operates at the instruction level, not the
  tensor level. Recovering MMA patterns is still a separate problem.
- **Maturity** — Aster is young ("pre-release", active since October 2025).
  APIs may change. But it's actively developed with 551 commits and backing
  from AMD.

## Option E: Layered approach (recommended direction)

A pragmatic architecture that uses **multiple IR levels** and lifts only as
high as is needed (or feasible) for each use case. The key update from Option D
is that **Aster replaces waveasm as the Level 1 target** — it provides a
strictly better representation with real SSA, register allocation, structured
wait counters, and an active upstream.

```
                    ┌─────────────────────────────────────────────┐
                    │    Binary (.text bytes)                      │
                    └──────────────────┬──────────────────────────┘
                                       │ MCDisassembler (mechanical)
                                       ▼
                    ┌─────────────────────────────────────────────┐
  Level 1           │    waveasm MLIR (current, temporary)         │
  (entry)           │    Thin wrapper: 1:1 with MCInst             │
                    └──────────────────┬──────────────────────────┘
                                       │ Instruction classification
                                       │ (table lookup, mechanical)
                                       ▼
                    ┌─────────────────────────────────────────────┐
  Level 1.5         │    Aster amdgcn dialect (target)             │
  (structured ISA)  │    Instruction-class ops, SSA, real regalloc │
                    │    Structured wait counters, kernel metadata  │
                    └────────┬──────────────────┬─────────────────┘
                             │                  │
            Cross-ISA        │                  │ Lift (analysis required)
            (remap + lower)  │                  │
                             ▼                  ▼
              ┌──────────────────┐   ┌──────────────────────────────┐
              │ Retargeted HSACO │   │  SPIR-V / GPU dialect         │
              │ (same-vendor,    │   │  (vendor-neutral, typed,      │
  Level 2     │  Aster pipeline) │   │   structured CFG)             │
              └──────────────────┘   └──────────┬───────────────────┘
                                                │
                                    Lower (LLVM backend)
                                                │
                                     ┌──────────┴──────────┐
                                     ▼                     ▼
                               AMD (ROCDL)          NVIDIA (NVVM)
```

**How it works:**

1. **Lift to waveasm first.** This is our existing mechanical lifter. It
   remains the first step because it's done and correct. Think of it as a
   "raw disassembly" layer.

2. **Convert waveasm → Aster amdgcn dialect.** This is a translation between
   two ISA-level MLIR representations. It classifies instruction formats,
   builds register types/ranges, preserves wait counter operands, and
   reconstructs kernel metadata. Mostly table-driven, some pattern matching.

3. **For same-vendor retargeting (RDNA ↔ CDNA):** use Aster's infrastructure
   for mnemonic remapping, addressing mode lowering, wave width translation,
   and register re-allocation. Emit HSACO through Aster's in-process pipeline.
   No text splicing, no `system()` calls.

4. **For cross-vendor retargeting (AMD → NVIDIA):** lift from Aster to SPIR-V
   or GPU dialect. The richer IR makes this easier — Aster already has SSA,
   typed registers, and instruction semantics. From SPIR-V, lower through
   LLVM's NVVM backend.

5. **For known ML patterns (optional, future):** pattern-match instruction
   sequences in Aster IR to recover high-level operations. Aster's
   "nanokernel" concept could help here: recognized patterns become reusable
   optimized fragments.

**Why this is the right order:**

- **Incremental migration.** The current waveasm pipeline continues working
  for same-ISA round-trips. We add the Aster layer as a new target
  representation, and can migrate cross-ISA translation to it incrementally
  without breaking the existing path.

- **Upstream leverage.** Aster is actively developed by AMD. Features we need
  (instruction format metadata, register allocator, HSACO generation) are
  being built anyway. Contributing to Aster gives us maintenance for free
  and aligns us with AMD's direction.

- **Graceful degradation.** If the Aster path fails for a specific kernel,
  fall back to the direct waveasm → assembly path. If the SPIR-V lifter
  can't handle something, fall back to Aster-level retargeting.

- **Solves most of our current shortcuts.** Moving to Aster as the working
  IR for cross-ISA translation directly addresses 8 of the 20 items in
  `SHORTCUTS_AND_LIMITATIONS.md` (the ones related to SSA, register
  allocation, text-based splicing, wait counters, and emitter fragility).

## Recommendation

**Short-term (0-3 months):** Continue hardening the Level 1 (waveasm) path for
same-ISA and cross-ISA retargeting. In parallel, evaluate Aster integration:
1. Build Aster and run its tests on our MI300 machines
2. Prototype a waveasm → Aster `amdgcn` dialect converter for the vecadd kernel
3. Verify that Aster's HSACO generation produces a working binary from
   the converted IR

This validates the migration path without committing to it.

**Medium-term (3-9 months):** Migrate cross-ISA translation to Aster:
1. Build the full waveasm → Aster lifter (instruction classification,
   register type mapping, wait counter preservation)
2. Re-implement mnemonic remapping and `scale_offset` lowering using Aster's
   instruction metadata rather than string-prefix heuristics
3. Use Aster's register allocator instead of hardcoded scratch registers
4. Use Aster's HSACO pipeline instead of `system()` + `llvm-mc` + `ld.lld`
5. Expand kernel coverage using the more robust pipeline

**Long-term (9+ months):** Two parallel tracks:
- **AMD-focused:** Contribute upstream to Aster. Their planned "raising from
  existing ASM" feature is exactly our binary lifter. If we contribute our
  MCDisassembler → MLIR work, it benefits both projects.
- **Cross-vendor:** Build an Aster → SPIR-V lifter. Aster's SSA and typed
  registers make this significantly easier than lifting from raw waveasm.
  This opens the AMD → NVIDIA retargeting path.

## Comparison Matrix

| Criterion                     | waveasm (L1) | Aster (L1.5) | SPIR-V (L2) | TileIR (L3) | Layered  |
|-------------------------------|:------------:|:------------:|:------------:|:------------:|:--------:|
| Lift from binary              | Mechanical   | Table-driven | Hard         | Research     | Staged   |
| AMD → AMD retarget            | Works today  | Better       | Overkill     | Overkill     | L1.5     |
| AMD → NVIDIA retarget         | No           | No           | Yes          | Yes          | L2 path  |
| Optimization potential        | Low          | Medium       | Medium       | High         | All      |
| Correctness risk at lift      | Minimal      | Low          | Moderate     | High         | Varies   |
| Engineering effort            | Done         | 2-4 months   | 6-12 months  | Research     | Staged   |
| Round-trip fidelity           | High         | High         | Low          | None         | High/L1.5|
| Works for non-ML kernels      | Yes          | Yes          | Yes          | No           | Yes      |
| Real register allocation      | No           | Yes          | Yes          | Yes          | Yes      |
| Wait counters preserved       | No           | Yes          | Yes          | N/A          | Yes      |
| Active upstream                | Yes (AMD/iree-org) | Yes (AMD/iree-org) | Yes (LLVM)   | Yes (NVIDIA) | Mixed    |
| Standards/ecosystem           | iree-org/wave | iree-org/aster | Khronos      | NVIDIA       | Mixed    |
