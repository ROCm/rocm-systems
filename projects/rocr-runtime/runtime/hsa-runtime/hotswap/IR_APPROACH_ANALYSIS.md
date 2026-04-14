# IR Approach Analysis: waveasm vs. Aster vs. LLVM MIR vs. LLVM IR vs. SPIR-V vs. TileIR

An analysis of intermediate representation choices for the HotSwap binary
translation pipeline, considering the current waveasm-based approach, Aster
(iree-org's AMDGCN MLIR dialect), LLVM Machine IR, LLVM IR (full SSA), and
higher-level alternatives.

**Status (April 2026):** Four prototypes have been built and GPU-verified
for the `vecadd` kernel on gfx942: waveasm, Aster, LLVM MIR, and LLVM IR.
This document reflects concrete findings from all four implementations.

## The Problem

We receive a **compiled GPU binary** (ELF with AMDGCN machine code) for one
ISA and need to produce a working binary for a different ISA. This is
fundamentally a decompilation/recompilation problem. The choice of IR
determines what transformations are possible, how much information we can
recover, and what targets we can reach.

## Abstraction Levels

There are five natural levels at which a GPU IR can sit:

```
Level 3 (Tensor/Tile)   TileIR, Triton IR, Wave IR
                         Operations: mma, load_tile, reduce
                         Lost from binary: everything

Level 2 (Thread/CFG)    SPIR-V, NVVM
                         Operations: load, add, branch, phi
                         Lost from binary: types, CFG structure, variable names

Level 2 (Typed SSA)     LLVM IR (with amdgcn intrinsics)
                         Operations: load float, fadd, getelementptr, br i1
                         llvm.amdgcn.workgroup.id.x(), implicitarg.ptr()
                         Lost from binary: types, SSA, structured CFG (all recovered)

Level 1.75 (Post-RA     LLVM MachineInstr (MIR)
 Machine IR)            Operations: S_LOAD_DWORD_IMM_vi, V_ADD_F32_e32_vi
                         Physical regs + implicit operands from TableGen
                         Lost from binary: labels, liveness (recoverable)

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

## Option A: waveasm (prototype complete)

**What it is.** A 1:1 representation of hardware instructions in MLIR, based on
the `waveasm` dialect from [iree-org/wave](https://github.com/iree-org/wave).
Each `waveasm.*` op corresponds to one machine instruction. Physical registers
are explicit. Control flow is represented as branches + labels (not structured).

**Prototype results (vecadd on gfx942).**
- 292/292 instructions lifted (100%), including ~268 `s_nop` padding
- Cross-ISA gfx1250 → gfx942 GPU-verified correct (1024 elements)
- Same-ISA round-trip GPU-verified correct
- Unknown instructions escape via `waveasm.raw` — 100% coverage by definition

**Observed strengths.**
- Lifting is mechanical: disassemble → create op. No analysis required.
- Round-trip fidelity: the emitter can always produce valid assembly because
  the IR never abstracts away from the ISA.
- Cross-ISA translation works for instructions with 1:1 mappings (renames,
  format changes, addressing mode lowering).
- `waveasm.raw` escape hatch means any kernel can round-trip, even with
  unknown instructions — coverage is never blocked.

**Observed weaknesses (from `SHORTCUTS_AND_LIMITATIONS.md`).**
- `s_waitcnt` operands are destroyed — emitter always drains all counters,
  killing IPC. The IR cannot represent "wait for 2 loads" vs "wait for all."
- Implicit register defs/uses (VCC, SCC, EXEC) are not modeled — def-use
  chains are incomplete, blocking any analysis pass.
- SSA construction is linear with no CFG awareness — no dominance, no phi
  nodes, no real register allocation possible.
- `scale_offset` lowering and wave width passes use hardcoded scratch
  registers (v3, v[4:5], s16) — will clobber live values in complex kernels.
- Emitter uses mnemonic-prefix heuristics for operand layout — breaks on
  unhandled instruction formats.
- Assembly splicing is fragile text-based pattern matching (searching for
  `s_cbranch_execz`, `s_endpgm` as strings).

**Best for.** Fast, correctness-focused retargeting between closely related AMD
ISAs (RDNA ↔ CDNA) where most instructions have direct equivalents, and
where coverage is more important than IR quality.

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

## Option D: Aster (Level 1.5 — structured ISA-level, prototype complete)

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

**Prototype results (vecadd on gfx942).**
- 23/23 unique instructions lifted (100%), no unsupported, no escape hatches
- Same-ISA round-trip GPU-verified correct (1024 elements)
- Cross-ISA not yet attempted (future work)
- Lifting goes **directly from MCInst → Aster amdgcn dialect**, bypassing
  waveasm entirely. This is simpler than the originally proposed two-stage
  path (MCInst → waveasm → Aster).
- Exec-mask control flow (`s_and_saveexec_b64` + `s_cbranch_execz`) was
  converted to Aster's native multi-block VCC-based branching — a principled
  semantic conversion rather than opaque passthrough.

**What Aster solves vs. waveasm.**

| waveasm limitation                       | Aster's answer                                      | Verified? |
|------------------------------------------|-----------------------------------------------------|-----------|
| Wait counter operands destroyed (#1)     | `s_waitcnt` has named counter fields in the IR      | Yes — prototype preserves vmcnt/lgkmcnt/expcnt |
| Implicit defs/uses not modeled (#2)      | TableGen-defined ops with instruction format info    | Partially — VCC handled, full implicit modeling not yet |
| No real SSA (#4, #8, #13)                | SSA from the ground up + real register allocator     | Yes — IR is true SSA |
| Emitter uses mnemonic heuristics (#10)   | Assembly emission via `translateModule`              | Yes — correct assembly generated |
| Hardcoded scratch regs (#5, #6)          | Incremental register allocator picks free regs      | Not yet used — prototype pins all registers |
| `system()` + `/tmp/` for rebuild (#14)   | In-process HSACO generation pipeline                | Partially — `translateModule` is in-process, but our pipeline still shells out for `llvm-mc` and `ld.lld` |
| No kernel descriptor validation (#9)     | `amdgcn.kernel` carries metadata as structured attrs| Partially — Aster generates metadata, but we patch it post-hoc for hidden args |
| Text-based splicing (#11, #12)           | Full kernel structure in MLIR, no text manipulation  | Partially — kernel structure is MLIR, but metadata patching is still string-based |

**Lifting: what we learned from building it.**

- **Instruction classification** — Mapping MCInst to Aster's instruction-class
  ops is mostly mechanical, but not a simple table lookup. The mnemonic string
  must be parsed to select the correct op class (VOP2, VOP3, SOP2, SOPP, etc.)
  and the correct `#amdgcn.inst<...>` attribute. We identified this as
  string-based and fragile (should use `MCInstrDesc::getOpcode()` instead).
- **Register types** — Directly derivable from MC register info. Register
  ranges (e.g., `s[0:1]` for 64-bit SGPR pairs) required careful handling —
  `classifyReg` must count only atomic sub-registers, not transitive ones.
  VCC, SCC, and EXEC need `Register(0)` (physical register 0), which was
  non-obvious.
- **Control flow** — The biggest architectural decision. Aster uses multi-block
  regions with `cbranch`/`branch` ops, not exec-mask predication. We convert
  the `s_and_saveexec_b64` + `s_cbranch_execz` pattern to VCC-based branching.
  This is **semantically correct only for wavefront-aligned dispatches** —
  the conversion loses per-lane exec-mask predication. This is documented as
  a high-severity limitation.
- **Wait counters** — The GFX9 packed encoding must be manually decoded
  (`vmcnt = bits[3:0]`, `expcnt = bits[6:4]`, `lgkmcnt = bits[13:8]`). This
  is generation-specific and would break silently on GFX10+ or GFX12.
- **Kernel metadata** — The hardest practical problem. Aster's `translateModule`
  generates kernel descriptors from the IR, but doesn't know about HIP's
  hidden implicit arguments (`hidden_group_size_x`, `hidden_block_count_x`,
  etc.). Our prototype must patch the generated assembly post-hoc to inject
  these, which is fragile string manipulation. This needs a principled
  solution (extract metadata from the original code object and pass it
  through the IR).

**What it enables (demonstrated).**

- **Principled instruction representation** — Every op in the IR has defined
  semantics. No opaque `waveasm.raw` escape hatch needed for the vecadd
  kernel.
- **Structured control flow** — Multi-block CFG with `cbranch`/`branch` ops
  enables analysis and optimization passes. waveasm keeps everything flat.
- **Register ranges** — First-class `!amdgcn.sgpr<[0 : 2]>` correctly models
  64-bit addresses and multi-dword loads. waveasm requires manual tracking.
- **Correct assembly emission** — `translateModule` produces valid assembly
  with proper register counts, float modes, and kernel descriptors.

**What it enables (not yet demonstrated but now clearly feasible).**

- **Real register allocation** — Aster's allocator could replace the pinned
  register assignments in our prototype. Requires lifting to unallocated form.
- **Cross-ISA translation** — Aster's instruction-class ops abstract over
  encoding differences. A `amdgcn.flat.global_load` is cleaner to retarget
  than a raw mnemonic string.
- **Instruction scheduling** — Aster has scheduling expressions. Could
  reschedule for target microarchitecture after translation.
- **Upstream alignment** — Aster is from `iree-org` (AMD's ML compiler team).
  Their planned "raising from existing ASM" feature is exactly our binary
  lifter. Contributing our MCDisassembler → MLIR work would benefit both
  projects.

**What it doesn't solve.**

- **Cross-vendor (AMD → NVIDIA)** — Aster is AMD-specific. Getting to NVIDIA
  still requires lifting to SPIR-V or a vendor-neutral IR.
- **Tile-level recovery** — Aster operates at the instruction level, not the
  tensor level. Recovering MMA patterns is still a separate problem.
- **Maturity** — Aster is young ("pre-release", active since October 2025).
  APIs may change. But it's actively developed with 551 commits and backing
  from AMD.
- **Hidden kernel args** — Aster has no built-in awareness of HIP's implicit
  argument ABI. This must be handled externally.

## Option F: LLVM MIR (Level 1.75 — post-RA Machine IR, prototype complete)

**What it is.** LLVM's `MachineFunction` / `MachineInstr` representation — the
form code is in after register allocation in LLVM's backend pipeline. Each
`MachineInstr` is created via `BuildMI()` from an `MCInstrDesc` (TableGen), which
automatically populates implicit operand defs/uses. The result is serializable
as human-readable YAML MIR, the same format used by LLVM developers for testing.

The prototype lives in `hotswap/llvm_mir_proto/`.

**Prototype results (vecadd on gfx942).**
- 23/23 instructions lifted (100%)
- Same-ISA round-trip GPU-verified correct (1024 elements)
- MIR serialized with full implicit operands visible:
  - `V_CMP_GT_I32_e32_vi $sgpr4, $vgpr0, implicit-def $vcc, implicit $exec`
  - `S_AND_SAVEEXEC_B64_vi $vcc, implicit-def $exec, implicit-def $scc, implicit $exec`
  - `V_ADD_F32_e32_vi $vgpr6, $vgpr7, implicit $mode, implicit $exec`
- Multi-block CFG with proper successor edges
- AMDGPU-specific `machineFunctionInfo` populated (kernarg size, entry function, etc.)

**Key insight: implicit operands are free.**
This was the primary hypothesis for the prototype, and it was confirmed. When
you call `BuildMI(*MBB, MBB->end(), DebugLoc(), TII->get(opcode))`, the
`MCInstrDesc` from TableGen automatically adds implicit-def and implicit-use
operands. For `V_CMP_GT_I32_e32`, this means `implicit-def $vcc` and
`implicit $exec` appear without any manual handling. In both waveasm and Aster,
modeling these implicit operands required per-instruction special cases.

**What LLVM MIR solves vs. waveasm and Aster.**

| Limitation | waveasm | Aster | LLVM MIR |
|------------|---------|-------|----------|
| Implicit operands (VCC, SCC, EXEC) | Not modeled | Partially (VCC only, manual) | **Automatic from TableGen** |
| Instruction identification | Mnemonic string | Mnemonic string | **Opcode integer** |
| Register classes | None | Typed regs | **MachineRegisterInfo** |
| Wait counter infrastructure | Destroyed | Decoded fields | **Path to SIInsertWaitcnts pass** |
| $mode tracking | No | No | **Automatic** |

**Architectural gap: MIR is inspection-only.**
The prototype generates MIR for analysis and demonstration, but the HSACO is
produced via a separate MCInstPrinter round-trip path. The MIR is not connected
to the assembly generation because:
1. `llc` cannot process our MIR — it requires a non-`declare` IR function body
2. Using LLVM's `AsmPrinter` in-process requires private AMDGPU headers
   (`GCNPassConfig`, `GCNTargetMachine`) not in the public install
3. Building a custom `MachineInstr` → `MCInst` lowering is feasible but
   requires understanding AMDGPU's `MCInstLowering` internals

This is the single biggest limitation — until the MIR feeds into the codegen
path, cross-ISA translation through MIR is not possible. The MIR currently
serves as a high-quality analysis artifact, not a production IR.

**Possible paths forward for the AsmPrinter gap:**
1. Build LLVM from source with AMDGPU private headers accessible
2. Contribute a public API for MIR → assembly lowering to upstream LLVM
3. Serialize MIR, create a minimal IR function body, and run `llc`
4. Use the MIR for analysis only and use Aster for the actual codegen path

**Best for.** Analysis of implicit operands, register pressure, wait counter
placement. Potential future use as the "ground truth" representation that other
IRs can be validated against. Not yet suitable as a production codegen path.

## Option G: LLVM IR (Level 2 — typed SSA, prototype complete)

**What it is.** Full LLVM IR: typed values in SSA form, structured control flow
with `br i1`, `llvm.amdgcn.*` intrinsics for GPU-specific operations, and
`amdgpu_kernel` calling convention. This is the same representation the compiler
produces before backend lowering — the "canonical" form for LLVM-based
compilation.

The prototype lives in `hotswap/llvm_ir_proto/`.

**Prototype results (vecadd on gfx942).**
- 20 semantic instructions matched from the binary
- LLVM IR raised with: `llvm.amdgcn.workgroup.id.x()`,
  `llvm.amdgcn.workitem.id.x()`, `llvm.amdgcn.implicitarg.ptr()`,
  typed `getelementptr`, `load float`, `store float`, `fadd float`,
  `icmp slt`, `br i1`, `ret void`
- `llc` compiled the IR to assembly with **no manual metadata patching**
- Same-ISA round-trip GPU-verified correct (1024 elements)
- Generated HSACO: 6064 bytes (vs 5752 original — larger due to different
  register allocation and metadata layout, both correct)

**The breakthrough: standard backend integration.**

This is the only prototype where the raised representation feeds directly into
LLVM's standard AMDGPU backend. When `llc` processes the lifted LLVM IR:
- Register allocation is performed from scratch
- `SIInsertWaitcnts` inserts correct wait counters
- Kernel descriptor (`.amdhsa_kernel`) is generated with correct VGPR/SGPR
  counts, accum_offset, float modes
- Full `.amdgpu_metadata` YAML is emitted with all hidden argument entries
- No manual assembly patching needed — everything the other three prototypes
  required manual metadata handling for is automatic here

**What we actually raised.**

The lifted IR is semantically equivalent to the compiler's output:

```llvm
define amdgpu_kernel void @_Z6vecaddPfS_S_i(
    ptr addrspace(1) %A, ptr addrspace(1) %B,
    ptr addrspace(1) %C, i32 %N) #0 {
entry:
  %wg_id_x = call i32 @llvm.amdgcn.workgroup.id.x()
  %implicitarg_ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gsz_ptr = getelementptr inbounds i8, ptr addrspace(4) %implicitarg_ptr, i64 12
  %group_size_i16 = load i16, ptr addrspace(4) %gsz_ptr
  %group_size = zext i16 %group_size_i16 to i32
  %base = mul i32 %wg_id_x, %group_size
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gid = add i32 %base, %tid
  %cmp = icmp slt i32 %gid, %N
  br i1 %cmp, label %body, label %exit
body:
  %idx = sext i32 %gid to i64
  %ptr_A = getelementptr inbounds float, ptr addrspace(1) %A, i64 %idx
  ; ... loads, fadd, store ...
  br label %exit
exit:
  ret void
}
```

**Key finding: the backend does everything.**

The three biggest engineering challenges in the other prototypes were:
1. Kernel descriptor metadata (manual `.amdhsa_*` directives)
2. Hidden argument YAML metadata (manual `.amdgpu_metadata`)
3. Wait counter placement (manual or elided)

All three are **automatically handled** by `llc` when given proper LLVM IR.
This eliminates entire categories of bugs that the other prototypes suffered
from (`accum_offset` mismatch, missing hidden args, incorrect `kernarg_size`).

**Current limitation: pattern-match based raiser.**

The prototype validates that the instruction stream matches the vecadd pattern
and emits the corresponding IR directly. It does not yet walk MachineInstrs
and raise per-instruction. Generalizing to arbitrary kernels requires building
the per-opcode pattern matcher described in the plan. This is the core
engineering challenge — the pipeline itself is proven.

**What it enables (not yet demonstrated but now clearly feasible).**
- **Cross-ISA via `-mcpu` flag**: Change `llc -mcpu=gfx942` to `gfx1100` or
  any supported target. The backend handles instruction selection, encoding,
  and metadata for the new target.
- **Optimization**: Standard LLVM passes (loop unrolling, vectorization,
  inlining) can operate on the lifted IR.
- **Validation**: The lifted IR can be `diff`ed against `hipcc -emit-llvm`
  output to verify semantic correctness.

**Best for.** Full semantic recovery from binaries. Production codegen via
LLVM's standard backend with no manual patching. Cross-ISA retargeting.

## Option E: Layered approach (recommended direction, updated)

A pragmatic architecture that uses **multiple IR levels** and lifts only as
high as is needed (or feasible) for each use case.

**Key update from prototyping:** The original plan proposed a two-stage lift
(MCInst → waveasm → Aster). The Aster prototype demonstrated that **direct
lifting from MCInst → Aster is simpler and more natural** — the intermediate
waveasm layer adds no value because both lifters start from the same MCInst
input. The updated architecture reflects this.

```
                    ┌─────────────────────────────────────────────┐
                    │    Binary (.text bytes)                      │
                    └──────────────────┬──────────────────────────┘
                                       │ MCDisassembler (mechanical)
                                       │ LLVM MCInst stream
                    ┌──────────────────┼──────────────┬────────────────┐
                    ▼                  ▼              ▼                ▼
  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌────────────┐
  │  LLVM IR (L2)     │  │  LLVM MIR (L1.75)│  │ Aster (L1.5) │  │ waveasm    │
  │  **Target path**  │  │  Analysis path   │  │ Alt codegen  │  │ (L1)       │
  │  Typed SSA,       │  │  Implicit ops    │  │ SSA, typed   │  │ 1:1, escape│
  │  llvm.amdgcn.*    │  │  from TableGen   │  │ regs, CFG    │  │ hatch      │
  └────────┬──────────┘  └────────┬─────────┘  └──────┬───────┘  └────────────┘
           │                      │                    │
           │ llc (standard        │ Analysis           │ Cross-ISA
           │ AMDGPU backend)      │                    │ (Aster pipeline)
           ▼                      ▼                    ▼
  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐
  │ HSACO            │  │ Analysis data    │  │ Retargeted    │
  │ (auto metadata,  │  │ (wait counters,  │  │ HSACO         │
  │  wait counters,  │  │  reg pressure,   │  │               │
  │  reg alloc)      │  │  implicit deps)  │  │               │
  └──────────────────┘  └──────────────────┘  └───────────────┘
           │
           │ Change -mcpu flag
           ▼
  ┌──────────────────┐
  │ Cross-ISA HSACO  │
  │ (any AMDGPU      │
  │  target)         │
  └──────────────────┘
```

**How it works (updated):**

1. **Lift directly to Aster amdgcn dialect.** MCDisassembler produces an
   MCInst stream. Each instruction is classified by format and mapped to
   the corresponding Aster op class with an `#amdgcn.inst<...>` attribute.
   Registers are typed (`!amdgcn.vgpr`, `!amdgcn.sgpr`, ranges). Wait
   counters are decoded into named fields. Kernel metadata is populated
   from the ELF's code object. This is the prototype we have working today.

2. **For same-vendor retargeting (RDNA ↔ CDNA):** use Aster's infrastructure
   for mnemonic remapping, addressing mode lowering, wave width translation,
   and register re-allocation. Emit HSACO through Aster's `translateModule`.

3. **For cross-vendor retargeting (AMD → NVIDIA):** lift from Aster to SPIR-V
   or GPU dialect. Aster's SSA, typed registers, and multi-block CFG make
   this significantly easier than lifting from waveasm or raw MCInst.

4. **For known ML patterns (optional, future):** pattern-match instruction
   sequences in Aster IR to recover high-level operations (e.g., MFMA
   sequences → tile-level ops).

**What waveasm's role is now:**

The waveasm prototype remains as a parallel implementation and reference point.
It has value for:
- Cross-ISA translation (gfx1250 → gfx942, already demonstrated)
- Quick coverage of new kernels (the `waveasm.raw` escape hatch)
- Comparison and validation against the Aster path

However, waveasm is **not a prerequisite** for the Aster path. They lift
independently from the same MCInst input.

**Why this architecture:**

- **Fail loudly, no fallbacks.** If the Aster lifter encounters an unsupported
  instruction, it reports a diagnostic and fails. No silent degradation to
  `waveasm.raw` or opaque bytes. This is a deliberate design choice —
  fallbacks hide bugs and produce binaries with unknown correctness.

- **Upstream leverage.** Aster is actively developed by AMD. Features we need
  (instruction format metadata, register allocator, HSACO generation) are
  being built anyway. Contributing our lifting work upstream aligns with
  Aster's planned "raising from existing ASM" feature.

- **Addresses most waveasm limitations.** The Aster path directly solves 5 of
  the 8 waveasm `SHORTCUTS_AND_LIMITATIONS.md` items we've verified in the
  prototype (SSA, wait counters, register types, assembly emission, kernel
  structure). The remaining 3 (register allocation, in-process assembly,
  metadata patching) have clear paths but need more work.

## Recommendation (updated April 2026)

**Short-term evaluation — DONE.** All four prototypes have been built and verified:
1. ~~Build waveasm and verify cross-ISA translation~~ — Done (gfx1250 → gfx942).
2. ~~Build Aster and verify MCInst → amdgcn dialect lifting~~ — Done.
3. ~~Verify Aster's HSACO generation produces a working binary~~ — Done.
4. ~~Build LLVM MIR and verify implicit operand modeling~~ — Done.
5. ~~Verify all three pipelines produce correct GPU execution~~ — Done (1024/1024).
6. ~~Build LLVM IR raiser and verify full backend integration~~ — Done (1024/1024).

**Key finding from LLVM IR prototype:** Raising to LLVM IR and using `llc` as
the backend **eliminates all manual metadata handling**. The kernel descriptor,
hidden argument metadata, wait counter insertion, and register allocation are
all performed automatically by the AMDGPU backend. This was the single biggest
source of bugs in the other three prototypes. The LLVM IR path is the only one
where the generated HSACO is **fully correct by construction** rather than
requiring hand-tuned assembly patching.

**Key finding from LLVM MIR prototype:** Implicit operands (VCC, SCC, EXEC,
$mode) are free from TableGen via `BuildMI()`. LLVM MIR is the right
representation for analysis tasks. However, the MIR is not connected to the
codegen path — using LLVM MIR for analysis while targeting LLVM IR for codegen
is the recommended layered approach.

**Updated recommendation: LLVM IR as the target representation.**

The LLVM IR path should be the primary codegen path because:
1. Standard backend integration (no manual patching)
2. Cross-ISA retargeting via `-mcpu` flag
3. Access to LLVM optimization passes
4. Validation by diffing against `hipcc -emit-llvm` output

Aster remains valuable as an alternative codegen path for cases where the full
IR raise is too difficult (complex kernels with patterns the raiser doesn't
recognize yet).

**Next phase (now):** Build the per-instruction raiser:
1. **Implement per-opcode MachineInstr → IR raising.** The current prototype
   pattern-matches the whole vecadd kernel. Generalize by walking each
   MachineInstr and emitting the corresponding IR operation. Use `MCInstrDesc`
   flags (`isBranch`, `mayLoad`, `mayStore`) to guide classification.
2. **Parse kernel metadata from ELF.** Extract kernarg layout, hidden arg
   offsets, and kernel signature from the original code object's `.note`
   section. Eliminate hardcoded signatures.
3. **Build SSA construction.** Track physical register → IR value mappings.
   Implement register-to-value translation for SGPR and VGPR operands.
4. **Expand instruction coverage.** Priority: MUBUF/scratch loads, DS (shared
   memory), atomics, additional comparison ops, MFMA (matrix ops).
5. **Test with partial wavefronts** (e.g., N=1000) to validate or fix the
   exec-mask → branch conversion.
6. **Cross-ISA demonstration.** Raise a gfx942 binary to LLVM IR, then
   compile with `llc -mcpu=gfx1100` to demonstrate cross-ISA via the standard
   backend.

**Medium-term (3-6 months):** Production hardening:
- Use Aster's register allocator for cross-ISA lowerings that introduce
  new registers (e.g., `scale_offset` decomposition).
- Handle multi-kernel code objects (parse ELF symbol table for kernel
  offsets).
- Contribute the MCInst → Aster lifter upstream — this aligns with Aster's
  planned "raising from existing ASM" feature.
- Build out a real test suite beyond vecadd: reduction, GEMM, attention,
  kernels with shared memory and divergent control flow.

**Long-term (6+ months):**
- **Cross-vendor path:** Build an Aster → SPIR-V lifter. Aster's SSA and
  typed registers make this significantly easier than lifting from raw
  waveasm. This opens AMD → NVIDIA retargeting.
- **Pattern recovery:** Detect MFMA sequences in Aster IR and lift to
  tile-level operations for higher-level optimization.

## Comparison Matrix (updated with all four prototype results)

| Criterion                     | waveasm (L1) | Aster (L1.5) | LLVM MIR (L1.75) | **LLVM IR (L2)** | SPIR-V (L2) | TileIR (L3) | Layered  |
|-------------------------------|:------------:|:------------:|:-----------------:|:----------------:|:------------:|:------------:|:--------:|
| Lift from binary              | Mechanical   | Mostly mech. | Mechanical        | **Pattern-match** | Hard        | Research     | Staged   |
| AMD → AMD retarget            | Demonstrated | Demo (same)  | Analysis only     | **Via -mcpu flag** | Overkill   | Overkill     | L2       |
| AMD → NVIDIA retarget         | No           | No           | No                | **No (AMD only)** | Yes        | Yes          | L2 path  |
| Optimization potential        | Low          | Medium       | High (backend)    | **Highest (LLVM opt)** | Medium | High         | All      |
| Correctness risk at lift      | Minimal      | Low          | Minimal           | **Medium (patterns)** | Moderate | High         | Varies   |
| Engineering effort to here    | Done         | Done         | Done              | **Done (prototype)** | 6-12 mo  | Research     | Staged   |
| Remaining to production       | Medium       | Medium       | High (AsmPrinter) | **High (raiser)** | High      | Very high    | Staged   |
| Round-trip fidelity           | High         | High         | High              | **Low (recompiled)** | Low      | None         | High     |
| Manual metadata patching      | **Yes**      | **Yes**      | **Yes**           | **No (automatic)** | No       | No           | Varies   |
| Implicit operands modeled     | No           | Partial      | Yes (automatic)   | **N/A (eliminated)** | Yes     | N/A          | Yes      |
| Instruction identification    | String       | String       | Opcode integer    | **Pattern match** | Typed ops  | High-level   | Mixed    |
| Real SSA                      | No           | Yes          | No (phys regs)    | **Yes**          | Yes        | Yes          | Yes      |
| Register classes              | No           | Typed regs   | MachineRegInfo    | **Virtual values** | Yes      | Yes          | Yes      |
| Wait counter handling         | No           | Fields only  | SIInsertWaitcnts  | **Auto (llc)**   | N/A        | N/A          | Yes      |
| Structured control flow       | No           | Yes          | Multi-block       | **br i1 (structured)** | Yes    | Yes          | Yes      |
| Kernel descriptor generation  | Manual       | Manual       | Manual            | **Automatic (llc)** | Auto    | Auto         | Varies   |
| Escape hatch for unknowns     | waveasm.raw  | Fail loudly  | All opcodes       | **Fail loudly**  | None       | None         | Fail     |
| Active upstream               | AMD/iree-org | AMD/iree-org | LLVM              | **LLVM**         | Khronos    | NVIDIA       | Mixed    |
