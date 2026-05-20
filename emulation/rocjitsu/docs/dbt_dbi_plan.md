# DBI and DBT Implementation Plan for rocjitsu

## Executive Summary

This plan describes the complete implementation of Dynamic Binary Instrumentation (DBI) and Dynamic Binary Translation (DBT) capabilities for the `rocjitsu` project. The document is organized into four pillars that build on each other, followed by an integration plan for ROCm/HSA, and a phased implementation roadmap.

### ISA Specifications Available

| File | Architecture | GFX | Notes |
|------|-------------|-----|-------|
| [cdna1](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi100-cdna1-shader-instruction-set-architecture.pdf) | CDNA1 | GFX908 | Wave64 only |
| [cdna2](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf) | CDNA2 | GFX90a | Wave64 only; XNACK (memory retry — affects DBI memory-op instrumentation) |
| [cdna3](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) | CDNA3 | GFX940/942 | Wave64 only; AccVGPR (separate MFMA register file — SpillManager/liveness must handle); XCD (multi-die topology) |
| [cdna4](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) | CDNA4 | GFX950 | Wave64 only |
| [rdna1](https://docs.amd.com/v/u/en-US/rdna-shader-instruction-set-architecture) | RDNA1 | GFX10xx | Wave32 default; Wave64 supported (`ENABLE_WAVEFRONT_SIZE32=0`) |
| [rdna2](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture) | RDNA2 | GFX10xx | Wave32 default; Wave64 supported (`ENABLE_WAVEFRONT_SIZE32=0`) |
| [rdna3](https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0) | RDNA3 | GFX11xx | Wave32 default; Wave64 supported (`ENABLE_WAVEFRONT_SIZE32=0`) |
| [rdna3.5](https://docs.amd.com/v/u/en-US/rdna35_instruction_set_architecture) | RDNA3.5 | GFX115x | Wave32 default; Wave64 supported (`ENABLE_WAVEFRONT_SIZE32=0`) |
| [rdna4](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture) | RDNA4 | GFX12xx | Wave32 default; Wave64 supported (`ENABLE_WAVEFRONT_SIZE32=0`) |

---

# Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Existing `isa/` Infrastructure Mapping](#isa-mapping)
3. [Pillar 1: ELF and Code Object Foundation](#pillar-1)
4. [Pillar 4: Def-Use and Dataflow Analysis](#pillar-4)
5. [Pillar 2: Dynamic Binary Instrumentation (DBI)](#pillar-2)
6. [Pillar 3: Dynamic Binary Translation (DBT)](#pillar-3)
7. [Pillar 5: ROCm Integration Hooks](#pillar-5)
8. [Source Attribution](#source-attribution)
9. [Implementation Roadmap](#roadmap)

---

## Architecture Overview

```
                     rocjitsu DBI/DBT Architecture
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        User Application                              │
  └──────────────────────┬───────────────────────────────────────────────┘
                         │  hsa_executable_load_agent_code_object()
                         │  hsa_executable_freeze()
  ┌──────────────────────▼───────────────────────────────────────────────┐
  │             ROCm Integration Hooks (Pillar 5)                        │
  │  HSA Tools Layer: HSA_TOOLS_LIB=librocjitsu_hooks.so                 │
  │  OnLoad() → CoreApiTable fn-ptr override + AMD intercept queue       │
  │  Load-time ELF patching ║ Dispatch-time kernel_object swap           │
  │  RjTranslationPolicy, env-var knobs, on-disk cache                   │
  └──────────────────────┬───────────────────────────────────────────────┘
                         │  rj_code_object_t (parsed ELF)
  ┌──────────────────────▼───────────────────────────────────────────────┐
  │           Dataflow & Analysis Engine (Pillar 4)                      │
  │  LivenessAnalysis, DefUseChain, RegisterSet                          │
  └────────────┬──────────────────────────────────┬──────────────────────┘
               │                                  │
  ┌────────────▼──────────┐            ┌──────────▼──────────────────────┐
  │  DBI Engine (Pillar 2)│            │  DBT Engine (Pillar 3)          │
  │  Instrumentor         │            │  BinaryTranslator               │
  │  TrampolineBuilder    │            │  LegalizationLookupFn           │
  │  BufferInjector       │            │  WaitcntTranslator              │
  │                       │            │  Wave emulation, MfmaStub       │
  └────────────┬──────────┘            └──────────┬──────────────────────┘
               │                                  │
  ┌────────────▼──────────────────────────────────▼──────────────────────┐
  │          ELF and Code Object Foundation (Pillar 1)                   │
  │  CodeObjectPatcher: text rewrite, branch fixup, symbol update,       │
  │  kernel descriptor update, ELF re-emission                           │
  │  SpillManager: scratch slot allocation, {isa}/addr_calc.h formulas   │
  │  {isa}/machine_insts.h · {isa}/encodings.h — instruction encoding    │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## Existing `isa/` Infrastructure Mapping {#isa-mapping}

The `isa/` and `code/` subsystems already provide most of the raw material each pillar needs. This section maps the existing classes to their roles in each pillar so implementations don't re-invent what already exists.

### Key Existing Classes

Each ISA variant lives under `isa/arch/amdgpu/{isa}/`. As of Phase A (branch `phaseA-multi-isa-targets`), **all 9 ISA targets are fully generated and compile-clean under `-Werror`**: cdna1, cdna2, cdna3, cdna4, rdna1, rdna2, rdna3, rdna3_5, rdna4. All files listed below exist identically in every `{isa}/` directory; the DBI/DBT implementation works against the abstract interfaces and templates them over `{isa}` — no per-ISA special-casing in the analysis or translation core.

> **Phase A note:** RDNA4 uses distinct encoding family names (`ENC_VFLAT`, `ENC_VGLOBAL`, `ENC_VSCRATCH`, `ENC_VDS`, `ENC_VBUFFER`) and field renames (`opsel`/`opsel_hi` without underscores, `vsrc` for flat store source, `nv` for coherency). These are fully handled by profile properties in the amdisa codegen pipeline (`Rdna4Profile`). The RDNA4 `addr_calc.h` provides overloads for `VglobalMachineInst` and `VscratchMachineInst` as Phase A stubs (throw `UnimplementedInst`) to be completed in Phase D.

| Class / File | What it provides |
|---|---|
| `Instruction` (`isa/instruction.h`) | Abstract instruction with `src_operands_` and `dst_operands_` vectors, `InstFlags` bitmask, size in bytes, mnemonic string |
| `IsaOperand<Isa>` / `Operand` (`isa/operand.h`) | Per-operand `encoding_value_` (register number), `OperandType` tag, `name()` string |
| `{isa}::OperandType` (`isa/arch/amdgpu/{isa}/operand_types.h`) | Per-ISA enum with per-class value ranges (e.g., `OPR_SGPR_MIN/MAX`, `OPR_VGPR_MIN/MAX`, `OPR_VCC`, `OPR_EXEC`, `OPR_M0`, `OPR_FLAT_SCRATCH`; CDNA ISAs also have `OPR_ACCVGPR_*`) — classifies operands by register file without string parsing |
| `{isa}::Isa::WF_SIZE` (`isa/arch/amdgpu/{isa}/isa.h`) | `static constexpr uint32_t WF_SIZE` — exec mask width at compile time via `GpuIsa` concept; 64 for all CDNA ISAs, 32 or 64 for RDNA ISAs depending on kernel descriptor mode |
| `{isa}/machine_insts.h` | Bitfield structs (`Sop1MachineInst`, `SoppMachineInst`, `SmemMachineInst`, `FlatMachineInst`, `Vop3pMachineInst`, ...) for both **decoding** guest instructions and **emitting** host instructions by filling fields and casting to `uint32_t` |
| `{isa}/encodings.h` | Instruction format base classes (`Sop1`, `Vop2`, `Vop3p`, ...) — serve as emitter templates when constructing translated instructions |
| `{isa}/addr_calc.h` | `flat_calculate_addresses()`, `mubuf_calculate_addresses()` — per-lane scratch address formulas reused by `SpillManager` to compute spill slot addresses |
| `{isa}/mfma_exec.h` | `input_loc()`, `output_loc_32()`, `output_loc_64()`, `exec_f32()`, `exec_i32_i8()`, `exec_f64()` — reference MFMA implementation including register layout; CDNA ISAs only (RDNA has no MFMA); used by DBI for AccVGPR clobber detection (CDNA2+ only — CDNA1 MFMA outputs to regular VGPRs, not AccVGPRs) and by DBT as the software fallback body |
| `{isa}::Decoder` (`isa/decoder.h`, `isa/arch/amdgpu/{isa}/decoder.h`) | Factory that decodes raw `uint32_t` words into typed `Instruction` subclasses; serves as the **guest decoder** in DBT |
| `BasicBlock` (`code/basic_block.h`) | Splits decoded instructions at branch terminators using `InstFlags::BRANCH`; provides `instructions()` iterator, `start_offset()` / `end_offset()`, and CFG predecessor/successor links for byte-level position in `.text` |
| `AmdGpuCodeObject` (`code/amdgpu_code_object.h`) | ELF parser; `kernel_descriptor_offset()` looks up `.kd` symbols; `image_data()` / `image_size()` give the raw ELF bytes to the patcher; `code_sections()` returns all executable sections, including DBT-generated `.rj_translations` |
| `amdgpu_elf.h` | `Elf64_Ehdr`, `Elf64_Shdr`, `Elf64_Sym`, `EF_AMDGPU_MACH_*` constants — used by `CodeObjectPatcher` to locate and rewrite sections and by `detect_arch_from_elf()` for ISA detection |
| `Executable` (`code/executable.h`) | HIP fat binary parser; `code_object(target, index)` extracts `AmdGpuCodeObject` per GFX target; used in Pillar 5 to parse application binaries before the pipeline runs |

### CFG Edges Belong Directly in `BasicBlock`

`BasicBlock::build()` returns a `vector<unique_ptr<BasicBlock>>` with predecessor/successor links populated. The liveness analysis (Pillar 4) consumes that connected graph directly.

**Design decision: embed edges in `BasicBlock` directly**, not in a separate `ControlFlowGraph` class.

```cpp
// BasicBlock fields (code/basic_block.h — hand-written, not autogenerated):
std::vector<BasicBlock *> successors_;    // raw non-owning pointers
std::vector<BasicBlock *> predecessors_;

const std::vector<BasicBlock *> &successors() const { return successors_; }
const std::vector<BasicBlock *> &predecessors() const { return predecessors_; }
```

`BasicBlock::build()` populates these in a second pass after all blocks are constructed: for each block's terminator, call `inst->branch_offset_bytes()` (a virtual method on `Instruction` overridden by ISA-specific subclasses) to get the signed byte delta to the branch target, compute the absolute target offset, look it up in an `offset → BasicBlock*` map built during the first pass, and link successor/predecessor pairs. Fall-through blocks (blocks whose terminator is not a branch — e.g., a non-branching ALU instruction at a basic-block boundary created by a subsequent block's incoming branch target) link the next sequential block in the flat list as their sole successor. Only true program-exit blocks (`s_endpgm`, `s_trap`, or the last block in the function with `InstFlags::PROGRAM_TERMINATOR`) get an empty successor list. Indirect branches (`InstFlags::INDIRECT_BRANCH`, e.g., `s_setpc_b64`) also get an empty successor list because their target is not statically known.

This gives an **implicit CFG** — the block list returned by `build()` *is* the CFG. No separate `ControlFlowGraph` class or object is needed. `LivenessAnalysis` takes `KernelBlockScope` (`span<BasicBlock *const>`) and traverses `bb->successors()` directly, with no auxiliary object to thread through call sites or manage lifetime for.

The `ControlFlowGraph` phase (Phase 2 in the roadmap) is therefore implemented as the two fields and the edge-building second pass in `BasicBlock::build()`. No new class file is required.

### Pillar-by-Pillar Usage Summary

**Pillar 1 (ELF foundation)** provides `CodeObjectPatcher` and `SpillManager` as the write layer; all pillars that produce modified binaries drive these two components. The `{isa}/machine_insts.h` and `{isa}/encodings.h` structs are the instruction encoding primitives shared across Pillars 2 and 3.

**Pillar 2 (DBI)** uses `{isa}/machine_insts.h` bitfield structs to emit new instructions (spill stores, restores, `s_swappc_b64`), uses `{isa}/addr_calc.h` formulas inside `SpillManager`, and consults `{isa}/mfma_exec.h`'s `output_loc_32/64()` to identify AccVGPR ranges clobbered by MFMA instructions at the patch point (CDNA ISAs only).

**Pillar 3 (DBT)** calls `{isa}::Decoder::decode()` to decode guest instructions, uses cross-ISA `OperandType` comparison to detect capability gaps (e.g., `OPR_ACCVGPR` present in CDNA `operand_types.h` but absent in RDNA), and reuses the guest ISA's `mfma_exec.h` `exec_f32()` / `exec_i32_i8()` / `exec_f64()` as the software fallback body when translating to a target that lacks MFMA.

**Pillar 4 (Liveness)** builds `InstDefUse` from explicit operand vectors plus `Instruction::implicit_uses()` / `implicit_defs()`, classifies ordinary SGPR/VGPR/AccVGPR operands through `Operand::to_register_ref()` (not by string-parsing `name()`), and runs `LivenessAnalysis` over one `KernelBlockScope` at a time. Special architectural state such as EXEC, VCC, SCC, M0, TTMP, and FLAT_SCRATCH is represented in `RegClass` but not tracked by `RegisterSet` yet; special-register liveness remains future work.

**Pillar 5 (ROCm hooks)** intercepts via the ROCR HSA Tools Layer (`HSA_TOOLS_LIB`) rather than LD_PRELOAD symbol shadowing. The `RjHsaLayer::OnLoad` callback receives ROCR's live `CoreApiTable` and replaces `hsa_executable_load_agent_code_object_fn` and `hsa_queue_create_fn` with wrappers. The load-time wrapper calls `rj_code_executable_create()` / `rj_code_executable_get_code_object()` on the intercepted ELF, uses `AmdGpuCodeObject::target_id()` to decide whether translation is needed, and reads `EF_AMDGPU_MACH` from `amdgpu_elf.h` in `detect_arch_from_elf()`. The queue-creation wrapper calls `hsa_amd_queue_intercept_create` (from `AmdExtTable`) to register a dispatch-time packet handler for lazy translation and per-kernel observability.

---

## Pillar 1: ELF and Code Object Foundation {#pillar-1}

The ELF/code-object layer is the substrate that every other pillar writes through. Pillars 2 (DBI) and 3 (DBT) both produce modified binaries; both do so by driving `CodeObjectPatcher` and `SpillManager`. Pillar 4 (analysis) reads the decoded instruction stream that lives inside the same ELF. Pillar 5 (ROCm hooks) hands raw ELF bytes into this layer at load time.

The detailed design of each component is specified inside the pillar that owns its primary use case, but the components themselves are general-purpose and shared:

| Component | Detailed spec | Role |
|---|---|---|
| `CodeObjectPatcher` | §2.6 (DBI) | Insert/overwrite bytes in `.text`; fix branch offsets; update `.symtab` and section headers; re-emit valid ELF |
| `SpillManager` | §2.3 (DBI) | Allocate non-overlapping flat-scratch slots from a `RegisterSet`; compute per-slot GPU VAs using `{isa}/addr_calc.h`; update `private_segment_fixed_size` in the kernel descriptor |
| `{isa}/machine_insts.h` | §2 (ISA mapping) | Bitfield structs for decoding guest instructions and emitting new host instructions |
| `{isa}/encodings.h` | §2 (ISA mapping) | Instruction format base classes used as emitter templates |
| `AmdGpuCodeObject` | §2 (ISA mapping) | ELF parser — exposes raw bytes, kernel descriptor offsets, and target ISA |
| `{isa}::Decoder` | §3 (DBT) | Decodes raw `uint32_t` words into typed `Instruction` subclasses; guest decoder in DBT |

---

## Pillar 4: Def-Use and Dataflow Analysis {#pillar-4}

### 4.1 Motivation and Design Rationale

The existing `rocjitsu` IR (the `Instruction` / `BasicBlock` / `InstructionList` hierarchy in `isa/instruction.h` and `code/basic_block.h`) provides decoded instructions with operand vectors and `InstFlags`, but exposes no register-level def-use information. Both DBI (spill-set calculation) and DBT (register renaming, waitcnt rewriting) require precise knowledge of which registers are live at every program point.

The key observation is that most necessary raw data already exists: every decoded instruction carries `src_operands_` and `dst_operands_` vectors of `Operand*`, each of which exposes an `encoding_value_` (the register number within its file) and an `OperandType` tag (from `{isa}/operand_types.h`) that classifies ordinary SGPR, VGPR, and AccVGPR operands. **No string parsing of `name()` is needed** — register classification is done through `Operand::to_register_ref()`, which is generated from `OperandType` metadata and value-range constants such as `OPR_SGPR_MIN/MAX`.

Hidden operands that are not printed in the explicit operand list use instruction-level implicit hooks. The current implementation uses this for FLAT `saddr` address dependencies, adding ordinary SGPR uses while leaving the hidden field out of disassembly.

The analysis subsystem must:
- Work across the entire `IsaOperand<{isa}::Isa>` family (CDNA1–4, RDNA1–4) without per-ISA special cases in generic analysis code.
- Provide O(|RegisterSet|/64) set operations (union, intersection, complement) via `std::bitset` word-parallel operations.
- Preserve old values for predicated definitions and EXEC-masked vector definitions conservatively, without modeling EXEC liveness yet.
- Be fast enough to run at kernel-load time — total complexity O(D · N · |RegisterSet|/64), where N is instruction count and D is the number of dataflow iterations to convergence (typically 3–5 for structured GPU kernels).

### 4.2 New File Layout

```
lib/rocjitsu/src/rocjitsu/analysis/
  def_use_chain.h/.cpp   # Per-instruction def and use sets
  liveness.h/.cpp        # Kernel-scoped backwards dataflow: block live-in/live-out,
                         # plus materialized live-before sets per instruction

lib/rocjitsu/src/rocjitsu/isa/
  register_set.h         # RegClass enum, RegisterRef, RegisterSet, expand/erase/contains

lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/{isa}/
  operand.h/.cpp         # (existing) — gains autogenerated to_register_ref() override body
  encodings.h/.cpp       # (existing) — encoding bases may override implicit_defs/uses()
```

### 4.3 `RegisterRef`, `RegisterSet`, and `RegClass`

`RegisterSet` uses per-class `std::bitset` members sized to the supported AMDGPU family maxima — no flat namespace. The current implementation tracks ordinary SGPRs, VGPRs, and AccVGPRs. Special register classes are present in `RegClass` so operands can classify them consistently, but `RegisterSet` ignores those classes until special-register liveness has a concrete consumer.

```cpp
// lib/rocjitsu/src/rocjitsu/isa/register_set.h
#pragma once
#include <bitset>
#include <cstdint>

namespace rocjitsu {

/// @brief ISA-independent register-file class.
enum class RegClass : uint8_t {
  SGPR, VGPR, ACC_VGPR,
  EXEC, VCC, SCC, M0, FLAT_SCRATCH, TTMP, PC // Classified, not tracked yet.
};

/// @brief A contiguous register reference within one register file.
struct RegisterRef {
  RegClass cls;
  uint16_t index;
  uint8_t  width = 1;

  constexpr bool operator==(const RegisterRef &) const = default;
};

/// @brief Per-class register sets for dataflow analysis.
class RegisterSet {
public:
  void expand(RegisterRef ref);
  void erase(RegisterRef ref);
  [[nodiscard]] bool contains(RegisterRef ref) const;
  [[nodiscard]] bool none() const;
  [[nodiscard]] bool intersects(const RegisterSet &rhs) const;

  RegisterSet &operator|=(const RegisterSet &rhs);
  RegisterSet &operator&=(const RegisterSet &rhs);
  RegisterSet &operator-=(const RegisterSet &rhs);

private:
  std::bitset<REGISTER_SET_MAX_SGPRS> sgprs_;
  std::bitset<REGISTER_SET_MAX_VGPRS> vgprs_;
  std::bitset<REGISTER_SET_MAX_ACC_VGPRS> acc_vgprs_;
};

} // namespace rocjitsu
```

`REGISTER_SET_MAX_SGPRS` / `REGISTER_SET_MAX_VGPRS` are derived from the shared CDNA/RDNA ISA base traits. `REGISTER_SET_ALLOCATABLE_SGPRS` is the conservative cross-family SGPR allocation bound used by `find_free_sgpr()` and `find_free_sgpr_pair()`.

### 4.4 Autogenerated `to_register_ref` in ISA Operand Classes

The codegen extends `{isa}/operand.cpp` with the `to_register_ref()` override. No new files; the override lives alongside the rest of the ISA-specific operand implementation:

```cpp
// isa/operand.h — add to Operand base class
/// @brief Map this operand to its register reference for analysis.
/// Returns nullopt for immediates, literals, labels, and other non-register operands.
[[nodiscard]] virtual std::optional<RegisterRef> to_register_ref() const;
```

```cpp
// isa/arch/amdgpu/cdna3/operand.cpp  (autogenerated section — do not edit)
std::optional<RegisterRef> cdna3::Operand::to_register_ref() const {
  const auto reg_width = static_cast<uint8_t>(size_bits_ > 32 ? size_bits_ / 32 : 1);
  switch (opr_type_) {
    case OperandType::OPR_SDST: case OperandType::OPR_SRC: case OperandType::OPR_SSRC:
      if (encoding_value_ >= OPR_SDST_SGPR_MIN && encoding_value_ <= OPR_SDST_SGPR_MAX)
        return RegisterRef{RegClass::SGPR, uint16_t(encoding_value_ - OPR_SDST_SGPR_MIN),
                           reg_width};
      break; // Special registers are not tracked by RegisterSet yet.
    case OperandType::OPR_VGPR: case OperandType::OPR_SRC_VGPR:
      return RegisterRef{RegClass::VGPR, uint16_t(encoding_value_ - OPR_VGPR_MIN), reg_width};
    case OperandType::OPR_ACCVGPR: case OperandType::OPR_SRC_ACCVGPR:
      return RegisterRef{RegClass::ACC_VGPR, uint16_t(encoding_value_ - OPR_ACCVGPR_MIN),
                         reg_width};
    default:
      return std::nullopt;
  }
  return std::nullopt;
}
```

Generated operands normalize `encoding_value_` to the per-class bitset index by subtracting the operand-selector minimum for that register class. The ISA is resolved at decode time via the concrete subclass type; no runtime ISA lookup occurs.

`implicit_defs()` and `implicit_uses()` on `Instruction` add hidden register effects directly to `RegisterSet`. The current generated use is FLAT `saddr`: non-null scratch/global addressing fields are added as ordinary SGPR uses without being printed as source operands.

Generic analysis code in `def_use_chain.h` calls only the virtuals:

```cpp
// lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.h (excerpt)

namespace rocjitsu {

/// @brief Def and use sets for a single instruction.
///
/// Constructed directly from an instruction — no separate factory function needed.
/// Calls operand.to_register_ref() and inst.implicit_defs/uses()
/// with no ISA headers or OPR_* constants in this class.
class InstDefUse {
public:
  InstDefUse(const Instruction &inst);

  RegisterSet defs;                        ///< Registers overwritten by the instruction.
  RegisterSet uses;                        ///< Registers read before the instruction writes defs.
  bool has_exec_masked_vector_def = false; ///< True if any vector def is predicated by EXEC.
  bool has_predicated_def = false;         ///< True if defs preserve old values on some paths.
};

} // namespace rocjitsu
```

Whether an operand is a def or use comes from its position in `dst_operands_` vs `src_operands_`, not from `opr_type_`. The virtual `to_register_ref()` tells you *which register*; the vector membership tells you *def or use*.

**Implicit defs and uses** (registers not encoded as explicit operands but read or written by an instruction) are ISA-specific and must not be hardcoded in generic analysis. They are handled by two virtual methods on the `Instruction` base class:

```cpp
// isa/instruction.h — add to Instruction base class

/// @brief Registers implicitly written by this instruction (not in dst_operands_).
virtual void implicit_defs(RegisterSet &defs) const {}

/// @brief Registers implicitly read by this instruction (not in src_operands_).
virtual void implicit_uses(RegisterSet &uses) const {}
```

ISA-specific encoding classes override these for hidden ordinary-register effects. For example, FLAT encodings add the hidden `saddr` field as a one-SGPR scratch use or a two-SGPR global use, ignoring the null sentinel. `InstDefUse(inst)` in generic analysis code calls both virtuals alongside iterating `dst_operands_`/`src_operands_`, with no knowledge of which instructions have hidden fields. Special-register def/use hooks are not implemented yet.

### 4.5 CFG Edges in `BasicBlock`

CFG edges are stored directly on `BasicBlock` (see §Pillar 1 / Phase 2). No separate `ControlFlowGraph` class is needed. Analysis code receives the block list from `BasicBlock::build()` and traverses `bb->successors()` / `bb->predecessors()` directly.

The edge-building second pass in `BasicBlock::build()` works as follows:

1. Build an `offset → BasicBlock*` map over `block->start_offset()` for all blocks.
2. For each block whose terminator has `InstFlags::BRANCH` or `InstFlags::COND_BRANCH`, call the virtual `inst->branch_offset_bytes()` to get the signed byte delta and compute:
   ```cpp
   uint64_t target = terminator_byte_offset + terminator->size() + terminator->branch_offset_bytes();
   ```
3. Look up the target in the offset map and call `link(this, target_block)` to append to both `successors_` and `predecessors_`.
4. For conditional branches, also link the fall-through block (next block in the flat list) as the second successor.
5. For indirect branches (`InstFlags::INDIRECT_BRANCH`): no outgoing edge; block treated as function-exit terminator (empty `successors_`).

RPO traversal (used by liveness convergence) is computed with a standard iterative DFS over `bb->successors()`, returning blocks in reverse finish order. It is a free function, not a class member:

```cpp
// analysis/liveness.h (or basic_block.h)
std::vector<const BasicBlock *> reverse_post_order(
    KernelBlockScope blocks);
```

### 4.6 `LivenessAnalysis`

```cpp
// lib/rocjitsu/src/rocjitsu/analysis/liveness.h

namespace rocjitsu {

/// @brief The basic blocks reachable from one kernel entry.
using KernelBlockScope = std::span<BasicBlock *const>;

/// @brief Per-basic-block liveness information.
struct BlockLiveness {
  RegisterSet live_in;   ///< Registers live on entry to this block.
  RegisterSet live_out;  ///< Registers live on exit from this block.
  RegisterSet gen;       ///< Upward-exposed uses (registers used before any definition in the block).
  RegisterSet kill;      ///< Locally defined registers (definitions that precede any upward-exposed use).
};

/// @brief Backward dataflow liveness analysis over a basic block list.
///
/// Algorithm: standard backward iterative dataflow.
///   live_out(B)  = ∪ { live_in(S) | S ∈ B.successors() }
///   live_in(B)   = gen(B) ∪ (live_out(B) \ kill(B))
///
/// Convergence: iterates in RPO order; terminates in at most |blocks| iterations.
/// Complexity per iteration: O(N · |RegisterSet|/64).
/// Total: O(D · N · |RegisterSet|/64), where D is iterations to convergence
/// (worst case O(N), typically 3–5 for structured GPU kernels).
class LivenessAnalysis {
public:
  /// @brief Compute liveness for one kernel's block set.
  ///
  /// @details Successor/predecessor edges that leave @p blocks are ignored.
  /// DBT callers must pass only the blocks reachable from the kernel descriptor
  /// entry being translated, not every block decoded from the containing code
  /// object.
  explicit LivenessAnalysis(KernelBlockScope blocks);

  /// @brief Block liveness by block object.
  [[nodiscard]] const BlockLiveness &block_liveness(const BasicBlock &block) const;

  /// @brief Registers live immediately before @p inst executes.
  [[nodiscard]] const RegisterSet &live_before(const Instruction &inst) const;

  /// @brief Convenience predicate for one register reference.
  [[nodiscard]] bool is_live_before(const Instruction &inst, RegisterRef ref) const;

  /// @brief Find N consecutive dead VGPRs immediately before an instruction.
  [[nodiscard]] std::optional<uint16_t> find_free_run(const Instruction *inst, uint16_t count,
                                                      uint16_t search_start = 0) const;

  /// @brief Find an even-aligned dead SGPR pair immediately before an instruction.
  [[nodiscard]] std::optional<uint16_t> find_free_sgpr_pair(const Instruction *inst,
                                                            uint16_t search_start = 0) const;

  /// @brief Find one dead SGPR immediately before an instruction.
  [[nodiscard]] std::optional<uint16_t> find_free_sgpr(const Instruction *inst,
                                                       uint16_t search_start = 0) const;

private:
  void analyze(KernelBlockScope blocks);

  std::vector<BlockLiveness> liveness_;
  std::unordered_map<const BasicBlock *, size_t> block_index_;
  std::unordered_map<const Instruction *, RegisterSet> live_before_;
  static constexpr RegisterSet empty_{};
};

} // namespace rocjitsu
```

Implementation notes:

- The constructor precomputes each block's local `gen`/`kill`, iterates backward dataflow over in-scope successors, and then materializes `live_before_` for every instruction.
- `live_before()` returns the materialized set. It returns an empty set for an instruction outside the analyzed scope.
- Predicated definitions and EXEC-masked vector definitions are conservative: they do not kill the old value unless the implementation can prove the write is unconditional for the relevant lanes. This is not special-register liveness; EXEC itself is still not tracked.

### 4.7 Public C API Extensions

The entire analysis pipeline (`RegisterSet`, `RegisterRef`, `InstDefUse`, `LivenessAnalysis`) is internal C++ — nothing in `rj_code.h` exposes it. The public API only needs to support the two end-user outcomes this library exists for: binary instrumentation and binary translation. Internal pipeline components (`SpillManager`, `Instrumentor`, `BinaryTranslator`) consume `LivenessAnalysis` and `InstDefUse` directly.

Analysis-derived features for external tools (e.g., register pressure queries for profilers) are deferred to a later extension point and will be designed when the internal pipeline is stable.

### 4.8 CDNA/RDNA ISA Differences for Analysis

| Property | CDNA1/2/3 (Wave64) | CDNA4 (Wave64) | RDNA1 (Wave32/64) | RDNA2/3/3.5 (Wave32/64) | RDNA4 (Wave32/64) |
|---|---|---|---|---|---|
| EXEC width | 64-bit (EXEC_LO+EXEC_HI) | 64-bit (EXEC_LO+EXEC_HI) | 32-bit or 64-bit per-kernel flag | 32-bit default; 64-bit in Wave64 mode | 32-bit default; 64-bit in Wave64 mode |
| VCC width | 64-bit (VCC_LO+VCC_HI) | 64-bit | 32-bit in Wave32 mode; 64-bit in Wave64 mode | 32-bit (Wave32 default); 64-bit in Wave64 mode | 32-bit (Wave32 default); 64-bit in Wave64 mode |
| AccVGPR | CDNA1: Absent; CDNA2/3: Present (256 regs) | Present (256 regs) | Absent | Absent | Absent |
| Max SGPR | 102 (CDNA) | 102 (CDNA4) | 106 (RDNA) | 106 (RDNA) | 106 (RDNA) |
| FLAT_SCRATCH | CDNA1/2: Via SGPR pair (ABI-defined); CDNA3: Dedicated hardware pair | Dedicated hardware pair | Via SGPR | Via SGPR | Via SGPR |
| s_waitcnt encoding | GFX9 classic: lgkmcnt[11:8], vmcnt[15:14,3:0], expcnt[6:4] | GFX9 classic (same as CDNA1-3; CDNA4 retains monolithic S_WAITCNT despite GFX11-gen hardware) | GFX10: same 16-bit simm16 + `s_waitcnt_vscnt` | GFX11: split `s_wait_loadcnt`/`s_wait_storecnt`/`s_wait_kmcnt`/`s_wait_expcnt`/`s_wait_dscnt` (no monolithic `s_waitcnt`) | GFX12: split `s_wait_*` with combined `s_wait_storecnt_dscnt` |

The current liveness pass does not parameterize on `wf_size`, because it does not track EXEC/VCC or other special mask registers yet. Ordinary SGPR/VGPR/AccVGPR liveness uses register indices and operand widths only. When special-register liveness is added, wavefront width must be derived from both the arch and the kernel descriptor's `ENABLE_WAVEFRONT_SIZE32` bit outside generic analysis code.

---

## Pillar 2: Dynamic Binary Instrumentation (DBI) {#pillar-2}

### 2.1 Motivation and Design Rationale

DBI in the AMDGPU context means patching an HSA code object's `.text` section in-place (before it is loaded into device memory) to insert calls into user-supplied instrumentation functions. The challenges unique to this architecture are:

1. **No caller-save convention at the wave level.** There is no equivalent of x86 `call`/`ret` for in-wave subroutines. We must use `s_swappc_b64` as the inter-function call primitive, but all registers are caller-managed.
2. **Scratch memory is the only safe spill target.** SGPRs and VGPRs cannot be pushed to any stack; the private segment (flat scratch) must be used.
3. **EXEC must be preserved.** Many instrumentation points live inside control-flow regions where EXEC is not all-ones. The trampoline must save/restore EXEC.
4. **`s_swappc_b64` requires a free SGPR pair.** The trampoline call instruction needs two consecutive live SGPRs to hold the return address; if all SGPRs are live, spill is required.
5. **Instruction size changes require branch-offset fixup** across the entire function.

### 2.2 New File Layout

```
lib/rocjitsu/src/rocjitsu/code/
  patch/
    code_object_patcher.h/.cpp   # ELF text section rewriter and re-emitter
    trampoline_builder.h/.cpp    # Generates save/call/restore ISA sequences
    spill_manager.h/.cpp         # Scratch spill slot allocator
    buffer_injector.h/.cpp       # Injects logging buffer pointer into kernels
    instrumentor.h/.cpp          # High-level Instrumentor class
```

### 2.3 Scratch Memory Layout for Spill Slots

On CDNA3/4 (and analogously on RDNA), the private segment (scratch) is the per-thread spill area. Each thread's scratch is `private_segment_fixed_size` bytes long, available via flat-scratch addressing. The wave-base scratch address is loaded from the `FLAT_SCRATCH` SGPR pair whose indices are ISA-specific and provided by the ISA's register info (not hardcoded in the generic spill logic).

The per-lane address formulas are implemented in `shared/addr_calc_scalar.h` (SMEM, MUBUF, MTBUF, DS) and `shared/addr_calc_flat.h` (FLAT/GLOBAL/SCRATCH), with per-ISA thin wrappers in `{isa}/addr_calc.cpp`. `SpillManager` reuses the shared logic directly — it does not reimplement scratch address derivation.

Additionally, when calculating the spill set at a patch site containing an MFMA instruction on **CDNA2+ targets** (GFX90A, GFX940/941/942, GFX950), `SpillManager` must account for AccVGPR ranges clobbered by the MFMA. The clobbered output registers are identified by calling `mfma_exec.h`'s `output_loc_32()` / `output_loc_64()` with the MFMA dimensions extracted from the `Vop3pMachineInst` encoding — these functions return the exact `{reg, lane}` pairs written, from which the AccVGPR range can be derived and added to the spill set. On CDNA1 (GFX908/MI100), MFMA instructions write to regular VGPRs (not AccVGPRs); there is no AccVGPR range to account for.

NOTE (05/06/2026): SpillManager was implemented in Phase 5. Accounting for 
AccVGPR ranges clobbered by the MFMA has been deferred.

The per-lane base address is:

```
flat_scratch_base_byte_addr[lane] = FLAT_SCRATCH_INIT + lane * private_segment_fixed_size
```

`v_readlane_b32` allows the scalar domain to read the VGPR holding the lane's flat scratch pointer into an SGPR:

```asm
; lane 0's scratch base into s_tmp
v_readlane_b32 s_tmp, v_flat_scratch_addr, 0
```

The `SpillManager` allocates slots within a reserved zone appended to the existing private segment. It must also update the kernel descriptor's `private_segment_fixed_size` field to account for the new slots.

```cpp
// lib/rocjitsu/src/rocjitsu/code/patch/spill_manager.h

namespace rocjitsu {

/// @brief Manages a reservation of scratch slots for DBI spill/fill.
///
/// The manager appends a "DBI spill zone" to the existing private segment.
/// Layout (from the per-lane perspective):
///   [0,   original_private_segment_size)   Original kernel scratch
///   [original_private_segment_size, ...]   DBI spill slots (added here)
///
/// Slot indices are assigned in allocation order and are 4 bytes each.
class SpillManager {
public:
  /// @brief Construct a spill manager for a given kernel.
  /// @param original_private_bytes  Existing private_segment_fixed_size from the kernel descriptor.
  explicit SpillManager(uint32_t original_private_bytes);

  /// @brief Allocate a 32-bit spill slot for a register.
  /// @param reg  Register to spill (must be 32-bit scalar or one lane of VGPR).
  /// @returns Byte offset within per-lane scratch of the allocated slot.
  uint32_t allocate_slot(RegisterRef reg);

  /// @brief Allocate slots for a contiguous range (e.g., a 64-bit SGPR pair).
  /// @param reg    First register.
  /// @param width  Number of 32-bit slots.
  /// @returns Byte offset of the first slot.
  uint32_t allocate_slots(RegisterRef reg, int width);

  /// @brief Total size of the per-lane scratch after DBI slots are added.
  uint32_t total_private_bytes() const { return total_bytes_; }

  /// @brief Retrieve the scratch offset for a previously allocated register.
  /// Returns -1 if not allocated.
  int offset_for(RegisterRef reg) const;

private:
  uint32_t base_offset_;   ///< First DBI slot byte offset = original_private_bytes rounded up to 16.
  uint32_t total_bytes_;
  std::map<std::pair<RegClass, uint16_t>, uint32_t> reg_to_offset_;  ///< (cls, index) -> byte offset
  uint32_t next_offset_;
};

} // namespace rocjitsu
```

### 2.4 Trampoline Code Structure

Each instrumentation site is patched with a call-site stub that transfers control to a per-site trampoline emitted into a new `.rj_trampolines` section. The generic structure of a trampoline is:

```
Trampoline for site S:
  ┌─────────────────────────────────────┐
  │ [Save EXEC register]                │
  │ [Spill live SGPRs to scratch]       │
  │ [Spill live VGPRs to scratch]       │
  │ [Optionally set EXEC = all-lanes]   │
  │ [Load probe function address]       │
  │ [Call probe via indirect call]      │
  │ [Restore VGPRs from scratch]        │  ← while EXEC still all-lanes active
  │ [s_waitcnt vmcnt(0)]                │  ← wait for all VGPR loads to complete
  │ [Restore EXEC]                      │  ← after vmcnt==0, so EXEC write races no loads
  │ [Restore SGPRs from scratch]        │
  │ [Return to original code]           │
  └─────────────────────────────────────┘
```

`TrampolineBuilder` is an abstract class. ISA-specific subclasses (created via `TrampolineBuilder::create(arch)`) emit the actual machine instructions using their ISA's machine instruction layout types from `{isa}/machine_insts.h`. No instruction encodings, opcodes, or register numbers appear in the generic class.

```cpp
// lib/rocjitsu/src/rocjitsu/code/patch/trampoline_builder.h

namespace rocjitsu {

/// @brief Abstract interface for emitting ISA-specific trampoline sequences.
///
/// The call-site stub overwrites the original instruction at the patch point.
/// The trampoline body lives in a .rj_trampolines section; it saves all live
/// registers, calls the probe device function, restores registers, and returns.
class TrampolineBuilder {
public:
  virtual ~TrampolineBuilder() = default;

  /// @brief Factory: returns the correct ISA-specific subclass for @p arch.
  static std::unique_ptr<TrampolineBuilder> create(rj_code_arch_t arch);

  /// @brief Size in bytes of the call-site stub that replaces the patched instruction.
  virtual uint32_t call_site_size() const = 0;

  /// @brief Emit the call-site stub bytes (overwrites the patched instruction in .text).
  /// @param trampoline_va  Virtual address of the trampoline body in .rj_trampolines.
  /// @param link_pair      SGPR pair index for storing the return address.
  virtual std::vector<uint8_t> build_call_site(uint64_t trampoline_va,
                                                uint32_t link_pair) const = 0;

  /// @brief Emit the full trampoline body for one patch site.
  /// @param spill_mgr       Scratch slot assignments for live registers.
  /// @param live_regs       Registers live at the patch point (from LivenessAnalysis).
  ///                        Must NOT include link_pair (s_swappc_b64 overwrites it;
  ///                        it is a TrampolineBuilder output, not a live input).
  /// @param probe_va        Virtual address of the probe device function.
  /// @param link_pair       Even-indexed SGPR pair for storing the return address.
  ///                        Caller must ensure link_pair is NOT in live_regs and
  ///                        link_pair is even (required for s_swappc_b64).
  /// @param force_full_exec If true, set all execution lanes active before calling probe.
  /// @returns Non-empty byte vector on success, or empty vector on failure (e.g., if
  ///          spill slots are exhausted — caller should fall back to PASSTHROUGH).
  virtual std::vector<uint8_t> build_trampoline(const SpillManager &spill_mgr,
                                                 const RegisterSet &live_regs,
                                                 uint64_t probe_va,
                                                 uint32_t link_pair,
                                                 bool force_full_exec) const = 0;
};

} // namespace rocjitsu
```

ISA-specific subclasses (e.g., `Cdna3TrampolineBuilder` in `isa/arch/amdgpu/cdna3/trampoline_builder.cpp`) implement `build_call_site` and `build_trampoline` using their ISA's machine instruction structs and opcode constants from `{isa}/machine_insts.h`. The choice of call instruction, spill/restore instruction family (SMEM vs flat-scratch), and EXEC register encoding are all ISA-specific and contained entirely within those subclasses.

**Probe calling convention.** The probe is a non-kernel `__device__` function called via `s_swappc_b64`. On the AMDGPU CC, non-kernel functions receive only their explicit arguments (no kernel-implicit SGPRs such as dispatch pointer or queue pointer), provided the function does not use scratch. The probe is compiled at `-O2` with a small, spill-free body by design — `TrampolineBuilder::build_trampoline` reads the probe code object's `COMPUTE_PGM_RSRC1.GRANULATED_WAVEFRONT_SGPR_COUNT` and `VGPR_COUNT` to verify this before emitting the call sequence. With no scratch, argument registers are:

| Argument | Register | Notes |
|---|---|---|
| `hdr` (buffer base, 64-bit) | `s[0:1]` | Trampoline copies from `s[buf_lo:buf_hi]` (loaded by `BufferInjector` prologue) |
| `record_type` (32-bit constant) | `s[2]` | Baked in by `TrampolineBuilder` per site |
| `inst_byte_offset` (32-bit constant) | `s[3]` | Baked in by `TrampolineBuilder` per site |
| `payload` (64-bit; uniform or per-lane) | `s[4:5]` or `v[0:1]` | Scalar if uniform (e.g., workgroup ID); vector if per-lane (e.g., effective address). Probe variants are compiled for both cases; the trampoline selects the correct symbol. |

The trampoline diagram gains two entries:

```
Trampoline for site S (logging variant):
  ┌─────────────────────────────────────────────┐
  │ [Save EXEC register]                        │
  │ [Spill live SGPRs to scratch]               │
  │ [Spill live VGPRs to scratch]               │
  │ [Optionally set EXEC = all-lanes]           │
  │ [Copy s[buf_lo:buf_hi] → s[0:1]]            │  ← buf VA for probe
  │ [Load record_type constant → s[2]]          │  ← per-site constant
  │ [Load inst_byte_offset constant → s[3]]     │  ← per-site constant
  │ [Collect payload → s[4:5] or v[0:1]]        │  ← probe-kind specific
  │ [Load probe function address]               │
  │ [Call probe via s_swappc_b64]               │
  │ [Restore VGPRs from scratch]                │  ← while EXEC still all-lanes active
  │ [s_waitcnt vmcnt(0)]                        │  ← wait for all VGPR loads to complete
  │ [Restore EXEC]                              │  ← after vmcnt==0, so EXEC write races no loads
  │ [Restore SGPRs from scratch]                │
  │ [Return to original code]                   │
  └─────────────────────────────────────────────┘
```

### 2.5 `Instrumentor` Class

```cpp
// lib/rocjitsu/src/rocjitsu/code/patch/instrumentor.h

#include "rocjitsu/code/rj_code.h"  // rj_code_instrument_kind_t

namespace rocjitsu {

/// @brief Instrumentation point descriptor: describes one location in the kernel to patch.
struct InstrumentationPoint {
  /// @brief Reference instruction (required for BEFORE/AFTER; used to identify block for BLOCK_*).
  const Instruction *anchor_inst = nullptr;

  /// @brief Where to insert relative to anchor_inst.
  /// Uses the public API enum directly — no parallel internal enum needed.
  rj_code_instrument_kind_t kind = RJ_CODE_INSTR_BEFORE_INST;

  /// @brief Bitmask of InstFlags; only instructions matching this mask are patched (0 = all).
  uint32_t filter_flags = 0;

  /// @brief Code object containing the probe device function.
  /// The probe ELF is merged into the output by CodeObjectPatcher::merge_code_object().
  const AmdGpuCodeObject *probe_obj = nullptr;

  /// @brief Symbol name of the probe device function in probe_obj.
  /// The probe is a GPU device function; its calling convention is defined by the trampoline:
  /// all registers live at the patch point are saved before the call and restored after.
  /// The probe may freely use any register; it receives no explicit arguments by default.
  std::string probe_symbol;

  /// @brief If true, restore exec to -1 before calling the probe (all lanes active).
  bool force_full_exec = false;
};

/// @brief Result of patching: the modified code object ELF as a byte vector.
struct PatchedCodeObject {
  std::vector<uint8_t> elf_bytes;     ///< Re-emitted ELF.
  uint32_t new_private_segment_size;  ///< Updated private_segment_fixed_size.
  std::vector<std::string> warnings;
  // The emitted ELF contains a .note.rocjitsu section recording what was done,
  // allowing the load-time hook to detect and skip re-patching on reload.
};

/// @brief Instruments one or more code objects by injecting trampoline calls.
///
/// Workflow:
///   1. Run LivenessAnalysis on the target function.
///   2. For each InstrumentationPoint, allocate spill slots via SpillManager.
///   3. Build a trampoline sequence via TrampolineBuilder.
///   4. Patch the .text section via CodeObjectPatcher.
///   5. Re-emit the ELF with updated kernel descriptors.
class Instrumentor {
public:
  /// @brief Construct an instrumentor for the given code object.
  /// @param obj          Decoded code object.
  /// @param arch         Target ISA.
  Instrumentor(const AmdGpuCodeObject &obj, rj_code_arch_t arch);

  /// @brief Add an instrumentation point.
  void add_point(InstrumentationPoint pt);

  /// @brief Execute all patches and return the modified ELF.
  ///
  /// Patch sites are processed in increasing byte-offset order. LivenessAnalysis
  /// is computed once over the original instruction sequence and remains valid for
  /// each site at its original position; liveness for site N is consumed before
  /// the patch at site N shifts subsequent offsets. Branch fixup happens last,
  /// after all insertions are complete.
  ///
  /// May fail if not enough free registers/scratch space can be found,
  /// or if a branch fixup range would overflow.
  /// @returns Empty PatchedCodeObject (elf_bytes.empty() == true) on failure.
  PatchedCodeObject patch();

  /// **Link pair selection** (called internally by patch() for each site):
  /// Finds the lowest even-indexed SGPR pair that is dead at the patch site.
  /// ```cpp
  /// int select_link_pair(const RegisterSet &live_regs) {
  ///   for (int s = 0; s <= 102; s += 2) {   // only even indices for s_swappc_b64
  ///     if (!live_regs.test(RegClass::SGPR, s) &&
  ///         !live_regs.test(RegClass::SGPR, s + 1))
  ///       return s;
  ///   }
  ///   return -1;  // all SGPR pairs live — spill required before link_pair can be used
  /// }
  /// ```
  /// If -1 is returned, patch() falls back to PASSTHROUGH for that site.

private:
  const AmdGpuCodeObject &obj_;
  rj_code_arch_t arch_;
  std::vector<InstrumentationPoint> points_;

  std::unique_ptr<LivenessAnalysis> liveness_;
  std::unique_ptr<SpillManager> spill_mgr_;
};

} // namespace rocjitsu
```

### 2.6 `CodeObjectPatcher` — ELF Rewriter

```cpp
// lib/rocjitsu/src/rocjitsu/code/patch/code_object_patcher.h

namespace rocjitsu {

/// @brief Modifies the raw ELF image of an AMD GPU code object.
///
/// Supports:
///   - Growing the .text section (inserting bytes at arbitrary offsets).
///   - Appending a .rj_trampolines section for trampoline code.
///   - Appending a .note.rocjitsu section with instrumentation/translation metadata
///     (allows load-time re-detection of already-patched objects; see emit_note()).
///   - Updating symbol st_value fields after insertions.
///   - Updating relative branch offsets in SOPP instructions.
///   - Updating the kernel descriptor's private_segment_fixed_size.
class CodeObjectPatcher {
public:
  explicit CodeObjectPatcher(const AmdGpuCodeObject &obj);

  /// @brief Insert @p bytes at @p byte_offset within the .text section.
  ///
  /// All branch targets in [byte_offset, end_of_text) are adjusted.
  /// All symbols in the .symtab whose st_value > byte_offset are incremented by bytes.size().
  /// Complexity: O(total_text_bytes + num_symbols + num_branch_instructions).
  void insert_at(uint64_t byte_offset, std::span<const uint8_t> bytes);

  /// @brief Overwrite bytes at @p byte_offset (no size change; for trampoline call patches).
  void overwrite_at(uint64_t byte_offset, std::span<const uint8_t> bytes);

  /// @brief Append a new ELF section to the code object.
  void append_section(std::string name, uint32_t sh_type, uint64_t sh_flags,
                      std::vector<uint8_t> data);

  /// @brief Update the kernel descriptor's private_segment_fixed_size.
  /// @param kernel_name  Kernel name (looks up .kd symbol).
  /// @param new_size     New private_segment_fixed_size in bytes.
  void update_private_segment_size(const std::string &kernel_name, uint32_t new_size);

  /// @brief Write a .note.rocjitsu section encoding what was done to this object.
  ///
  /// The note records: kind (instrument | translate | both), source arch, target arch,
  /// and a SHA256 of the original .text bytes. The load-time hook reads this to skip
  /// re-patching already-instrumented/translated code objects.
  void emit_note(uint32_t kind_flags, rj_code_arch_t src_arch, rj_code_arch_t dst_arch,
                 const uint8_t sha256[32]);

  /// @brief Emit the modified ELF as a byte vector.
  std::vector<uint8_t> emit();

private:
  std::vector<uint8_t> image_;  ///< Working copy of the ELF image.

  void fixup_branches_after_insertion(uint64_t insert_offset, int32_t delta_bytes,
                                       std::span<const BranchReloc> branch_relocs);
  void fixup_symbols_after_insertion(uint64_t insert_offset, int32_t delta_bytes);
  Elf64_Shdr *find_section_by_name(const std::string &name);
  const Elf64_Sym *find_symbol(const std::string &name) const;
};

} // namespace rocjitsu
```

The branch-offset fixup uses a precomputed list of branch locations derived from the decoded basic blocks — no ISA-specific encoding knowledge is needed in the patcher:

```cpp
/// @brief Branch relocation entry: precomputed from decoded BasicBlock list
/// by the Instrumentor before any patching begins.
struct BranchReloc {
  uint64_t inst_byte_offset;    ///< Byte offset of the branch instruction in .text.
  uint64_t target_byte_offset;  ///< Byte offset of the branch target in .text.
  uint32_t inst_size;           ///< Instruction size in bytes (4 or 8).
};

/// @brief Build the branch relocation list from a decoded basic block list.
/// Called by Instrumentor before constructing CodeObjectPatcher.
std::vector<BranchReloc> build_branch_relocs(
    const std::vector<std::unique_ptr<BasicBlock>> &blocks);
```

```cpp
void CodeObjectPatcher::fixup_branches_after_insertion(
    uint64_t insert_offset,
    int32_t delta_bytes,
    std::span<const BranchReloc> branch_relocs) {

  auto *text_sec = find_section_by_name(".text");
  if (!text_sec) return;
  uint8_t *text_base = image_.data() + text_sec->sh_offset;

  for (const auto &br : branch_relocs) {
    bool inst_before   = br.inst_byte_offset   < insert_offset;
    bool target_before = br.target_byte_offset < insert_offset;

    if (inst_before == target_before)
      continue;  // Branch does not cross insertion point; no fixup needed.

    // Branch crosses the insertion: patch the encoded offset in-place.
    // The ISA-specific subclass knows the offset field layout; the patcher
    // delegates to a virtual patch_branch_offset() call.
    int32_t adjustment = inst_before ? delta_bytes : -delta_bytes;
    patch_branch_offset(text_base + br.inst_byte_offset, adjustment);
  }
}
```

`patch_branch_offset` is a pure virtual method on `CodeObjectPatcher` — ISA-specific subclasses (e.g., `Cdna3CodeObjectPatcher`) override it using their ISA's branch instruction layout structs from `{isa}/machine_insts.h`. The generic patcher carries no encoding knowledge. The method declaration in `CodeObjectPatcher`:
```cpp
  /// @brief Patch the encoded branch offset in-place at @p branch_ptr by @p adjustment bytes.
  /// Called from fixup_branches_after_insertion for every branch that crosses an insertion.
  virtual void patch_branch_offset(uint8_t *branch_ptr, int32_t adjustment_bytes) = 0;
```

### 2.7 Passing a Logging Buffer to Instrumentation Functions

For memory-access tracing or other logging, the instrumentation function needs a pointer to a per-kernel output buffer. The buffer VA cannot be baked into the ELF at load time — it is only known at dispatch time and must be unique per concurrent dispatch. The design splits the work across two phases:

**Load-time (ELF patch, `BufferInjector`):** Patch the kernel entry point with an `s_load_dwordx2` that reads the buffer VA from a reserved slot at the end of the kernarg segment. Record the reserved slot's byte offset so the dispatch-time handler knows where to write it. `BufferInjector` is responsible only for the ELF patch and reporting the offset; it does not supply the buffer VA.

**Dispatch-time (AQL packet handler, §4.3.4):** Extend the kernarg segment by 8 bytes. Write the buffer VA into the reserved slot. Update `kernarg_address` in the AQL packet to point to the extended buffer. The extended buffer is allocated from a pre-allocated per-queue pool so the hot path requires no dynamic allocation.

```cpp
// lib/rocjitsu/src/rocjitsu/code/patch/buffer_injector.h

namespace rocjitsu {

/// @brief Injects a logging buffer pointer into a kernel's SGPR argument space.
///
/// Load-time responsibility only:
///   1. Parse COMPUTE_PGM_RSRC2.USER_SGPR_COUNT from the kernel descriptor.
///   2. Determine the first free SGPR pair after the existing user SGPRs.
///      If USER_SGPR_COUNT == 16 (all user SGPRs occupied — common in HPC kernels),
///      no additional user SGPR can be added. Fallback: read the buffer VA via an
///      s_load_dwordx2 from a fixed kernarg offset using the existing kernarg_ptr
///      SGPR pair, temporarily spilling one live SGPR to scratch if needed.
///   3. Prepend (to the kernel entry point) an s_load_dwordx2 that reads
///      the buffer VA from the reserved kernarg slot (original_kernarg_size + 0).
///   4. Update the kernel descriptor and metadata to account for the extended kernarg.
///
/// The actual buffer VA is NOT set here. It is injected per-dispatch by
/// rj_packet_handler (§4.3.4), which writes the VA into the reserved slot in a
/// copy of the kernarg segment and updates the AQL packet's kernarg_address.
class BufferInjector {
public:
  /// @brief Load-time injection result.
  struct Result {
    int sgpr_lo;                  ///< First SGPR holding the buffer address (low 32 bits).
    int sgpr_hi;                  ///< Second SGPR (high 32 bits).
    uint32_t original_kernarg_size; ///< Kernarg size before extension (bytes).
    uint32_t buffer_va_kernarg_offset; ///< Byte offset of the reserved VA slot = original_kernarg_size.
  };

  static Result inject(CodeObjectPatcher &patcher,
                       const std::string &kernel_name,
                       const AmdGpuCodeObject &obj,
                       rj_code_arch_t arch);
};

} // namespace rocjitsu
```

The injected prologue reads the buffer VA from the extended kernarg segment that the dispatch-time handler supplies:

```asm
; Injected at kernel entry point (before original first instruction):
; kernarg_ptr is in s[kernarg_lo:kernarg_hi] (ENABLE_SGPR_KERNARG_SEGMENT_PTR).
; buffer_va_kernarg_offset = original_kernarg_size (first byte after original args).
s_load_dwordx2  s[buf_lo:buf_hi], s[kernarg_lo:kernarg_hi], <buffer_va_kernarg_offset>
s_waitcnt lgkmcnt(0)
; Original kernel instructions follow. s[buf_lo:buf_hi] holds the buffer VA.
```

The load-time hook stores `Result::original_kernarg_size` and `Result::buffer_va_kernarg_offset` in `KernelDescriptorRegistry::DispatchInfo` (§4.3.4). The dispatch-time hook uses these offsets to build the extended kernarg buffer without any ELF knowledge.

#### 2.7.1 Logging Buffer Protocol

The logging buffer is a **fixed-size MPSC lock-free circular ring buffer** in fine-grained system memory (coherent between GPU and CPU without explicit cache flushes). Many GPU waves write concurrently (M producers); one host drain thread reads (one consumer).

**Why the buffer can never grow:** Private segment and kernarg sizes are fixed at load time. The buffer VA is injected per-dispatch but the physical buffer is pre-allocated at instrumentation time. When the ring is full, the GPU drops records and increments an overflow counter — blocking the GPU to wait for host draining would risk a GPU timeout.

**Memory layout:**

```cpp
// lib/rocjitsu/src/rocjitsu/hooks/log_buffer.h

namespace rocjitsu {

/// @brief Fixed-size record written by each probe invocation.
///
/// All fields are written by the GPU wave before the valid flag is set.
/// Size is a power of 2 so that slot addressing is a single mask.
struct alignas(32) LogRecord {
  std::atomic<uint32_t> valid;   ///< 0 = slot free, 1 = record ready for host.
                                  ///< GPU writes last (release); host reads first (acquire).
  uint32_t  record_type;         ///< Which probe kind fired (maps to rj_code_instrument_kind_t).
  uint32_t  inst_byte_offset;    ///< Byte offset of the patched instruction in .text.
  uint32_t  wave_id;             ///< From s_getreg(HW_REG_HW_ID) or equivalent.
  uint64_t  workgroup_id;        ///< Packed workgroup index: (z<<42)|(y<<21)|x (21 bits each).
  uint64_t  payload;             ///< Probe-specific data (e.g., effective address for memory ops).
};
static_assert(sizeof(LogRecord) == 32);

/// @brief Header of the ring buffer. Placed at the start of the allocation.
///
/// write_ptr and read_ptr are monotonically increasing byte-counts of RECORDS
/// (not bytes). Slot index = ptr % slot_count.
struct alignas(64) LogBufferHeader {
  std::atomic<uint64_t> write_ptr;      ///< Next slot to claim; GPU increments atomically.
  std::atomic<uint64_t> read_ptr;       ///< Next slot to drain; host increments and publishes.
  uint64_t              slot_count;     ///< Capacity in records (power of 2).
  std::atomic<uint64_t> overflow_count; ///< Records dropped because the ring was full.
  uint64_t              signal_handle;  ///< Raw hsa_signal_t value; GPU calls rj_signal_wake
                                        ///< when write_ptr − read_ptr > hwm_threshold.
                                        ///< Written once at allocation; GPU reads relaxed.
  uint8_t               _pad[64 - 40];
};
static_assert(sizeof(LogBufferHeader) == 64,
              "LogBufferHeader layout mismatch with device RjLogBufferHeader");

} // namespace rocjitsu
```

The physical allocation is one contiguous fine-grained memory region:
```
[ LogBufferHeader (64 bytes) | LogRecord[slot_count] ]
```

**GPU write protocol — HIP device-only compilation:**

#### Compilation approach: plain AMDGPU C++, no HIP

**HIP is not needed, and standard `__atomic_*` is the correct scope for most operations.**

HIP's `__hip_atomic_fetch_add` with `__HIP_MEMORY_SCOPE_AGENT` is not a call into the ROCm device libs (ockl/ocml). It is a compiler builtin that emits LLVM IR `atomicrmw add syncscope("agent")`, which the AMDGPU backend lowers directly to the `global_atomic_add_u64` hardware instruction with the agent-scope bits set in the instruction encoding. The ROCm device libs are for math functions (sin, sqrt, etc.) — they have nothing to do with atomics.

Standard C++ `__atomic_fetch_add` on the AMDGPU target (`-target amdgcn-amd-amdhsa`) defaults to `syncscope("")` (system scope). This maps to `global_atomic_add_u64` with system-scope encoding. For our protocol:

| Operation | Scope needed | Reason | `__atomic_*` default |
|---|---|---|---|
| `write_ptr` claim (GPU→GPU) | agent sufficient | Only GPU waves observe each other's increments | system scope — correct, negligible overhead on logging path |
| `read_ptr` load (host→GPU) | system required | Host publishes this; GPU must see the updated value cross-chip | system scope — correct |
| `overflow_count` increment | agent sufficient | Host only reads at drain, no ordering dependency | system scope — correct |
| `valid` store (GPU→host) | system required | Host CPU reads this; must be visible across the CPU-GPU interconnect | plain store after fence — see below |
| fence before `valid` | agent sufficient | `s_waitcnt vmcnt(0)` drains GPU write buffer; fine-grained memory is coherent to CPU without L2 flush | `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")` |

Using standard `__atomic_*` (system scope for all pointer-level atomics) is **correct for every case** in this table. For `write_ptr` it is slightly conservative (system scope instead of agent scope), but on a logging path — already far off the kernel's critical path — this overhead is irrelevant. The important correctness point the user raises is real: `read_ptr` and `valid` cross the CPU-GPU boundary and genuinely require at minimum system-scope visibility, which standard `__atomic_*` already provides.

All `__builtin_amdgcn_*` builtins needed by the probe (mbcnt, read_exec, readlane, s_getreg, workgroup_id_x/y/z, fence) are available with `-target amdgcn-amd-amdhsa -x c++` — no HIP required. rocjitsu therefore has no HIP compile-time dependency and can be used under any HSA runtime (IREE, OpenCL runtimes, custom loaders, etc.) without dragging in HIP toolchain machinery.

The probe is compiled once per GFX target to a bare AMDGPU ELF code object, embedded in `librocjitsu.so` as byte arrays, and selected by `rj_code_arch_t` at instrumentation time.

#### `rj_log_probe.cpp` — probe source

One source file, compiled nine times (once per GFX target). Uses:
- `__atomic_fetch_add` / `__atomic_load_n` (standard C, system scope — correct for CPU-GPU visibility) for all ring-buffer pointer operations.
- `__builtin_amdgcn_mbcnt_lo` / `__builtin_amdgcn_mbcnt_hi` for wave-collaborative slot claiming (one atomic per wave, each active lane computes its own slot offset via bit-count below its lane position in EXEC).
- `__builtin_amdgcn_readlane` to broadcast the claimed base slot from lane 0 to all lanes.
- `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")` for the payload-before-valid release fence; compiles to `s_waitcnt vmcnt(0)` on all targets.
- `__builtin_amdgcn_s_getreg` for hardware wave ID.
- `__builtin_amdgcn_workgroup_id_{x,y,z}` for workgroup coordinates.

```cpp
// lib/rocjitsu/device/rj_log_probe.cpp
// Compiled with: clang -target amdgcn-amd-amdhsa -mcpu=<gfx-id> -x c++
// Single source; compiled per-target — see §2.7.2 for build details.
// No HIP headers. No ROCm device libs. Keep in sync with log_buffer.h layout.

#include <stdint.h>

// Device-side mirror of LogBufferHeader (log_buffer.h). Plain integer fields;
// accessed via __atomic_* builtins (system scope) — correct for CPU-GPU sync.
struct RjLogBufferHeader {
    uint64_t write_ptr;       // +0
    uint64_t read_ptr;        // +8
    uint64_t slot_count;      // +16  constant after allocation; plain load is safe
    uint64_t overflow_count;  // +24
    uint64_t signal_handle;   // +32  raw hsa_signal_t value; GPU reads with __atomic_load_n relaxed
    uint8_t  _pad[24];        // +40  pad to 64 bytes
};
static_assert(sizeof(RjLogBufferHeader) == 64, "layout mismatch with host LogBufferHeader");

// Device-side mirror of LogRecord (log_buffer.h).
struct RjLogRecord {
    uint32_t valid;             // +0   publication flag; GPU writes last after fence
    uint32_t record_type;       // +4
    uint32_t inst_byte_offset;  // +8
    uint32_t wave_id;           // +12
    uint64_t workgroup_id;      // +16  packed (z<<42)|(y<<21)|x
    uint64_t payload;           // +24  probe-specific (e.g., effective address)
};
static_assert(sizeof(RjLogRecord) == 32, "layout mismatch with host LogRecord");

// rj_log_write — called from the trampoline via s_swappc_b64.
//
// Arguments (AMDGPU CC, no scratch, no kernel implicit SGPRs):
//   hdr              : s[0:1]  — buffer base VA (placed by trampoline from buf_lo:buf_hi)
//   record_type      : s[2]    — per-site constant
//   inst_byte_offset : s[3]    — per-site constant
//   payload          : s[4:5] or v[0:1] depending on probe variant
//
// __attribute__((noinline)): required so the symbol exists as a callable entry
//   point in the code object for s_swappc_b64. Without it the compiler may
//   inline into a caller that does not exist in device code.
__attribute__((noinline))
void rj_log_write(RjLogBufferHeader *hdr,
                  uint32_t           record_type,
                  uint32_t           inst_byte_offset,
                  uint64_t           payload)
{
    // ── Step 1: Wave-collaborative slot claiming ─────────────────────────────
    //
    // Read the full 64-bit EXEC mask. On Wave32 targets (RDNA) the upper 32
    // bits are always zero; __builtin_popcountll handles both widths uniformly.
    uint64_t exec = __builtin_amdgcn_read_exec();
    uint32_t active_count = __builtin_popcountll(exec);

    // Lane 0 claims active_count consecutive slots with a single atomic, then
    // each lane takes its own slot via MBCNT — one atomic per wave instead of
    // one atomic per active lane.
    //
    // Lane-0 check without HIP: mbcnt_lo(~0, 0) == 0 is true only for lane 0
    // because MBCNT counts bits set below the current lane in the mask, and
    // lane 0 has no bits below it regardless of wave width.
    //
    // Scope: __atomic_fetch_add emits system-scope atomic on AMDGPU. This is
    // correct: write_ptr is in fine-grained system memory readable by the host.
    uint64_t base_slot = 0;
    if (__builtin_amdgcn_mbcnt_lo(~0u, 0u) == 0) {
        base_slot = __atomic_fetch_add(&hdr->write_ptr,
                                       (uint64_t)active_count,
                                       __ATOMIC_RELAXED);
    }

    // Broadcast base_slot from lane 0. __builtin_amdgcn_readlane is 32-bit;
    // broadcast the 64-bit value in two halves.
    uint32_t base_lo = __builtin_amdgcn_readlane((uint32_t)base_slot,        0);
    uint32_t base_hi = __builtin_amdgcn_readlane((uint32_t)(base_slot >> 32), 0);
    base_slot = ((uint64_t)base_hi << 32) | base_lo;

    // High-watermark check: if the ring is > 3/4 full, wake the host drain thread.
    // Done by lane 0 only, immediately after the slot claim.
    // base_slot equals the write_ptr value BEFORE our add, so
    // (base_slot + active_count - read_ptr) is the used slot count after this wave.
    // We use base_slot alone for the check (conservative: slightly underestimates).
    // Multiple waves may fire rj_signal_wake concurrently — that is harmless; the
    // host drains once and the subsequent waves will find the ring no longer near-full.
    if (__builtin_amdgcn_mbcnt_lo(~0u, 0u) == 0) {
        uint64_t rptr     = __atomic_load_n(&hdr->read_ptr, __ATOMIC_RELAXED);
        uint64_t used     = base_slot - rptr;
        uint64_t sc       = hdr->slot_count; // constant; plain load
        uint64_t sig      = __atomic_load_n(&hdr->signal_handle, __ATOMIC_RELAXED);
        if (sig && used > (sc - (sc >> 2))) {  // > 3/4 full
            rj_signal_wake(sig, 1LL);
        }
    }

    // Each lane's position within the claimed block = popcount(exec bits below
    // this lane's index) = MBCNT. mbcnt_lo covers bits [0:31], mbcnt_hi extends
    // the count through bits [32:63] using the result of mbcnt_lo as the init.
    uint32_t exec_lo     = (uint32_t)exec;
    uint32_t exec_hi     = (uint32_t)(exec >> 32);
    uint32_t lane_offset = __builtin_amdgcn_mbcnt_hi(
                               exec_hi,
                               __builtin_amdgcn_mbcnt_lo(exec_lo, 0u));
    uint64_t my_slot = base_slot + lane_offset;

    // ── Step 2: Overflow check ───────────────────────────────────────────────
    //
    // If the ring is full (my_slot - read_ptr >= slot_count), drop and count.
    // read_ptr is relaxed-loaded; the host publishes it with release after each
    // drain batch, so we may see a slightly stale value — that is acceptable
    // (it only means we drop slightly earlier than necessary).
    //
    // __atomic_load_n emits system-scope load — correct because read_ptr is
    // written by the host CPU and must be visible across the CPU-GPU interconnect.
    uint64_t rptr       = __atomic_load_n(&hdr->read_ptr, __ATOMIC_RELAXED);
    uint64_t slot_count = hdr->slot_count;  // constant after allocation; plain load
    // Protocol invariant: read_ptr is monotonically non-decreasing and is never
    // published past write_ptr (the host only advances read_ptr after consuming
    // completed records). Therefore my_slot >= rptr always holds and the unsigned
    // subtraction does not wrap. If this invariant were violated it would indicate
    // a host-side drain bug, not a GPU-side race.
    if ((my_slot - rptr) >= slot_count) {
        __atomic_fetch_add(&hdr->overflow_count, 1ULL, __ATOMIC_RELAXED);
        return;
    }

    // ── Step 3: Compute record pointer ───────────────────────────────────────
    //
    // slot_count is a power of 2 (enforced at allocation); bitwise AND replaces
    // modulo division.
    uint64_t    idx = my_slot & (slot_count - 1);
    RjLogRecord *rec = (RjLogRecord *)((char *)hdr
                        + sizeof(RjLogBufferHeader)
                        + idx * (uint64_t)sizeof(RjLogRecord));

    // ── Step 4: Collect hardware identifiers ─────────────────────────────────
    //
    // HW_REG_HW_ID (id=4): encodes WAVE_ID, SIMD_ID, PIPE_ID, CU_ID in one
    // 32-bit value. Encoding: bits[15:11]=size-1=31, bits[10:6]=start_bit=0,
    // bits[5:0]=reg_id=4  →  (31u<<11)|(0u<<6)|4u = 0xF804.
    //
    // Portability note: HW_REG_HW_ID (reg_id=4) exists on all GFX9/10 targets.
    // On GFX11+ (RDNA3/4, CDNA4), HW_REG_HW_ID1 (reg_id=4) and HW_REG_HW_ID2
    // (reg_id=15) split the information; the 32-bit read still works but only
    // returns the lower sub-fields. For uniqueness purposes the lower 32 bits are
    // sufficient — if full wave identity is required on GFX11+, a second
    // `s_getreg (31u<<11)|(0u<<6)|15u` call for HW_REG_HW_ID2 is needed.
    uint32_t wave_id = __builtin_amdgcn_s_getreg(
                           (31u << 11) | (0u << 6) | 4u);

    // Pack workgroup X/Y/Z into a single 64-bit word (21 bits each; max grid
    // dimension 2^21 = 2M, which exceeds all current AMDGPU HW limits).
    uint64_t wg_id =
        ((uint64_t)__builtin_amdgcn_workgroup_id_z() << 42) |
        ((uint64_t)__builtin_amdgcn_workgroup_id_y() << 21) |
        ((uint64_t)__builtin_amdgcn_workgroup_id_x());

    // ── Step 5: Write payload fields ─────────────────────────────────────────
    //
    // Plain stores. No memory ordering is required on these stores — the
    // release fence in step 6 ensures they are all visible before valid=1.
    rec->record_type      = record_type;
    rec->inst_byte_offset = inst_byte_offset;
    rec->wave_id          = wave_id;
    rec->workgroup_id     = wg_id;
    rec->payload          = payload;

    // ── Step 6: Release fence then publish valid flag ─────────────────────────
    //
    // The fence drains all outstanding global stores from the GPU's write pipeline
    // before the valid store below.  On GFX9/GFX10 targets this compiles to
    // `s_waitcnt vmcnt(0)`; on GFX11+ (RDNA3/3.5, CDNA4, RDNA4) the compiler emits
    // the equivalent split-wait sequence (e.g., `s_wait_loadcnt 0; s_wait_storecnt 0`).
    // The valid flag is read by the host CPU; it must be visible across the
    // CPU-GPU interconnect. Use a system-scope atomic store (not a plain store)
    // so the store is not reordered past the fence by the GPU write coalescer
    // and is fully visible to the host's acquire load on valid.
    // Release-store the valid flag: the release ordering ensures all preceding
    // plain stores (record_type, wave_id, payload, etc.) are visible to the host
    // before the host's acquire-load of valid sees 1.
    // The __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent") above drains the GPU
    // write pipeline before this atomic store; the fence generates the correct
    // target-specific wait sequence (see note above), which carries its own
    // release ordering at the system scope — no separate fence is needed.
    __atomic_store_n(&rec->valid, 1u, __ATOMIC_RELEASE);  // system-scope release store
}
```

#### 2.7.2 Probe Compilation Model

The probe is a plain C++ file compiled directly to a bare AMDGPU ELF code object. No HIP, no OpenCL, no ROCm device libs. Each supported GFX target gets one independent compilation.

**Compilation command (per target):**

```sh
clang -target amdgcn-amd-amdhsa \
      -mcpu=<gfx-id> \               # e.g. gfx942, gfx950, gfx1100
      -x c++ \
      -std=c++17 \
      -O2 \
      -nogpulib \                    # skip ROCm device libs — not needed
      -fno-exceptions \
      -fno-rtti \
      -ffreestanding \               # no host stdlib; __atomic_* are compiler builtins
      lib/rocjitsu/device/rj_log_probe.cpp \
      -o lib/rocjitsu/device/rj_log_probe_<gfx-id>.co
```

The output is a bare `amdgcn-amd-amdhsa` ELF that `AmdGpuCodeObject` can parse directly, containing exactly the `rj_log_write` symbol with no external dependencies.

The 9 targets and their ISA family:

| GFX ID | ISA family | Wave default |
|---|---|---|
| `gfx908` | CDNA1 | Wave64 |
| `gfx90a` | CDNA2 | Wave64 |
| `gfx942` | CDNA3 | Wave64 |
| `gfx950` | CDNA4 | Wave64 |
| `gfx1010` | RDNA1 | Wave32 |
| `gfx1030` | RDNA2 | Wave32 |
| `gfx1100` | RDNA3 | Wave32 |
| `gfx1150` | RDNA3.5 | Wave32 |
| `gfx1200` | RDNA4 | Wave32 |

Wave size is not controlled by a compiler flag here because `rj_log_write` reads the full 64-bit EXEC register via `__builtin_amdgcn_read_exec()` and uses `__builtin_popcountll` — the upper 32 bits are zero on Wave32 targets, so the same source is correct for both widths.

**Embedding in `librocjitsu.so`:**

Each `.co` file is embedded as a read-only byte array using a CMake-generated `.S` assembler file:

```cmake
# cmake/EmbedCodeObject.cmake
function(embed_code_object target gfx_id)
    set(src_file "${CMAKE_SOURCE_DIR}/lib/rocjitsu/device/rj_log_probe.cpp")
    set(co_file "${CMAKE_CURRENT_BINARY_DIR}/rj_log_probe_${gfx_id}.co")
    set(asm_file "${CMAKE_CURRENT_BINARY_DIR}/rj_log_probe_${gfx_id}.s")
    add_custom_command(
        OUTPUT  ${asm_file}
        COMMAND ${CMAKE_COMMAND}
                -DINPUT=${co_file}
                -DGFX=${gfx_id}
                -DOUTPUT=${asm_file}
                -P ${CMAKE_SOURCE_DIR}/cmake/GenIncbin.cmake
        DEPENDS ${co_file}
    )
    target_sources(${target} PRIVATE ${asm_file})
endfunction()
```

`GenIncbin.cmake` emits:
```asm
    .section .rodata
    .globl rj_log_probe_<gfx_id>_start
    .globl rj_log_probe_<gfx_id>_end
rj_log_probe_<gfx_id>_start:
    .incbin "rj_log_probe_<gfx_id>.co"
rj_log_probe_<gfx_id>_end:
```

At runtime, `ProbeLibrary::get(rj_code_arch_t arch)` maps `arch` to the correct `{start, end}` symbol pair, wraps it in an `AmdGpuCodeObject`, and returns the `AmdGpuCodeObject*` for `rj_log_write` via symbol lookup. `Instrumentor` passes this to `InstrumentationPoint::probe_obj` / `probe_symbol`.

```cpp
// lib/rocjitsu/src/rocjitsu/code/patch/probe_library.h

namespace rocjitsu {

/// @brief Provides access to the pre-compiled built-in probe code objects
///        embedded in librocjitsu.so as read-only byte arrays.
///
/// Each probe code object is compiled once per GFX target at build time
/// and embedded via GenIncbin.cmake. ProbeLibrary wraps each embedded byte
/// range as an AmdGpuCodeObject for use by Instrumentor.
class ProbeLibrary {
public:
  /// @brief Returns the pre-compiled log-write probe code object for @p arch.
  /// Returns nullptr for unsupported architectures.
  static const AmdGpuCodeObject *get(rj_code_arch_t arch);
};

} // namespace rocjitsu
```

**Verification step in the build:**

After compilation, a CMake post-build command runs:
```sh
llvm-readobj --symbols rj_log_probe_<gfx_id>.co | grep rj_log_write
llvm-readobj --file-header rj_log_probe_<gfx_id>.co | grep PrivateSegmentFixedSize
```
The second check asserts `PrivateSegmentFixedSize == 0` — confirming no scratch use and therefore the simple calling convention documented in §2.4.

#### 2.7.3 Device-Side Code Policy and `rj_signal_wake`

**Policy: no HIP in the rocjitsu repository.**

All device code that rocjitsu compiles and embeds (`rj_log_probe.cpp`, `rj_hostcall_device.c`, any future built-in probes) is written in standard C or standard C++ and compiled with `clang -target amdgcn-amd-amdhsa`. No `#include <hip/hip_runtime.h>`, no `__device__` (that is a HIP/CUDA-ism), no `__hip_atomic_*`. The `__builtin_amdgcn_*` compiler builtins used throughout are LLVM AMDGPU target builtins and do not require HIP.

**User-provided probes may use HIP.** The `InstrumentationPoint::probe_obj` field accepts any `AmdGpuCodeObject*` regardless of what language or toolchain produced it. A user who writes a probe in HIP, compiles it with `hipcc`, and extracts the device code object can pass it directly to `rj_code_instrumentor_add_probe`. rocjitsu only requires that the resulting ELF contains a callable `__device__`-convention (AMDGPU non-kernel CC) symbol matching `probe_symbol`. The source language is irrelevant.

**`rj_signal_wake` — device-side HSA signal notification:**

`rj_signal_wake` implements the same operation as `__ockl_hsa_signal_add` from device-libs but with no ockl or HIP dependency. It uses the `amd_signal_t` memory layout, which is part of the stable AMD HSA ABI (documented in `amd_hsa_signal.h`).

```c
// lib/rocjitsu/device/rj_signal.h
// Included by rj_log_probe.cpp and rj_hostcall_device.c.
// Standard C; compiled with clang -target amdgcn-amd-amdhsa.

#pragma once
#include <stdint.h>

// amd_signal_t memory layout offsets (AMD HSA ABI — amd_hsa_signal.h).
// hsa_signal_t is a uint64_t whose value is the VA of an amd_signal_t struct
// in fine-grained system memory.
//
//   +0  : kind        (int64_t)  AMD_SIGNAL_KIND_USER = 1
//   +8  : value       (int64_t)  current signal value (for USER signals)
//   +16 : event_mailbox_ptr (uint64_t) VA of KFD event mailbox page; 0 if no event
//   +24 : event_id    (uint32_t) ID to write into mailbox to fire KFD event
//
// The KFD event mailbox write is what allows hsa_signal_wait_scacquire with
// HSA_WAIT_STATE_BLOCKED to sleep the host thread in the OS rather than
// busy-spinning. Without it the host must busy-poll.

#define RJ_AMD_SIGNAL_VALUE_OFFSET    8u
#define RJ_AMD_SIGNAL_MAILBOX_OFFSET 16u
#define RJ_AMD_SIGNAL_EVENT_ID_OFFSET 24u

// Atomically add `delta` to the signal value (release ordering) and fire the
// KFD event mailbox to wake any host thread sleeping in hsa_signal_wait.
//
// Equivalent to __ockl_hsa_signal_add(signal, delta, memory_order_release)
// from device-libs, without any ockl or HIP dependency.
static __attribute__((always_inline))
void rj_signal_wake(uint64_t signal_handle, int64_t delta)
{
    // Release atomic add: ensures all preceding stores (payload, valid flags)
    // are globally visible before the signal value changes.
    // __atomic_fetch_add on the AMDGPU target emits global_atomic_add_u64
    // with system scope — correct for CPU-GPU visibility on fine-grained memory.
    __atomic_fetch_add(
        (volatile int64_t *)(signal_handle + RJ_AMD_SIGNAL_VALUE_OFFSET),
        delta, __ATOMIC_RELEASE);

    // Fire the KFD event to wake a host thread sleeping in hsa_signal_wait.
    // mailbox_ptr is zero when the signal was created without an associated
    // OS event (e.g., HSA_WAIT_STATE_ACTIVE polling signals); skip in that case.
    uint64_t event_mailbox_va = *(volatile uint64_t *)(
        signal_handle + RJ_AMD_SIGNAL_MAILBOX_OFFSET);
    if (event_mailbox_va) {
        uint32_t eid = *(volatile uint32_t *)(
            signal_handle + RJ_AMD_SIGNAL_EVENT_ID_OFFSET);
        // Writing event_id to the mailbox page triggers the OS-level interrupt
        // via the KFD event mechanism — same approach as __ockl_hsa_signal_add.
        __atomic_store_n((volatile uint32_t *)event_mailbox_va, eid, __ATOMIC_RELAXED);
    }
}
```

The host creates the `hsa_signal_t` with `hsa_signal_create(initial_value=1, ...)`. Whether ROCR allocates an interrupt-capable signal (with a valid `event_mailbox_ptr` / `event_id`) depends on ROCR's internal `g_use_interrupt_wait` flag (controlled by `HSA_ENABLE_INTERRUPT`, default on in most builds). If interrupt mode is disabled, `event_mailbox_ptr` is null, `rj_signal_wake`'s mailbox write is skipped, and `hsa_signal_wait_scacquire` with `HSA_WAIT_STATE_BLOCKED` busy-spins. After creating the signal, verify `reinterpret_cast<amd_signal_t *>(signal.handle)->event_mailbox_ptr != 0`; if zero, log a warning that drain latency may be high. The initial signal value of 1 means the first `hsa_signal_wait_scacquire(signal, NE, 1, ...)` blocks until the GPU increments to ≥ 2.

**Host drain thread pattern:**

```cpp
// lib/rocjitsu/src/rocjitsu/hooks/log_drain_thread.cpp
void LogDrainThread::run() {
    // Unconditional initial drain: the GPU may have already fired rj_signal_wake
    // (and written records) before this thread started. The initial load captures
    // the current value; the drain below processes any already-written records.
    uint64_t observed = hsa_signal_load_scacquire(signal_);
    for (auto &[kd_va, buf] : registry_.buffers()) {
        if (buf->has_pending_records())
            buf->drain(callback_);
    }

    while (!stop_requested_.load(std::memory_order_relaxed)) {
        // Sleep until GPU fires rj_signal_wake or timeout expires.
        // HSA_WAIT_STATE_BLOCKED: OS may sleep this thread (KFD interrupt path).
        // Timeout prevents permanent sleep if the GPU never signals (e.g., kernel
        // finishes without crossing the HWM threshold).
        constexpr uint64_t kTimeoutNs = 5'000'000; // 5 ms
        uint64_t new_value = hsa_signal_wait_scacquire(
            signal_,
            HSA_SIGNAL_CONDITION_NE,
            observed,
            kTimeoutNs,
            HSA_WAIT_STATE_BLOCKED);

        observed = new_value; // update for next wait

        // Drain all registered log buffers for this listener.
        for (auto &[kd_va, buf] : registry_.buffers()) {
            if (buf->has_pending_records())
                buf->drain(callback_);
        }
    }
}
```

**Host drain protocol:**

The drain is called from a background host thread or from the post-dispatch completion handler (for short-running kernels):

```cpp
void LogBuffer::drain(std::function<void(const LogRecord &)> callback) {
  const uint64_t wptr = header_->write_ptr.load(std::memory_order_acquire);

  while (local_read_ptr_ < wptr) {
    uint64_t slot = local_read_ptr_ % header_->slot_count;
    LogRecord *rec = &records_[slot];

    // Spin until the GPU's release-store of valid=1 is visible (acquire).
    // Head-of-line blocking: if slot N is not yet valid, the drain stalls even
    // if slot N+1 is already valid. In practice this spin is very short because
    // the GPU fills slots in order and typically finishes before the host reaches
    // the slot. For pathological cases where a lane is active but writes slowly,
    // the drain thread may add a bounded retry limit and skip to the next batch
    // rather than blocking indefinitely (future improvement; acceptable for MVP).
    while (rec->valid.load(std::memory_order_acquire) == 0)
      std::this_thread::yield();

    callback(*rec);

    // Clear the slot for reuse before advancing read_ptr.
    rec->valid.store(0, std::memory_order_relaxed);
    ++local_read_ptr_;
  }

  // Publish the updated read_ptr so the GPU knows these slots are free.
  header_->read_ptr.store(local_read_ptr_, std::memory_order_release);
}
```

The host must publish `read_ptr` so the GPU's overflow check (step 2 above) sees the freed space and stops dropping records. Without this publication, a long-running kernel would overflow the ring after `slot_count` records even if the host is draining.

**Drain triggers:**

| Scenario | Drain mechanism |
|---|---|
| Short kernel (< 1ms) | Post-dispatch: drain after HSA completion signal fires |
| Long kernel | Background host thread sleeping in `hsa_signal_wait_scacquire` on the buffer's `signal_handle`; woken by GPU via `rj_signal_wake` (§2.7.3) |
| Buffer > 3/4 full | GPU probe calls `rj_signal_wake(hdr->signal_handle)` immediately after claiming a slot (see probe code above); host wakes within microseconds |

The `signal_handle` is a raw `hsa_signal_t` value stored in `LogBufferHeader` at offset +32, written once at allocation time by the host (never modified). The GPU reads it with a relaxed load. The host background thread calls `hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_NE, current_value, timeout_ns, HSA_WAIT_STATE_BLOCKED)` — the `BLOCKED` hint allows the OS (via KFD) to sleep the thread rather than busy-spinning. `rj_signal_wake` (§2.7.3) increments the signal value with a release atomic and fires the KFD event mailbox write that wakes the sleeping host thread; it does not use HIP or ockl.

**Buffer sizing:**

At 32 bytes per record, the default `slot_count = 131072` (4 MiB total) holds 128K records before overflow. For memory-op tracing (one record per memory instruction per wave), this fills in roughly `128K / (waves_in_flight * records_per_wave_per_ms)` milliseconds — configurable via `RJ_LOG_BUFFER_RECORDS` env var.

**`LogBuffer` class:**

```cpp
// lib/rocjitsu/src/rocjitsu/hooks/log_buffer.h

namespace rocjitsu {

class LogBuffer {
public:
  /// @brief Allocate a log buffer for @p agent in fine-grained system memory.
  static std::unique_ptr<LogBuffer> create(hsa_agent_t agent,
                                            uint32_t slot_count,
                                            hsa_signal_t hwm_signal);
  ~LogBuffer();

  /// @brief GPU VA of the buffer base (written into kernarg by rj_packet_handler).
  uint64_t gpu_va() const;

  /// @brief Returns true if any records have been written but not yet drained.
  /// Checked by LogDrainThread::run() to avoid a full drain() pass on empty buffers.
  bool has_pending_records() const;

  /// @brief Drain all ready records and invoke @p callback for each.
  void drain(std::function<void(const LogRecord &)> callback);

  uint64_t overflow_count() const;

private:
  LogBufferHeader *header_;    ///< Host VA of the header.
  LogRecord       *records_;   ///< Host VA of the record array.
  uint64_t         gpu_va_;    ///< GPU VA (= host VA on hUMA; may differ on dGPU).
  uint64_t         local_read_ptr_ = 0;
};

} // namespace rocjitsu
```

`LoggingBufferRegistry` maps each original KD VA to a `LogBuffer` instance, allocated once when the kernel is instrumented at load time and reused across all dispatches of that kernel.

```cpp
// lib/rocjitsu/src/rocjitsu/hooks/logging_buffer_registry.h

namespace rocjitsu {

/// @brief Thread-safe registry mapping original KD GPU VAs → LogBuffer instances.
///
/// Written once per kernel at load time (by Instrumentor after patching).
/// Read at dispatch time by rj_packet_handler to look up the buffer VA.
/// Also read by LogDrainThread to iterate all registered buffers.
class LoggingBufferRegistry {
public:
  static LoggingBufferRegistry &instance();

  /// @brief Register a log buffer for a kernel.
  /// @param original_kd_va  GPU VA of the original (un-patched) kernel descriptor.
  /// @param buf             The pre-allocated log buffer for this kernel.
  /// @param attributor      Source attributor for enriching drained records.
  void register_buffer(uint64_t original_kd_va,
                       std::unique_ptr<LogBuffer> buf,
                       std::unique_ptr<SourceAttributor> attributor);

  /// @brief Returns the GPU VA of the log buffer for @p kd_va, or 0 if not found.
  uint64_t buffer_for(uint64_t original_kd_va) const;

  /// @brief Returns a snapshot of all registered (kd_va, LogBuffer*) pairs for drain iteration.
  ///
  /// Returns a copy (not a reference) of the live buffer list taken under a shared_lock,
  /// preventing iterator invalidation if register_buffer() is called concurrently from
  /// another thread. The raw LogBuffer* pointers are valid for the registry lifetime —
  /// buffers are never removed while the registry is alive.
  std::vector<std::pair<uint64_t, LogBuffer *>> buffers() const;

private:
  LoggingBufferRegistry() = default;
  LoggingBufferRegistry(const LoggingBufferRegistry &) = delete;
  LoggingBufferRegistry &operator=(const LoggingBufferRegistry &) = delete;
  LoggingBufferRegistry(LoggingBufferRegistry &&) = delete;
  LoggingBufferRegistry &operator=(LoggingBufferRegistry &&) = delete;

  mutable std::shared_mutex mutex_;
  std::unordered_map<uint64_t, std::unique_ptr<LogBuffer>>     buf_map_;
  std::unordered_map<uint64_t, std::unique_ptr<SourceAttributor>> attr_map_;
};

} // namespace rocjitsu
```

### 2.8 C API for DBI

The public API exposes only the outcome: given a target code object and a user-supplied probe device function, produce an instrumented ELF. Analysis internals (`LivenessAnalysis`, `RegisterSet`, `SpillManager`) are not exposed.

```c
// New additions to rj_code.h (or new header rj_code_dbi.h)

/// @brief Opaque handle to an instrumentor.
typedef struct rj_code_instrumentor_t rj_code_instrumentor_t;

/// @brief Where relative to matching instructions the probe fires.
typedef enum rj_code_instrument_kind_e {
  RJ_CODE_INSTR_BEFORE_INST,  ///< Before each matching instruction.
  RJ_CODE_INSTR_AFTER_INST,   ///< After each matching instruction.
  RJ_CODE_INSTR_BLOCK_ENTRY,  ///< At entry of each basic block containing a match.
  RJ_CODE_INSTR_BLOCK_EXIT,   ///< Before every terminator of each basic block containing a match.
} rj_code_instrument_kind_t;

/// @brief Create an instrumentor for the given target code object.
/// @param[in]  target  Target code object to instrument.
/// @param[in]  arch    Target ISA.
/// @param[out] instrumentor  Newly created instrumentor handle.
[[nodiscard]] RJ_API_EXPORT rj_status_t
rj_code_instrumentor_create(rj_code_object_t *target,
                             rj_code_arch_t arch,
                             rj_code_instrumentor_t **instrumentor);

/// @brief Destroy an instrumentor.
RJ_API_EXPORT void rj_code_instrumentor_destroy(rj_code_instrumentor_t *instrumentor);

/// @brief Add a probe: trampoline to a GPU device function at each matching point.
///
/// The probe is a device function in @p probe_obj identified by @p probe_symbol.
/// rocjitsu merges @p probe_obj into the instrumented output ELF and generates a
/// save/call/restore trampoline at each instruction matching @p filter_flags.
///
/// Probe calling convention: the trampoline saves all registers live at the patch
/// point before the call and restores them after. The probe receives no explicit
/// arguments and may use any register freely.
///
/// @param instrumentor   Instrumentor handle.
/// @param probe_obj      Code object containing the probe device function.
/// @param probe_symbol   Symbol name of the probe function in @p probe_obj.
/// @param kind           When to fire relative to matching instructions.
/// @param filter_flags   Bitmask of rj_code_inst_flags_t; 0 = all instructions.
/// @param force_full_exec If true, restore exec to -1 before calling the probe.
[[nodiscard]] RJ_API_EXPORT rj_status_t
rj_code_instrumentor_add_probe(rj_code_instrumentor_t *instrumentor,
                                const rj_code_object_t *probe_obj,
                                const char *probe_symbol,
                                rj_code_instrument_kind_t kind,
                                uint32_t filter_flags,
                                bool force_full_exec);

/// @brief Execute all patches and return a merged instrumented ELF.
///
/// The output ELF contains the patched target .text, the merged probe code,
/// the trampoline section (.rj_trampolines), and a .note.rocjitsu metadata section.
/// @param[out] patched_elf       Heap-allocated buffer; caller frees with free().
/// @param[out] patched_elf_size  Size of @p patched_elf in bytes.
[[nodiscard]] RJ_API_EXPORT rj_status_t
rj_code_instrumentor_patch(rj_code_instrumentor_t *instrumentor,
                            uint8_t **patched_elf,
                            size_t *patched_elf_size);
```

> **Future extension — host-generated probe bytes:** A second mode where the caller
> supplies a host-side callback that returns raw GPU ISA bytes for each patch point
> (analogous to DynamoRIO's `dr_insert_*` model) is a useful capability but is
> deferred. It requires the callback to make decisions based on liveness information
> at each patch point without the public API exposing `RegisterSet` or `InstDefUse`
> directly. The right design — an opaque "probe context" handle exposing only
> what a callback needs — will be specified once the internal pipeline is stable.

---

### 2.9 RjHostcall — Generic GPU-to-Host Call Protocol

#### 2.9.1 Design Rationale

The log buffer's `rj_signal_wake` mechanism (§2.7.3) is sufficient for asynchronous host notification — the GPU fires and continues. For future use cases where the GPU needs a **synchronous response** from the host (GPU memory allocation, device-side printf, wave-level assertions that must abort the dispatch), a two-way blocking protocol is required.

ROCm device-libs implements exactly this in its `hostcall` subsystem (`ockl/src/hostcall*.cl`, `rocclr/device/devhostcall.*`). rocjitsu's `RjHostcall` copies the same fundamental design — Treiber stack free/ready queues, per-wave packets, KFD-interrupt signaling, `readfirstlane`-gated spin-wait — but with no ockl or HIP dependency, using the same standard-C device compilation approach as the rest of rocjitsu's device code. The buffer is passed to kernels via the existing kernarg extension mechanism (§4.3.4) rather than via the OpenCL implicit-arg pointer.

**Key design choices matching device-libs:**
- One packet per in-flight wave; at most one concurrent hostcall per wave.
- Tagged 64-bit pointers (Treiber stacks) for ABA-safe lock-free stack operations.
- GPU spins on `control` READY_FLAG with `__builtin_amdgcn_readfirstlane` to prevent load-hoisting and `__builtin_amdgcn_s_sleep(1)` to yield the SIMD to other waves.
- One shared HSA signal (doorbell) per listener; all per-kernel buffers share it.
- Host steals the entire ready stack with `atomic_exchange(..., 0)` to batch-process packets.

**Key differences from device-libs:**
- No ockl, no HIP — pure standard C device code.
- Buffer VA passed via kernarg extension, not via implicit kernel arg pointer.
- Service IDs are rocjitsu-specific; no OpenCL printf/devmem services.
- `RjHostcall` is a purely future-facing mechanism; the log buffer uses `rj_signal_wake` (async, no packet overhead).

#### 2.9.2 Buffer and Packet Layout

The entire buffer is one contiguous fine-grained system memory allocation:

```
[ RjHostcallHeader   (64 bytes)              ]  ← VA passed to kernel via kernarg
[ RjPacketHeader[N]  (32 bytes × N)          ]  ← headers immediately follow
[ padding to alignof(RjPayload) = 8 bytes    ]
[ RjPayload[N]       (4096 bytes × N)        ]
```

`N` must be at least the maximum number of waves that can be in-flight on the device simultaneously (`max_waves_per_queue`), rounded up to the next power of two. `N` is fixed at allocation time.

```c
// lib/rocjitsu/device/rj_hostcall.h
// Standard C; shared between device-side (compiled with -target amdgcn-amd-amdhsa)
// and host-side (compiled normally). Layout must be identical on both sides.

#pragma once
#include <stdint.h>

// ── Control block (at the start of the allocation, offset 0) ─────────────────
//
// Device code reads headers_va, payloads_va, signal_handle, free_stack,
// ready_stack, and index_mask. The host initialises all fields.
typedef struct {
    uint64_t headers_va;    // +0:  VA of RjPacketHeader array
    uint64_t payloads_va;   // +8:  VA of RjPayload array
    uint64_t signal_handle; // +16: raw hsa_signal_t; GPU calls rj_signal_wake after push
    uint64_t free_stack;    // +24: head of free-packet Treiber stack (host initialises)
    uint64_t ready_stack;   // +32: head of ready-packet Treiber stack (GPU pushes)
    uint64_t index_mask;    // +40: (N - 1); fast modulo: packet_index = ptr & index_mask
    uint8_t  _pad[64 - 48]; // +48: pad to 64 bytes
} RjHostcallHeader;         // 64 bytes; alignas(64)

// ── Packet header (32 bytes) ─────────────────────────────────────────────────
//
// One per packet. Lives in the headers array immediately after RjHostcallHeader.
typedef struct {
    uint64_t next;        // +0:  tagged pointer to next packet in intrusive stack
    uint64_t activemask;  // +8:  64-bit EXEC mask at call site (from __builtin_amdgcn_read_exec)
    uint32_t service_id;  // +16: which service to invoke on the host
    uint32_t control;     // +20: bit 0 = READY_FLAG: 1=awaiting host, 0=host done
    uint32_t _pad[2];     // +24: pad to 32 bytes
} RjPacketHeader;         // 32 bytes; alignas(8)

// ── Payload (4096 bytes) ──────────────────────────────────────────────────────
//
// 64 lanes × 8 × 8 bytes = 4096 bytes, matching device-libs payload layout.
// GPU writes input args into slots[lane][0..7] before setting READY_FLAG.
// Host writes response into slots[lane][0..1] before clearing READY_FLAG.
typedef struct {
    uint64_t slots[64][8];
} RjPayload;              // 4096 bytes; alignas(8)

// ── Service IDs ──────────────────────────────────────────────────────────────
#define RJ_HOSTCALL_SERVICE_RESERVED  0u  // must not be dispatched
#define RJ_HOSTCALL_SERVICE_LOG_FLUSH 1u  // slot[0] = log_buffer_va; drain it; no return
#define RJ_HOSTCALL_SERVICE_ASSERT    2u  // slot[0] = message_va; host prints wave dump, aborts
#define RJ_HOSTCALL_SERVICE_USER      3u  // slot[0] = fn ptr; host calls fn(output, input)

// ── Tagged pointer helpers ────────────────────────────────────────────────────
//
// Packets are referenced by tagged 64-bit pointers. The packet index lives in
// the low log2(N) bits (= index_mask bits); the tag lives in the remaining upper
// bits. Incrementing the tag on every pop prevents the ABA problem in the Treiber
// stacks. One tag unit = index_mask + 1 (the smallest value that increments only
// the tag field without touching the index field).
//
// This is identical to device-libs' inc_ptr_tag / tagged pointer scheme.
static inline uint64_t rj_ptr_index(uint64_t ptr, uint64_t index_mask) {
    return ptr & index_mask;
}
static inline uint64_t rj_ptr_inc_tag(uint64_t ptr, uint64_t index_mask) {
    uint64_t tag_unit = index_mask + 1;
    uint64_t result   = ptr + tag_unit;
    return result ? result : tag_unit; // never return zero (null sentinel)
}
```

#### 2.9.3 GPU Device-Side Implementation

The device-side implementation is one standard-C source file:

```c
// lib/rocjitsu/device/rj_hostcall_device.c
// Compiled with: clang -target amdgcn-amd-amdhsa -mcpu=<gfx-id> -x c -O2 -nogpulib
// No HIP. No ockl. All synchronisation via __atomic_* and __builtin_amdgcn_*.

#include "rj_hostcall.h"
#include "rj_signal.h"   // rj_signal_wake
#include <stdint.h>

// ── Treiber stack pop (lock-free, ABA-safe) ──────────────────────────────────
//
// Returns the popped tagged pointer, or 0 if the stack is empty.
// Spins with s_sleep(1) on CAS failure to yield the SIMD to other waves.
// Memory scope: system (__atomic_* default on AMDGPU) — correct because
// both GPU waves and the host CPU touch these stacks.
static uint64_t treiber_pop(uint64_t *stack_head,
                             const RjPacketHeader *headers_base,
                             uint64_t index_mask)
{
    while (1) {
        uint64_t head = __atomic_load_n(stack_head, __ATOMIC_ACQUIRE);
        if (!head) return 0; // empty

        uint64_t idx  = rj_ptr_index(head, index_mask);
        uint64_t next = __atomic_load_n(
            &headers_base[idx].next, __ATOMIC_RELAXED);

        // New head: next's index, head's tag incremented.
        uint64_t new_head = rj_ptr_index(next, index_mask)
                          | (rj_ptr_inc_tag(head, index_mask) & ~index_mask);

        if (__atomic_compare_exchange_n(stack_head, &head, new_head,
                                        /*weak=*/1,
                                        __ATOMIC_RELEASE, __ATOMIC_ACQUIRE))
            return head;

        __builtin_amdgcn_s_sleep(1); // yield on contention
    }
}

// ── Treiber stack push (lock-free) ───────────────────────────────────────────
static void treiber_push(uint64_t *stack_head, uint64_t ptr,
                          RjPacketHeader *header, uint64_t index_mask)
{
    while (1) {
        uint64_t top = __atomic_load_n(stack_head, __ATOMIC_RELAXED);
        __atomic_store_n(&header->next, top, __ATOMIC_RELAXED);
        if (__atomic_compare_exchange_n(stack_head, &top, ptr,
                                        /*weak=*/1,
                                        __ATOMIC_RELEASE, __ATOMIC_ACQUIRE))
            return;
        __builtin_amdgcn_s_sleep(1);
    }
}

// ── rj_hostcall_invoke ───────────────────────────────────────────────────────
//
// Synchronous GPU-to-host call. Blocks the entire wave until the host responds.
//
// ctrl_va     : VA of RjHostcallHeader (from kernarg extension).
// service_id  : which host service to invoke (RJ_HOSTCALL_SERVICE_*).
// args        : up to 8 uint64_t args written into payload->slots[lane][0..7]
//               for each active lane.
// Returns     : host response in slots[this_lane][0..1] as a uint64_t[2].
//
// Only lane 0 (determined by MBCNT) drives the stack operations; other lanes
// wait via the readfirstlane-gated spin. This mirrors device-libs' pattern.
__attribute__((noinline))
void rj_hostcall_invoke(uint64_t ctrl_va, uint32_t service_id,
                         const uint64_t args[8], uint64_t response[2])
{
    RjHostcallHeader *ctrl =
        (RjHostcallHeader *)(uintptr_t)ctrl_va;
    RjPacketHeader   *headers =
        (RjPacketHeader *)(uintptr_t)ctrl->headers_va;
    RjPayload        *payloads =
        (RjPayload *)(uintptr_t)ctrl->payloads_va;
    uint64_t          index_mask  = ctrl->index_mask;

    uint64_t exec   = __builtin_amdgcn_read_exec();
    uint32_t my_lane = __builtin_amdgcn_mbcnt_hi(
                           (uint32_t)(exec >> 32),
                           __builtin_amdgcn_mbcnt_lo((uint32_t)exec, 0u));
    int is_lane0 = (__builtin_amdgcn_mbcnt_lo(~0u, 0u) == 0);

    // ── Step 1: lane 0 pops a free packet ────────────────────────────────────
    uint64_t packet_ptr = 0;
    if (is_lane0)
        packet_ptr = treiber_pop(&ctrl->free_stack, headers, index_mask);

    // Broadcast packet_ptr to all active lanes.
    uint32_t ptr_lo = __builtin_amdgcn_readfirstlane((uint32_t)packet_ptr);
    uint32_t ptr_hi = __builtin_amdgcn_readfirstlane((uint32_t)(packet_ptr >> 32));
    packet_ptr = ((uint64_t)ptr_hi << 32) | ptr_lo;

    // Spin until a free packet is available (should be rare under normal load).
    while (!packet_ptr) {
        __builtin_amdgcn_s_sleep(1);
        if (is_lane0)
            packet_ptr = treiber_pop(&ctrl->free_stack, headers, index_mask);
        ptr_lo = __builtin_amdgcn_readfirstlane((uint32_t)packet_ptr);
        ptr_hi = __builtin_amdgcn_readfirstlane((uint32_t)(packet_ptr >> 32));
        packet_ptr = ((uint64_t)ptr_hi << 32) | ptr_lo;
    }

    uint64_t         idx = rj_ptr_index(packet_ptr, index_mask);
    RjPacketHeader  *hdr = &headers[idx];
    RjPayload       *pld = &payloads[idx];

    // ── Step 2: fill header and payload ──────────────────────────────────────
    hdr->activemask = exec;
    hdr->service_id = service_id;

    // Each active lane writes its own args into its payload slot.
    if (my_lane < 64) {
        for (int i = 0; i < 8; ++i)
            pld->slots[my_lane][i] = args[i];
    }

    // Set READY_FLAG = 1. Plain store — ordering provided by the push CAS below.
    __atomic_store_n(&hdr->control, 1u, __ATOMIC_RELAXED);

    // ── Step 3: lane 0 pushes to ready stack, then wakes the host ────────────
    if (is_lane0) {
        treiber_push(&ctrl->ready_stack, packet_ptr, hdr, index_mask);
        rj_signal_wake(ctrl->signal_handle, 1LL);
    }

    // ── Step 4: spin until host clears READY_FLAG ─────────────────────────────
    //
    // readfirstlane serialises all lanes through lane 0's acquire load.
    // Without it the compiler could hoist payload reads above the fence.
    // s_sleep(1) yields the SIMD so the host-side thread can get CPU time
    // to process the packet — same approach as device-libs.
    while (1) {
        uint32_t ready = 1;
        if (is_lane0)
            ready = __atomic_load_n(&hdr->control, __ATOMIC_ACQUIRE) & 1u;
        ready = __builtin_amdgcn_readfirstlane(ready);
        if (!ready) break;
        __builtin_amdgcn_s_sleep(1);
    }

    // ── Step 5: each lane reads host response from its slot ──────────────────
    if (response && my_lane < 64) {
        response[0] = pld->slots[my_lane][0];
        response[1] = pld->slots[my_lane][1];
    }

    // ── Step 6: lane 0 returns packet to free stack ───────────────────────────
    if (is_lane0)
        treiber_push(&ctrl->free_stack, packet_ptr, hdr, index_mask);
}
```

#### 2.9.4 Host-Side Listener

```cpp
// lib/rocjitsu/src/rocjitsu/hooks/hostcall_listener.h
#pragma once
#include <hsa/hsa.h>
#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

class RjHostcallBuffer;

/// @brief Service handler signature.
/// @p slots  Per-lane slot array: input is slots[lane][0..7];
///           write response (up to 2 uint64_t) into slots[lane][0..1].
/// @p activemask  64-bit EXEC mask from the GPU at call site.
using HostcallHandler =
    std::function<void(uint64_t slots[64][8], uint64_t activemask)>;

/// @brief Runs a single background thread that sleeps on the shared doorbell
///        signal and dispatches host-call packets to registered service handlers.
///        Mirrors the device-libs HostcallListener pattern.
class RjHostcallListener {
public:
    /// @brief Create the listener. @p doorbell must be an event-capable HSA
    ///        signal (hsa_signal_create with initial value 1 so the first wait
    ///        blocks immediately).
    explicit RjHostcallListener(hsa_signal_t doorbell);
    ~RjHostcallListener();

    /// @brief Register a buffer belonging to one HSA queue.
    void add_buffer(std::shared_ptr<RjHostcallBuffer> buf);

    /// @brief Register a handler for @p service_id.
    void register_service(uint32_t service_id, HostcallHandler handler);

    void start();
    void stop();  // signals shutdown, joins thread

private:
    void run();
    void process_all_buffers();

    hsa_signal_t                                      doorbell_;
    std::vector<std::shared_ptr<RjHostcallBuffer>>    buffers_;
    std::unordered_map<uint32_t, HostcallHandler>     services_;
    std::thread                                        thread_;
    std::atomic<bool>                                  stop_{false};

    static constexpr uint64_t kTimeoutFloor = 4ULL  * 1024 * 1024; // ~4  ms
    static constexpr uint64_t kTimeoutCeil  = 16ULL * 1024 * 1024; // ~16 ms
};

} // namespace rocjitsu
```

`RjHostcallListener::run` mirrors the device-libs listener loop exactly: exponential backoff on the timeout (halve on activity, double on inactivity, clamp to `[kTimeoutFloor, kTimeoutCeil]`), `hsa_signal_wait_scacquire` with `HSA_WAIT_STATE_BLOCKED`, then `process_all_buffers` which atomic-exchanges `ready_stack` to 0, walks the stolen chain, dispatches each packet to its service handler for each active lane, and clears `control` (READY_FLAG → 0) with a release store.

**Built-in service registrations (at listener startup):**

| Service | Handler |
|---|---|
| `RJ_HOSTCALL_SERVICE_LOG_FLUSH` | Look up `LogBuffer` by `slots[0][0]` (= log_buffer_va) in `LoggingBufferRegistry`; call `drain()`. |
| `RJ_HOSTCALL_SERVICE_ASSERT` | Print wave dump: service_id, activemask, wave_id from `slots[0][0]`; set a per-buffer abort flag; write `slots[lane][0] = 0` (no meaningful return). |
| `RJ_HOSTCALL_SERVICE_USER` | Cast `slots[0][0]` to `void (*fn)(uint64_t *out, const uint64_t *in)`; call with `slots[lane]+1` as input; write return values back. |

#### 2.9.5 Relationship Between `rj_signal_wake` and `RjHostcall`

These are two distinct mechanisms for different access patterns:

| | `rj_signal_wake` | `RjHostcall` |
|---|---|---|
| GPU blocks? | No — fire and continue | Yes — spin-waits for response |
| Overhead | One atomic add + mailbox write | Treiber pop + payload fill + Treiber push + signal + spin-wait |
| Use case | Log buffer HWM notification; any async wake | Future: GPU-side assert, printf, GPU memory alloc |
| Packet buffer needed? | No | Yes (pre-allocated with `add_buffer`) |
| Host shares doorbell? | Yes (same `signal_handle`) | Yes (same doorbell signal on `RjHostcallHeader`) |

Both mechanisms share the same host listener thread (same doorbell signal). The host always drains log buffers in `process_all_buffers` whether it woke from a `rj_signal_wake` (no hostcall packets) or from an `RjHostcall` push (has packets to process). This is safe — draining an empty log buffer is a no-op.

---

## Pillar 3: Dynamic Binary Translation (DBT) {#pillar-3}

### 3.1 Motivation and Design Rationale

The DBT goal is to run any CDNA1/2/3/4 or RDNA1/2/3/4 binary on any CDNA or RDNA hardware without source recompilation. The primary use cases are:

- **Compatibility**: run a CDNA3 (GFX942) kernel on a CDNA4 (GFX950) machine and vice versa.
- **Emulation**: run CDNA (Wave64) kernels on RDNA (Wave32) hardware, or develop/test RDNA kernels on CDNA machines.
- **Feature emulation**: run MFMA-using kernels on non-MFMA hardware via software fallback.

The approach taken here is **static pre-translation** (translation happens at code-object-load time, before any kernel execution), stored in a disk cache, with transparent JIT translation as a fallback. This is analogous to Apple Rosetta 2's AOT model.

### 3.2 ISA Capability Matrix

| Feature | CDNA1 | CDNA2 | CDNA3 | CDNA4 | RDNA1 | RDNA2 | RDNA3 | RDNA3.5 | RDNA4 |
|---|---|---|---|---|---|---|---|---|---|
| Wave size | 64 | 64 | 64 | 64 | 32 or 64¹ | 32 or 64¹ | 32 or 64¹ | 32 or 64¹ | 32 or 64¹ |
| EXEC width | 64b | 64b | 64b | 64b | 32b or 64b¹ | 32b or 64b¹ | 32b or 64b¹ | 32b or 64b¹ | 32b or 64b¹ |
| AccVGPR | No | Yes | Yes | Yes | No | No | No | No | No |
| MFMA | Yes | Yes | Yes | Yes | No | No | No | No | No |
| WMMA | No | No | No | No | No | No | Yes | Yes | Yes |
| Flat scratch | SGPRs | SGPRs | Dedicated | Dedicated | SGPRs | SGPRs | SGPRs | SGPRs | SGPRs |
| s_waitcnt | GFX9 enc | GFX9 enc | GFX9 enc | GFX9 enc² | GFX10 enc (+vscnt) | GFX10 enc (+vscnt) | GFX11 enc (new layout + named variants) | GFX11 enc (new layout + named variants) | GFX12 enc (split S_WAIT_*, no monolithic) |
| GFX | 908 | 90a | 940/942 | 950 | 1010-1012 | 1030-1036 | 1100-1103 | 1150-1151 | 1200-1201 |

¹ All RDNA generations (RDNA1–4) support both Wave32 (default) and Wave64. Wave size is selected per-kernel via bit 10 (`ENABLE_WAVEFRONT_SIZE32`) of the `kernel_code_properties` field in the kernel descriptor: 1 = Wave32 (default), 0 = Wave64. EXEC width, VCC width, and branch-condition semantics all scale with the selected wave size. CDNA ISAs are Wave64-only and do not expose this bit.

² CDNA4 (GFX950), while GFX11-generation hardware, retains the single monolithic S_WAITCNT instruction from the GFX9 encoding family — it does **not** use the GFX11 split wait instructions. The SIMM16 layout is identical to CDNA1-3: vmcnt[3:0] at bits [3:0], expcnt[2:0] at bits [6:4], lgkmcnt[3:0] at bits [11:8], vmcnt[5:4] at bits [15:14].

### 3.3 `BinaryTranslator` Class

```cpp
// lib/rocjitsu/src/rocjitsu/code/dbt/binary_translator.h

namespace rocjitsu {

/// @brief Identifies an (ISA → ISA) translation pair.
struct TranslationKey {
  rj_code_arch_t source_arch;
  rj_code_arch_t target_arch;
  uint64_t binary_hash;   ///< xxHash64 of the source .text section content.

  bool operator==(const TranslationKey &o) const {
    return source_arch == o.source_arch && target_arch == o.target_arch &&
           binary_hash == o.binary_hash;
  }
};

struct TranslationKeyHash {
  size_t operator()(const TranslationKey &k) const {
    size_t h = std::hash<int>{}((int)k.source_arch);
    h ^= std::hash<int>{}((int)k.target_arch) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<uint64_t>{}(k.binary_hash) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

/// @brief Result of a DBT translation operation.
struct TranslatedCodeObject {
  std::vector<uint8_t> elf_bytes;   ///< Translated ELF for the target ISA.
  rj_code_arch_t target_arch;
  std::vector<std::string> warnings;  ///< Non-fatal translation issues.
  std::vector<std::string> errors;    ///< Fatal issues (if non-empty, translation failed).
};

/// @brief Top-level dynamic binary translator.
///
/// Translates an AmdGpuCodeObject from source_arch to target_arch by:
///   1. Decoding all instructions via the EXISTING Decoder::decode() factory
///      (cdna3::Decoder or cdna4::Decoder per source_arch) — no new decoder code
///      is written for supported ISAs.
///   2. Running liveness analysis (Pillar 4) on the decoded blocks.
///   3. Legalizing each instruction for the target ISA by consulting InstructionMapper
///      (§3.3.1) for the legalization action, then applying it:
///      - IDENTITY:   instruction is already legal; copy word(s) verbatim.
///      - SUBSTITUTE: direct opcode substitution; identical semantics, different encoding.
///        Patched in-place using target ISA machine_insts.h structs; size unchanged.
///      - LOWER:      instruction requires lowering to a target-native sequence.
///        Same-size lowering (e.g., flat→global, same 8-byte FLAT encoding) is done in-place.
///        Size-expanding lowering (e.g., s_waitcnt→split, s_barrier→signal/wait) uses a
///        code cave: the original instruction is replaced in-place with a same-size branch
///        stub; the lowering sequence is emitted into .rj_translations (§3.3.2).
///      - EXPAND:     no target equivalent; always uses a code cave (§3.3.2).
///        Original instruction replaced in-place with a same-size branch stub; the
///        emulation sequence is emitted into .rj_translations.
///      - ILLEGAL:    no legalization rule exists for this pair; translation fails.
///   4. Re-emitting a valid ELF for target_arch via CodeObjectPatcher.
///
/// Because every in-place replacement preserves the original instruction's size
/// (same-size for IDENTITY/SUBSTITUTE/same-size LOWER; branch stub for caves),
/// no instruction in .text ever shifts. Branch offsets in the original .text are
/// therefore always valid — no global branch fixup pass is required.
class BinaryTranslator {
public:
  explicit BinaryTranslator(rj_code_arch_t source_arch, rj_code_arch_t target_arch);

  /// @brief Translate a decoded code object.
  [[nodiscard]] TranslatedCodeObject translate(const AmdGpuCodeObject &obj);

  /// @brief Whether Wave32↔Wave64 emulation is required for this pair.
  bool requires_wave_shim() const;

  /// @brief Whether waitcnt encoding differs between source and target.
  bool requires_waitcnt_translation() const;

private:
  /// @brief Legalize and emit target words for a single source instruction.
  ///
  /// Looks up the legalization action from mapper_ and dispatches to the
  /// appropriate lowering or expansion rule. Returns zero or more target
  /// instruction words (zero is valid — instruction is a legal no-op on target).
  std::vector<uint32_t> translate_instruction(const Instruction &inst,
                                               const InstDefUse &du);

  // --- Same-size lowering rules (LOWER action, no cave needed) ---
  std::vector<uint32_t> lower_flat_memory(const Instruction &inst);
  std::vector<uint32_t> lower_wave_exec(const Instruction &inst,
                                         const InstDefUse &du);

  // --- Cave-based lowering rules (LOWER action, size expands) ---
  // These return the body to emit in .rj_translations; emit_cave_stub() is
  // called separately to write the branch stub in-place.
  std::vector<uint32_t> lower_waitcnt_body(const Instruction &inst);
  std::vector<uint32_t> lower_barrier_body(const Instruction &inst);

  // --- Expansion rules (EXPAND action, always cave-based) ---
  std::vector<uint32_t> expand_mfma_body(const Instruction &inst);

  /// @brief Build a same-size branch stub for a code cave entry.
  ///
  /// Replaces @p inst in-place with an s_branch to the cave entry at
  /// @p cave_byte_offset, padded with s_nop to match inst.size().
  /// Called by translate_instruction() for LOWER (size-expanding) and EXPAND.
  /// See §3.3.2 for the full code cave strategy.
  std::vector<uint32_t> emit_cave_stub(const Instruction &inst,
                                        uint64_t cave_byte_offset) const;

  rj_code_arch_t src_arch_;
  rj_code_arch_t dst_arch_;
  std::unique_ptr<InstructionMapper> mapper_;  ///< Opcode mapping table for this pair.
  std::vector<uint8_t> cave_body_;             ///< Accumulates .rj_translations content.
  std::vector<std::string> errors_;            ///< Collected per-instruction legalization errors.
};

} // namespace rocjitsu
```

#### 3.3.1 Legalization Rule Table

> **Implementation note:** The plan below describes `InstructionMapper` as a class. In the actual implementation, this was simplified to a `LegalizationLookupFn` function pointer that queries the auto-generated legalization table directly. No `InstructionMapper` class exists. The concept and responsibility are the same — the references to `InstructionMapper` throughout this document should be read as `LegalizationLookupFn`.

The legalization rule table answers one question per instruction for a specific (src_arch, dst_arch) pair: *is this instruction legal on the target, and if not, how should it be legalized?* It does not perform any transformation itself — that is `BinaryTranslator`'s job.

**Where rules are defined:**

- **Classification rules** (which `Action` each instruction gets) — auto-generated by `amdisa/legalization.py` from the ISA XML specs, stored as `inline constexpr InstructionLegalization[]` arrays in per-pair headers under `code/dbt/generated/legalization_*.h`. Each entry maps a `(src_encoding_id, src_opcode)` pair to an `Action` value and a `target_opcode` (populated for Identity, Substitute, and Lower entries; used by `BinaryTranslator`). Adding a new rule means updating the mnemonic rename map or domain-specific rules in `legalization.py` and regenerating.
- **Lowering logic** (how to actually transform `Action::Lower` instructions) — implemented as private methods on `BinaryTranslator` (e.g., `lower_waitcnt_body()`, `lower_flat_memory()`, `lower_encoding()`). The classification tells the translator *what* to do; the lowering method knows *how*.
- **Expansion logic** (software emulation for `Action::Expand` instructions) — same split: `expand_mfma_body()`, `expand_wmma_body()` in `BinaryTranslator`.

**Where rules are applied:**

`BinaryTranslator::translate_instruction()` is the single rule application point. It queries the mapper for the legalization action and dispatches:

```cpp
// lib/rocjitsu/src/rocjitsu/code/dbt/instruction_mapper.h
//
// Thin wrapper around the auto-generated legalization tables in
// code/dbt/generated/legalization_tables.h.  The types Action, InstructionLegalization,
// and lookup() are defined ONCE in the generated header — InstructionMapper
// does NOT duplicate them.

#include "rocjitsu/code/dbt/generated/legalization_tables.h"

namespace rocjitsu {

/// @brief Legalization rule table for a (src_arch, dst_arch) pair.
///
/// Wraps a std::span<const InstructionLegalization> from the auto-generated tables.
/// The Action enum (Identity, Substitute, Lower, Expand, Illegal) and
/// InstructionLegalization struct are defined in code/dbt/generated/legalization_tables.h
/// and used directly — no parallel type definitions.
///
/// The generated tables cover all 27 supported translation pairs.  If no
/// table covers a (src, dst) pair, create() returns nullptr and
/// BinaryTranslator::translate() returns ROCJITSU_STATUS_UNSUPPORTED.
class InstructionMapper {
public:
  explicit InstructionMapper(std::span<const InstructionLegalization> table) : table_(table) {}

  /// @brief Factory.  Returns nullptr for unsupported (src, dst) pairs.
  static std::unique_ptr<InstructionMapper> create(rj_code_arch_t src_arch,
                                                    rj_code_arch_t dst_arch);

  /// @brief Query the legalization action for a decoded source instruction.
  /// Uses the encoding_id and opcode to look up the entry via binary search.
  const InstructionLegalization *legalize(uint16_t encoding_id, uint16_t opcode) const {
    return lookup(table_, encoding_id, opcode);
  }

private:
  std::span<const InstructionLegalization> table_;
};

} // namespace rocjitsu
```

`translate_instruction()` applies the legalization rule and returns the words to write in-place at the original instruction's location. For cave-based rules it returns a same-size branch stub and appends the expansion body to `cave_body_`; for in-place rules it returns the replacement words directly. Either way, the returned word count always equals `inst.size() / 4`, so the surrounding layout is never disturbed.

```cpp
std::vector<uint32_t> BinaryTranslator::translate_instruction(
    const Instruction &inst, const InstDefUse &du) {
  const InstructionLegalization *entry = mapper_->legalize(encoding_id_of(inst), opcode_of(inst));
  if (!entry) {
    // Not in the table — treat as generic LOWER (decode-and-re-encode).
    return lower_encoding(inst);
  }

  switch (entry->action) {

  case Action::Identity:
    // In-place: encoding-compatible — copy verbatim.
    return inst.raw_encoding();

  case Action::Substitute:
    // In-place: swap the opcode field; operand encoding and size unchanged.
    return inst.with_opcode(entry->target_opcode).raw_words();

  case Action::Lower: {
    // Same-size lowering: rewrite in-place; no cave needed.
    if (inst.is_flat_memory())  return lower_flat_memory(inst);
    if (inst.is_exec_write())   return lower_wave_exec(inst, du);

    // Size-expanding lowering: emit body into cave, return branch stub in-place.
    std::vector<uint32_t> body;
    if (inst.is_waitcnt())  body = lower_waitcnt_body(inst);
    if (inst.is_barrier())  body = lower_barrier_body(inst);
    // Additional LOWER categories added alongside their lower_*_body() method.
    if (!body.empty()) {
      uint64_t cave_off = cave_byte_offset();
      append_cave_entry(body, inst.byte_offset() + inst.size());
      return emit_cave_stub(inst, cave_off);
    }
    // Default: decode-and-re-encode for encoding-level differences.
    return lower_encoding(inst);
  }

  case Action::Expand: {
    // Expansion: always cave-based; no target equivalent exists.
    std::vector<uint32_t> body;
    if (inst.is_mfma())  body = expand_mfma_body(inst);
    if (inst.is_wmma())  body = expand_wmma_body(inst);
    if (!body.empty()) {
      uint64_t cave_off = cave_byte_offset();
      append_cave_entry(body, inst.byte_offset() + inst.size());
      return emit_cave_stub(inst, cave_off);
    }
    errors_.push_back("EXPAND: no expansion rule for " + inst.mnemonic() +
                      " at offset 0x" + hex_str(inst.byte_offset()));
    return inst.raw_encoding();
  }

  case Action::Illegal:
    // Should not occur — the generated tables have zero ILLEGAL entries.
    // Defensive: fall back to decode-and-re-encode.
    return lower_encoding(inst);
  }
  return inst.raw_encoding();
}
```

After all instructions are processed, `translate()` calls `CodeObjectPatcher::append_section(".rj_translations", ...)` to emit `cave_body_` as a new ELF section. The cave stubs written in-place already contain the correct `s_branch` offsets computed relative to the start of that section.

#### 3.3.2 Code Cave Strategy

> **Implementation note (current state):** The DBT implementation now places cave bodies in a separate executable `.rj_translations` section immediately after `.text`, so expansion capacity no longer depends on compiler-emitted NOP padding after `s_endpgm`. Direct branch stubs are implemented; trampoline islands for exceptionally large kernels remain future work.

Size-expanding legalization (LOWER with size growth, all EXPAND) uses **code caves** to preserve the original `.text` layout. The invariant is:

> Every word written in-place at the original instruction's location is exactly `inst.size() / 4` words — the returned replacement always has the same word count as the source instruction.

**Branch stub format** (`emit_cave_stub`):

For a 4-byte source instruction (e.g., `s_waitcnt`, `s_barrier`):
```
[s_branch <cave_entry_offset>]   // 4 bytes — exact fit
```

For an 8-byte source instruction (e.g., MFMA, FLAT load/store):
```
[s_branch <cave_entry_offset>]   // 4 bytes
[s_nop]                          // 4 bytes — dead, never reached
```

**Cave entry format** (emitted into `.rj_translations`):
```
<lowering or expansion sequence>     // N bytes, ISA-specific
[s_branch <return_offset>]           // 4 bytes — jumps to inst_byte_offset + inst.size()
```

The return address (`inst_byte_offset + inst.size()`) is the instruction immediately following the original in `.text`. Because `.text` is never shifted, this address is stable and can be computed at the time the cave entry is assembled.

**Branch range.** `s_branch simm16` encodes a signed offset in dwords with a ±128 KB range. The range check is per-stub, computed as the distance from the stub's location in `.text` to the cave entry in `.rj_translations`. The `.rj_translations` section is appended immediately after `.text` in the output ELF, so for kernels whose `.text` is ≤ 128 KB, all stubs reach their cave entries with a direct `s_branch`.

**ELF load alignment.** The section header for `.rj_translations` covers only the cave words. When later `PT_LOAD` segments would be shifted by the insertion, `CodeObjectPatcher` pads the inserted file range after `.rj_translations` so every shifted load segment still satisfies `p_offset % p_align == p_vaddr % p_align`. The padding may be part of the executable load segment, but it is outside the `.rj_translations` section.

**Far stub constraint.** For larger kernels where the per-stub range exceeds ±128 KB, a far stub requires 16 bytes (`s_getpc_b64` [4B] + `s_add_u32` [4B] + `s_addc_u32` [4B] + `s_setpc_b64` [4B]). This is incompatible with 4-byte source instructions (e.g., `s_barrier`, `s_waitcnt`) that only provide 4 bytes in-place. The solution is a **trampoline island**: a short landing pad allocated within near-branch reach of the 4-byte stub containing the full 16-byte far stub sequence. The 4-byte in-place stub branches to the island (near `s_branch`); the island jumps to the cave entry (far indirect jump).

Island allocation rules:
- One `.rj_islands` section is inserted for each contiguous 128 KB span of `.text` that contains at least one far stub.
- Section size = `num_far_stubs_in_span × 16 bytes`. Maximum 4096 entries per island (= 64 KB island, still reachable by the ±128 KB near branch from the span's far stub).
- If `num_far_stubs_in_span > 4096`, `BinaryTranslator::translate()` returns an error (non-empty `errors` field): the kernel is too large/dense for DBT and the caller falls back to PASSTHROUGH.
- **DBI does NOT use far stubs or islands.** DBI trampolines are inserted directly into `.text` via `CodeObjectPatcher::insert_at()`, growing the section in-place; there is no separate `.rj_translations` section and no branch-range constraint (the inserted call stub is adjacent to the patched instruction). Island logic is a DBT-only concern.
- For 8-byte source instructions the 16-byte far stub does not fit in-place either; these also use an island entry.

In practice this path is only needed for exceptionally large shaders (>128 KB `.text`).

**Effect on branch fixup.** Since no instruction in `.text` changes size, all branch offsets in the original `.text` remain valid after translation. The global `fixup_branch_targets()` pass that would otherwise scan the entire translated text is not needed and is not part of `BinaryTranslator`.

#### 3.3.3 Semantic Translation (Tier 2)

`BinaryTranslator` uses a three-tier translation architecture:

| Tier | Scope | Coverage | Mechanism |
|---|---|---|---|
| **Tier 1** | Instruction-at-a-time | ~80% of instructions | Legalization table lookup + encoding translate (§3.3.1, current system) |
| **Tier 2** | Semantic translation | ~15% of matrix-heavy kernels | Data-driven rules for instructions whose semantics change across ISAs (this section) |
| **Tier 3** | Fallback expansion | ~5% remaining | Software emulation stubs via trampoline (§3.6) |

Tier 1 handles Identity, Substitute, and simple Lower/Expand actions via per-instruction dispatch. However, certain instruction families — particularly MFMA data preparation, matrix execution, and AccVGPR readback — form multi-instruction idioms where the correct translation requires recognizing and replacing the complete sequence as a unit. Translating these instruction-at-a-time produces incorrect results because:

1. **Data preparation is ISA-specific.** CDNA4's `ds_read_b64_tr_b16` (transpose load from LDS) has no single-instruction RDNA4 equivalent. The entire data staging pipeline must be replaced.
2. **Matrix dimensions change.** A 32×32 MFMA decomposes into 4× 16×16 WMMAs, requiring register redistribution across the tiled operations.
3. **Register topology changes.** AccVGPR (CDNA) → unified VGPR (RDNA) requires remapping the entire accumulator data flow, not just individual move instructions.
4. **Wave width changes.** Wave64 MFMA lane-to-element mapping differs fundamentally from Wave32 WMMA mapping.

**Semantic translator design.** The recognizer runs per-basic-block *before* the per-instruction translation loop. For each block, it scans for anchor instructions (MFMA, transpose loads, split barriers) and attempts to match known patterns in a window around each anchor. Matched sequences are emitted as a unit (into a code cave if size-expanding); the source instructions are marked as consumed and skipped by the per-instruction loop. Unmatched instructions fall through to Tier 1.

```cpp
// lib/rocjitsu/src/rocjitsu/code/dbt/semantic_translator.h

namespace rocjitsu {

struct SemanticAnchor {
  uint16_t encoding_id;
  uint16_t opcode;
  uint8_t window_before;
  uint8_t window_after;
};

struct SemanticMatch {
  uint64_t start_offset;
  uint64_t end_offset;
  uint16_t reg_bindings[16];
  uint32_t imm_bindings[4];
  uint8_t num_source_insts;
};

struct SemanticRule {
  const char *name;
  rj_code_arch_t guest_arch;
  rj_code_arch_t host_arch;
  SemanticAnchor anchor;

  using MatchFn = bool (*)(const BasicBlock &block,
                           const Instruction &anchor_inst,
                           SemanticMatch &match);
  MatchFn match;

  using EmitFn = std::vector<uint32_t> (*)(const SemanticMatch &match,
                                           rj_code_arch_t host_arch);
  EmitFn emit;
};

class SemanticTranslator {
public:
  SemanticTranslator(rj_code_arch_t guest, rj_code_arch_t host);

  // Scan a basic block for known idioms. Returns a list of matched regions
  // (offset ranges) and their replacement byte sequences. The caller writes
  // replacements into the translated text and marks source instructions
  // as consumed.
  struct Replacement {
    uint64_t start_offset;
    uint64_t end_offset;
    std::vector<uint32_t> target_words;
  };

  std::vector<Replacement> recognize(const BasicBlock &block);

private:
  std::span<const SemanticRule> rules_;
};

} // namespace rocjitsu
```

**Rule selection.** The semantic rule table is selected by `(guest_arch, host_arch)` pair at `SemanticTranslator` construction, mirroring the function-pointer selection pattern used by `EncodingTranslateFn` and `LegalizationLookupFn`. For the MVP, only CDNA4→RDNA4 semantic rules are implemented. Adding rules for new pairs requires only defining new match/emit function pairs and registering them in the pair's rule table.

**Why data-driven rules, not hardcoded if/else.** Research on binary translation systems (Bansal & Aiken, OSDI 2008; XED's datafile-per-extension pattern) consistently shows that declarative rule tables scale better than procedural translation logic. Each semantic rule is self-contained and independently testable. Adding a new idiom means adding one `SemanticRule` entry — no modifications to the recognizer framework or the translation loop.

**Concrete semantic rules for CDNA4→RDNA4:**

| Rule name | Source pattern (CDNA4) | Target pattern (RDNA4) | Notes |
|---|---|---|---|
| `mfma_f16_16x16x16_with_tr_load` | `ds_read_b64_tr_b16` + `s_waitcnt` + `v_mfma_f32_16x16x16_f16` + `v_accvgpr_read` × N | `ds_load_b128` + `s_wait_loadcnt` + `v_wmma_f32_16x16x16_f16` | 1:1 shape; AccVGPR eliminated |
| `mfma_f16_32x32x16_with_tr_load` | `ds_read_b64_tr_b16` × 2 + `s_waitcnt` + `v_mfma_f32_32x32x16_f16` + `v_accvgpr_read` × N | `ds_load_b128` × 4 + `s_wait_loadcnt` + `v_wmma_f32_16x16x16_f16` × 4 | 2×2 tiling; 32×32 → 4× 16×16 |
| `accvgpr_standalone` | `v_accvgpr_write acc[N], v[M]` | `v_mov_b32 v[N + base], v[M]` | Remap to unified VGPR |
| `accvgpr_standalone` | `v_accvgpr_read v[M], acc[N]` | `v_mov_b32 v[M], v[N + base]` | Remap to unified VGPR |
| `ds_read_tr_standalone` | `ds_read_b64_tr_b16 v[dst], v[addr]` | `ds_load_b64 v[dst], v[addr]` + permute | No adjacent MFMA; standalone transpose |

**Interaction with Tier 1.** When the semantic translator matches a sequence, it produces a `Replacement` covering the byte range `[start_offset, end_offset)`. The Tier 1 per-instruction loop skips any instruction whose offset falls within a replacement range. If the replacement is larger than the source range, the excess is emitted into a code cave with the same branch stub mechanism as §3.3.2.

**Performance impact.** Idiom recognition adds a linear scan per basic block (one pass over instructions to check anchor opcodes, then a small window check per anchor). For non-matrix kernels this adds negligible overhead. For matrix-heavy kernels, the Tier 2 path avoids the much more expensive Tier 3 software fallback, providing 10-50× better performance for MFMA→WMMA translation compared to the trampoline stubs.

#### 3.3.4 Architectural Justification

The three-tier direct translation architecture is informed by analysis of production binary translation systems:

**Why not raise to LLVM IR and lower?** Raising binary code to LLVM IR is a decompilation problem — the binary has already been through register allocation, instruction scheduling, and loop optimization. Raising discards those decisions; LLVM's backend redoes them differently, often worse, because source-level information (types, aliasing, loop bounds) is lost. ZLUDA's raise-to-LLVM-IR approach (PTX → LLVM IR → AMDGPU) achieves correct low-level translation for only ~2.5% of assembly patterns. MFMA/WMMA are LLVM intrinsics, not generic IR ops — raising to IR and lowering back requires the same semantic translator we build directly, making the IR a round-trip with no value added.

**Why not a generic intermediate representation?** QEMU's TCG IR achieves any-to-any translation with 2N effort but incurs 5-20× performance overhead. Valgrind's VEX IR requires 30K+ lines of handwritten C per guest architecture and still can't keep up with AVX/AVX-512 after 10+ years. Rosetta 2 proves that direct translation with minimal inter-instruction optimization (only dead flags + push/pop combining) achieves 1.2× overhead when the ISAs are structurally similar. CDNA and RDNA share the same vendor, mnemonic conventions, encoding format families, and ~85% instruction overlap — the semantic gap is far smaller than x86→ARM64.

**Why semantic-level pattern matching?** FEX-EMU's multiblock analysis for x87 stack operations reduced translated instruction count by 93% — proving that recognizing and replacing complete multi-instruction patterns is the highest-leverage optimization for complex instruction families. The MFMA→WMMA problem is analogous: `ds_read_b64_tr_b16` + `v_mfma_*` + `v_accvgpr_read` is a stereotyped idiom produced by GPU compilers, not an arbitrary code sequence. A small number of pattern rules (~10-20 for the CDNA4→RDNA4 pair) covers the vast majority of matrix computation in real HPC/AI kernels.

**Why data-driven rule tables?** Intel XED's datafile-per-ISA-extension pattern demonstrates that declarative instruction specifications scale far better than procedural decoders (VEX's 30K-line `guest_amd64_toIR.c` vs. XED's ~200-line `.xed.txt` files per extension). Our XML-driven legalization tables, auto-generated encoding translators, and semantic rule structs follow the same principle: the rule *data* is separated from the rule *application engine*, so adding a new ISA pair or a new idiom requires only new data entries, not new framework code.

### 3.4 `s_waitcnt` Encoding Translation

`s_waitcnt` is a SOPP instruction where the `simm16` field packs counter thresholds differently across generations. This is one of the most common sources of incompatibility between GFX9-class and GFX10/11-class hardware.

#### GFX9 (CDNA1/2/3) `s_waitcnt` encoding:

```
simm16[15:14]  = vmcnt_hi[1:0]    (bits 15-14 are high 2 bits of vmcnt)
simm16[13:12]  = reserved (write 1 in GFX9 — assembler convention)
simm16[11:8]   = lgkmcnt[3:0]    (4 bits; max 15)
simm16[7]      = reserved (write 1 in GFX9 — assembler convention)
simm16[6:4]    = expcnt[2:0]
simm16[3:0]    = vmcnt_lo[3:0]   (low 4 bits of vmcnt)
```

Full vmcnt = `{vmcnt_hi, vmcnt_lo}` = 6-bit value (max 63).

#### GFX10 (RDNA1/2) `s_waitcnt` encoding:

```
simm16[15:14]  = vmcnt_hi[1:0]
simm16[13:12]  = reserved (write 0 on GFX10 — distinct from GFX9 convention)
simm16[11:8]   = lgkmcnt[3:0]    (4 bits; max 15; same width as GFX9/CDNA)
simm16[7]      = reserved (write 0 on GFX10)
simm16[6:4]    = expcnt[2:0]
simm16[3:0]    = vmcnt_lo[3:0]
```

#### GFX11+ (RDNA3/3.5) and GFX12 (RDNA4): Split wait instructions

GFX11 replaces `s_waitcnt` with individual per-counter instructions (there is no
monolithic `s_waitcnt` on RDNA3/3.5/4 — do not pass these ISAs as `src_arch` to
`decode_waitcnt()`). **Note:** CDNA4 is GFX11-generation hardware but retains
the monolithic `s_waitcnt` with GFX9-style encoding (see footnote 2 in §3.2).

```
s_wait_loadcnt   imm16   — wait for vmcnt (load) counter  (4-bit, max 15)
s_wait_storecnt  imm16   — wait for store counter         (4-bit, max 15)
s_wait_expcnt    imm16   — wait for export counter        (3-bit, max 7)
s_wait_kmcnt     imm16   — wait for scalar memory counter (4-bit, max 15)
s_wait_dscnt     imm16   — wait for DS/LDS counter        (4-bit, max 15)
s_wait_samplecnt imm16   — wait for image sample counter
s_wait_bvhcnt    imm16   — wait for BVH traversal counter
```

GFX12 (RDNA4) additionally combines `s_wait_storecnt` + `s_wait_dscnt` into a
single `s_wait_storecnt_dscnt imm16` instruction (the two 4-bit fields are packed
into one 8-bit immediate). All other split-wait instructions remain the same.

**Note: CDNA4 (GFX950) retains the monolithic `s_waitcnt` instruction with GFX9-style encoding** — it does **not** use the GFX11-style split `s_wait_*` instructions, despite being GFX11-generation hardware. The `simm16` layout is identical to CDNA1-3: vmcnt[3:0] at bits [3:0], expcnt[2:0] at bits [6:4], lgkmcnt[3:0] at bits [11:8], vmcnt[5:4] at bits [15:14]. CDNA3→CDNA4 `s_waitcnt` translation is therefore IDENTITY (same encoding). RDNA4 (GFX12-style) uses `s_wait_storecnt_dscnt` (combined) in place of the separate `s_wait_storecnt` + `s_wait_dscnt` pair. A GFX9-source `s_waitcnt` translating to RDNA3/4 targets must be expanded to a sequence of `s_wait_*` instructions via a code cave (LOWER action, size-expanding). The `WaitcntTranslator` handles this via `encode_waitcnt()` returning a multi-word vector; RDNA4 is dispatched separately to emit the combined `s_wait_storecnt_dscnt`.

Translation function:

```cpp
// lib/rocjitsu/src/rocjitsu/code/dbt/semantic_translator.h (waitcnt decode/encode merged here)

namespace rocjitsu {

/// @brief Decoded counter thresholds from s_waitcnt (and s_waitcnt_vscnt for RDNA1/2).
///
/// Default values represent "don't wait" (all counters relaxed).
///
/// Sentinel values use the maximums (vmcnt=0x3F, lgkmcnt=0x0F, expcnt=0x07).
/// All CDNA and RDNA ISAs use 4-bit lgkmcnt (max 0x0F); no normalization needed.
/// so encode_waitcnt() can use uniform != sentinel checks regardless of source ISA.
///
/// is_noop() compares all counters to their sentinel values.
/// Sentinels are: vmcnt=0x3F, lgkmcnt=0x0F, expcnt=0x07, vscnt=0x3F.
struct WaitcntValues {
  uint8_t vmcnt   = 0x3F;  ///< 6-bit normalized sentinel (GFX9/CDNA/RDNA1/2 native split field);
                           ///< encode_waitcnt() clamps to 4-bit for GFX11-style targets (max 15);
                           ///< 0x3F = "don't wait"
  uint8_t lgkmcnt = 0x0F;  ///< max 4-bit (GFX9/CDNA and GFX10/RDNA1/2); 0x0F = "don't wait"
  uint8_t expcnt  = 0x07;  ///< max 3-bit; 0x07 = "don't wait"
  /// GFX10/RDNA1/2 only: store counter from the separate `s_waitcnt_vscnt` instruction.
  /// Not part of the s_waitcnt simm16; encode_waitcnt() emits a second `s_waitcnt_vscnt`
  /// word when dst is RDNA1/2 and vscnt != 0x3F.  Always 0x3F (ignored) on other ISAs.
  uint8_t vscnt   = 0x3F;  ///< max 6-bit; 0x3F = "don't wait"

  /// True if all counters are at their "don't wait" values.
  bool is_noop() const {
    return vmcnt == 0x3F && lgkmcnt == 0x0F && expcnt == 0x07 && vscnt == 0x3F;
  }
};

/// @brief Decode s_waitcnt simm16 for the given source architecture.
///
/// @param src_arch Must be CDNA1/2/3 or RDNA1/2.  CDNA4 and RDNA3/3.5/4 use
///                 split s_wait_* instructions and never produce a single
///                 s_waitcnt simm16 — passing them here is a precondition
///                 violation.  GFX10 (RDNA1/2) "don't wait" values (0xF
///                 for lgkmcnt) use the sentinel values (0x0F)
///                 so that callers may use uniform comparisons at encode time.
///
/// @note  vscnt (GFX10/RDNA1/2 store counter) lives in a separate
///        `s_waitcnt_vscnt` instruction, not in the simm16 field.
///        Callers that process GFX10 code must decode any adjacent
///        `s_waitcnt_vscnt` word independently and merge it into the
///        returned WaitcntValues::vscnt field before calling encode_waitcnt().
WaitcntValues decode_waitcnt(uint16_t simm16, rj_code_arch_t src_arch);

/// @brief Encode WaitcntValues for the given target architecture.
/// Returns a vector of one or more uint32_t words:
///   - GFX9/CDNA1/2/3: exactly one s_waitcnt word.
///   - GFX10/RDNA1/2: one s_waitcnt word; plus one s_waitcnt_vscnt word when vscnt != 0x3F.
///   - GFX11-style (RDNA3, RDNA3.5): up to four words (one per active counter);
///     fully relaxed emits a single s_nop.
///   - GFX12-style (RDNA4): same as GFX11 but s_wait_storecnt_dscnt replaces the
///     separate storecnt + dscnt words.
/// @returns A non-empty vector of uint32_t words. Always contains at least one word:
///          if all counters are fully relaxed (no wait needed), emits a single s_nop
///          so call-sites can unconditionally overwrite a fixed-size hole.
std::vector<uint32_t> encode_waitcnt(WaitcntValues vals, rj_code_arch_t dst_arch);

} // namespace rocjitsu
```

The implementation dispatches on ISA capability booleans (not ISA generation names). Instruction encoding constants (opcode values, bit field positions) are defined in ISA-specific constant headers (e.g. `{isa}/machine_insts.h`) and referenced symbolically — no magic numbers inline:

```cpp
WaitcntValues decode_waitcnt(uint16_t s, rj_code_arch_t src) {
  WaitcntValues v;
  // CDNA1/2/3/4 and RDNA1/2 all encode vmcnt as a split 6-bit field (lo in [3:0], hi in [15:14]).
  // CDNA4 retains GFX9-style monolithic s_waitcnt despite being GFX11-generation hardware.
  // GFX11+ (RDNA3/3.5/4) are a precondition violation — they have no monolithic s_waitcnt.
  const bool has_split_vmcnt = (src == ROCJITSU_CODE_ARCH_CDNA1 ||
                                 src == ROCJITSU_CODE_ARCH_CDNA2 ||
                                 src == ROCJITSU_CODE_ARCH_CDNA3 ||
                                 src == ROCJITSU_CODE_ARCH_CDNA4 ||
                                 src == ROCJITSU_CODE_ARCH_RDNA1 ||
                                 src == ROCJITSU_CODE_ARCH_RDNA2);
  // GFX9/CDNA1/2/3/4: lgkmcnt is 4-bit [11:8].
  // GFX10/RDNA1/2:  lgkmcnt is 4-bit [11:8]; same as GFX9/CDNA.
  const bool rdna1_or_2 = (src == ROCJITSU_CODE_ARCH_RDNA1 ||
                            src == ROCJITSU_CODE_ARCH_RDNA2);
  if (has_split_vmcnt) {
    uint8_t vmcnt_lo    = (s >> 0)  & 0xF;
    uint8_t expcnt      = (s >> 4)  & 0x7;
    uint8_t raw_lgkmcnt = (s >> 8) & 0xF;  // 4-bit on all ISAs
    uint8_t vmcnt_hi    = (s >> 14) & 0x3;
    v.vmcnt   = (vmcnt_hi << 4) | vmcnt_lo;
    v.expcnt  = expcnt;
    v.lgkmcnt = raw_lgkmcnt;
  } else {
    // Only reached if src_arch is invalid (GFX11+ or unknown) — precondition violation.
    assert(false && "decode_waitcnt: src_arch has no monolithic s_waitcnt encoding");
    (void)s;
  }
  return v;
}

std::vector<uint32_t> encode_waitcnt(WaitcntValues v, rj_code_arch_t dst) {
  // CDNA1/2/3/4: 6-bit split vmcnt field; 4-bit lgkmcnt.
  // CDNA1/2/3/4 all use monolithic s_waitcnt with GFX9-style split vmcnt field.
  // CDNA4 retains GFX9 encoding despite being GFX11-generation hardware.
  const bool has_split_vmcnt = (dst == ROCJITSU_CODE_ARCH_CDNA1 ||
                                 dst == ROCJITSU_CODE_ARCH_CDNA2 ||
                                 dst == ROCJITSU_CODE_ARCH_CDNA3 ||
                                 dst == ROCJITSU_CODE_ARCH_CDNA4);
  // GFX11+ (RDNA3, RDNA3.5, RDNA4): s_waitcnt expands to separate s_wait_* instructions.
  // encode_waitcnt() returns a multi-word sequence for these targets.
  // RDNA3.5 (GFX115x) uses the same GFX11-style split encoding as RDNA3.
  // Note: CDNA4 is NOT in this set — it retains monolithic s_waitcnt.
  const bool use_split_wait = (dst == ROCJITSU_CODE_ARCH_RDNA3 ||
                                dst == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                                dst == ROCJITSU_CODE_ARCH_RDNA4);

  // Opcode and encoding prefix come from the ISA's machine_insts.h constants.
  const auto enc = waitcnt_encoding_for(dst);  // ISA-specific lookup; returns {sopp_prefix, waitcnt_op}

  if (has_split_vmcnt) {
    uint8_t vmcnt_lo = v.vmcnt & 0xF;
    uint8_t vmcnt_hi = (v.vmcnt >> 4) & 0x3;
    uint8_t lgk = std::min<uint8_t>(v.lgkmcnt, enc.lgkmcnt_max);
    uint16_t simm16 = enc.pack_waitcnt(vmcnt_lo, vmcnt_hi, v.expcnt, lgk);
    return {enc.make_word(simm16)};
  } else if (use_split_wait) {
    // GFX11-style (RDNA3, RDNA3.5) or GFX12-style (RDNA4): emit split s_wait_* instructions.
    // Note: CDNA4 is NOT in this set — it retains monolithic s_waitcnt (GFX9 encoding).
    // Each counter gets its own instruction; "don't wait" counters (sentinels 0x3F/0x0F/0x07)
    // are omitted. The caller (translate_instruction) handles size expansion via a code cave
    // when the source was a single 4-byte s_waitcnt.
    //
    // NOTE: RDNA4 (GFX12-style) uses a combined s_wait_storecnt_dscnt that replaces the
    // separate s_wait_storecnt + s_wait_dscnt of the GFX11-style model.
    // s_wait_kmcnt remains a distinct instruction on GFX12 and is NOT merged.
    std::vector<uint32_t> words;
    const auto split = split_wait_encoding_for(dst);  // ISA-specific split wait opcodes
    if (v.vmcnt   != 0x3F) words.push_back(split.make_loadcnt(std::min<uint8_t>(v.vmcnt, split.loadcnt_max)));
    // GFX9 lgkmcnt covers BOTH scalar-memory (KMEM) and LDS (DS) operations.
    // Translate conservatively: emit the KMEM counter separately (s_wait_kmcnt), then
    // the DS counter (s_wait_dscnt), potentially combined with the store counter.
    if (v.lgkmcnt != 0x0F) {
      // kmcnt (scalar memory) is a separate instruction on both GFX11 and GFX12.
      words.push_back(split.make_kmcnt(std::min<uint8_t>(v.lgkmcnt, split.kmcnt_max)));
    }
    if (dst == ROCJITSU_CODE_ARCH_RDNA4) {
      // GFX12-style: s_wait_storecnt_dscnt packs storecnt[7:4]+dscnt[3:0].
      // storecnt comes from vscnt (the GFX10 store counter, separate from lgkmcnt).
      // dscnt comes from lgkmcnt — because GFX9 lgkmcnt covered BOTH scalar-memory
      // (KMEM) and LDS (DS) with a single counter, we conservatively emit lgkmcnt's
      // value for both s_wait_kmcnt (above) and the dscnt field here.
      const bool need_sc = (v.vscnt   != 0x3F);
      const bool need_dc = (v.lgkmcnt != 0x0F);
      if (need_sc || need_dc) {
        // split.storecnt_max / split.dscnt_max are the "don't wait" sentinels for
        // each 4-bit field (value = 15, meaning "no pending ops to wait for").
        // When a counter is relaxed, pass its sentinel so make_storecnt_dscnt
        // encodes it as "don't wait" in the packed imm16 (storecnt in [7:4], dscnt in [3:0]).
        uint8_t sc = need_sc ? std::min<uint8_t>(v.vscnt,   split.storecnt_max) : split.storecnt_max;
        uint8_t dc = need_dc ? std::min<uint8_t>(v.lgkmcnt, split.dscnt_max)    : split.dscnt_max;
        words.push_back(split.make_storecnt_dscnt(sc, dc));  // packs sc<<4|dc internally
      }
    } else {
      // GFX11-style (RDNA3, RDNA3.5): separate dscnt instruction.
      if (v.lgkmcnt != 0x0F)
        words.push_back(split.make_dscnt(std::min<uint8_t>(v.lgkmcnt, split.dscnt_max)));
    }
    // CDNA4 has no export pipeline; skip expcnt entirely (bits are reserved on CDNA4).
    // RDNA3/4 have export queues and use expcnt normally.
    if (dst != ROCJITSU_CODE_ARCH_CDNA4 && v.expcnt != 0x07)
      words.push_back(split.make_expcnt(v.expcnt));
    if (words.empty()) words.push_back(split.make_nop());  // fully relaxed → s_nop
    return words;
  } else {
    // GFX10 (RDNA1/2): same split vmcnt field as GFX9 (lo in [3:0], hi in [15:14]).
    // lgkmcnt is 4-bit (bits [11:8], max 15); same width as GFX9/CDNA.
    // vscnt is in a separate `s_waitcnt_vscnt` instruction; emit it when vscnt != "don't wait".
    std::vector<uint32_t> words;
    uint8_t vmcnt_lo = v.vmcnt & 0xF;
    uint8_t vmcnt_hi = (v.vmcnt >> 4) & 0x3;
    uint8_t lgk = std::min<uint8_t>(v.lgkmcnt, enc.lgkmcnt_max);  // clamp to target width
    uint16_t simm16 = enc.pack_waitcnt(vmcnt_lo, vmcnt_hi, v.expcnt, lgk);
    words.push_back(enc.make_word(simm16));
    if (v.vscnt != 0x3F)
      words.push_back(enc.make_vscnt_word(std::min<uint8_t>(v.vscnt, enc.vscnt_max)));
    return words;
  }
}
```

`waitcnt_encoding_for(dst)` is an ISA-specific dispatch function (implemented alongside the ISA's machine instruction definitions) that returns a `WaitcntEncoding` struct carrying the opcode constant, SOPP prefix, field widths, and `pack_waitcnt`/`make_word` helpers for that ISA. No encoding values appear in the generic translation logic.

### 3.5 Wave32 ↔ Wave64 Emulation

#### Correction: All RDNA Generations Support Wave64

The AMD RDNA ISA specifications confirm that **all four RDNA generations natively support Wave64 mode**. Wave64 is selectable per kernel dispatch via a kernel descriptor bit (not an ISA extension or emulation layer):

> *"Wave64 uses all 64 bits of the exec mask. Wave32 waves use only bits 31:0 and hardware does not act upon the upper bits."*
> — RDNA3 ISA §3.2.2, RDNA4 ISA §3.2.2

All generations document Wave64-specific VGPR allocation blocks, Wave64-specific instruction skipping behavior, and treat Wave64 as a first-class execution mode. RDNA1 VGPR allocation is in groups of 4 DWords for Wave64 and 8 for Wave32. RDNA2, RDNA3, and RDNA4 use groups of 8 DWords for Wave64 and 16 for Wave32; some 1536-VGPR/SIMD RDNA3/RDNA4 devices use groups of 12 for Wave64 and 24 for Wave32. These are physical allocation blocks from the ISA manuals. The AMDHSA `GRANULATED_WORKITEM_VGPR_COUNT` descriptor field has its own encoding described later. The underlying hardware issues Wave64 instructions in two 32-lane passes internally, but this is fully transparent to ISA-level code.

**Implication for DBT:** A CDNA Wave64 kernel can run on RDNA hardware in Wave64 mode with no wavefront-count changes. The translator simply sets the Wave64 mode bit in the target kernel descriptor and handles the `s_waitcnt` encoding differences and any missing/changed opcodes. The complex wave-splitting strategies described below are therefore **not needed for the CDNA→RDNA direction** — they are retained only for hypothetical future scenarios where a Wave32-only host is targeted.

---

#### 3.5.1 Wave64 Guest on Wave64-Capable Host (CDNA → RDNA): Primary Strategy

Since all RDNA ISAs support Wave64, the translator sets the Wave64 mode bit in the target kernel descriptor and proceeds with standard instruction translation. The primary changes required are:

**Kernel descriptor update:**

The AMDGPU kernel descriptor's `COMPUTE_PGM_RSRC1` register contains a wave-size control field. The translator sets this field to request Wave64 in the target descriptor:

```cpp
// In BinaryTranslator::translate(), after copying the kernel descriptor:
// ENABLE_WAVEFRONT_SIZE32 is bit 10 of kernel_code_properties (the AMDHSA kernel
// descriptor field), not a bit of COMPUTE_PGM_RSRC1. Clearing it requests Wave64.
// Note: COMPUTE_PGM_RSRC1 bit 14 is WGP_MODE (CU vs WGP scheduling), unrelated
// to wave size. Do not modify COMPUTE_PGM_RSRC1 for wave-size control.
kd.kernel_code_properties &= ~(1u << 10);  // clear ENABLE_WAVEFRONT_SIZE32 → Wave64 mode
```

**VGPR descriptor adjustment:**

Physical VGPR allocation blocks differ by ISA and by device, but the translated kernel descriptor's `GRANULATED_WORKITEM_VGPR_COUNT` field must be recomputed using the AMDHSA descriptor formula:

```cpp
// CDNA3/CDNA4 (Wave64): descriptor granularity = 8 DWords.
// RDNA GFX10-GFX12 (Wave64): descriptor granularity = 4 DWords.
uint32_t actual_vgprs = (cdna_gran + 1) * 8;  // recover actual count from CDNA encoding
uint32_t rdna_gran    = (actual_vgprs + 3) / 4 - 1;  // re-encode for RDNA Wave64
kd.compute_pgm_rsrc1 = (kd.compute_pgm_rsrc1 & ~vgpr_gran_mask) | (rdna_gran << vgpr_gran_shift);
```

**Instruction translation in Wave64 mode:**

Most instructions are identical between CDNA Wave64 and RDNA Wave64. The remaining translation work is:
- `s_waitcnt` encoding (§3.4) — the dominant source of incompatibility.
- Opcode remapping for instructions that changed encoding between GFX9 and GFX10/11 (a small set, mostly in the SMEM and FLAT encodings).
- MFMA/AccVGPR instructions on non-CDNA RDNA hosts (§3.6).

**EXEC-modifying instructions:** No special handling needed. All RDNA ISAs support 64-bit EXEC natively. `s_mov_b64 exec, ...`, `s_and_b64 exec, exec, ...`, and `v_cmp_* exec, ...` work identically on RDNA in Wave64 mode. The translator emits these verbatim (subject to opcode encoding differences only).

**Cross-lane operations:** `v_readlane_b32`, `v_writelane_b32`, `ds_permute_b32` all operate over all 64 lanes on both CDNA and RDNA in Wave64 mode. No lane remapping is required.

---

#### 3.5.2 Wave64 Guest on a Hypothetical Wave32-Only Host: Wave Splitting

This case does not arise for the current RDNA target set (all generations support Wave64). It is documented here for completeness and as a fallback if a future RDNA variant removes Wave64 support.

**Concept:** Replace every guest Wave64 wavefront with two Wave32 wavefronts within the same workgroup. Wave-half 0 executes guest lanes 0–31; wave-half 1 executes guest lanes 32–63. A workgroup of 64 threads on Wave64 produces 1 wavefront; on Wave32 it produces 2 wavefronts sharing one LDS allocation, so `ds_*` and `s_barrier` semantics are preserved without dispatch-level changes.

**EXEC translation:** Each half holds one 32-bit slice of the guest's 64-bit exec in its own 32-bit exec register. Instructions that write exec with a 64-bit SGPR pair (`s_mov_b64 exec, s[4:5]`) become `s_mov_b32 exec, s4` on half 0 and `s_mov_b32 exec, s5` on half 1. Scalar boolean ops on exec (`s_and_b64 exec, exec, s[x:x+1]`) become `s_and_b32` with the appropriate half's operand. `v_cmp_* exec, ...` is translated identically since it writes only the 32 bits covering that half's lanes.

**Cross-lane ops with dynamic indices spanning both halves** require an LDS exchange zone — the most expensive case. This is the primary reason wave splitting is undesirable and why leveraging native Wave64 mode on RDNA is strongly preferred.

**64-bit register packing (alternative, opt-in):** For element-wise kernels with no cross-lane ops, two guest lanes' 32-bit data can be packed into one host lane's 64-bit register using `v_pk_*` instructions (GFX11+). This avoids wave splitting but changes the lane-to-VGPR address mapping, breaking memory ops, LDS, `v_cmp`, and any instruction that uses lane ID as an address offset. Not suitable as a general strategy.

---

#### 3.5.3 Wave32 Guest on Wave64 Host (RDNA → CDNA)

A Wave32 binary compiled for RDNA uses only 32 lanes. On Wave64 CDNA hardware, activate only the low 32 lanes by constraining exec at kernel entry:

```asm
; Injected at translated kernel entry (64-bit literals cannot be encoded in a single
; s_mov_b64 instruction; use two 32-bit s_mov_b32 writes instead):
s_mov_b32 exec_lo, 0xFFFFFFFF   ; lanes 0..31 active
s_mov_b32 exec_hi, 0            ; lanes 32..63 inactive

; Guest instructions that write exec_lo (Wave32 canonical exec write) pass through:
s_mov_b32 exec_lo, s0           ; exec_hi remains 0 (no change needed)

; Guest: v_cmp_gt_f32 exec, v0, v1 (SDST == 126, i.e., writing EXEC pair)
; On Wave64, v_cmp writes all 64 bits. Lanes 32..63 hold uninitialized VGPR data
; that may spuriously activate upper lanes. Clear exec_hi after every such v_cmp:
v_cmp_gt_f32 exec, v0, v1
s_mov_b32    exec_hi, 0         ; clear upper 32 lanes (4 bytes, always encodable)
;
; v_cmpx instructions always write EXEC directly. Every v_cmpx must also be followed
; by the exec_hi clear shim:
v_cmpx_gt_f32 v0, v1            ; guest instruction (writes exec directly)
s_mov_b32    exec_hi, 0         ; clear upper 32 lanes
```

**Thread IDs:** The hardware populates `v0` with thread IDs 0–63 for a Wave64 wavefront. The active lanes 0–31 already hold IDs 0–31, which matches the Wave32 guest's expectation. No adjustment needed.

---

#### 3.5.4 Translation Decision Table

| Guest → Host | Wave64 available on host? | Strategy | Kernel descriptor change | Remaining work |
|---|---|---|---|---|
| Wave64 CDNA → RDNA1 | Yes (all RDNA gens) | Set Wave64 mode bit in KD; translate opcodes | kernel_code_properties bit 10 cleared (ENABLE_WAVEFRONT_SIZE32=0); re-encode VGPR granularity | s_waitcnt encoding, MFMA stubs, opcode remaps |
| Wave64 CDNA → Wave32-only (hypothetical) | No | Wave splitting: 2× Wave32 per workgroup | No KD change; dual entry points | exec split, LDS exchange zone, cross-lane ops |
| Wave32 RDNA → CDNA Wave64 | N/A (guest is Wave32) | exec masking: exec_hi=0 at entry | No KD change | `s_mov_b32 exec_hi, 0` after every v_cmp/v_cmpx targeting EXEC; s_waitcnt encoding |

### 3.6 MFMA Expansion

On non-CDNA hardware (RDNA), MFMA instructions have no target equivalent and are classified `Action::Expand` in the auto-generated legalization table. `BinaryTranslator::expand_mfma_body()` handles the expansion, dispatching on the mnemonic prefix `v_mfma_` or `v_smfmac_`.

**The reference implementation of the matrix math lives in `shared/mfma_exec.h`** (factored from `cdna3/mfma_exec.h` in Phase B, with `AccMode` enum for Unified/Separate/VgprOnly). That file contains:
- `input_loc(dim, K, B, i, k, b, data_bits)` — maps `(i, k, b)` indices to `{vgpr_offset, lane, sub_element}` for A and B matrices.
- `output_loc_32(M, N, i, j, b)` / `output_loc_64(M, N, i, j, b)` — maps `(i, j, b)` to `{reg, lane}` for the D/C matrix.
- `exec_f32<ExtractA, ExtractB>()`, `exec_i32_i8()`, `exec_f64()` — the full accumulate loops over `(blocks, rows, cols, K)` calling the above.

The DBT MFMA fallback stub is a **pre-compiled device-side library** (built as a separate AMDGPU device ELF for the target ISA) that wraps exactly these `mfma_exec.h` templates, compiled once at rocjitsu library build time via device-side C++. The translator inserts a call to the appropriate stub via the same `s_swappc_b64` trampoline mechanism as DBI — no new matrix math code is written.

```cpp
std::vector<uint32_t> BinaryTranslator::expand_mfma_body(
    const Instruction &mfma_inst) {
  // 1. Parse the MFMA encoding to determine dimensions and data types.
  //    The Vop3pMfmaMachineInst struct's op field encodes the variant.
  // 2. Look up the corresponding stub symbol in a pre-linked stub library ELF.
  // 3. Emit a trampoline call:
  //    - Spill conflicting registers (determined by liveness analysis).
  //    - Load stub address (known at translation time).
  //    - s_swappc_b64.
  //    - Restore registers.
  // 4. Return the trampoline call sequence as a byte vector.

  // Simplified: emit a call to a stub that panics (for now).
  // Production: emit call to mfma_f32_16x16x4f16_stub or equivalent.
  std::vector<uint32_t> seq;
  // Placeholder: emit a trap until the stub library is wired in.
  // emit_trap() is an ISA-specific free function from {isa}/machine_insts.h
  // that encodes the appropriate trap instruction for arch_.
  seq.push_back(emit_trap(arch_, /*trap_id=*/1));
  return seq;
}
```

### 3.7 Translation Cache

```cpp
// lib/rocjitsu/src/rocjitsu/code/dbt/translation_cache.h

namespace rocjitsu {

/// @brief On-disk result cache for completed ELF translations.
///
/// This is NOT a rewrite-rule database. The translation rules live in
/// InstructionMapper (§3.3.1). TranslationCache stores finished translated
/// ELFs on disk so that BinaryTranslator::translate() can be skipped entirely
/// on subsequent runs for the same binary.
///
/// Cache directory: ${XDG_CACHE_HOME:-~/.cache}/rocjitsu/dbt/
/// File naming: <hex(binary_hash)>_<src_arch>_<dst_arch>.elf
///
/// The cache is keyed on (binary_hash, source_arch, target_arch).
/// A translated ELF is valid as long as the source binary hash matches.
class TranslationCache {
public:
  static TranslationCache &instance();

  /// @brief Look up a translation in the cache.
  /// @returns Path to the cached ELF, or empty if not found.
  std::optional<std::filesystem::path>
  lookup(const TranslationKey &key) const;

  /// @brief Store a translated ELF in the cache.
  void store(const TranslationKey &key, std::span<const uint8_t> elf_bytes);

  /// @brief Compute the xxHash64 of a code object's text sections.
  static uint64_t hash_code_object(const AmdGpuCodeObject &obj);

  static constexpr const char *CACHE_DIR_ENV = "RJ_DBT_CACHE_DIR";
  static constexpr const char *DEFAULT_SUBDIR = "rocjitsu/dbt";

private:
  TranslationCache() = default;
  TranslationCache(const TranslationCache &) = delete;
  TranslationCache &operator=(const TranslationCache &) = delete;
  TranslationCache(TranslationCache &&) = delete;
  TranslationCache &operator=(TranslationCache &&) = delete;

  std::filesystem::path cache_dir_;
  std::filesystem::path key_to_path(const TranslationKey &key) const;
};

} // namespace rocjitsu
```

### 3.8 Branch Target Fixup — Not Required

The code cave strategy (§3.3.2) ensures that every legalization action writes exactly `inst.size()` bytes in-place in `.text`. No instruction ever grows or shrinks in-place, so no instruction's address changes, and no existing branch offset in `.text` is invalidated.

The `patch_branch_offset` virtual method and `BranchReloc` types defined in §2.6 are still used by `CodeObjectPatcher` for the **DBI** path (where `insert_at()` genuinely inserts bytes and shifts subsequent code). They are not needed by `BinaryTranslator`.

The only branch-offset work `BinaryTranslator` performs is computing the two-instruction offsets inside each cave entry (`s_branch <return_offset>`) and inside each cave stub (`s_branch <cave_offset>`), both of which are computed locally at cave-assembly time with no need to scan the surrounding text.

### 3.9 Instruction-Level Translation Table

Translation actions fall into five categories:

| Term | `Action` enum value | Meaning |
|---|---|---|
| **Identity** | `Action::Identity` | Instruction encoding is identical on both ISAs; re-encode with `target_opcode` (same value as source). |
| **Substitute** | `Action::Substitute` | Same encoding layout, different opcode; re-encode with `InstructionLegalization::target_opcode`. |
| **Lower** | `Action::Lower` | Encoding layout differs; the generated decode function extracts fields into an ISA-neutral struct, then the encode function packs them into the target struct using `InstructionLegalization::target_opcode`. Semantic lowering (waitcnt, barrier) is in `lower_*()` methods. |
| **Expand** | `Action::Expand` | No target equivalent; instruction is expanded to a software emulation sequence (e.g., MFMA on RDNA, WMMA on CDNA). Expansion logic is in the matching `expand_*()` method on `BinaryTranslator`. See §3.6. |
| **N/A** | — | Instruction class does not exist on the source ISA; no action needed. |

> **Type system note:** The `Action` enum and `InstructionLegalization` struct are defined once in the auto-generated header `code/dbt/generated/legalization_tables.h`. `InstructionMapper` in `code/dbt/instruction_mapper.h` includes this header and uses these types directly — no parallel `LegalizationAction` or `InstructionMapping` types exist.

#### 3.9.1 ISA Feature Summary

The table below summarizes which instruction categories are available on each supported ISA family. This drives which translation actions apply for a given guest→host pair.

| Feature / Category | CDNA1 GFX908 | CDNA2 GFX90a | CDNA3 GFX942 | CDNA4 GFX950 | RDNA1 GFX1010 | RDNA2 GFX1030 | RDNA3 GFX1100 | RDNA3.5 GFX1150 | RDNA4 GFX1200 |
|---|---|---|---|---|---|---|---|---|---|
| Default wave size | Wave64 | Wave64 | Wave64 | Wave64 | Wave32 | Wave32 | Wave32 | Wave32 | Wave32 |
| Wave64 capable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| `s_waitcnt` encoding | GFX9 classic | GFX9 classic | GFX9 classic | GFX9 classic² | GFX10 (+vscnt) | GFX10 (+vscnt) | Split (GFX11-style) | Split (GFX11-style) | Split (GFX12-style) |
| `s_barrier` | SOPP single | SOPP single | SOPP single | `signal`/`wait` pair | SOPP single | SOPP single | `signal`/`wait` pair | `signal`/`wait` pair | `signal`/`wait` pair |
| Memory addressing | `flat` + `seg` field | `flat` + `seg` field | Separate `global`/`scratch` | Separate `global`/`scratch` | Separate `global`/`scratch` | Separate `global`/`scratch` | Separate `global`/`scratch` | Separate `global`/`scratch` | Separate `global`/`scratch` |
| SMEM mnemonic family | `s_load_dword` | `s_load_dword` | `s_load_dword` | `s_load_b32` | `s_load_dword` | `s_load_dword` | `s_load_b32` | `s_load_b32` | `s_load_b32` |
| MFMA ops | Yes (v1, no AccVGPR) | Yes (v2, +XDLOPS, +AccVGPR) | Yes (v3, +sparse) | Yes (v4, +FP8/BF8) | No | No | No | No | No |
| WMMA ops | No | No | No | No | No | No | Yes (v1) | Yes (v1) | Yes (v2) |
| ACC VGPRs (`v_accvgpr_*`) | No (MFMA uses regular VGPRs) | Yes | Yes | Yes | No | No | No | No | No |
| Hardware reconvergence insts | No | No | No | No | No | No | Yes | Yes | Yes |
| `s_sleep` / `s_nop` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

**`s_waitcnt` encoding variants:**
- **GFX9 classic** — 16-bit imm: vmcnt\[3:0\]+\[15:14\], expcnt\[6:4\], lgkmcnt\[11:8\] (4-bit).
- **GFX10 (+vscnt)** — Same 16-bit `s_waitcnt` plus new `s_waitcnt_vscnt` for store completion.
- **Split (GFX11-style)** — `s_wait_loadcnt`, `s_wait_storecnt`, `s_wait_expcnt`, `s_wait_kmcnt`, `s_wait_dscnt`, `s_wait_samplecnt`, `s_wait_bvhcnt`. `WaitcntTranslator` maps from the source encoding to the target split using capability booleans (§3.4).
- **Split (GFX12-style)** — As GFX11 but `s_wait_storecnt_dscnt` replaces the separate store and DS waits.

#### 3.9.2 Translation Action Matrix

Each cell gives the action `BinaryTranslator` takes for that instruction category when translating from the row's source ISA family to the column's target ISA family. Same-family minor-version upgrades (e.g., CDNA1→CDNA2) are shown on the left; cross-family pairs on the right.

**Legend:** `=` Identity · `S` Substitute · `L` Lower · `E` Expand · `–` N/A (not in source ISA) · `?` Requires ISA spec confirmation before implementation

| Instruction Category | CDNA1→CDNA2 | CDNA2→CDNA3 | CDNA3→CDNA4 | CDNA1/2/3→RDNA1/2 | CDNA1/2/3→RDNA3/4 | CDNA4→RDNA3/4 | RDNA1/2→RDNA3/4 | RDNA3→RDNA4 |
|---|---|---|---|---|---|---|---|---|
| SOP1/SOP2 integer ALU | `=` | `=` | `=` | `=` | `=` | `=` | `=` | `=` |
| VOP2/VOP3 fp32/int32 | `=` | `=` | `=` | `=` | `=` | `=` | `=` | `=` |
| `s_waitcnt` | `=` | `=` | `=`² | `L` §3.4 | `L` §3.4 | `L` §3.4 | `L` §3.4 | `L` §3.4 |
| `s_barrier` | `=` | `=` | `L` (signal/wait) | `=` | `L` (signal/wait) | `=` | `L` (signal/wait) | `=` |
| `flat_load/store` (seg=0) | `=` | `L` → `global_load/store` (pointer provably global; else `ILLEGAL`) | `=` | `L` → `global_load/store` (only if pointer provably global; otherwise `ILLEGAL`) | `L` → `global_load/store` (same caveat) | `=` | `=` | `=` |
| `scratch_load/store` (seg=2) | `=` | `S` → `scratch_load/store` | `=` | `L` → `scratch_load/store` | `L` → `scratch_load/store` | `=` | `=` | `=` |
| `global_load/store` | `–` | `–` | `–` | `–` | `–` | `=` | `=` | `=` |
| SMEM (`s_load_dword` family) | `=` | `=` | `S` → `s_load_b32` family | `=` | `S` → `s_load_b32` family | `=` | `S` → `s_load_b32` family | `=` |
| MFMA ops (compatible subset) | `=` | `=` | `=` | `E` §3.6 | `E` §3.6 | `E` §3.6 | `–` | `–` |
| MFMA ops (source-only ops) | `–` | `–` | `E` §3.6 (CDNA3-only ops on CDNA4) | `E` §3.6 | `E` §3.6 | `E` §3.6 | `–` | `–` |
| MFMA ops (dest-only new ops) | `–` | `–` | `–` | `–` | `–` | `–` | `–` | `–` |
| ACC VGPR insts (`v_accvgpr_*`) | `–` | `=` | `=` | `E` §3.6 | `E` §3.6 | `E` §3.6 | `–` | `–` |
| WMMA ops | `–` | `–` | `–` | `–` | `–` | `–` | `=` | `S` ? |
| Wave32 exec handling | `–` | `–` | `–` | `L` §3.5 (force Wave64) | `L` §3.5 (force Wave64) | `L` §3.5 (force Wave64) | `=` | `=` |
| Reconvergence insts | `–` | `–` | `–` | `–` | `L` §3.5 (insert at join points) | `L` §3.5 (insert at join points) | `=` | `=` |
| `s_sleep` / `s_nop` / SOPP misc | `=` | `=` | `=` | `=` | `=` | `=` | `=` | `=` |

**Notes:**

- **CDNA3→CDNA4 MFMA (source-only):** CDNA3 has MFMA shapes not yet present on CDNA4. `expand_mfma()` emulates them with sequences of supported MFMA shapes, or traps. See §3.6.
- **CDNA→RDNA ACC VGPRs:** RDNA lacks ACC VGPRs entirely. Any kernel that uses `v_accvgpr_write`/`v_accvgpr_read` for MFMA accumulation must be replaced with WMMA equivalents (RDNA3/4) or rejected as `ILLEGAL`.
- **RDNA3→RDNA4 WMMA:** RDNA4 added new WMMA shapes (FP8/BF8, mixed precision). Shared shapes use direct substitution; source-only shapes require expansion. Spec confirmation needed before finalizing the substitution table (`?`).
- **`flat` seg=0/2 on CDNA4/RDNA:** The `flat_load/store` instruction with `seg` field is deprecated in GFX11+. `seg=2` (explicit scratch) rewrites safely to `scratch_load/store`. `seg=0` (generic flat) can address global **or** LDS **or** private memory at runtime depending on the pointer value — a blind rewrite to `global_load/store` is incorrect for LDS/private pointers. `lower_flat_memory()` must classify the pointer origin statically via liveness/def-use: if the pointer is derived exclusively from kernarg or device-memory allocations, `global_load/store` is safe; otherwise the instruction is classified `ILLEGAL` and translation fails with an explanatory diagnostic. The diagnostic includes the instruction offset and the unknown-origin pointer SGPR so the user can annotate the source.
- **Cross-pair combinations not listed** (e.g., RDNA→CDNA, CDNA4→CDNA3 downgrade) are not planned for initial implementation. `rj_code_translate` returns `ROCJITSU_STATUS_UNSUPPORTED` for these pairs.

#### 3.9.3 E-Graph Rewrite Database for Legalization Table Generation

The translation action matrix in §3.9.2 is the authoritative specification, but manually implementing one `InstructionMapper` subclass per (src, dst) pair for all 72 directional translation pairs does not scale. Instead, the existing `amdisa` Python library is extended with two new modules — `legalization.py` and `legalization_codegen.py` — that automate the production of legalization tables from the parsed ISA specifications. No new language or build toolchain is introduced; the legalization codegen runs as part of the existing `python -m amdisa --multi ... --gen-legalization` pipeline.

##### Design Overview

The legalization generator reuses `amdisa`'s existing XML parser (`parser.py`), semantics derivation (`semantics.py`), ISA profiles (`isa_profile.py`), and cross-ISA structural comparison (`cross_isa.py`). It adds a mnemonic rename map for cross-generation name changes, a union-find for transitive equivalence closure, domain-specific override rules, and a C++ codegen backend that emits sorted `InstructionLegalization[]` tables.

```
python -m amdisa --multi cdna3:cdna3.xml cdna4:cdna4.xml ... --gen-legalization
        │
        ▼
┌──────────────────────────────────────────┐
│  amdisa (Python)                         │
│                                          │
│  parser.py        (existing — reused)    │
│  semantics.py     (existing — reused)    │
│  cross_isa.py     (existing — reused)    │
│  isa_profile.py   (existing — reused)    │
│                                          │
│  legalization.py  (NEW — ~300 lines)     │
│    ├── mnemonic rename map               │
│    ├── union-find equivalence classes    │
│    ├── legalization classification       │
│    └── domain-specific override rules    │
│                                          │
│  legalization_codegen.py (NEW — ~150 lines) │
│    └── emit C++ InstructionLegalization[] tables        │
└────────────┬─────────────────────────────┘
             │
             ▼
dbt/generated/legalization_tables.h    ← shared header
dbt/generated/legalize_cdna3_cdna4.cpp ← one per pair
```

**Why Python, not a separate Rust crate.** The legalization classifier is a build-time tool that processes ~10K instructions across 9 ISAs. The core data structure is a 15-line union-find over mnemonic equivalence classes. Python handles this in seconds with zero duplication of the XML parser, semantics, or profile logic. A Rust crate would duplicate ~600 lines of parsing code, add Cargo as a build dependency, and provide no performance benefit for a build-time tool. If algebraic equality saturation (e.g., via the `egg` library) is needed in the future for peephole superoptimization, a Rust extension module can be added via PyO3 and called from `legalization.py` — but that decision is deferred until a concrete workload justifies it.

##### Semantic Equivalence Without Execution Semantics

The XML specs contain only structural information (encodings, operands, opcodes, field layouts) — not execution semantics. Semantic equivalence is determined by:

1. **Mnemonic identity**: Same instruction name on two ISAs = same semantics. AMD never changes an instruction's semantics while keeping its mnemonic.

2. **Mnemonic rename map**: A manually curated table (~50 entries) in `legalization.py` mapping known cross-generation name changes that preserve semantics:

   | Source family | Target family | Boundary | Example |
   |---|---|---|---|
   | `S_LOAD_DWORD*` | `S_LOAD_B32*` | GFX9→GFX11 | `S_LOAD_DWORDX4` → `S_LOAD_B128` |
   | `FLAT_LOAD_DWORD*` | `GLOBAL_LOAD_B32*` | GFX9→GFX11 | `FLAT_LOAD_DWORD` → `GLOBAL_LOAD_B32` |
   | `S_WAITCNT` | `S_WAIT_LOADCNT`+... | GFX9→GFX11 | 1:N split |
   | `S_BARRIER` | `S_BARRIER_SIGNAL`+`S_BARRIER_WAIT` | GFX9→GFX11 | 1:2 split |

3. **Encoding/operand comparison**: Reuses `cross_isa.py`'s `_field_signature()` and `_operand_signature()` to determine whether semantically equivalent instructions are encoding-compatible (IDENTITY/SUBSTITUTE) or require lowering (LOWER).

##### E-Graph / Union-Find Structure

The equivalence engine is a minimal union-find (~15 lines of Python) over instruction IDs. Each instruction on each ISA gets a unique ID. Instructions with the same canonical mnemonic (after applying the rename map) across ISAs are merged into the same equivalence class. The union-find provides transitive closure automatically: if `S_LOAD_DWORD` (CDNA1) ≡ `S_LOAD_B32` (CDNA4) via rename, and `S_LOAD_B32` (CDNA4) ≡ `S_LOAD_B32` (RDNA4) via mnemonic identity, then `S_LOAD_DWORD` (CDNA1) ≡ `S_LOAD_B32` (RDNA4) is discovered without explicit three-way mapping.

This is the same data structure as a full e-graph's equivalence layer, without the hash-consing and rewrite-rule saturation machinery. If algebraic rewrite rules are added later (e.g., `v_add_f32 x, 0 ≡ v_mov_b32 x` for peephole optimization), a Rust-based `egg` wrapper can be integrated via PyO3 as a drop-in replacement for the union-find, called from the same `legalization.py` orchestration code.

##### Legalization Classification

For each (src, dst) pair and each source instruction:

1. Find the source instruction's equivalence class (union-find root).
2. Search the class for a target-ISA instruction.
3. If found and encodings are identical (same field layout + same opcode): **IDENTITY**.
4. If found and encodings are compatible (same field layout, different opcode): **SUBSTITUTE**.
5. If found but encodings differ: **LOWER** (with a lowering-kind tag for dispatch to the correct `lower_*()` method).
6. If not found: **LOWER** (generic lowering) or **EXPAND** (for MFMA/AccVGPR/WMMA). The classifier never returns ILLEGAL — every instruction has a correct lowering or expansion path.

Domain-specific override rules are applied after structural classification:
- Any `s_waitcnt` crossing a waitcnt-model boundary → LOWER (§3.4).
- Any `s_barrier` where the target uses split barrier → LOWER.
- Any `flat_load/store` where the target uses separate global/scratch → LOWER.
- Any MFMA instruction on a non-MFMA target → EXPAND.
- Any AccVGPR instruction on a non-AccVGPR target → EXPAND.

##### Generated C++ Interface

The generated C++20 header uses `operator<=>` for sorted-table ordering, `std::span` for type-safe table access, and `std::lower_bound` for O(log n) lookup:

```cpp
// code/dbt/generated/legalization_tables.h — AUTO-GENERATED by amdisa

namespace rocjitsu {

enum class Action : uint8_t {
    Identity = 0, Substitute = 1, Lower = 2, Expand = 3, Illegal = 4,
};

struct InstructionLegalization {
    uint16_t src_opcode;
    uint16_t src_encoding_id;
    Action   action;
    uint16_t target_opcode;  // populated for Identity, Substitute, and Lower

    constexpr auto operator<=>(const InstructionLegalization &rhs) const {
        if (auto cmp = src_encoding_id <=> rhs.src_encoding_id; cmp != 0)
            return cmp;
        return src_opcode <=> rhs.src_opcode;
    }
    constexpr bool operator==(const InstructionLegalization &rhs) const = default;
};

// Per-pair tables (e.g., kLegalization_cdna3_cdna4, kLegalization_cdna3_rdna4, ...)
inline constexpr size_t kLegalization_cdna3_cdna4_size = 1401;
extern const InstructionLegalization kLegalization_cdna3_cdna4[kLegalization_cdna3_cdna4_size];
// ... 27 total tables covering all supported translation pairs

/// Binary-search a sorted legalization table by (encoding_id, opcode).
inline const InstructionLegalization *lookup(
    std::span<const InstructionLegalization> table,
    uint16_t encoding_id, uint16_t opcode);

} // namespace rocjitsu
```

`InstructionMapper::create(src, dst)` returns a mapper backed by the appropriate table:

```cpp
std::unique_ptr<InstructionMapper> InstructionMapper::create(
    rj_code_arch_t src, rj_code_arch_t dst) {
    // Select the correct kLegalization_* array for this (src, dst) pair.
    auto table = select_legalization_table(src, dst);
    if (table.empty()) return nullptr;
    return std::make_unique<InstructionMapper>(table);
}
```

##### Module Location

```
lib/python/amdisa/
├── legalization.py          # NEW — LegalizationGenerator, union-find,
│                            #   mnemonic rename map, domain rules (~300 lines)
└── legalization_codegen.py  # NEW — emit C++ InstructionLegalization tables (~150 lines)
```

Invoked via the existing CLI: `python -m amdisa --multi ... --gen-legalization --legalization-output <dir>`

##### Future: Rust E-Graph Extension via PyO3

If algebraic equality saturation is needed (peephole superoptimization, cost-driven extraction over thousands of rewrite rules), a Rust extension module wrapping the `egg` library can be added:

```
tools/rj-egraph/           # Future — Rust PyO3 extension
├── Cargo.toml
└── src/lib.rs             # Exposes IsaEGraph with saturate() to Python
```

`legalization.py` would call `from rj_egraph import IsaEGraph` as a drop-in replacement for the built-in union-find, with no changes to the rest of the pipeline. This decision is deferred until a concrete workload justifies the added build complexity.

##### Roadmap Integration

This is **Phase 8.5** (parallel with Phases 1–8, no dependencies beyond the existing `amdisa` library). Estimated 3–5 days (reduced from 1–2 weeks since no XML parser or semantics porting is needed). Phase 9 is updated to consume the generated tables instead of hand-written mappers.

#### 3.9.4 Resource Translation: Never-Fail DBT via Tiered Virtualization

Instruction-level legalization (§3.9.1–§3.9.3) handles opcode and encoding differences. A second, equally critical dimension is **resource translation**: the guest kernel may require more VGPRs, SGPRs, LDS, or a different scratch addressing model than the target hardware provides. The design principle is that translation **never fails** — every resource mismatch has a correctness-preserving transformation, and performance degrades gracefully through a tiered fallback chain.

This approach is informed by state-of-the-art GPU resource virtualization research:
- **Zorua** (MICRO 2016, Ausavarungnirun et al.) demonstrated that GPU registers, scratchpad memory, and thread slots can all be virtualized simultaneously by oversubscribing on-chip resources with a global-memory swap space, yielding 55% reduction in porting performance loss across GPU architectures.
- **RegDem** (Volkov 2019) showed that excess registers can be spilled to *shared memory* rather than global memory when shared memory is underutilized, achieving 9% geometric mean speedup over the default compiler strategy.
- **NVIDIA PTX→SASS JIT** handles arbitrary resource differences across GPU generations without ever rejecting a valid PTX program; the JIT compiler adapts register allocation, shared memory usage, and occupancy to the target hardware.

##### 3.9.4.1 VGPR Count Mismatch

**Problem.** The guest kernel uses N VGPRs; the target ISA provides at most M_max VGPRs per wavefront (e.g., CDNA3 Wave64: 512 VGPRs; RDNA3 Wave32: 256 VGPRs; RDNA3 Wave64: 512 VGPRs). When N > M_max, the kernel cannot run without transformation.

**Tiered fallback (cheapest first):**

| Tier | Condition | Strategy | Overhead |
|---|---|---|---|
| 0 | N ≤ M at desired occupancy | **Direct mapping.** Re-encode `GRANULATED_WORKITEM_VGPR_COUNT` using the target AMDHSA descriptor encoding, not the target ISA's physical VGPR allocation block. | Zero |
| 1 | N > M at desired occupancy, but N ≤ M_max | **Accept reduced occupancy.** Set the KD to request N VGPRs. The target runs fewer concurrent wavefronts but the kernel is correct. The occupancy cost is logged as a translation warning. | Occupancy loss; no instruction changes |
| 2 | N > M_max, but target has spare LDS capacity | **RegDem-style spill to LDS.** Run `LivenessAnalysis` (Pillar 4) to identify cold VGPRs (long live ranges, infrequent accesses). Allocate a reserved LDS zone at the top of the workgroup's LDS allocation. Insert `ds_write_b32` at spill points and `ds_read_b32` at fill points. Update the KD's `group_segment_fixed_size` to include the spill zone. Each spill slot is `wf_size × 4` bytes (one 32-bit value per lane). The reserved zone starts at `original_lds_size` and grows upward. | LDS bandwidth; reduced LDS available to kernel; VGPR count reduced to M_max |
| 3 | N > M_max and LDS budget exhausted | **Spill to scratch (private memory).** Use `SpillManager` (§2.3) to allocate flat-scratch slots for the coldest remaining VGPRs. Insert `scratch_store_b32` / `scratch_load_b32` at spill/fill points. Update the KD's `private_segment_fixed_size`. | Global memory bandwidth (slowest tier) |

The translator selects the tier at translation time. `LivenessAnalysis` live-before sets identify dead registers that do not need spilling; only truly live registers that exceed the target's capacity are spill candidates. The spill set is computed once per kernel, not per instruction.

**VGPR descriptor re-encoding** is always performed regardless of tier. This uses the AMDHSA `compute_pgm_rsrc1.GRANULATED_WORKITEM_VGPR_COUNT` formula, which is distinct from the physical VGPR allocation block size in the ISA manuals:

```
Guest CDNA3/CDNA4 (Wave64): descriptor granularity = 8 DWords
  actual_vgprs = (cdna_granulated + 1) × 8

Target RDNA GFX10-GFX12 (Wave64): descriptor granularity = 4 DWords
  rdna_granulated = ceil(actual_vgprs / 4) − 1

Target RDNA GFX10-GFX12 (Wave32): descriptor granularity = 8 DWords
  rdna_granulated = ceil(actual_vgprs / 8) − 1
```

##### 3.9.4.2 SGPR Count Mismatch

**Problem.** RDNA ISAs provide 106 SGPRs; CDNA ISAs provide 102. An RDNA kernel using 103+ SGPRs cannot run on CDNA without transformation.

**Strategy.** The same tiered approach as VGPRs, but cheaper because SGPRs are scalar (shared across all lanes in the wavefront):

| Tier | Strategy | Cost per spill/fill |
|---|---|---|
| 0 | Direct mapping (guest count ≤ target max) | Zero |
| 1 | Spill to scratch via `s_store_dword` / `s_load_dword` | One SMEM instruction each (no per-lane cost) |

LDS spill is not used for SGPRs because LDS is vector-addressed (per-lane), while SGPRs are scalar. Writing a scalar to LDS would require broadcasting to all lanes and is wasteful. Scratch spill for SGPRs costs one SMEM operation per spill/fill — far cheaper than VGPR spill.

`LivenessAnalysis` provides the live-before sets needed to choose SGPR spill candidates. Spill candidates are selected starting from the highest-numbered live SGPRs (closest to the target's limit).

##### 3.9.4.3 AccVGPR on Non-AccVGPR Target

**Problem.** CDNA2+ kernels use `v_accvgpr_write` / `v_accvgpr_read` to access a separate 256-register accumulator file used by MFMA instructions. RDNA and CDNA1 have no AccVGPR file.

**Strategy.** AccVGPR accesses are remapped to regular VGPRs:

```asm
; Source (CDNA2+):
v_accvgpr_write acc[N], v[M]      ; write VGPR M to AccVGPR N
v_accvgpr_read  v[M], acc[N]      ; read AccVGPR N to VGPR M

; Translated (target without AccVGPR):
v_mov_b32 v[N + accvgpr_base], v[M]   ; map acc[N] → v[N + accvgpr_base]
v_mov_b32 v[M], v[N + accvgpr_base]
```

`accvgpr_base` is the first VGPR index beyond the guest's regular VGPR range: `accvgpr_base = guest_vgpr_count`. The KD's total VGPR count is increased to `guest_vgpr_count + guest_accvgpr_count`. If this exceeds the target's M_max, the VGPR tiered fallback (§3.9.4.1) handles the overflow — AccVGPR-mapped VGPRs are typically cold between MFMA invocations and are good spill candidates.

MFMA instructions themselves are handled by the EXPAND rule (§3.6), which emits software fallback stubs using the reference implementations in `shared/mfma_exec.h`.

##### 3.9.4.4 LDS Size Mismatch

**Problem.** The guest kernel requests `group_segment_fixed_size` bytes of LDS, but the target hardware provides less per CU (e.g., a hypothetical target with 32KB LDS vs. a guest compiled for 64KB).

**Note on current hardware.** All current CDNA and RDNA ISAs provide at least 64KB LDS per CU: CDNA1-4 provide 64KB, RDNA1-4 provide 64KB-128KB. LDS overflow is therefore unlikely for same-generation or forward translations. However, the mechanism is needed for correctness guarantees and for future hardware that may have different LDS budgets. The RegDem-style VGPR-to-LDS spill (§3.9.4.1 tier 2) also consumes LDS and may push total LDS usage beyond the target's capacity.

**Strategy: Zorua-style LDS virtualization.** LDS accesses beyond the physical limit are redirected to a per-workgroup overflow buffer in device memory:

1. **Partition the LDS address space.** Addresses in `[0, physical_lds_max)` stay in real LDS. Addresses in `[physical_lds_max, guest_lds_size)` map to a global-memory overflow buffer.

2. **Rewrite overflow `ds_*` instructions.** Every `ds_read_b32` / `ds_write_b32` whose address may fall in the overflow range is replaced with a conditional sequence:
   ```asm
   ; Original: ds_read_b32 v_dst, v_addr
   ; Translated (address may overflow):
   v_cmp_lt_u32 vcc, v_addr, physical_lds_max
   s_cbranch_vccnz .use_lds
   ; Overflow path: redirect to global memory
   v_add_u32 v_tmp, v_addr, s[overflow_base:overflow_base+1]  ; add overflow buffer VA
   global_load_b32 v_dst, v_tmp, off
   s_branch .done
   .use_lds:
   ds_read_b32 v_dst, v_addr
   .done:
   ```

3. **Overflow buffer allocation.** The overflow buffer is `(guest_lds_size − physical_lds_max) × num_workgroups_per_cu` bytes, allocated in device memory at dispatch time. Its base VA is injected via the kernarg extension mechanism (§2.7, §4.3.4) — one additional 8-byte kernarg slot.

4. **KD update.** `group_segment_fixed_size` is set to `min(guest_lds_size, physical_lds_max)`. The overflow portion is not reflected in the KD because it lives in device memory, not in the hardware LDS.

**Static analysis optimization.** Most `ds_*` instructions in GPU kernels use statically-known address ranges (e.g., `threadIdx.x * 4` with a known max). When the translator can prove at translation time that all addresses in a kernel are below `physical_lds_max`, the conditional sequence is omitted entirely and the instruction passes through unchanged. Only kernels with dynamically-computed LDS addresses that *may* exceed the limit pay the conditional overhead.

##### 3.9.4.5 Scratch/Private Segment Addressing Model

**Problem.** CDNA1/2 initialize the flat-scratch base via an SGPR pair (`ENABLE_SGPR_FLAT_SCRATCH_INIT`). CDNA3/4 use a dedicated hardware `FLAT_SCRATCH` register pair — no SGPR is consumed.

**Kernel descriptor fixup:**

| Direction | KD change | Prologue |
|---|---|---|
| CDNA1/2 → CDNA3/4 | Clear `ENABLE_SGPR_FLAT_SCRATCH_INIT`; decrement `USER_SGPR_COUNT` by 2 | None (hardware provides scratch base automatically) |
| CDNA3/4 → CDNA1/2 | Set `ENABLE_SGPR_FLAT_SCRATCH_INIT`; increment `USER_SGPR_COUNT` by 2 | Inject `s_mov_b32 flat_scratch_lo, s[N]` + `s_mov_b32 flat_scratch_hi, s[N+1]` prologue to initialize the hardware pair from the runtime-provided SGPRs |
| CDNA → RDNA or RDNA → CDNA | Varies per generation; follows the same pattern | Per-direction prologue |

The SGPR pair freed (CDNA1/2→CDNA3/4) or consumed (CDNA3/4→CDNA1/2) is accounted for in the user SGPR layout translation (§3.9.4.6).

##### 3.9.4.6 User SGPR Layout Translation

**Problem.** The AMDGPU kernel descriptor's `COMPUTE_PGM_RSRC2` register contains `USER_SGPR_COUNT` and a set of `ENABLE_SGPR_*` bits that define which implicit SGPRs the hardware pre-loads at kernel entry. The layout and ordering of these SGPRs differ across ISA generations:

| SGPR slot | CDNA1/2 | CDNA3/4 | RDNA1/2 | RDNA3/4 |
|---|---|---|---|---|
| Dispatch pointer | s[0:1] (if enabled) | s[0:1] (if enabled) | s[0:1] (if enabled) | s[0:1] (if enabled) |
| Queue pointer | s[2:3] (if enabled) | s[2:3] (if enabled) | s[2:3] (if enabled) | s[2:3] (if enabled) |
| Kernarg pointer | s[4:5] (if enabled) | s[4:5] (if enabled) | s[4:5] (if enabled) | s[4:5] (if enabled) |
| Flat scratch init | s[6:7] (if enabled) | N/A (hardware) | s[6:7] (if enabled) | s[6:7] (if enabled) |

When `ENABLE_SGPR_FLAT_SCRATCH_INIT` changes between source and target, all subsequent user SGPRs shift by ±2. The translator must:

1. Parse the source KD's `ENABLE_SGPR_*` bits to build the source user SGPR assignment map.
2. Build the target user SGPR map based on which features the target ISA needs.
3. If the layouts differ, inject a **prologue shuffle** that moves SGPRs from the source layout to the target layout using `s_mov_b32` instructions. Worst case: ~10 `s_mov_b32` instructions (all user SGPRs shift by 2).
4. Update `USER_SGPR_COUNT` in the target KD.

The prologue is emitted by `KernelDescriptorTranslator`, placed in a code cave by `CodeObjectPatcher`, and made the kernel entry by updating the descriptor's `kernel_code_entry_byte_offset`. The original entry instruction is left untouched, so descriptor ABI prologues do not depend on relocating an arbitrary first instruction.

##### 3.9.4.7 Occupancy Impact

Resource translation may reduce occupancy (the number of concurrent wavefronts per CU). This is a **performance cost, not a correctness issue**. The translator logs the occupancy impact as a warning:

```
[rocjitsu-dbt] kernel_foo: CDNA3→RDNA3 translation
  VGPRs: 256 → 256 (tier 0, direct mapping)
  SGPRs: 96 → 96 (tier 0, direct mapping)
  LDS: 32768 → 32768 (no overflow)
  Occupancy: 4 waves/CU → 4 waves/CU (unchanged)

[rocjitsu-dbt] kernel_bar: CDNA3→RDNA3 translation
  VGPRs: 512 → 512 (tier 1, reduced occupancy)
  AccVGPR remap: 256 AccVGPRs → 256 extra VGPRs (total: 768)
  VGPRs after remap: 768 → 512 (tier 2, spilled 256 cold VGPRs to LDS)
  LDS: 0 + 4096 spill zone = 4096 bytes
  Occupancy: 2 waves/CU → 1 wave/CU (reduced due to VGPR+LDS pressure)
```

For performance-critical deployments, the user can set `RJ_DBT_MIN_OCCUPANCY=N` to force the translator to target at least N waves/CU. If the translated kernel cannot meet this threshold at any tier, the translator emits the best-effort translation and logs a warning — it still **never fails**.

##### 3.9.4.8 `KernelDescriptorTranslator`

All resource-level translations are driven by a single component that systematically patches every KD field:

```cpp
// lib/rocjitsu/src/rocjitsu/code/dbt/kernel_descriptor_translator.h

namespace rocjitsu {

/// @brief Per-kernel resource translation result.
///
/// Computed by KernelDescriptorTranslator::translate() and consumed by
/// BinaryTranslator to (a) hand descriptor bytes and prologue words to
/// CodeObjectPatcher, (b) redirect the KD entry point when a prologue is needed,
/// and (c) decide which VGPR/SGPR spill transformations to apply.
struct KdTranslation {
  // --- VGPR ---
  uint32_t target_vgpr_count;         ///< After AccVGPR remapping + occupancy/spill adjustment.
  uint32_t target_vgpr_granulated;    ///< Re-encoded for target ISA's granularity.
  uint32_t accvgpr_base;              ///< First VGPR index for AccVGPR remapping (0 if no remap).
  uint32_t vgpr_spill_to_lds_count;   ///< VGPRs spilled to LDS (tier 2).
  uint32_t vgpr_spill_to_scratch_count; ///< VGPRs spilled to scratch (tier 3).

  // --- SGPR ---
  uint32_t target_sgpr_count;
  uint32_t sgpr_spill_count;          ///< SGPRs spilled to scratch.

  // --- LDS ---
  uint32_t target_lds_size;           ///< min(guest_lds + spill_zone, target_max_lds).
  uint32_t lds_spill_zone_bytes;      ///< LDS reserved for VGPR spill (tier 2).
  uint32_t lds_overflow_size;         ///< Guest LDS exceeding target capacity (Zorua overflow).
  bool     needs_lds_overflow_buf;    ///< True if lds_overflow_size > 0.

  // --- Scratch ---
  uint32_t target_private_size;       ///< After spill slot allocation for VGPRs + SGPRs.

  // --- Wave size ---
  uint8_t  target_wave_size;          ///< 32 or 64.
  bool     force_wave64;              ///< Set ENABLE_WAVEFRONT_SIZE32=0 on RDNA targets.

  // --- User SGPR layout ---
  uint8_t  target_user_sgpr_count;
  bool     needs_flat_scratch_init_sgpr; ///< Target uses SGPR-based scratch init.
  std::vector<uint32_t> user_sgpr_shuffle; ///< Prologue s_mov_b32 sequence for layout fixup.

  // --- Combined prologue ---
  /// All injected prologue instructions (user SGPR shuffle + scratch init + exec mask).
  /// Placed in a cave by CodeObjectPatcher; the KD entry point is redirected to it.
  std::vector<uint32_t> prologue_words;

  // --- Diagnostics ---
  uint32_t source_occupancy;          ///< Wavefronts/CU on source hardware.
  uint32_t target_occupancy;          ///< Wavefronts/CU on target hardware (after translation).
  std::vector<std::string> warnings;  ///< Occupancy reduction, spill notices, etc.
};

/// @brief Translates the kernel descriptor and computes resource adjustments.
///
/// Does NOT modify the ELF — it produces a KdTranslation struct that
/// BinaryTranslator applies via CodeObjectPatcher. This separation allows
/// BinaryTranslator to make instruction-level decisions (spill insertion,
/// AccVGPR remapping, LDS overflow conditionals) based on the KdTranslation
/// before committing any changes.
class KernelDescriptorTranslator {
public:
  KernelDescriptorTranslator(rj_code_arch_t src_arch, rj_code_arch_t dst_arch);

  /// @brief Translate a kernel descriptor.
  /// @param src_kd        Raw bytes of the source kernel descriptor (64 bytes).
  /// @param liveness      Liveness analysis for the kernel (provides VGPR/SGPR
  ///                      pressure and spill candidate identification).
  /// @param target_limits Hardware resource limits for the target agent,
  ///                      queried via hsa_agent_get_info or from ISA profile.
  KdTranslation translate(const uint8_t src_kd[64],
                           const LivenessAnalysis &liveness,
                           const TargetResourceLimits &limits) const;

private:
  rj_code_arch_t src_arch_;
  rj_code_arch_t dst_arch_;

  uint32_t compute_vgpr_tier(uint32_t guest_vgprs, uint32_t accvgpr_remapped,
                              const TargetResourceLimits &limits,
                              uint32_t available_lds) const;
  uint32_t compute_sgpr_tier(uint32_t guest_sgprs,
                              const TargetResourceLimits &limits) const;
  uint32_t compute_lds_overflow(uint32_t guest_lds, uint32_t lds_spill_zone,
                                 const TargetResourceLimits &limits) const;
  std::vector<uint32_t> build_user_sgpr_shuffle(
      const uint8_t src_kd[64]) const;
};

/// @brief Target hardware resource limits, queried at load time.
struct TargetResourceLimits {
  uint32_t max_vgprs_per_wf;       ///< Absolute max VGPRs (e.g., 256 for RDNA3 Wave32).
  uint32_t max_sgprs_per_wf;       ///< Absolute max SGPRs (e.g., 102 for CDNA, 106 for RDNA).
  uint32_t max_lds_per_cu;         ///< Max LDS bytes per CU (e.g., 65536).
  uint32_t max_scratch_per_thread;  ///< Max private segment bytes.
  uint32_t vgpr_granularity;        ///< Allocation block size in DWords.
  uint32_t max_waves_per_cu;        ///< Maximum concurrent wavefronts per CU.
  uint8_t  wave_size;               ///< Native wave size (32 or 64).
};

} // namespace rocjitsu
```

**Occupancy computation.** `KernelDescriptorTranslator` computes the resulting occupancy using the standard AMDGPU occupancy formula:

```
waves_limited_by_vgpr = floor(total_vgprs_per_simd / ceil(vgpr_count / granularity) / granularity)
waves_limited_by_sgpr = floor(total_sgprs_per_cu / ceil(sgpr_count / sgpr_granularity) / sgpr_granularity)
waves_limited_by_lds  = floor(max_lds_per_cu / group_segment_fixed_size)  // if lds > 0
occupancy = min(waves_limited_by_vgpr, waves_limited_by_sgpr, waves_limited_by_lds, max_waves_per_cu)
```

This is computed for both source and target to produce the `source_occupancy` and `target_occupancy` diagnostic fields.

##### 3.9.4.9 Why This Never Fails

Every resource mismatch has a fallback chain that bottoms out at a correct (if slower) transformation:

```
VGPR overflow  → reduce occupancy → spill to LDS → spill to scratch
SGPR overflow  → spill to scratch
AccVGPR absent → remap to VGPR → (VGPR overflow chain if needed)
LDS overflow   → virtualize to device memory
Scratch model  → KD fixup + prologue
User SGPR layout → prologue shuffle
Wave size      → force Wave64 (RDNA) or exec-mask shim (CDNA)
```

No branch of this chain produces `Action::Illegal`. The only theoretical failure mode would be if the guest kernel requires more total state than the target can physically address (e.g., >4GB scratch on 32-bit scratch address space), which does not arise across the CDNA/RDNA family.

#### 3.9.5 Concrete Lowering and Expansion Strategies

Every source instruction is classified as `Action::Identity`, `Action::Substitute`, `Action::Lower`, or `Action::Expand` — never `Action::Illegal`. This section specifies the concrete target-native sequence for each Lower and Expand category. `BinaryTranslator::translate_instruction()` dispatches to the appropriate `lower_*()` or `expand_*()` handler based on the instruction's properties (mnemonic prefix, encoding class, `InstFlags`).

##### 3.9.5.1 Encoding-Level Lowering (Encoding Layout Differs)

The dominant source of LOWER entries is encoding layout differences between ISA generations. The same instruction exists on both source and target with the same semantics, but the microcode field layout (bit positions, field widths, operand encoding ranges) differs. These are handled by **decode-and-re-encode**: decode the source instruction's fields using the source ISA's `{src}/machine_insts.h` bitfield struct, then construct the target instruction using the target ISA's `{dst}/machine_insts.h` bitfield struct with the same field values.

> **Field-level remap rules:** The encoding translator does not blindly copy field values. Each field is classified as COPY, REMAP, DROP, or INSERT based on cross-ISA field analysis (see §3.9.6.1). This ensures that coherency bits, scope fields, and other semantically-changed fields are correctly remapped even when the overall encoding layout appears similar.

**Categories:**
- **VOP1/VOP2/VOP3/VOPC encoding changes** (GFX9→GFX10→GFX11→GFX12): Field positions shift across generations; VOP3 gains new modifier bits; VOPC comparison encoding changes.
- **SOP1/SOP2/SOPC/SOPK encoding changes**: Opcode field widths and positions shift.
- **SMEM encoding changes**: Offset field width changes (GFX9: 20-bit; GFX10+: 21-bit); `s_load_dword` → `s_load_b32` mnemonic change already handled by rename map; encoding layout requires re-encode.
- **MUBUF/MTBUF → VBUFFER** (GFX9→GFX12): `BUFFER_LOAD_DWORD` (MUBUF encoding) → `BUFFER_LOAD_B32` (VBUFFER encoding); field layout completely restructured; requires full decode-and-re-encode.
- **FLAT/GLOBAL/SCRATCH encoding changes**: Field positions for `offset`, coherency bits (`sc0`/`sc1`/`nt` vs `glc`/`slc`/`dlc`), and address fields shift across generations.
- **DS encoding changes**: Offset fields, address/data operand positions.
- **DPP/SDWA alternate encodings**: These encoding variants exist on some ISAs but not others. When the target lacks the DPP/SDWA encoding, lower to the base VOP encoding plus explicit data permutation/masking instructions.

**Lowering procedure for encoding-level differences:**

The encoding translators are auto-generated by `amdisa/encoding_translator_codegen.py` (Phase 8.5b) using a three-layer architecture with an ISA-neutral intermediate:

1. **Neutral field structs** (`SmemFields`, `Vop3Fields`, etc.) — one per encoding format, containing the union of all semantic fields across all 9 ISAs. Generated into a shared `encoding_fields.h` header from all ISA specs at once.
2. **Decode functions** (`decode_smem_cdna4`, etc.) — extract named fields from the source ISA's `machine_insts.h` bitfield struct into the neutral struct. One per (format × source ISA). Reusable across all pairs involving that source ISA.
3. **Encode functions** (`encode_smem_rdna4`, etc.) — pack neutral struct fields into the target ISA's bitfield struct with coherency remapping. One per (format × target ISA). Reusable across all pairs targeting that ISA.
4. **Dispatch function** — composes decode + encode: `encode_smem_rdna4(decode_smem_cdna4(w0, w1), dst_op)`.

One header is generated per (src, dst) pair (e.g., `code/dbt/generated/encoding_cdna4_to_rdna4.h`). Adding a new pair reuses existing decode/encode functions and only generates the missing ones.

`BinaryTranslator` dispatches encoding translation through an ISA-agnostic function pointer (`EncodingTranslateFn`) selected at construction time based on the (guest, host) pair. No ISA-specific code appears in the translation loop:

```cpp
// In binary_translator.h:
using EncodingTranslateFn = std::function<TranslationResult(
    uint32_t encoding_id, uint32_t w0, uint32_t w1, uint32_t w2,
    uint16_t dst_op, uint8_t seg)>;

// In binary_translator.cpp — pair selection at construction:
EncodingTranslateFn select_encoding_translator(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return translate_encoding_cdna4_to_rdna4;
  return nullptr;
}
```

This is the most common lowering case (~60-80% of all LOWER entries). Same-size encodings produce exactly the same number of target instruction words (1:1). Size-changing encodings (64-bit MUBUF → 96-bit VBUFFER, 64-bit FLAT → 96-bit VFLAT/VGLOBAL/VSCRATCH) produce more words and require a code cave.

##### 3.9.5.2 Waitcnt Lowering

Covered in §3.4. Summary:
- **GFX9 `s_waitcnt` → GFX11/GFX12 split**: Decode the 16-bit `simm16` field into `{vmcnt, lgkmcnt, expcnt}` counters; emit up to 4 separate `s_wait_loadcnt`, `s_wait_storecnt`, `s_wait_kmcnt`, `s_wait_expcnt` instructions. Uses code cave (1→N expansion).
- **GFX10 `s_waitcnt` + `s_waitcnt_vscnt` → GFX11/GFX12 split**: Same plus vscnt → storecnt mapping.
- **GFX11/GFX12 split → GFX9 `s_waitcnt`**: Merge individual counter waits into a single `simm16` encoding. Multiple consecutive `s_wait_*` instructions that form a complete set are merged into one `s_waitcnt`; individual `s_wait_*` that appear alone emit an `s_waitcnt` with other counters set to "don't wait" (sentinel values).

##### 3.9.5.3 Barrier Lowering

- **`s_barrier` (monolithic, GFX9) → `s_barrier_signal` + `s_barrier_wait` (GFX11+)**: Code cave; emit two instructions.
- **`s_barrier_signal` + `s_barrier_wait` (GFX11+) → `s_barrier` (GFX9)**: When a `s_barrier_signal`/`s_barrier_wait` pair is adjacent, replace with single `s_barrier`. When they are separated (split-phase barrier), emit `s_barrier` at the `s_barrier_wait` site and `s_nop` at the `s_barrier_signal` site. Note: this changes the semantics from split-phase to full barrier — correct but potentially slower.

##### 3.9.5.4 Memory Instruction Lowering

**FLAT → GLOBAL/SCRATCH (GFX9 → GFX11+):**
- `flat_load_dword v_dst, v_addr` → `global_load_b32 v_dst, v_addr, off`
- Requires knowing pointer provenance. Conservative: if pointer origin cannot be proven global, emit `global_load_b32` with the assumption that HPC kernels only use global flat accesses. Log a warning for unproven flat addresses.
- Scratch-segment flat (`seg=2`): `flat_load_dword v_dst, v_addr [seg=2]` → `scratch_load_b32 v_dst, v_addr, off`

**MUBUF → VBUFFER (GFX9 → GFX12):**
- `BUFFER_LOAD_DWORD` (MUBUF encoding) → `BUFFER_LOAD_B32` (VBUFFER encoding)
- Decode all MUBUF fields (`offset`, `offen`, `idxen`, `glc`, `slc`, `lds`, `vaddr`, `vdata`, `srsrc`, `soffset`); re-encode into VBUFFER format with field position changes and coherency bit remapping (`glc/slc` → `sc0/sc1/nt` or `scope/th`).

**Coherency bit remapping (all memory instructions):**

| Source model | Target model | Mapping |
|---|---|---|
| GFX9 `glc` | GFX940 `sc0`/`sc1`/`nt` | `glc=1` → `sc0=1, sc1=0, nt=0` (agent-scope read-through) |
| GFX9 `glc` | GFX10 `glc`/`dlc`/`slc` | Direct mapping; `dlc` set to 0 by default |
| GFX9 `glc` | GFX11 `sc0`/`sc1`/`nt` | `glc=1` → `sc0=1, sc1=0, nt=0` |
| GFX9 `glc` | GFX12 `scope`/`th` | `glc=1` → `scope=SCOPE_SE, th=TH_LOAD_NT` |
| GFX940 `sc0`/`sc1`/`nt` | GFX9 `glc` | `sc0=1` → `glc=1`; `nt` has no GFX9 equivalent (dropped, conservative) |
| GFX12 `scope`/`th` | GFX9 `glc` | `scope>=SCOPE_SE` → `glc=1` |

##### 3.9.5.5 DPP/SDWA Lowering

DPP (Data-Parallel Primitives) and SDWA (Sub-Dword Addressing) are alternate encoding variants that add cross-lane permutation or sub-dword selection to VOP instructions. When the target ISA lacks a specific DPP/SDWA variant:

- **DPP → base VOP + explicit permute**: Decode the DPP control field (`dpp_ctrl`, `row_mask`, `bank_mask`, `bound_ctrl`). Emit a `v_mov_b32` with the DPP modifier as a preliminary permutation step, then the base VOP instruction on the permuted data. This always works because DPP operations are defined as data permutation followed by ALU.
- **SDWA → base VOP + explicit mask/shift**: Decode the SDWA `src_sel`, `dst_sel`, `dst_unused` fields. Emit `v_bfe_u32` / `v_bfi_b32` (bit-field extract/insert) instructions to perform the sub-dword selection, then the base VOP instruction, then pack the result back.

##### 3.9.5.6 Comparison Instruction Lowering (V_CMP/V_CMPX)

V_CMP and V_CMPX instructions have encoding changes across generations:
- **VOPC encoding (32-bit, GFX9) → VOP3 encoding (64-bit, GFX11+)**: Some comparison operations that had compact VOPC encodings on GFX9 require the wider VOP3 encoding on later ISAs (or vice versa). Lower by decode-and-re-encode into the target's available encoding.
- **Removed comparison predicates**: `V_CMP_F_*` (always-false) and `V_CMP_T_*` / `V_CMP_TRU_*` (always-true) are removed in GFX11+. Lower to: `s_mov_b{32,64} vcc, 0` (always-false) or `s_mov_b{32,64} vcc, exec` (always-true). For CMPX variants, `V_CMPX_F_*` → `s_mov_b{32,64} exec, 0` (kills all lanes — correct but program will halt); `V_CMPX_T_*` → no-op (exec unchanged, which is the always-true identity).

##### 3.9.5.7 V_MAD → V_FMA Lowering

`V_MAD_F32` (multiply-add without intermediate rounding) is removed in GFX11+. The replacement is `V_FMA_F32` (fused multiply-add with IEEE intermediate rounding). These are *not* bit-identical — MAD may produce different results in the last ULP for some inputs. However:

- For `V_MAD_F32` → `V_FMA_F32`: The result differs by at most 1 ULP. This is acceptable for GPU compute workloads (HIP/OpenCL do not guarantee MAD vs FMA semantics unless `-cl-mad-enable` is specified).
- `V_MAD_F16` → `V_FMA_F16`: Same 1-ULP tolerance.
- `V_MAD_U32_U24` / `V_MAD_I32_I24` → `V_MAD_U32_U24` / `V_MAD_I32_I24` (integer MAD is retained on all ISAs; no lowering needed for integer variants).
- `V_MADMK_F32` → `V_FMAMK_F32`; `V_MADAK_F32` → `V_FMAAK_F32` (literal-constant variants).

##### 3.9.5.8 Scalar Instruction Lowering

- **S_CBRANCH_CDBGSYS/S_CBRANCH_CDBGUSER/S_CBRANCH_CDBGSYS_OR_USER/S_CBRANCH_CDBGSYS_AND_USER**: Debug-conditional branches removed in GFX11+. Lower to `s_nop` (debug branches are no-ops in non-debug execution; the translator logs a warning that debug breakpoints are disabled).
- **S_ANDN2_B{32,64} → S_AND_NOT1_B{32,64}**: Already handled by rename map. When encoding also differs, decode-and-re-encode.
- **S_SETVSKIP, S_ATC_PROBE, S_ATC_PROBE_BUFFER**: Removed in GFX11+. Lower to `s_nop` (these are performance hints or debug instructions with no semantic effect on compute kernels).
- **S_MEMTIME / S_MEMREALTIME**: Removed in GFX11+. Lower to `s_getreg_b32` reading `HW_REG_SHADER_CYCLES` (GFX11+ equivalent for cycle counting).
- **S_CMPK_* (SOPK comparisons)**: Encoding layout changes across generations. Decode-and-re-encode with the target SOPK field layout.
- **S_RFE_B64**: Return from exception — removed in GFX11+. Lower to `s_nop` (not used in compute kernels).

##### 3.9.5.9 DS (Data Share / LDS) Lowering

- **DS_READ_* → DS_LOAD_*; DS_WRITE_* → DS_STORE_***: Rename already handled. Encoding layout differences are decode-and-re-encode.
- **DS_CMPST_* → DS_CMPSTORE_***: Same.
- **DS_GWS_* (Group Work Sharing)**: Removed in CDNA4 and GFX11+. GWS operations coordinate workgroups across CUs — on targets without hardware GWS, lower to a software barrier using atomics in device memory:
  - `DS_GWS_BARRIER count` → atomic counter in a pre-allocated device-memory barrier buffer. Each workgroup atomically decrements the counter; when it reaches zero, all workgroups are released via a polling loop with `s_sleep(1)`. The barrier buffer VA is injected via kernarg extension (same mechanism as §2.7).
  - `DS_GWS_INIT`, `DS_GWS_SEMA_*`: Lowered to equivalent atomic operations on the device-memory barrier buffer.
- **DS_ADD_F64, DS_MIN_F64, DS_MAX_F64, etc.**: 64-bit float LDS atomics removed in some ISAs. Lower to a CAS loop: `ds_read_b64` → compare-exchange loop using `ds_cmpstore_b64` with the computed result.
- **DS_CONDXCHG32_RTN_B64**: Lower to `ds_cmpstore_b64` equivalent.
- **DS_READ_B64_TR_B16 / DS_READ_B64_TR_B8 / DS_READ_B64_TR_B4 / DS_READ_B96_TR_B6**: CDNA4-only (GFX950) transpose load instructions that read from LDS and rearrange data for MFMA input. These have no equivalent on RDNA or earlier CDNA generations. When adjacent to an MFMA instruction, the semantic translator (§3.3.3) replaces the entire data-prep + MFMA + AccVGPR-readback sequence with the target-native equivalent (e.g., `ds_load_b128` + `v_wmma_*` on RDNA4). When standalone (no adjacent MFMA), lower to `ds_load_b64` + an explicit register permute sequence that reproduces the transpose.

##### 3.9.5.10 Image Instruction Lowering

IMAGE_SAMPLE, IMAGE_LOAD, IMAGE_STORE, IMAGE_GATHER4, and IMAGE_ATOMIC instructions have encoding changes across generations (MIMG encoding field layout, RSRC descriptor format, coherency bits). All image instructions are lowered via decode-and-re-encode of the MIMG encoding fields.

For cross-family translation (CDNA → RDNA or RDNA → CDNA), IMAGE instruction support differs:
- CDNA ISAs have limited image support (no texture filtering hardware on compute-only GPUs). IMAGE_SAMPLE instructions are rare in HPC kernels.
- RDNA ISAs have full image support including all filtering modes.
- When translating RDNA image instructions to CDNA targets that lack the hardware, lower to equivalent flat/global memory loads using the image descriptor to compute the linear address (software image addressing). This is slow but correct.

##### 3.9.5.11 V_INTERP Lowering

Vertex interpolation instructions changed significantly in GFX11+:
- **GFX9 `V_INTERP_P1_F32`/`V_INTERP_P2_F32`/`V_INTERP_MOV_F32`** (VINTRP encoding) → **GFX11+ `V_INTERP_P10_F32_INREG`/`V_INTERP_P2_F32_INREG`** (VOP3 encoding): These use entirely different encoding formats and operand conventions. Lower by decoding the VINTRP fields (attr, attrchan, vsrc) and re-encoding as VOP3 with INREG operands.
- V_INTERP instructions are rare in compute kernels (they are primarily used in pixel shaders). The translator logs a warning when encountering them.

##### 3.9.5.12 MFMA/SMFMAC Expansion (EXPAND)

Covered in §3.6. All MFMA and SMFMAC instructions on non-CDNA targets are expanded using a **tiered strategy** (cheapest first). **Tier 0 is implemented via the semantic translator (§3.3.3)**, which replaces the complete data-prep + MFMA + AccVGPR-readback sequence as a unit. This is critical because CDNA4's transpose load instructions (`ds_read_b64_tr_b16`, etc.) have no single-instruction RDNA4 equivalent — the entire data staging pipeline must be replaced alongside the matrix instruction.

**Tier 0 — Semantic-level MFMA→WMMA translation (CDNA→RDNA3/4 only):**
When the target has WMMA hardware (RDNA3/4), the semantic translator identifies MFMA instructions and their surrounding data preparation and result extraction instructions, replacing the complete idiom with a target-native WMMA sequence. The MFMA output matrix is decomposed into WMMA-sized tiles (16×16 per WMMA instruction), AccVGPR-resident accumulators are remapped to unified VGPRs, and CDNA4-specific transpose loads are replaced with standard LDS loads. The WMMA instructions run at hardware speed (~10-50× faster than software stubs). Not all MFMA shapes map cleanly to WMMA tiles — unsupported shapes fall through to Tier 1.

Compatible MFMA→WMMA mappings:

| MFMA shape | WMMA tile | Tiling | Notes |
|---|---|---|---|
| `v_mfma_f32_16x16x16_f16` | `v_wmma_f32_16x16x16_f16` | 1:1 direct | Same shape |
| `v_mfma_f32_16x16x16_bf16` | `v_wmma_f32_16x16x16_bf16` | 1:1 direct | Same shape |
| `v_mfma_f32_32x32x8_f16` | `v_wmma_f32_16x16x16_f16` | 2×2 tiles | 4 WMMA per MFMA |
| `v_mfma_i32_16x16x32_i8` | `v_wmma_i32_16x16x16_iu8` | K-split: 2 passes | Accumulate |
| `v_mfma_f32_16x16x32_fp8_*` | `v_wmma_f32_16x16x16_fp8_*` | K-split: 2 passes | RDNA4 only |

**Tier 1 — Software fallback stubs:**
For MFMA shapes with no WMMA equivalent (e.g., `v_mfma_f64_*`, `v_mfma_f32_4x4x1_*`, sparse SMFMAC), expand to software fallback stubs compiled from `shared/mfma_exec.h` reference implementations. The expansion uses the same `s_swappc_b64` trampoline mechanism as DBI (§2.4). Performance is ~100x slower than hardware.

**AccVGPR instructions** (`V_ACCVGPR_WRITE`, `V_ACCVGPR_READ`, `V_ACCVGPR_MOV_B32`) are expanded to `V_MOV_B32` with remapped register indices as described in §3.9.4.3.

##### 3.9.5.13 WMMA Expansion (EXPAND)

RDNA3/4 WMMA instructions have no direct equivalent on CDNA or RDNA1/2. Expansion uses a **tiered strategy** mirroring §3.9.5.12:

**Tier 0 — WMMA→MFMA hardware mapping (RDNA3/4→CDNA only):**
When the target has MFMA hardware (CDNA1-4), WMMA instructions with compatible shapes are lowered to MFMA sequences. AccVGPR staging is inserted (CDNA2+: `v_accvgpr_write` to load accumulators; `v_accvgpr_read` to extract results). On CDNA1 (no AccVGPR), MFMA outputs go to regular VGPRs directly.

| WMMA shape | MFMA equivalent | Notes |
|---|---|---|
| `v_wmma_f32_16x16x16_f16` | `v_mfma_f32_16x16x16_f16` | Direct; add AccVGPR staging on CDNA2+ |
| `v_wmma_f32_16x16x16_bf16` | `v_mfma_f32_16x16x16_bf16` | Direct; add AccVGPR staging on CDNA2+ |
| `v_wmma_i32_16x16x16_iu8` | `v_mfma_i32_16x16x32_i8` | K-dimension mismatch; pad or split |
| `v_wmma_f32_16x16x16_fp8_*` | `v_mfma_f32_16x16x32_fp8_*` | CDNA4 only; K-dimension mismatch |

**Tier 1 — Software fallback stubs (RDNA3/4→RDNA1/2):**
When the target has neither WMMA nor MFMA, expand using software matrix multiplication stubs with standard VOP instructions (`v_fma_f32`, `v_dot2_f32_f16`, etc.) and explicit lane shuffle via `ds_permute_b32` / `v_readlane_b32`. Performance is ~50x slower than hardware WMMA. Log a warning.

##### 3.9.5.14 Lane-Sensitive Instruction Lowering

Cross-lane operations (`v_readlane_b32`, `v_writelane_b32`, `ds_bpermute_b32`, `ds_permute_b32`, `v_readfirstlane_b32`) and wave-collective operations (ballot, reductions) have width-dependent semantics when translating between Wave32 and Wave64 execution modes.

**Wave64→Wave64 (CDNA→RDNA in Wave64 mode):** All lane operations work identically — no lowering needed. This is the primary strategy (§3.5.1).

**Wave32→Wave64 (RDNA→CDNA, exec-masking shim):** Lane operations that reference the exec mask or lane count require width-aware adaptation:

| Instruction | Wave32 behavior | Wave64 adaptation |
|---|---|---|
| `v_readlane_b32 dst, src, lane_idx` | Reads lane 0–31 | No change (upper lanes inactive via exec_hi=0 shim) |
| `ds_bpermute_b32 dst, addr, src` | Permutes within 32 lanes | No change (addr computed mod 32 by hardware) |
| `v_readfirstlane_b32 dst, src` | Reads lowest active lane in 32-bit exec | No change (exec_hi=0 ensures lowest is in [0,31]) |
| `s_wqm_b32 dst, src` | Whole-quad-mode on 32-bit mask | Lower to `s_wqm_b64` on Wave64 (or `s_wqm_b32` on exec_lo only since exec_hi=0) |
| Ballot/popcount on exec | 32-bit popcount | No change (exec_hi=0 ensures popcount(exec) == popcount(exec_lo)) |

The exec-masking shim (§3.5.3) handles the critical invariant: `exec_hi = 0` at all times. This means Wave32 lane operations produce correct results on Wave64 hardware without per-instruction adaptation, because the hardware's 64-lane operations on the upper 32 inactive lanes have no observable effect.

##### 3.9.5.15 Atomic and Memory Scope Harmonization

Memory scope semantics differ across ISA generations. The coherency bit remapping in §3.9.5.4 handles the encoding-level translation; this section covers semantic gaps where the source scope has no exact target equivalent.

**Scope model overview:**

| ISA | Scope mechanism | Available scopes |
|---|---|---|
| GFX9 (CDNA1-3) | `glc` bit (binary) | Agent-scope (glc=1) or CU-scope (glc=0) |
| GFX940 (CDNA3) | `sc0`/`sc1`/`nt` | CU, Agent, System (via sc0/sc1 combination) |
| GFX10 (RDNA1/2) | `glc`/`dlc`/`slc` | CU, SA (shader array), Agent |
| GFX11 (RDNA3, CDNA4) | `sc0`/`sc1`/`nt`/`th` | CU, SE (shader engine), Agent, System |
| GFX12 (RDNA4) | `scope` (2-bit) + `th` | CU, SE, Agent, System |

**Translation rules for scope gaps:**

- **Wider scope → narrower scope**: Always safe. GFX12 `scope=System` → GFX9 `glc=1` (agent-scope is the widest available; correct because agent-scope on a single-GPU system is equivalent to system-scope).
- **Narrower scope → wider scope**: Conservatively widen. GFX9 `glc=0` (CU-scope) → GFX12 `scope=SCOPE_CU`. No information is lost.
- **Missing scopes**: GFX9 has no SE (shader-engine) scope. GFX12 `scope=SCOPE_SE` → GFX9 `glc=1` (widen to agent-scope; conservative but correct).
- **XNACK (memory retry)**: CDNA2/3 support XNACK for GPU page fault retry. When translating to a non-XNACK target, the translator clears the `ENABLE_XNACK` bit in the kernel descriptor. Atomic operations that relied on XNACK retry semantics (e.g., atomics to pageable memory) may require a software retry loop on the target — but this is rare in practice and logged as a warning.

**Fence insertion for scope widening:**

When the target lacks a scope available on the source, the translator may need to insert explicit fence instructions to maintain ordering guarantees:

```asm
; Source (GFX12): global_load_b32 v0, v[2:3], off scope:SCOPE_SE th:TH_LOAD_NT
; Target (GFX9):  s_waitcnt vmcnt(0)          ; fence: drain prior loads
;                 global_load_dword v0, v[2:3], off glc  ; agent-scope (wider than SE)
```

The fence is only inserted when the translation widens the scope AND there are prior memory operations that could be reordered. `LivenessAnalysis` detects whether a fence is needed by checking for preceding memory instructions in the same basic block.

##### 3.9.5.16 Miscellaneous Lowering

- **EXP → EXPORT**: Rename handled; encoding differences are decode-and-re-encode.
- **BUFFER_WBL2 / BUFFER_INV / BUFFER_GL0_INV / BUFFER_GL1_INV**: Cache management instructions. Map to the target's equivalent cache invalidation instruction. When the exact equivalent doesn't exist, use the nearest conservative substitute (e.g., `BUFFER_WBL2` → `BUFFER_GL1_INV` is safe because invalidation is a superset of write-back in terms of coherency guarantees).
- **V_SWAP_B32**: Removed in some encodings. Lower to `v_mov_b32` pair with a temporary register (allocated via liveness analysis `find_free_run()`): `v_mov_b32 v_tmp, v_a; v_mov_b32 v_a, v_b; v_mov_b32 v_b, v_tmp`.
- **New RDNA4-only instructions** (V_MINIMUM/V_MAXIMUM, V_SWMMAC, DS_BVH_STACK, etc.): When translating RDNA4 → older targets, these are lowered to equivalent multi-instruction sequences. `V_MINIMUM_F32` → `V_MIN_F32` + NaN handling; `V_MAXIMUM_F32` → `V_MAX_F32` + NaN handling. BVH stack operations are lowered to explicit DS load/store sequences.

##### 3.9.5.17 Lowering Summary Table

| Category | LOWER/EXPAND | Mechanism | Cave needed? | Typical expansion ratio |
|---|---|---|---|---|
| Encoding layout differs (same instruction) | LOWER | Decode-and-re-encode | No (1:1) | 1:1 |
| Waitcnt model crossing | LOWER | `WaitcntTranslator` | Yes (1:N) | 1:2–1:5 |
| Barrier split/merge | LOWER | Pair emission / merge | Yes (1:2) or in-place (2:1) | 1:2 or 2:1 |
| FLAT → GLOBAL/SCRATCH | LOWER | Re-encode with address mode | No (1:1) | 1:1 |
| MUBUF → VBUFFER | LOWER | Decode-and-re-encode | No (1:1) | 1:1 |
| Coherency bit remap | LOWER | Field substitution | No (1:1) | 1:1 |
| DPP → base VOP + permute | LOWER | Decode + prepend permute | Yes (1:2) | 1:2 |
| SDWA → base VOP + mask/shift | LOWER | Decode + BFE/BFI | Yes (1:3–1:5) | 1:3 |
| V_CMP_F/T (removed predicates) | LOWER | `s_mov` of 0 or exec | No (1:1) | 1:1 |
| V_MAD → V_FMA | LOWER | Opcode swap (1 ULP change) | No (1:1) | 1:1 |
| DS_GWS_* (no hardware GWS) | LOWER | Software atomic barrier | Yes (1:~10) | 1:~10 |
| DS F64 atomics (removed) | LOWER | CAS loop | Yes (1:~8) | 1:~8 |
| S_MEMTIME → S_GETREG | LOWER | Instruction swap | No (1:1) | 1:1 |
| Debug branches (removed) | LOWER | `s_nop` | No (1:1) | 1:1 |
| IMAGE encoding changes | LOWER | Decode-and-re-encode | No (1:1) | 1:1 |
| V_INTERP (VINTRP→VOP3) | LOWER | Decode-and-re-encode | No (1:1) | 1:1 |
| MFMA → WMMA (CDNA→RDNA3/4) | EXPAND | Tier 0: semantic translator (§3.3.3) — replaces data-prep + MFMA + AccVGPR readback as a unit | Yes (1:4–1:8) | 1:4 |
| MFMA (no WMMA target) | EXPAND | Tier 1: software stub via trampoline | Yes (1:~100+) | 1:~100+ |
| WMMA → MFMA (RDNA3/4→CDNA) | EXPAND | Tier 0: semantic translator — MFMA hardware mapping + AccVGPR staging | Yes (1:3–1:6) | 1:4 |
| WMMA (no MFMA target) | EXPAND | Tier 1: software matrix multiply | Yes (1:~50+) | 1:~50+ |
| AccVGPR ops | EXPAND | Remap to VGPR `v_mov_b32` | No (1:1) | 1:1 |
| Lane-sensitive ops (Wave32→Wave64) | LOWER | No change (exec_hi=0 shim handles) | No (1:1) | 1:1 |
| Memory scope widening | LOWER | Scope remap + optional fence insertion | No or Yes (1:1–1:2) | 1:1 |
| Cache management (WBL2/INV) | LOWER | Map to target equivalent | No (1:1) | 1:1 |
| V_SWAP_B32 | LOWER | 3× `v_mov_b32` with temp | Yes (1:3) | 1:3 |

#### 3.9.6 Cross-ISA Semantic Execution Differences

The legalization table classifies instructions by encoding structure (IDENTITY/SUBSTITUTE/LOWER/EXPAND). However, encoding identity does NOT guarantee semantic identity. Some instructions produce different results on different ISAs despite having the same mnemonic and encoding layout. The encoding translator must account for these differences via field-level remap rules and compensating instruction insertion.

##### 3.9.6.1 Field-Level Remap Classification

The auto-generated encoding translators classify each field in each encoding format as one of:
- **COPY** — identical semantics across ISAs; copy verbatim (registers, offsets, most operand fields)
- **REMAP** — same concept, different encoding (coherency bits: `glc` → `scope/th`; documented in `shared/gfx*_cache_flags.h`)
- **DROP** — field doesn't exist on target (SDWA selectors on RDNA3+; DPP `wave_shl` modes on RDNA)
- **INSERT** — field exists on target but not source (`scope` on RDNA4, `s_delay_alu` scheduling hints)

##### 3.9.6.2 DPP (Data-Parallel Primitives) Semantic Differences

**DPP encoding split (CRITICAL):** GFX9 (CDNA1-4) uses `VOP_DPP` encoding; GFX10+ (RDNA1+) uses `DPP16` and `DPP8`. DPP8 allows arbitrary 8-lane swizzle with per-lane 3-bit indices and does not exist on GFX9.

**Wave-level DPP modes removed on RDNA (CRITICAL):** `wave_shl` (0x130), `wave_srl` (0x138), `wave_ror` (0x13C) exist on GFX9 only. Translation to RDNA must use `ds_permute_b32` or `v_permlane*` sequences.

**row_bcast removed on RDNA (CRITICAL):** `row_bcast:15` (0x142) and `row_bcast:31` (0x143) are GFX9 only. Translation must use `v_readlane_b32` + broadcast.

**row_xmask added on RDNA (GFX10+):** Not available on CDNA. Translation to CDNA requires `ds_swizzle_b32` with equivalent XOR pattern.

**DPP on VOP3/VOP3P (RDNA3+ only):** 3-source instructions can use DPP permutation on RDNA3+, with no equivalent on earlier ISAs.

**bound_ctrl semantics:** Consistent across ISAs (1=zero for OOB, 0=retain old value), but which lanes are OOB differs between Wave32 and Wave64.

##### 3.9.6.3 SDWA Removal on RDNA3+

SDWA encodings (`VOP1_VOP_SDWA`, `VOP2_VOP_SDWA`, `VOPC_VOP_SDWA_SDST_ENC`) exist on CDNA1-4 and RDNA1/2 but are removed on RDNA3/3.5/4. Translation must decompose SDWA operations:
- `dst_sel` (BYTE_0/1/2/3, WORD_0/1, DWORD) → `v_bfi_b32` / `v_bfe_u32` bit-field insert/extract
- `src_sel` → explicit shift + mask before the ALU operation
- `dst_unused` (PAD=zero extend, SEXT=sign extend, PRESERVE=merge) → post-ALU merge instruction

##### 3.9.6.4 Floating-Point Semantic Differences

**v_mad_f32 → v_fma_f32 (CRITICAL):** `v_mad_f32` (two roundings, always flushes denorms) exists only on CDNA1/2 and RDNA1. Replacement `v_fma_f32` (single rounding, respects MODE register denorm setting) produces results differing by up to 1 ULP. For iterative computations (e.g., Newton-Raphson in `v_rcp_f32` refinement), this can cascade. The translator accepts this as a known 1-ULP tolerance.

**v_mul_legacy_f32 / v_mad_legacy_f32 (CRITICAL):** DX9 "legacy" instructions treat `0 × ∞ = 0` instead of IEEE `0 × ∞ = NaN`. Renamed on RDNA3+ to `v_mul_dx9_zero_f32` / `v_fma_dx9_zero_f32`. The rename map handles the mnemonic change; the semantic behavior is preserved.

**v_min/v_max NaN semantics (CRITICAL):** Pre-RDNA4 `v_min_f32`/`v_max_f32` use IEEE 754-2008 `minNum`/`maxNum` (one NaN operand returns the number). RDNA4 splits into:
- `v_min_num_f32` / `v_max_num_f32`: same as pre-RDNA4 behavior (NaN-as-missing)
- `v_minimum_f32` / `v_maximum_f32`: IEEE 754-2019 (NaN propagates)

The rename map correctly maps `v_min_f32` → `v_min_num_f32` (same semantics). The new `v_minimum_f32` family has no pre-RDNA4 equivalent.

**Denorm handling defaults:** f32 denorms default to flush-to-zero on ALL ISAs; f16/f64 denorms default to preserve. The kernel descriptor's `FLOAT_MODE` bits control this. The `KernelDescriptorTranslator` copies `FLOAT_MODE` verbatim to preserve the source kernel's denorm behavior.

**MFMA denorm flushing (CDNA2):** On MI200 (CDNA2), MFMA and V_DOT2 instructions flush f16/bf16 input denorms regardless of the MODE register setting. This is a hardware behavior not controlled by any mode bit. When translating CDNA2 MFMA to WMMA on RDNA3/4 (which preserves denorms), results may differ for subnormal f16/bf16 inputs. Documented as a known semantic gap.

**MODE register bit layout:** The MODE register fields (`FP_ROUND`, `FP_DENORM`, `DX10_CLAMP`, `IEEE`) have consistent bit positions across CDNA and RDNA ISAs. However, RDNA1+ adds `s_denorm_mode` and `s_round_mode` instructions as shortcuts for `s_setreg_b32 MODE`. Translation from RDNA to CDNA must convert these to `s_setreg_b32`.

##### 3.9.6.5 Memory Coherency Model Differences

The coherency bit encoding evolution is covered in §3.9.5.4 (encoding-level remapping) and §3.9.5.15 (scope harmonization). This section summarizes the full semantic model for cross-reference:

**Cache flag evolution:**

| ISA | Model | Fields | Semantic mapping |
|---|---|---|---|
| GFX9 (CDNA1/2) | GLC-only | `glc` | glc=0: read-write cached; glc=1: coherent (bypass L1 for reads, write-through for writes) |
| GFX940 (CDNA3/4) | SC0/SC1/NT | `sc0`, `sc1`, `nt` | 2-bit scope (CU/Agent/System) + non-temporal hint |
| GFX10 (RDNA1/2) | GLC+DLC+SLC | `glc`, `dlc`, `slc` | glc: L1 bypass; dlc: L2 bypass; slc: system-level coherency |
| GFX11 (RDNA3/3.5) | Reinterpreted | `glc`, `dlc`, `slc` | scope = (slc<<1)\|glc; th = dlc (temporal hint) |
| GFX12 (RDNA4) | SCOPE+TH | `scope` (2-bit), `th` (3-bit) | No glc/dlc/slc fields at all |

**Remap rules for encoding translator:**
- CDNA4 `sc0=1,sc1=0` (Agent scope) → RDNA4 `scope=0x2` (SCOPE_DEV)
- CDNA4 `nt=1` (non-temporal) → RDNA4 `th=0x3` (TH_NT)
- CDNA4 `sc0=0,sc1=0` (CU scope) → RDNA4 `scope=0x0` (SCOPE_CU)

These remappings are implemented in the auto-generated encoding translators using the lookup tables already defined in `shared/gfx940_cache_flags.h` and `shared/gfx12_cache_flags.h`.

**Cache invalidation instruction mapping:**

| CDNA1/2 | CDNA3/4 | RDNA1/2/3 | RDNA4 | Semantic equivalent |
|---|---|---|---|---|
| `buffer_wbinvl1` | `buffer_inv` | `buffer_gl0_inv` | `global_inv` | Invalidate L0/L1 vector cache |
| `buffer_wbinvl1_vol` | — | `buffer_gl1_inv` | `global_wbinv` | Write-back + invalidate L1/L2 |
| — | — | `s_dcache_inv` | `s_dcache_inv` | Invalidate scalar cache |

##### 3.9.6.6 Wait Counter Semantic Differences

Covered in §3.4. Key semantic gap beyond encoding: on GFX9, `vmcnt` counts ALL vector memory operations (loads AND stores). On GFX10+, `vmcnt` counts only loads; stores have a separate `vscnt` counter. A translated `s_waitcnt vmcnt(0)` on RDNA4 must emit BOTH `s_wait_loadcnt(0)` AND `s_wait_storecnt(0)` to achieve the same fence effect.

The `WaitcntTranslator` handles this by always emitting the full set of target-native wait instructions from the decoded counter values, never passing through `s_waitcnt` as IDENTITY even when the encoding appears similar.

##### 3.9.6.7 Permutation and Lane Operation Portability

| Instruction | GFX9 (CDNA) | GFX10 (RDNA1/2) | GFX11 (RDNA3) | GFX12 (RDNA4) | Translation |
|---|---|---|---|---|---|
| `ds_permute_b32` | Yes (64 lanes) | Yes (32 or 64 lanes) | Yes | Yes | IDENTITY in Wave64 mode |
| `ds_bpermute_b32` | Yes (64 lanes) | Yes | Yes | Yes | IDENTITY in Wave64 mode |
| `ds_swizzle_b32` | Yes | Yes | Yes | Yes | IDENTITY — swizzle patterns identical |
| `v_readlane_b32` | Yes | Yes | Yes | Yes | IDENTITY — lane index interpretation same in Wave64 |
| `v_writelane_b32` | Yes | Yes | Yes | Yes | IDENTITY |
| `v_permlane16_b32` | No | Yes | Yes | Yes | CDNA→RDNA: N/A. RDNA→CDNA: LOWER to `ds_bpermute` |
| `v_permlanex16_b32` | No | Yes | Yes | Yes | Same as above |
| `v_permlane64_b32` | No | No | Yes | Yes | RDNA3+→earlier: LOWER to `ds_bpermute` |

All lane operations are semantically identical in Wave64 mode (which is forced for CDNA→RDNA translation). The `exec_hi=0` shim for RDNA→CDNA (Wave32→Wave64) ensures lane operations in the lower 32 lanes produce correct results.

##### 3.9.6.8 Semantic Override Table

A manually-curated table of instructions where encoding-identical does NOT imply semantics-identical. The legalization classifier applies these overrides after structural classification, promoting IDENTITY/SUBSTITUTE to LOWER when a known semantic difference exists:

```python
# In amdisa/legalization.py — applied during domain rule pass
SEMANTIC_OVERRIDES = {
    # Cache invalidation: same concept, different cache hierarchy targeting
    ('BUFFER_WBINVL1', 'gfx9', 'gfx10+'): 'lower',
    ('BUFFER_INV', 'gfx940', 'gfx12'): 'lower',

    # s_endpgm: cache flush behavior may differ
    ('S_ENDPGM', '*', '*'): 'lower',  # always re-encode + add compensating flush if needed

    # MODE register shortcuts: RDNA-only instructions
    ('S_DENORM_MODE', 'gfx10+', 'gfx9'): 'lower',  # convert to s_setreg_b32
    ('S_ROUND_MODE', 'gfx10+', 'gfx9'): 'lower',

    # v_mad_legacy → v_fma_dx9_zero: mnemonic rename + MAD→FMA rounding change
    ('V_MAD_LEGACY_F32', 'gfx9', 'gfx11+'): 'lower',

    # Hardware fork/join: CDNA only, no RDNA equivalent
    ('S_CBRANCH_G_FORK', 'gfx9', '*'): 'lower',
    ('S_CBRANCH_I_FORK', 'gfx9', '*'): 'lower',
    ('S_CBRANCH_JOIN', 'gfx9', '*'): 'lower',
}
```

### 3.10 C API for DBT

```c
// New additions to rj_code.h (or new header rj_code_dbt.h)

/// @brief Options controlling a DBT translation.
///
/// Passed to rj_code_translate(); all fields have safe defaults (zero = auto-detect).
typedef struct rj_code_dbt_options_t {
  /// @brief Source ISA. 0 = auto-detect from ELF e_flags.
  rj_code_arch_t guest_arch;

  /// @brief Target ISA. 0 = use detected native hardware ISA.
  rj_code_arch_t host_arch;

  /// @brief If non-zero, force Wave64 mode on the translated kernel even when the
  ///        host ISA defaults to Wave32 (i.e., set ENABLE_WAVEFRONT_SIZE32=0 in the
  ///        kernel descriptor). All RDNA ISAs support this; preferred over wave-splitting.
  uint8_t force_wave64;

  /// @brief Insert reconvergence fixups when translating to RDNA targets.
  ///
  /// RDNA3+ introduces hardware reconvergence that differs from CDNA's implicit
  /// wavefront reconvergence model. When enabled, the translator inserts
  /// s_wait_event instructions at post-divergence join points
  /// to ensure correct reconvergence semantics.
  /// (s_barrier_signal is a workgroup-barrier instruction, not a reconvergence inst.)
  /// Only required for kernels that rely on implicit reconvergence; off by default.
  uint8_t enable_reconvergence_fixups;

  /// @brief Disable the on-disk translation cache for this call.
  uint8_t disable_cache;
} rj_code_dbt_options_t;

/// @brief Translate a code object from guest_arch to host_arch.
///
/// @param[in]  source     Source code object to translate.
/// @param[in]  options    Translation options (NULL = use defaults / env-var policy).
/// @param[out] translated Newly created translated code object (refcount = 0; caller owns it).
/// @returns ROCJITSU_STATUS_SUCCESS on success.
///          ROCJITSU_STATUS_UNSUPPORTED if the translation pair is not yet implemented.
///          ROCJITSU_STATUS_ERROR on fatal translation failure.
[[nodiscard]] RJ_API_EXPORT rj_status_t
rj_code_translate(const rj_code_object_t *source,
                  const rj_code_dbt_options_t *options,
                  rj_code_object_t **translated);
```

The `rj_code_dbt_options_t` struct is also used internally by `RjHsaLayer` (Pillar 5) to carry the env-var policy into `BinaryTranslator`, so both the C API and the internal pipeline share the same options representation.

---

## Pillar 5: ROCm Integration Hooks {#pillar-5}

### 4.1 Motivation and Design Rationale

DBI and DBT need to intercept real application code objects before the GPU executes them. ROCR provides two **first-class instrumentation mechanisms** that do not require LD_PRELOAD symbol shadowing for HSA calls:

**Mechanism 1 — HSA Tools Layer (`HSA_TOOLS_LIB`).** ROCR's `hsa_init()` checks the `HSA_TOOLS_LIB` environment variable and, if set, `dlopen`s the named shared library and calls `OnLoad(HsaApiTable *table, ...)`. `OnLoad` receives a live pointer to the `CoreApiTable` and `AmdExtTable`, allowing the tool to:

1. Save the original function pointers.
2. Replace specific entries (e.g., `hsa_executable_load_agent_code_object_fn`, `hsa_queue_create_fn`) with wrappers that run the DBI/DBT pipeline.
3. Call the saved originals from inside the wrapper — a true call-chain layer with no symbol conflicts and no `dlsym(RTLD_NEXT, ...)` fragility.

The key structs are defined in `hsa_api_trace.h` (ROCR source tree):
- `HsaApiTable` — top-level container; `core_` points to `CoreApiTable`; `amd_ext_` points to `AmdExtTable`
- `CoreApiTable` — contains `hsa_executable_load_agent_code_object_fn`, `hsa_queue_create_fn`, `hsa_code_object_reader_create_from_memory_fn`, etc.
- `AmdExtTable` — contains `hsa_amd_queue_intercept_create_fn`, `hsa_amd_queue_intercept_register_fn`

**Mechanism 2 — Intercept Queue (`hsa_amd_queue_intercept_create` / `hsa_amd_queue_intercept_register`).** The AMD extension API allows any hardware queue to be wrapped as an "intercept queue." Every AQL packet written to the queue passes through registered handler callbacks before being submitted to hardware. Handlers can read, modify, or replace packets — including swapping the `kernel_object` field in `hsa_kernel_dispatch_packet_t` to point to a translated/instrumented kernel descriptor, or replacing `kernarg_address` to inject instrumentation arguments.

For DBI/DBT the two mechanisms are **complementary and layered**:

| Mechanism | When | Primary use |
|---|---|---|
| `CoreApiTable::hsa_executable_load_agent_code_object_fn` override | Code object load time | Run DBI/DBT pipeline on the ELF; store patched ELF via new `hsa_code_object_reader_t`; register GPU VA → patched KD mappings |
| `CoreApiTable::hsa_code_object_reader_create_from_memory_fn` override | Reader creation | Stash ELF bytes for the code-object reader registry |
| `CoreApiTable::hsa_queue_create_fn` override | Queue creation | Transparently wrap every queue as an intercept queue |
| Intercept queue packet handler | Kernel dispatch time | Inspect dispatch parameters; swap `kernel_object` to translated KD if lazy translation needed; log per-dispatch metrics |

The load-time hook is the **primary DBI/DBT path** — the patched ELF is what ROCR loads into the GPU address space, so the translation happens exactly once per code object regardless of how many kernels are dispatched. The dispatch-time hook adds observability and enables **lazy per-kernel translation** for cases where load-time is too early (e.g., JIT-compiled code objects or when only certain kernels need translation).

**Future path — full loader replacement.** Because the HSA Tools Layer gives rocjitsu full control of every `hsa_executable_*` and `hsa_queue_*` function pointer, it also provides the natural transition point to eventually replace ROCR's entire executable/loader implementation with rocjitsu's own `code/` layer. See §4.8 for the replacement design.

### 4.2 Policy and Environment Variables

```cpp
// lib/rocjitsu/include/rocjitsu/code/rj_translation_policy.h

namespace rocjitsu {

/// @brief Governs what rocjitsu does to each code object at load time.
enum class RjTranslationPolicy {
  /// @brief Pass code objects through unchanged.
  PASSTHROUGH,
  /// @brief Run DBI instrumentation only; no ISA translation.
  INSTRUMENT_ONLY,
  /// @brief Run DBT translation only; no instrumentation.
  TRANSLATE_ONLY,
  /// @brief Run both DBI and DBT (translate first, then instrument).
  INSTRUMENT_AND_TRANSLATE,
};

/// @brief Parse policy from environment variable RJ_DBI_POLICY.
RjTranslationPolicy policy_from_env();

/// @brief Parse source ISA from RJ_DBT_SOURCE_ISA (e.g., "cdna3", "gfx942").
/// Returns ROCJITSU_CODE_ARCH_INVALID if not set.
rj_code_arch_t source_isa_from_env();

/// @brief Parse target ISA from RJ_DBT_TARGET_ISA.
/// Returns ROCJITSU_CODE_ARCH_INVALID if not set (means "use hardware native ISA").
rj_code_arch_t target_isa_from_env();

} // namespace rocjitsu
```

Environment variable semantics:

| Variable | Values | Effect |
|---|---|---|
| `HSA_TOOLS_LIB` | `librocjitsu_hooks.so` | Load the rocjitsu HSA tools layer; required for any DBI/DBT interception |
| `RJ_DBI_POLICY` | `passthrough`, `instrument`, `translate`, `both` | Controls the pipeline mode |
| `RJ_DBT_SOURCE_ISA` | `cdna1`, `cdna2`, `cdna3`, `cdna4`, `gfx908`, `gfx90a`, `gfx942`, `gfx950` | Override source ISA detection |
| `RJ_DBT_TARGET_ISA` | Same values | Force target ISA (default: detected from hardware) |
| `RJ_DBI_FILTER` | `all`, `memory`, `branch` | Filter which instructions are instrumented |
| `RJ_DBT_CACHE_DIR` | path | Override the disk translation cache location |
| `RJ_DBT_DISABLE_CACHE` | `1` | Disable disk cache (always translate) |
| `RJ_DBT_LOG` | `0`-`3` | Verbosity level |

### 4.3 HSA Interception Implementation

#### 4.3.1 Tools Layer Entry Points

rocjitsu ships a dedicated shared library `librocjitsu_hooks.so`. ROCR calls its `OnLoad` / `OnUnload` functions during `hsa_init()` / `hsa_shut_down()`:

```cpp
// New file: lib/rocjitsu/src/rocjitsu/hooks/rj_hsa_layer.cpp

/// @brief Called by ROCR when HSA_TOOLS_LIB=librocjitsu_hooks.so.
///        Receives a live pointer to ROCR's CoreApiTable and AmdExtTable.
extern "C" bool OnLoad(HsaApiTable *table,
                       uint64_t runtime_version,
                       uint64_t failed_tool_count,
                       const char *const *failed_tool_names) {
  return rocjitsu::RjHsaLayer::instance().init(table);
}

/// @brief Called by ROCR during hsa_shut_down().
extern "C" void OnUnload() {
  rocjitsu::RjHsaLayer::instance().shutdown();
}
```

#### 4.3.2 `RjHsaLayer` — CoreApiTable Replacement

```cpp
// lib/rocjitsu/src/rocjitsu/hooks/rj_hsa_layer.h

namespace rocjitsu {

/// @brief Singleton that holds the saved CoreApiTable function pointers and
///        installs rocjitsu wrappers in their place.
class RjHsaLayer {
public:
  static RjHsaLayer &instance();

  bool init(HsaApiTable *table);
  void shutdown();

  // Saved originals — called by our wrappers after the pipeline runs.
  hsa_status_t (*real_load_agent_code_object)(
      hsa_executable_t, hsa_agent_t,
      hsa_code_object_reader_t, const char *,
      hsa_loaded_code_object_t *) = nullptr;

  hsa_status_t (*real_reader_create_from_memory)(
      const void *, size_t, hsa_code_object_reader_t *) = nullptr;

  hsa_status_t (*real_queue_create)(
      hsa_agent_t, uint32_t, hsa_queue_type32_t,
      void (*)(hsa_status_t, hsa_queue_t *, void *), void *,
      uint32_t, uint32_t, hsa_queue_t **) = nullptr;

  hsa_status_t (*real_reader_destroy)(hsa_code_object_reader_t) = nullptr;
  hsa_status_t (*real_queue_destroy)(hsa_queue_t *) = nullptr;
  hsa_status_t (*real_executable_destroy)(hsa_executable_t) = nullptr;

  // AMD extension table — needed to create intercept queues.
  AmdExtTable *amd_ext_ = nullptr;

private:
  RjHsaLayer() = default;
  RjHsaLayer(const RjHsaLayer &) = delete;
  RjHsaLayer &operator=(const RjHsaLayer &) = delete;
  RjHsaLayer(RjHsaLayer &&) = delete;
  RjHsaLayer &operator=(RjHsaLayer &&) = delete;

  std::atomic<bool> active_{false};  ///< Written in init(), read from multiple interposer threads.
};

} // namespace rocjitsu
```

```cpp
// rj_hsa_layer.cpp — init()

bool RjHsaLayer::init(HsaApiTable *table) {
  auto policy = policy_from_env();
  if (policy == RjTranslationPolicy::PASSTHROUGH) return true;

  CoreApiTable *core = table->core_;
  amd_ext_ = table->amd_ext_;

  // Save originals.
  real_load_agent_code_object  = core->hsa_executable_load_agent_code_object_fn;
  real_reader_create_from_memory = core->hsa_code_object_reader_create_from_memory_fn;
  real_queue_create            = core->hsa_queue_create_fn;

  // Install wrappers.
  core->hsa_executable_load_agent_code_object_fn       = rj_load_agent_code_object;
  core->hsa_code_object_reader_create_from_memory_fn   = rj_reader_create_from_memory;
  core->hsa_code_object_reader_destroy_fn              = rj_reader_destroy;
  core->hsa_queue_create_fn                            = rj_queue_create;
  core->hsa_queue_destroy_fn                           = rj_queue_destroy;
  core->hsa_executable_destroy_fn                      = rj_executable_destroy;

  active_ = true;
  return true;
}
```

#### 4.3.2a `run_rj_pipeline` — Top-Level ELF Pipeline

```cpp
// lib/rocjitsu/src/rocjitsu/hooks/rj_hsa_layer.cpp (internal helper)

/// @brief Run the DBI and/or DBT pipeline on a raw ELF image.
///
/// Depending on @p policy:
///   - PASSTHROUGH: returns empty vector immediately.
///   - INSTRUMENT_ONLY: runs DBI instrumentation; returns patched ELF.
///   - TRANSLATE_ONLY: runs DBT translation; returns translated ELF.
///   - INSTRUMENT_AND_TRANSLATE: translates first, then instruments; returns final ELF.
///
/// If translation cache contains a valid entry for (src_arch, dst_arch, hash), the
/// cached ELF is returned without invoking BinaryTranslator.
///
/// @returns Patched ELF bytes, or empty vector if the pipeline was skipped or failed.
static std::vector<uint8_t> run_rj_pipeline(
    const uint8_t *elf_bytes,
    size_t elf_size,
    rj_code_arch_t src_arch,
    rj_code_arch_t dst_arch,
    RjTranslationPolicy policy);
```

#### 4.3.3 Load-Time ELF Hook — `hsa_executable_load_agent_code_object`

```cpp
// Wrapper installed in CoreApiTable::hsa_executable_load_agent_code_object_fn

static hsa_status_t rj_load_agent_code_object(
    hsa_executable_t executable,
    hsa_agent_t agent,
    hsa_code_object_reader_t code_object_reader,
    const char *options,
    hsa_loaded_code_object_t *loaded_code_object)
{
  auto &layer = RjHsaLayer::instance();
  auto policy = policy_from_env();

  // Look up the ELF bytes stashed when the reader was created.
  const uint8_t *elf_bytes = nullptr;
  size_t elf_size = 0;
  CodeObjectReaderRegistry::instance().lookup(code_object_reader, &elf_bytes, &elf_size);

  if (!elf_bytes || elf_size == 0)
    return layer.real_load_agent_code_object(
        executable, agent, code_object_reader, options, loaded_code_object);

  // Detect ISA from ELF flags or env override.
  rj_code_arch_t src_arch = source_isa_from_env();
  rj_code_arch_t dst_arch = target_isa_from_env();
  if (src_arch == ROCJITSU_CODE_ARCH_INVALID)
    src_arch = detect_arch_from_elf(elf_bytes, elf_size);
  if (dst_arch == ROCJITSU_CODE_ARCH_INVALID)
    dst_arch = detect_native_arch_from_agent(agent);

  // Run DBT and/or DBI pipeline (may be a cache hit → fast path).
  std::vector<uint8_t> patched = run_rj_pipeline(elf_bytes, elf_size,
                                                  src_arch, dst_arch, policy);
  if (patched.empty())
    return layer.real_load_agent_code_object(
        executable, agent, code_object_reader, options, loaded_code_object);

  // Create a new reader from the patched bytes; stash so the buffer survives
  // the lifetime of the executable.
  auto owned = std::make_unique<std::vector<uint8_t>>(std::move(patched));
  hsa_code_object_reader_t patched_reader{};
  hsa_status_t s = layer.real_reader_create_from_memory(
      owned->data(), owned->size(), &patched_reader);
  if (s != HSA_STATUS_SUCCESS) goto passthrough;  // owned released by unique_ptr dtor

  s = layer.real_load_agent_code_object(
      executable, agent, patched_reader, options, loaded_code_object);
  if (s != HSA_STATUS_SUCCESS) {
    // Clean up the reader we created; owned is still held by unique_ptr.
    layer.real_reader_destroy(patched_reader);
    goto passthrough;
  }

  // Success: transfer ownership to the registry via std::move (RAII — no raw release()).
  const uint8_t *data_ptr = owned->data();
  const size_t   data_sz  = owned->size();
  CodeObjectReaderRegistry::instance().store(
      patched_reader, data_ptr, data_sz, std::move(owned));
  return s;

passthrough:
  return layer.real_load_agent_code_object(
      executable, agent, code_object_reader, options, loaded_code_object);
}

/// @brief Stash ELF bytes for every reader so the load-time hook can access them.
static hsa_status_t rj_reader_create_from_memory(
    const void *code_object, size_t size,
    hsa_code_object_reader_t *out_reader)
{
  auto &layer = RjHsaLayer::instance();
  hsa_status_t s = layer.real_reader_create_from_memory(code_object, size, out_reader);
  if (s == HSA_STATUS_SUCCESS)
    CodeObjectReaderRegistry::instance().store(
        *out_reader,
        reinterpret_cast<const uint8_t *>(code_object), size,
        /*owned=*/nullptr);
  return s;
}
```

#### 4.3.4 Dispatch-Time Hook — Intercept Queue

Every queue is transparently created as an intercept queue. The packet handler performs two mutations per dispatch packet:

1. **KD swap** — if the kernel was translated or instrumented at load time, swap `kernel_object` to the patched KD GPU VA.
2. **Kernarg extension** — if the kernel was instrumented by `BufferInjector`, extend the kernarg segment with the logging buffer VA for this dispatch.

**Why kernarg extension must be dispatch-time:** The logging buffer VA cannot be baked into the ELF at load time. It is only known at dispatch submission, and must be unique per concurrent dispatch — a static VA shared across concurrent dispatches would cause them to overwrite each other's output. The extension is a `memcpy` of the original kernargs plus an 8-byte append; no ELF knowledge is needed at dispatch time.

**Scratch memory and the scratch ring — a correctness requirement at queue creation.** The GPU determines per-wave scratch from `private_segment_fixed_size` in the kernel descriptor. The scratch ring buffer is sized at queue creation time from the `private_segment_size` argument to `hsa_queue_create`. If `rj_queue_create` passes the application's original value through unchanged — and many kernels declare `private_segment_fixed_size = 0` — the scratch ring has no headroom. When the dispatch-time handler then swaps `kernel_object` to a patched KD whose `private_segment_fixed_size` was increased by `SpillManager`, the GPU addresses scratch beyond the end of the ring: **memory corruption**.

The fix is in `rj_queue_create`: inflate `private_segment_size` by the maximum additional private segment bytes that any patched kernel on this agent will need. Since all code objects on this agent have been loaded before queues are created in the typical HSA program flow, `KernelDescriptorRegistry` already has this information at queue creation time via `DispatchInfo::additional_private_segment_bytes`. `rj_queue_create` queries the registry for the maximum and adds it to the application's requested size. This is a one-time cost at queue creation; the scratch ring is thereafter large enough for any KD we swap to.

This also enables **selective per-dispatch instrumentation**: the dispatch-time handler can choose between the original KD (no probe overhead, original scratch) and the patched KD (probe active, larger scratch) based on a policy. Since the scratch ring accommodates the larger size, both choices are safe.

**What dispatch-time injection does NOT help with:**
- **Trampolining:** Probe VAs are known when the probe ELF is merged at load time. Nothing dispatch-time is needed.
- **Translation:** The KD swap is the entirety of what dispatch-time needs for DBT. No kernarg changes are required for translated kernels.

**Per-queue `KernargExtensionPool`:** Kernarg extension buffers are pre-allocated at queue creation time in HSA fine-grained system memory (host + device accessible). The pool holds `queue_size` slots each of `max_kernarg_size + 8` bytes. Slots are consumed in round-robin order indexed by `next_slot_ % slot_count`. Correctness constraint: a slot must not be reused while the dispatch that filled it is still in-flight. **The AQL ring depth alone is insufficient as a safety bound.** An AQL packet slot is freed by the CP when the packet is consumed (i.e., when the dispatch is *queued* to hardware), not when the kernel *finishes*. A queue with depth `queue_size` can therefore have `queue_size` kernels in-flight simultaneously, all of whose kernarg slots must remain valid concurrently. The `slot_count == queue_size` sizing is safe only if the hardware's maximum concurrent in-flight dispatches does not exceed `queue_size`; on hardware with deep dispatch queues this may not hold. Safer sizing: set `slot_count` to `num_cu * max_waves_per_cu * avg_waves_per_dispatch` rounded up to a power of two. For typical HPC kernels with high occupancy, `slot_count = queue_size * 4` provides adequate headroom with negligible memory overhead. If any patched kernel uses more than `max_kernarg_size` bytes, `max_kernarg_size` must be sized to the maximum across all registered kernels (tracked by `KernelDescriptorRegistry`); at queue creation this maximum is queried and used to size slots.

```cpp
// lib/rocjitsu/src/rocjitsu/hooks/kernarg_extension_pool.h

namespace rocjitsu {

/// @brief Per-queue pool of pre-allocated kernarg extension buffers.
///
/// Allocated in HSA fine-grained system memory so both host (handler) and
/// device (kernel SGPR load) can access the same physical pages.
/// Slot size = max_kernarg_size + 8 (8 bytes for the buffer VA extension).
class KernargExtensionPool {
public:
  /// @pre @p queue_size must be a power of two (enforced by the constructor via
  ///      `assert((queue_size & (queue_size - 1)) == 0)`), as the round-robin slot
  ///      index uses bitwise AND (`next_slot_ & (slot_count_ - 1)`) for correctness.
  KernargExtensionPool(hsa_agent_t agent, uint32_t queue_size,
                       uint32_t max_kernarg_size);
  ~KernargExtensionPool();

  /// @brief Acquire the next slot. Returns its host-accessible VA.
  /// Thread-safe; uses an atomic index for round-robin slot selection.
  void *acquire_slot();

  /// @brief GPU VA of a slot (for writing into the AQL packet's kernarg_address).
  uint64_t gpu_va(const void *host_ptr) const;

private:
  void    *pool_host_va_;   ///< Base host VA (fine-grained system memory).
  uint64_t pool_gpu_va_;    ///< Corresponding GPU VA.
  uint32_t slot_size_;      ///< Bytes per slot.
  uint32_t slot_count_;
  std::atomic<uint32_t> next_slot_{0};
};

/// @brief Singleton registry mapping queue handles to their KernargExtensionPool.
///
/// Populated in rj_queue_create(); cleared in rj_queue_destroy(). Dispatch-time
/// packet handlers receive a raw pool pointer as user_data (stable for the queue
/// lifetime), so they do NOT need to consult this registry on the hot path.
/// The registry's sole purpose is lifetime management: it ensures the pool is
/// freed exactly once when the queue is destroyed.
///
/// Thread-safe: multiple queues may be created/destroyed concurrently; accesses
/// are protected by shared_mutex (shared for reads, exclusive for writes).
class KernargPoolRegistry {
public:
  static KernargPoolRegistry &instance();

  /// @brief Take ownership of @p pool for the given @p queue_handle.
  void store(uint64_t queue_handle, std::unique_ptr<KernargExtensionPool> pool);

  /// @brief Release and destroy the pool for @p queue_handle (called on queue destroy).
  void erase(uint64_t queue_handle);

private:
  KernargPoolRegistry() = default;
  KernargPoolRegistry(const KernargPoolRegistry &) = delete;
  KernargPoolRegistry &operator=(const KernargPoolRegistry &) = delete;
  KernargPoolRegistry(KernargPoolRegistry &&) = delete;
  KernargPoolRegistry &operator=(KernargPoolRegistry &&) = delete;

  mutable std::shared_mutex mutex_;
  std::unordered_map<uint64_t, std::unique_ptr<KernargExtensionPool>> map_;
};

} // namespace rocjitsu
```

```cpp
// Wrapper installed in CoreApiTable::hsa_queue_create_fn.

static hsa_status_t rj_queue_create(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*err_cb)(hsa_status_t, hsa_queue_t *, void *), void *err_data,
    uint32_t private_segment_size, uint32_t group_segment_size,
    hsa_queue_t **queue)
{
  auto &layer = RjHsaLayer::instance();

  // Inflate private_segment_size by the maximum additional scratch any patched
  // kernel on this agent will use. This ensures the scratch ring is large
  // enough for dispatch-time KD swaps to patched KDs whose
  // private_segment_fixed_size was grown by SpillManager.
  //
  // Ordering note: HIP creates queues during hsa_init() BEFORE any code objects
  // are loaded. At that point KernelDescriptorRegistry has no entries and
  // max_additional_private_segment_bytes() returns RJ_SCRATCH_HEADROOM_BYTES
  // (default: 256 bytes). This is a conservative fallback — it allows DBI to work
  // for kernels whose spill zone is ≤ 256 bytes. For kernels needing more scratch,
  // the application must create queues after loading code objects, or the user must
  // set RJ_SCRATCH_HEADROOM_BYTES to a larger value via the environment.
  //
  // This is a one-time cost; the scratch ring is thereafter large enough for
  // any KD swap the dispatch-time handler performs on this queue.
  //
  // Default fallback: RJ_SCRATCH_HEADROOM_BYTES defaults to 65,536 bytes
  // (= 256 VGPRs × 4 bytes × 64 lanes), covering the worst-case VGPR spill zone
  // for kernels loaded after queue creation. AccVGPR-heavy kernels on CDNA2+ may
  // require up to 131,072 bytes; set RJ_SCRATCH_HEADROOM_BYTES accordingly.
  uint32_t max_extra = KernelDescriptorRegistry::instance()
                           .max_additional_private_segment_bytes(agent);
  // Guard against uint32_t overflow before adding scratch headroom.
  // Use a runtime check (not assert) so release builds are also protected.
  if (private_segment_size > std::numeric_limits<uint32_t>::max() - max_extra) {
    // Extremely unlikely: the sum of the kernel's existing scratch and the DBI
    // headroom would exceed 4 GB. Return a hard error rather than silently
    // truncating the queue's scratch ring.
    return HSA_STATUS_ERROR;
  }
  uint32_t inflated_private_segment_size = private_segment_size + max_extra;

  hsa_status_t s = layer.amd_ext_->hsa_amd_queue_intercept_create_fn(
      agent, size, type, err_cb, err_data,
      inflated_private_segment_size, group_segment_size, queue);
  if (s != HSA_STATUS_SUCCESS) return s;

  // Allocate the extension pool. Ownership is held in KernargPoolRegistry
  // (an unordered_map<uint64_t, unique_ptr<KernargExtensionPool>> keyed by
  // queue->handle), which is cleared when the hsa_queue_destroy wrapper fires.
  // Using a registry — rather than relying on a "not shown" destructor — ensures
  // the pool is freed even if hsa_queue_destroy interposition is never added.
  auto owned_pool = std::make_unique<KernargExtensionPool>(agent, size, MAX_KERNARG_BYTES);
  auto *pool_ptr  = owned_pool.get();
  hsa_status_t reg_s = layer.amd_ext_->hsa_amd_queue_intercept_register_fn(
      *queue, rj_packet_handler, /*user_data=*/pool_ptr);
  if (reg_s != HSA_STATUS_SUCCESS) return reg_s;  // owned_pool freed by unique_ptr
  KernargPoolRegistry::instance().store((*queue)->handle, std::move(owned_pool));
  return HSA_STATUS_SUCCESS;
}

// Wrapper installed in CoreApiTable::hsa_queue_destroy_fn.
static hsa_status_t rj_queue_destroy(hsa_queue_t *queue)
{
  if (queue)
    KernargPoolRegistry::instance().erase(queue->handle);  // frees KernargExtensionPool
  return RjHsaLayer::instance().real_queue_destroy(queue);
}
```

```cpp
/// @brief Packet handler: receives AQL packets before they reach hardware.
///        Signature matches hsa_amd_queue_intercept_handler.
static void rj_packet_handler(
    const void *pkts,
    uint64_t pkt_count,
    uint64_t user_pkt_index,
    void *data,
    hsa_amd_queue_intercept_packet_writer writer)
{
  auto *pool = static_cast<KernargExtensionPool *>(data);
  assert(pool != nullptr && "rj_packet_handler: null user_data — KernargExtensionPool not registered");

  // Work on a mutable copy — pkts may be read-only ring buffer memory.
  std::vector<uint8_t> buf(pkt_count * sizeof(hsa_kernel_dispatch_packet_t));
  std::memcpy(buf.data(), pkts, buf.size());
  auto *packets = reinterpret_cast<hsa_kernel_dispatch_packet_t *>(buf.data());

  for (uint64_t i = 0; i < pkt_count; ++i) {
    uint16_t pkt_type = (packets[i].header >> HSA_PACKET_HEADER_TYPE)
                        & ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1);
    if (pkt_type != HSA_PACKET_TYPE_KERNEL_DISPATCH) continue;

    uint64_t kd_va = packets[i].kernel_object;
    const auto *info = KernelDescriptorRegistry::instance().lookup(kd_va);
    if (!info) continue;  // Not rocjitsu-managed; pass through unchanged.

    // Step 1: swap kernel_object to the patched/translated KD VA if needed.
    // Because rj_queue_create inflated private_segment_size at queue creation,
    // the scratch ring is large enough for the patched KD's larger
    // private_segment_fixed_size. This also enables selective instrumentation:
    // leave kernel_object unchanged here to skip instrumentation for this
    // dispatch (uses original KD + original scratch, no probe overhead) while
    // the scratch ring safely accommodates either choice.
    if (info->patched_kd_va != 0)
      packets[i].kernel_object = info->patched_kd_va;

    // Step 2: extend the kernarg segment with the logging buffer VA.
    // Only required for kernels patched by BufferInjector; buffer_va_kernarg_offset
    // is 0 for translated-only kernels.
    if (info->buffer_va_kernarg_offset != 0) {
      void *slot = pool->acquire_slot();

      // Copy original kernargs verbatim.
      std::memcpy(slot,
                  reinterpret_cast<const void *>(packets[i].kernarg_address),
                  info->original_kernarg_size);

      // Append the logging buffer VA in the reserved 8-byte slot.
      // LoggingBufferRegistry provides the per-kernel output buffer GPU VA,
      // allocated once at instrumentation time and reused across dispatches.
      uint64_t buf_va = LoggingBufferRegistry::instance().buffer_for(kd_va);
      std::memcpy(static_cast<uint8_t *>(slot) + info->buffer_va_kernarg_offset,
                  &buf_va, sizeof(buf_va));

      // Redirect the packet to the extended kernarg buffer.
      packets[i].kernarg_address = pool->gpu_va(slot);
    }
  }

  writer(buf.data(), pkt_count);
}
```

**`KernelDescriptorRegistry`** — upgraded from a simple VA→VA map to a per-KD dispatch metadata store:

```cpp
// lib/rocjitsu/src/rocjitsu/hooks/kernel_descriptor_registry.h

namespace rocjitsu {

/// @brief Per-kernel-descriptor metadata needed at dispatch time.
struct DispatchInfo {
  /// @brief Patched KD GPU VA (translated or instrumented). 0 = not patched.
  uint64_t patched_kd_va = 0;

  /// @brief HSA agent this kernel belongs to.
  /// Used by max_additional_private_segment_bytes() to filter per-agent.
  hsa_agent_t agent{};

  /// @brief How many bytes SpillManager added to private_segment_fixed_size
  /// in the patched KD relative to the original. Used by rj_queue_create to
  /// compute the scratch ring inflation needed for safe dispatch-time KD swaps.
  /// 0 when no scratch growth was required (translation-only, no spill slots).
  uint32_t additional_private_segment_bytes = 0;

  /// @brief Original (pre-extension) kernarg segment size in bytes.
  /// 0 when BufferInjector was not applied to this kernel.
  uint32_t original_kernarg_size = 0;

  /// @brief Byte offset in the extended kernarg segment where the logging
  /// buffer VA is written. Equal to original_kernarg_size when non-zero.
  /// 0 when BufferInjector was not applied to this kernel.
  uint32_t buffer_va_kernarg_offset = 0;
};

/// @brief Thread-safe registry mapping original KD GPU VAs → DispatchInfo.
///
/// Written once per kernel at load time; read many times at dispatch time.
/// Uses shared_mutex so concurrent dispatches read without blocking each other.
class KernelDescriptorRegistry {
public:
  static KernelDescriptorRegistry &instance();

  void register_kernel(uint64_t original_kd_va, DispatchInfo info);

  /// @brief Returns nullptr if kd_va is not a rocjitsu-managed kernel.
  [[nodiscard]] const DispatchInfo *lookup(uint64_t kd_va) const;

  /// @brief Maximum additional_private_segment_bytes across all kernels on
  /// @p agent. Called by rj_queue_create to compute the scratch ring inflation.
  /// Returns RJ_SCRATCH_HEADROOM_BYTES if no kernels are registered for this
  /// agent (conservative fallback for code objects loaded after queue creation).
  uint32_t max_additional_private_segment_bytes(hsa_agent_t agent) const;

private:
  KernelDescriptorRegistry() = default;
  KernelDescriptorRegistry(const KernelDescriptorRegistry &) = delete;
  KernelDescriptorRegistry &operator=(const KernelDescriptorRegistry &) = delete;
  KernelDescriptorRegistry(KernelDescriptorRegistry &&) = delete;
  KernelDescriptorRegistry &operator=(KernelDescriptorRegistry &&) = delete;

  mutable std::shared_mutex mutex_;
  std::unordered_map<uint64_t, DispatchInfo> map_;
};

} // namespace rocjitsu
```

The handler is a zero-cost pass-through for any packet whose `kernel_object` is not in the registry. For managed kernels the critical path is: one `shared_lock` + hash lookup, one optional pointer swap, one optional `memcpy` pair — no dynamic allocation on the dispatch hot path.

> **Note:** The `amd_aql_intercept_marker_t` vendor packet can be inserted into the queue to receive a callback when the packet immediately following it is placed on the underlying hardware queue — useful for GPU timeline correlation in future profiling work.

### 4.4 `CodeObjectReaderRegistry`

Populated by the `rj_reader_create_from_memory` wrapper (§4.3.3) every time ROCR creates a code object reader. Because `hsa_code_object_reader_t` is an opaque handle with no public API to recover the underlying bytes, this registry is the only way for the load-time hook to access the ELF. The `owned_buffer` field is non-null when the registry holds a patched-ELF buffer that must outlive the executable.

```cpp
// lib/rocjitsu/src/rocjitsu/code/patch/code_object_reader_registry.h

namespace rocjitsu {

/// @brief Thread-safe registry that maps hsa_code_object_reader_t handles
///        to the underlying ELF byte buffers.
///
/// Populated by the RjHsaLayer's hsa_code_object_reader_create_from_memory
/// wrapper (§4.3.3). Thread-safe; multiple HSA threads may create readers
/// concurrently.
class CodeObjectReaderRegistry {
public:
  static CodeObjectReaderRegistry &instance();

  /// @param owned_buffer  If non-null, the registry takes ownership and frees it on remove().
  ///                      Pass nullptr for byte buffers owned externally (e.g., the HSA reader's
  ///                      original ELF — the reader's lifetime guarantees the bytes are live).
  void store(hsa_code_object_reader_t reader,
             const uint8_t *bytes,
             size_t size,
             std::unique_ptr<std::vector<uint8_t>> owned_buffer);

  [[nodiscard]] bool lookup(hsa_code_object_reader_t reader,
                            const uint8_t **elf_bytes,
                            size_t *elf_size) const;

  /// @brief Remove a reader from the registry (called on reader_destroy interposition).
  void remove(hsa_code_object_reader_t reader);

private:
  CodeObjectReaderRegistry() = default;
  CodeObjectReaderRegistry(const CodeObjectReaderRegistry &) = delete;
  CodeObjectReaderRegistry &operator=(const CodeObjectReaderRegistry &) = delete;
  CodeObjectReaderRegistry(CodeObjectReaderRegistry &&) = delete;
  CodeObjectReaderRegistry &operator=(CodeObjectReaderRegistry &&) = delete;

  struct Entry {
    const uint8_t *bytes;
    size_t size;
    std::unique_ptr<std::vector<uint8_t>> owned_buffer;  ///< Non-null if registry owns the memory; freed on remove().
  };

  mutable std::shared_mutex mutex_;
  std::unordered_map<uint64_t, Entry> map_;  ///< keyed by reader.handle
};

} // namespace rocjitsu
```

### 4.5 Detecting the Native ISA from an HSA Agent

```cpp
// lib/rocjitsu/src/rocjitsu/code/rj_arch_detect.h

namespace rocjitsu {

/// @brief Detect the ISA architecture of an HSA agent.
///
/// Uses hsa_agent_get_info(HSA_AGENT_INFO_ISA) and parses the ISA name string
/// (e.g., "amdgcn-amd-amdhsa--gfx942") to extract the GFX target.
rj_code_arch_t detect_native_arch_from_agent(hsa_agent_t agent);

/// @brief Detect the ISA of an ELF by reading e_flags (EF_AMDGPU_MACH field).
rj_code_arch_t detect_arch_from_elf(const uint8_t *elf_bytes, size_t size);

/// @brief Map a GFX target string ("gfx942", "gfx950", etc.) to rj_code_arch_t.
rj_code_arch_t arch_from_gfx_string(std::string_view gfx);

} // namespace rocjitsu
```

Implementation uses the `EF_AMDGPU_MACH` field and constants already defined in `code/amdgpu_elf.h` (e.g., `EF_AMDGPU_MACH_AMDGCN_GFX942 = 0x4C`, `EF_AMDGPU_MACH_AMDGCN_GFX950 = 0x4F`) — no new ELF parsing code is needed:

```cpp
rj_code_arch_t detect_arch_from_elf(const uint8_t *bytes, size_t size) {
  if (size < sizeof(Elf64_Ehdr)) return ROCJITSU_CODE_ARCH_INVALID;
  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(bytes);
  uint32_t mach = ehdr->e_flags & EF_AMDGPU_MACH;
  switch (mach) {
    // CDNA family
    case 0x30: return ROCJITSU_CODE_ARCH_CDNA1;  // GFX908 (MI100)
    case 0x3f: return ROCJITSU_CODE_ARCH_CDNA2;  // GFX90A (MI200)
    case 0x40: return ROCJITSU_CODE_ARCH_CDNA3;  // GFX940
    case 0x4b: return ROCJITSU_CODE_ARCH_CDNA3;  // GFX941
    case 0x4c: return ROCJITSU_CODE_ARCH_CDNA3;  // GFX942 (MI300X)
    case 0x4f: return ROCJITSU_CODE_ARCH_CDNA4;  // GFX950 (MI350)
    // RDNA1 family (GFX10.1)
    case 0x33: return ROCJITSU_CODE_ARCH_RDNA1;  // GFX1010
    case 0x34: return ROCJITSU_CODE_ARCH_RDNA1;  // GFX1011
    case 0x35: return ROCJITSU_CODE_ARCH_RDNA1;  // GFX1012
    // RDNA2 family (GFX10.3)
    case 0x36: return ROCJITSU_CODE_ARCH_RDNA2;  // GFX1030
    case 0x37: return ROCJITSU_CODE_ARCH_RDNA2;  // GFX1031
    case 0x38: return ROCJITSU_CODE_ARCH_RDNA2;  // GFX1032
    case 0x39: return ROCJITSU_CODE_ARCH_RDNA2;  // GFX1033
    case 0x3e: return ROCJITSU_CODE_ARCH_RDNA2;  // GFX1034
    case 0x3d: return ROCJITSU_CODE_ARCH_RDNA2;  // GFX1035
    // RDNA3 family (GFX11)
    case 0x41: return ROCJITSU_CODE_ARCH_RDNA3;  // GFX1100
    case 0x46: return ROCJITSU_CODE_ARCH_RDNA3;  // GFX1101
    case 0x47: return ROCJITSU_CODE_ARCH_RDNA3;  // GFX1102
    case 0x44: return ROCJITSU_CODE_ARCH_RDNA3;  // GFX1103
    // RDNA3.5 family (GFX11.5)
    case 0x43: return ROCJITSU_CODE_ARCH_RDNA3_5; // GFX1150
    case 0x4a: return ROCJITSU_CODE_ARCH_RDNA3_5; // GFX1151
    // RDNA4 family (GFX12)
    case 0x48: return ROCJITSU_CODE_ARCH_RDNA4;  // GFX1200
    case 0x4e: return ROCJITSU_CODE_ARCH_RDNA4;  // GFX1201
    default:   return ROCJITSU_CODE_ARCH_INVALID;
  }
}
```

### 4.6 Disk Translation Cache Integration

```cpp
// In the translate_elf() pipeline:
auto key = dbt::TranslationKey{src_arch, dst_arch,
                                dbt::TranslationCache::hash_code_object(obj)};
auto &cache = dbt::TranslationCache::instance();

if (auto cached_path = cache.lookup(key)) {
  // Load cached ELF from disk.
  std::ifstream f(*cached_path, std::ios::binary);
  return {std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {}), dst_arch, {}, {}};
}

// Translate and store.
auto result = translator.translate(obj);
if (result.errors.empty())
  cache.store(key, result.elf_bytes);
return result;
```

Cache file format: raw ELF bytes, with a 16-byte header prepended for validation:

```
[0:7]   uint64_t magic = 0x524A444254434143ULL  ("RJDBTCAC" as big-endian ASCII; on a
                         little-endian host the bytes in memory read 43 41 43 54 42 44 4A 52
                         = "CACTBDJR" when hexdumped — use memcmp on a byte array rather than
                         a uint64_t comparison to avoid endianness confusion in validators)
[8:11]  uint32_t src_arch
[12:15] uint32_t dst_arch
[16...]  ELF bytes
```

### 4.7 `SimulatedDriver` Integration

For the simulation path (where rocjitsu acts as a full simulator rather than a runtime shim), the translation pipeline hooks into `SimulatedDriver::alloc_memory_ioctl` / `create_queue_ioctl`. When the simulated driver receives an ELF to load (via `KFD_IOC_ALLOC_MEMORY_OF_GPU` with the program code object flag), it passes the ELF through the same `translate_elf` + `instrument_elf` pipeline before committing it to the simulated GPU memory:

```cpp
// In simulated_driver.cpp, map_memory_ioctl():
if (flags & KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE) {
  auto policy = rocjitsu::policy_from_env();
  if (policy != RjTranslationPolicy::PASSTHROUGH) {
    // Run pipeline on the ELF bytes at host_ptr.
    auto patched = run_rj_pipeline(host_ptr, size, src_arch, dst_arch, policy);
    if (!patched.empty()) {
      // Safety: the patched ELF may be larger than the original (new ELF sections added).
      // The allocation at host_ptr must be grown before the memcpy if patched.size() > size.
      // In the simulated driver, KFD_IOC_ALLOC_MEMORY_OF_GPU maps a full GPU VA range; the
      // caller must verify the mapping covers patched.size() bytes, or re-map if needed.
      assert(patched.size() <= mapped_size);  // mapped_size = the full allocation size (>= size)
      std::memcpy(host_ptr, patched.data(), patched.size());
      size = patched.size();  // update tracked size to reflect the grown ELF
    }
  }
}
```

### 4.8 Path to Full Loader Replacement

The HSA Tools Layer is also the mechanism by which rocjitsu can eventually **replace ROCR's entire executable and loader implementation** with its own `code/` layer (`Executable`, `AmdGpuCodeObject`, etc.), without forking or patching ROCR. Because `OnLoad` grants ownership of every function pointer in `CoreApiTable`, the replacement is a table-surgery operation, not a library fork.

The transition proceeds in three stages:

**Stage 1 (current plan) — Intercept and patch.**
Override only `hsa_executable_load_agent_code_object_fn`, `hsa_code_object_reader_create_from_memory_fn`, and `hsa_queue_create_fn`. ROCR still owns executable lifecycle, symbol lookup, and loading; rocjitsu only intercepts the ELF bytes and dispatch packets.

**Stage 2 — Own the executable layer.**
Replace `hsa_executable_create_alt_fn`, `hsa_executable_freeze_fn`, `hsa_executable_get_symbol_by_name_fn`, `hsa_executable_iterate_symbols_fn`, and `hsa_executable_load_agent_code_object_fn` with a rocjitsu implementation backed by `code/executable.h`. ROCR's loader is bypassed; rocjitsu's `Executable::load()` handles ELF parsing, kernel descriptor extraction, and GPU memory placement. The `hsa_executable_t` handle becomes a rocjitsu object.

**Stage 3 — Own the full code object lifecycle.**
Replace all remaining `hsa_code_object_*` and `hsa_executable_*` function pointers. ROCR becomes a thin HAL providing only signal, queue, and memory primitives. rocjitsu's `code/` layer handles all loader responsibilities.

**Design constraint:** All stages must maintain binary compatibility with existing ROCm applications. The `hsa_executable_t` and `hsa_code_object_reader_t` handle types remain ROCR opaque types until Stage 3; rocjitsu uses parallel internal registries to associate them with its own objects.

**Why this path is sound:**
`HsaApiTable` is a stable ABI: ROCR increments `ApiTableVersion::minor_id` on additive changes and `major_id` on breaking ones. Stage 1 can validate the version in `OnLoad` and refuse to intercept on incompatible major versions, falling back to PASSTHROUGH safely.

```cpp
// In RjHsaLayer::init():
if (table->core_->version.major_id != EXPECTED_CORE_TABLE_MAJOR) {
  fprintf(stderr, "[rocjitsu] CoreApiTable major version mismatch — PASSTHROUGH\n");
  return true;  // don't install wrappers; let ROCR run unmodified
}
```

The `code/executable.h` implementation is the immediate beneficiary of the DBI/DBT work (it already parses fat binaries and extracts per-target code objects), making Stage 2 a natural follow-on once the tools layer is established.

---

## Source Attribution: Mapping Patched Instructions to Original Source Lines {#source-attribution}

### 5.1 Overview and Feasibility

GPU code objects compiled with `-g` (or `--generate-line-info`) embed standard DWARF debug information, specifically a `.debug_line` section whose line number program (LNP) maps instruction PCs to `(file, line, column)` tuples. The DWARF line number program format is identical to that used for CPU code — only the PC values differ (they are GPU virtual addresses or section-relative offsets). **No external libdwarf dependency is needed**: the LNP state machine interpreter is self-contained and ~150 lines of C++.

There are three distinct scenarios in the DBI/DBT pipeline, each with a different PC correctness situation:

| Scenario | PC shifts? | DWARF usable as-is? | Fix required |
|---|---|---|---|
| DBT — code cave stub in `.text` | No (§3.3.2 invariant) | Yes, for stub offset | None |
| DBT — expansion body in `.rj_translations` | N/A (new section) | No (no coverage) | Optional synthetic DWARF |
| DBI — trampoline inserted into `.text` | Yes — all subsequent PCs shift | No, after first insertion | `PatchOffsetMap` reverse translation |
| No `-g` (no DWARF) | — | No `.debug_line` exists | Symbol-level attribution fallback |

### 5.2 `DwarfLineTable` — Parsing `.debug_line`

The DWARF line number program produces a sorted matrix of rows: `{address, file_index, line, column, is_stmt, end_sequence}`. The interpreter advances a state machine using standard, special, and extended opcodes to emit rows; `DW_LNE_end_sequence` rows delimit ranges and are excluded from the lookup table.

```cpp
// lib/rocjitsu/src/rocjitsu/code/dwarf_line_table.h
#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu {

struct SourceLocation {
    std::string file_path;       // path from the compilation directory
    uint32_t    line   = 0;      // 1-based
    uint32_t    column = 0;      // 1-based; 0 = unknown
    bool        is_stmt = false; // recommended statement boundary
};

class DwarfLineTable {
public:
    /// Parse all .debug_line programs from a raw AMDGPU ELF image.
    /// Returns nullptr if no .debug_line section exists or parsing fails.
    static std::unique_ptr<DwarfLineTable> parse(const uint8_t *elf_data,
                                                 size_t elf_size);

    /// Translate a .text section-relative byte offset to a source location.
    /// Uses lower_bound on the sorted row table and returns the last row whose
    /// address <= offset — the standard DWARF "closest preceding" convention.
    /// Returns nullopt only if the table is empty or offset precedes all rows.
    std::optional<SourceLocation> lookup(uint64_t text_offset) const;

    bool   empty()     const { return rows_.empty(); }
    size_t row_count() const { return rows_.size(); }

private:
    struct Row {
        uint64_t address  = 0;
        uint32_t file_idx = 0;   // index into file_paths_
        uint32_t line     = 0;
        uint32_t column   = 0;
        bool     is_stmt  = false;
    };

    std::vector<std::string> file_paths_;  // built from LNP header file_names table
    std::vector<Row>         rows_;        // sorted ascending by address after parse
};

} // namespace rocjitsu
```

**Address convention.** In AMDGPU executable ELFs (the objects ROCR loads), `.debug_line` addresses are absolute VAs matching the `.text` section's `sh_addr`. `DwarfLineTable::lookup` takes a section-relative offset; the caller normalizes: `text_offset = abs_va - co.text_section_va()`. `BasicBlock::start_offset()` and `Instruction::byte_offset()` already return section-relative offsets, so no conversion is needed in the common path.

**Parser state machine.** For each line number program in `.debug_line`:
1. Read the LNP header: `unit_length`, `version`, `header_length`, `minimum_instruction_length`, `default_is_stmt`, `line_base`, `line_range`, `opcode_base`, `standard_opcode_lengths[]`, `include_directories[]`, `file_names[]`.
2. Initialize state: `{address=0, file=1, line=1, column=0, is_stmt=default_is_stmt}`.
3. Dispatch on each byte — standard opcodes (`DW_LNS_advance_pc`, `DW_LNS_advance_line`, `DW_LNS_copy`, `DW_LNS_set_file`, `DW_LNS_set_column`, `DW_LNS_const_add_pc`, `DW_LNS_fixed_advance_pc`), extended opcodes (`DW_LNE_set_address`, `DW_LNE_end_sequence`, `DW_LNE_define_file`), or special opcodes: `adjusted = op - opcode_base`; `address += (adjusted / line_range) * min_inst_length`; `line += line_base + (adjusted % line_range)`; emit row, clear `basic_block`.
4. `DW_LNE_end_sequence`: emit row marked `end_sequence = true`, reset state, do not add to output table.

After all programs are processed, sort `rows_` by `address`. The `.debug_line` section may contain one program per compilation unit; all programs are merged into the single sorted table.

**DWARF version handling.** LLVM HIP emits DWARF v4 or v5. The v5 LNP header adds `address_size` (1 byte at offset 6) and `segment_selector_size` (1 byte at offset 7) before `header_length`, and replaces the null-terminated `include_directories` / `file_names` lists with typed entry-format tables using `DW_LNCT_path`, `DW_LNCT_directory_index`, etc. The parser branches on the `version` field (bytes 4–5 of each LNP header) to handle both formats. v2 and v3 use the same layout as v4. DWARF v1 and split DWARF (`.dwo`) are not emitted by the HIP toolchain and can be treated as parse errors.

### 5.3 PC Correctness Under DBI — `PatchOffsetMap`

When `CodeObjectPatcher` inserts trampoline bytes into `.text`, all instructions after the insertion point shift forward by the trampoline size. The original `.debug_line` addresses remain unchanged and become stale for every instruction that follows. The fix is a side-table that translates any patched offset back to its original offset, after which the unchanged DWARF is queried correctly.

`CodeObjectPatcher` accumulates edits as it patches and exposes the reverse mapping:

```cpp
// Addition to CodeObjectPatcher (§2.6):
struct PatchEdit {
    uint64_t original_offset;  // byte offset in the unpatched .text
    int64_t  size_delta;       // > 0: insertion; < 0: deletion; 0: in-place overwrite
};

// Accumulated in original_offset order as edits are applied:
std::vector<PatchEdit> edits_;

/// Translate a patched-binary .text offset back to the original .text offset.
/// Subtracts the cumulative size_delta of all edits whose original_offset
/// precedes the query point. O(log n) via prefix-sum table built on first call.
/// The prefix-sum table is lazily initialized using std::once_flag so that
/// concurrent calls from multiple drain-callback threads are safe without an
/// explicit external lock.
uint64_t original_offset_of(uint64_t patched_offset) const;
```

**Usage:** Any report that stores a `patched_text_offset` — most importantly `LogRecord::inst_byte_offset` — passes it through `original_offset_of()` before calling `DwarfLineTable::lookup()`. The `LogRecord` always stores the patched offset (that is what the GPU actually executed); the host-side drain callback performs the reverse translation for human-readable output.

The `PatchOffsetMap` is per-kernel (one per `CodeObjectPatcher` invocation). It lives alongside the `LogBuffer` in `LoggingBufferRegistry`, accessed by the drain callback that enriches log records.

**For DBT code caves, no `PatchOffsetMap` is needed.** The code cave invariant (§3.3.2) guarantees that no instruction in `.text` ever changes its byte offset — the stub sits at the same offset as the original instruction it replaced. `DwarfLineTable::lookup(stub_offset)` returns the correct source location with no fixup.

### 5.4 Fallback When DWARF Is Absent

Binaries compiled without `-g` (production HIP libraries, most ROCm system libraries) have no `.debug_line`. `DwarfLineTable::parse()` returns `nullptr`. Attribution falls back to symbol granularity via `.symtab`:

```cpp
struct SymbolAttribution {
    std::string kernel_name;               // demangled symbol name
    uint64_t    kernel_start_offset;       // .text section-relative
    uint64_t    inst_offset_within_kernel; // offset - kernel_start_offset
};

/// Find the STT_FUNC symbol in .symtab whose [st_value, st_value + st_size)
/// interval contains text_offset. AmdGpuCodeObject already provides raw
/// .symtab access via amdgpu_elf.h; this is a linear scan over STT_FUNC entries.
std::optional<SymbolAttribution> attribute_by_symbol(
    const AmdGpuCodeObject &co, uint64_t text_offset);
```

This yields `"kernel_foo+0x1a0"` — coarser than source lines but unambiguous, zero-config, and always available (every kernel has a `.symtab` entry by definition). `AmdGpuCodeObject::kernel_descriptor_offset()` already iterates `.symtab` symbols, so the infrastructure is present.

### 5.5 Optional Synthetic DWARF for `.rj_translations`

If the original ELF had DWARF, the expansion body in `.rj_translations` executes at an offset with no DWARF coverage. This surfaces in debuggers as an unmapped address. `BinaryTranslator` already records `{cave_body_offset → original_text_offset}` in the `TranslationSiteMap`; `CodeObjectPatcher` can use this to emit a minimal synthetic `.debug_line` program for `.rj_translations`:

- One compilation unit header covering the entire `.rj_translations` section.
- For each cave body at `cave_body_start`:
  - `DW_LNE_set_address(rj_translations_va + cave_body_start)`
  - `DW_LNS_set_file` / `DW_LNS_advance_line` / `DW_LNS_set_column` from the source location of the original instruction, looked up via `DwarfLineTable`.
  - `DW_LNS_copy` to emit the row.
- `DW_LNE_end_sequence` at end.

`TranslationSiteMap` holds the `{cave_body_offset → original_text_offset}` mapping:

```cpp
// lib/rocjitsu/src/rocjitsu/code/dbt/translation_site_map.h

namespace rocjitsu {

/// @brief Maps cave body byte offsets (in .rj_translations) back to the
/// original .text byte offset of the instruction that was replaced.
///
/// Populated by BinaryTranslator::translate() for every LOWER (size-expanding)
/// and EXPAND instruction. Used by:
///   - CodeObjectPatcher to emit optional synthetic .debug_line for cave bodies.
///   - SourceAttributor to attribute cave-body log records to source lines.
class TranslationSiteMap {
public:
  /// @brief Record a mapping from a cave body start offset to an original text offset.
  void add(uint64_t cave_body_offset, uint64_t original_text_offset);

  /// @brief Look up the original text offset for a given cave body offset.
  /// Returns nullopt if offset is not the start of a known cave body.
  std::optional<uint64_t> original_offset_of(uint64_t cave_body_offset) const;

  /// @brief Iterate all mappings: for synthetic DWARF emission.
  const std::vector<std::pair<uint64_t, uint64_t>> &entries() const;

private:
  std::vector<std::pair<uint64_t, uint64_t>> entries_;  ///< (cave_offset, text_offset)
};

} // namespace rocjitsu
```

This is **optional** and gated on `RJ_EMIT_CAVE_DWARF=1`. Defer until the core pipeline is working. Its primary value is allowing `rocgdb` to step through the expansion body and attribute the time to the correct source line rather than showing it as unmapped.

### 5.6 `SourceAttributor` — Unified Interface

The DWARF, `PatchOffsetMap`, and symbol fallback mechanisms are composed behind a single class so no caller needs to know which is available:

```cpp
// lib/rocjitsu/src/rocjitsu/code/source_attributor.h
#pragma once
#include "rocjitsu/code/dwarf_line_table.h"
#include <memory>
#include <optional>
#include <string>

namespace rocjitsu {

class AmdGpuCodeObject;
class CodeObjectPatcher;  // provides PatchOffsetMap; may be nullptr for DBT

class SourceAttributor {
public:
    /// Build from an original (unpatched) code object.
    /// patcher is optional: if provided, offsets are reverse-translated through
    /// PatchOffsetMap before DWARF lookup; if nullptr, offsets are used directly
    /// (correct for DBT code caves and for offline inspection of unpatched ELFs).
    static std::unique_ptr<SourceAttributor> create(
        const AmdGpuCodeObject  &original_co,
        const CodeObjectPatcher *patcher = nullptr);

    /// Returns "foo.hip:42:8" (DWARF present) or "kernel_foo+0x1a0" (fallback).
    /// Returns nullopt only if the code object has neither DWARF nor symbols.
    std::optional<std::string> attribute(uint64_t patched_offset) const;

    /// Same as attribute() but returns structured data for programmatic use.
    std::optional<SourceLocation> lookup(uint64_t patched_offset) const;

    bool has_dwarf() const { return dwarf_ && !dwarf_->empty(); }

private:
    const AmdGpuCodeObject  *co_      = nullptr;
    const CodeObjectPatcher *patcher_ = nullptr;
    std::unique_ptr<DwarfLineTable> dwarf_;
};

} // namespace rocjitsu
```

### 5.7 Integration Points

| Where | How |
|---|---|
| `LogBuffer::drain()` callback | Drain receives `LogRecord::inst_byte_offset` (patched offset). `SourceAttributor::attribute(record.inst_byte_offset)` enriches each record. `LoggingBufferRegistry` stores a `SourceAttributor` per kernel alongside the `LogBuffer`, created during load-time patching. |
| `BinaryTranslator` output | After translation, `SourceAttributor::create(original_co, nullptr)` is stored in `TranslationCache` alongside the translated ELF so attribution survives process restarts via the on-disk cache. The `TranslationSiteMap` provides the cave body → original offset mapping for the drain callback. |
| `rj_code_inst_t` C API | A future `rj_code_inst_source_location()` wraps `SourceAttributor::lookup()` for offline inspection tools and annotated disassembly printers. |
| Disassembly output | Any `rj_code_inst_disassemble()` caller can prefix each line with the attribution string, gated on `RJ_SHOW_SOURCE=1`. |

---

## Implementation Roadmap {#roadmap}

Each phase produces a self-contained, compilable, tested changeset. A phase is complete only when its tests pass. Phases marked **[parallel]** have no ordering constraint relative to other phases at the same level and can be developed concurrently.

---

### Phase 0: InstFlags Extensions and BasicBlock Signature (1–2 days)

| Task | Files |
|---|---|
| Add `EXEC_MODIFY` flag to `InstFlags`; annotate SOPP/SOP1/VOP* instructions that write EXEC. (`INDIRECT_BRANCH` and `MEMORY_OP` flags already exist in `InstFlags`.) | `instruction.h`, all 9 ISA files |
| Verify `INDIRECT_BRANCH` is correctly annotated on `s_setpc_b64` / `s_swappc_b64` in all 9 ISA decoder outputs (flag already defined) | Verify in ISA instruction subclasses |
| Add entry-leader-aware `BasicBlock::build()` overload so kernel descriptor entry offsets start blocks | `basic_block.h/.cpp` |
| Add RDNA1/2/3/3.5 `EF_AMDGPU_MACH` constants to `amdgpu_elf.h` (currently only GFX942 and GFX950 are defined) | `code/amdgpu_elf.h` |

> **Design note:** `BasicBlock::build()` adds CFG edges in a second pass: after constructing all blocks it populates `successors_` and `predecessors_` on each block using the virtual `Instruction::branch_offset_bytes()` and an `offset → BasicBlock*` map. `basic_block.h` is hand-written and safe to modify. No separate `ControlFlowGraph` class is needed — the returned block list is the implicit CFG.

> **Existing infrastructure note:** The `GpuIsa` concept in `isa_traits.h` already includes `MAX_ACC_VGPRS_PER_WF`, and derived concepts `HasAccVgpr` and `HasMonolithicWaitcnt` are already defined. The `Instruction` class uses intrusive list membership via `IListNode<Instruction, IListParent<BasicBlock>>` which provides parent-block access — the plan's `parent_` pointer addition should leverage this existing mechanism rather than adding a redundant field. All 9 GPU ISAs are already enumerated in `rj_code_arch_t`.

**Tests:** Existing CTest suite passes unchanged. Spot-check: decode a CDNA3 or RDNA3 kernel (decoders for all 9 ISAs are available as of Phase A); assert at least one instruction has `EXEC_MODIFY` set; assert `s_setpc_b64` has `INDIRECT_BRANCH` set.

---

### Phase 1: RegisterRef, RegisterSet, and `to_register_ref` (3–5 days)

| Task | Files |
|---|---|
| `RegClass` enum, `RegisterRef` struct, `RegisterSet` class with `expand()` / `erase()` / `contains()` | `isa/register_set.h` |
| Add `virtual to_register_ref()` to `Operand` base | `isa/operand.h` |
| Extend codegen to emit `to_register_ref()` override body in `{isa}/operand.cpp` using `OpSel*` ranges from `operand_types.h` | `{isa}/operand.cpp` (autogenerated section) |
| Add `virtual implicit_defs/uses(RegisterSet &)` to `Instruction` base; encoding classes override it for hidden ordinary-register effects such as FLAT `saddr` | `isa/instruction.h`, `{isa}/encodings.h/.cpp` (autogenerated section) |
| `InstDefUse(inst)` constructor iterating `dst_operands_`/`src_operands_` + calling both implicit virtuals | `analysis/def_use_chain.h/.cpp` |

**Current status:** Implemented for ordinary SGPR, VGPR, and AccVGPR liveness, including hidden FLAT `saddr` ordinary SGPR uses. Special-register liveness (`EXEC`, `VCC`, `SCC`, `M0`, `TTMP`, `FLAT_SCRATCH`) is not implemented.

**Tests:** For all 9 ISAs (decoders available as of Phase A; prioritize CDNA3, CDNA4, RDNA3, RDNA4): call `to_register_ref()` on every ordinary register `OperandType` variant and assert the returned `RegisterRef` is stable across identical calls; assert two operands referencing the same register return equal `RegisterRef`; assert VGPR and SGPR with the same `encoding_value` return `RegisterRef` with distinct `cls`; verify `RegisterSet` union/intersection; verify `InstDefUse(inst)` produces non-empty def sets for VGPR-writing instructions and non-empty use sets for SGPR-reading instructions; verify implicit FLAT `saddr` uses. For CDNA ISAs also verify AccVGPR operands map to a distinct `RegClass`. Special-register tests are deferred until special-register liveness is implemented.

---

### Phase 2: CFG Edge Population in `BasicBlock::build()` (3–5 days)

**Depends on:** Phase 0

| Task | Modified Files |
|---|---|
| Add `successors_` and `predecessors_` (`std::vector<BasicBlock*>`) to `BasicBlock` with accessors | `code/basic_block.h` |
| Add `virtual std::optional<int64_t> branch_offset_bytes() const` to `Instruction` base; codegen emits override body in ISA subclasses for branch instructions | `isa/instruction.h`, autogenerated ISA files |
| Add second pass to `BasicBlock::build()`: build `offset → BasicBlock*` map, call `terminator->branch_offset_bytes()` to compute target offset, link successor/predecessor pairs; mark `INDIRECT_BRANCH` blocks as exits | `code/basic_block.cpp` |

No new files. No `ControlFlowGraph` class — the block list returned by `build()` is the implicit CFG.

**Tests:** Decode a CDNA3 kernel containing a `for` loop and an `if`/`else`. Assert: correct block count; loop back-edge present (`bb->successors()` contains an earlier block); `if`-head has exactly two successors; exit block has no successors; every predecessor list is the inverse of the successor lists.

---

### Phase 3: LivenessAnalysis (1 week)

**Depends on:** Phase 1, Phase 2

| Task | New Files |
|---|---|
| `LivenessAnalysis` backward dataflow over `KernelBlockScope`: `live_in(block)`, `live_out(block)`, `live_before(inst)` | `analysis/liveness.h/.cpp` |

No public C API additions — `LivenessAnalysis` is consumed internally by `SpillManager`, `Instrumentor`, and `BinaryTranslator`.

**Current status:** Implemented for ordinary SGPR, VGPR, and AccVGPR dataflow over a kernel-scoped CFG. DBT builds one `KernelBlockScope` per kernel descriptor entry and ignores edges leaving that scope.

**Tests:** Decode a CDNA3 matmul kernel. Assert dead SGPRs after their last use are absent from `live_before()`. For an instruction that reads and writes the same register — e.g., `s_add_u32 s0, s0, s1` — assert `live_before(inst)` contains `s0` and `s1` (the uses), not just the definition; this validates the standard transfer formula: `before = (after \ defs) ∪ uses`. Assert implicit FLAT `saddr` SGPR dependencies are live before the FLAT instruction. EXEC/VCC/SCC/M0/FLAT_SCRATCH/TTMP liveness tests are deferred until special-register liveness is implemented.

---

### Phase 4: CodeObjectPatcher — ELF Rewriter (1 week) [parallel with Phases 1–3] ✅ DONE

**Depends on:** `amdgpu_elf.h` only (no analysis dependency)

| Task | New Files |
|---|---|
| `CodeObjectPatcher`: insert/overwrite bytes in `.text`; fix all `s_branch`/`s_cbranch`/`s_setpc` offsets after insert; update `.symtab` symbol sizes; update section headers; re-emit valid ELF | `patch/code_object_patcher.h/.cpp` |

**Tests:** Load a compiled CDNA3 kernel ELF. Insert a 4-byte NOP at three distinct offsets (before a branch, after a branch target, at end of kernel). After each insertion: verify `readelf -a` reports no broken sections; verify all branch offsets in the patched text section are correct; verify symbol start addresses (`st_value`) in `.symtab` are correctly adjusted for symbols whose start is beyond the insertion point; verify the enclosing function symbol's `st_size` grows by 4 to reflect the inserted bytes.

---

### Phase 5: SpillManager (3–5 days) ✅

**Depends on:** Phase 1, Phase 4

| Task | New Files |
|---|---|
| `SpillManager`: allocate non-overlapping flat-scratch slots for a given `RegisterSet`; compute per-slot GPU VA using `addr_calc.h` formulas; bump `private_segment_fixed_size` in the kernel descriptor | `patch/spill_manager.h/.cpp` |

**Tests:** Create a synthetic `RegisterSet` containing 4 SGPRs and 2 VGPRs. Assert: 6 non-overlapping slots; SGPR slots are SGPR-sized (4 bytes each; an SGPR pair occupies two consecutive slots, 8 bytes total); VGPR slots are 4 bytes each in per-lane scratch (the hardware replicates per-lane scratch across all lanes automatically; no wave_size factor is applied to the per-lane slot size); flat scratch address for wave 0 / thread 0 / slot 0 matches the formula in `addr_calc.h` exactly. Assert `spill_mgr.total_private_bytes() == 24` (4 SGPRs × 4 bytes + 2 VGPRs × 4 bytes); this value feeds `private_segment_fixed_size` in the kernel descriptor patch.

---

### Phase 6: TrampolineBuilder and BufferInjector (1 week)

**Depends on:** Phase 5

| Task | New Files |
|---|---|
| `TrampolineBuilder`: emit spill-before/restore-after ISA sequences bracketing an `s_swappc_b64`; generate the trampoline code object stub; use `SpillManager` for slot assignment | `patch/trampoline_builder.h/.cpp` |
| `BufferInjector`: add a pointer-sized kernarg entry to the kernel descriptor for the logging buffer; adjust `kernarg_size` | `patch/buffer_injector.h/.cpp` |

**Tests:** Generate a trampoline for a patch point with a 4-SGPR / 2-VGPR live set. Disassemble with `rocm-dis` or objdump. Assert: spill count matches live register count; restore sequence is the symmetric inverse; `s_swappc_b64` is between them; no live register is referenced between spill and restore without being reloaded. Verify `BufferInjector` correctly increments `kernarg_size` by 8.

---

### Phase 7: Instrumentor and DBI C API (1 week)

**Depends on:** Phase 6

| Task | New Files |
|---|---|
| `Instrumentor`: orchestrates LivenessAnalysis + SpillManager + TrampolineBuilder + CodeObjectPatcher per probe point; merges probe code object into output ELF | `patch/instrumentor.h/.cpp` |
| `rj_code_instrumentor_create/destroy`, `rj_code_instrumentor_add_probe`, `rj_code_instrumentor_patch` C API | `rj_code.h` additions |
| `instrument_elf()` top-level function (ELF in → instrumented ELF out) | `patch/instrumentor.cpp` |
| `LogDrainThread`: background host thread sleeping in `hsa_signal_wait_scacquire`; wakes on GPU signal or 5 ms timeout; calls `LogBuffer::drain()` for each registered buffer | `hooks/log_drain_thread.h/.cpp` |
| `LoggingBufferRegistry`: maps original KD GPU VAs → `LogBuffer` instances; populated at instrumentation time; queried at dispatch time and by drain thread | `hooks/logging_buffer_registry.h/.cpp` |
| `ProbeLibrary`: wrapper around embedded probe code object byte arrays; maps `rj_code_arch_t` → `AmdGpuCodeObject*` | `patch/probe_library.h/.cpp` |

**Tests:** *(DBI milestone)* Instrument a CDNA3 `vector_add` kernel to count `flat_load_dword` operations via an atomic counter in the logging buffer. Run through the rocjitsu simulator. Assert counter value equals `grid_size_x` × (ops-per-workitem). Verify un-instrumented run produces identical output values.

---

### Phase 8: WaitcntTranslator (3–5 days) [parallel with any phase] ✅ DONE

**Depends on:** nothing

| Task | New Files |
|---|---|
| Waitcnt decode/encode: GFX9 `s_waitcnt` → GFX12 split `s_wait_*` instructions; descriptor ABI prologues emitted by `KernelDescriptorTranslator`, placed in caves by `CodeObjectPatcher`, and reached via KD entry redirect | `dbt/semantic_translator.h/.cpp`, `dbt/kernel_descriptor_translator.h/.cpp`, `patch/code_object_patcher.h/.cpp` |

**Tests:**
- **GFX9 ↔ GFX11 round-trip:** Golden-value round-trip for `vmcnt(0)`, `lgkmcnt(0)`, `expcnt(0)`, combined `vmcnt(15) lgkmcnt(0) expcnt(0)`. Verify GFX9 encoding encodes/decodes for every case.
- **GFX9 vmcnt clamping:** GFX9-source `s_waitcnt` with `vmcnt=16` (valid 6-bit, exceeds GFX11's 4-bit max of 15) clamps to `vmcnt=15` when encoding for GFX11/GFX12 — emits `s_wait_loadcnt(15)`.
- **GFX9 sentinel omission:** `vmcnt=63` (GFX9 "don't wait") is omitted on GFX11+ targets (no `s_wait_loadcnt` emitted).
- **GFX10 lgkmcnt 4-bit normalization:** Decode a GFX10/RDNA1-source `s_waitcnt` with `lgkmcnt=0xF` (4-bit sentinel); assert returned `WaitcntValues.lgkmcnt == 0x0F` (normalized to GFX9 sentinel).
- **GFX10 vscnt emission:** Call `encode_waitcnt({vmcnt=0x3F, lgkmcnt=0x0F, expcnt=0x07, vscnt=3}, RDNA2)` — assert the returned vector contains exactly two words: one `s_waitcnt` (all other fields relaxed, i.e., `s_waitcnt 0xFFF7`) and one `s_waitcnt_vscnt 3`.
- **GFX10 no vscnt when relaxed:** `encode_waitcnt({..., vscnt=0x3F}, RDNA2)` returns exactly one word (no `s_waitcnt_vscnt`).
- **RDNA4 lgkmcnt → kmcnt + storecnt_dscnt (vscnt relaxed):** `encode_waitcnt({lgkmcnt=2, vscnt=0x3F}, RDNA4)` — vscnt is the "don't wait" sentinel, so storecnt is left at max (relaxed). Emits exactly two words: `s_wait_kmcnt(2)` AND `s_wait_storecnt_dscnt(storecnt_max, dscnt=2)`; assert no `s_wait_loadcnt` (vmcnt relaxed). Storecnt field rule: `storecnt = (vscnt == 0x3F) ? storecnt_max : min(vscnt, storecnt_max)`.
- **RDNA4 vscnt + lgkmcnt combined (vscnt active):** `encode_waitcnt({lgkmcnt=2, vscnt=5}, RDNA4)` emits three words: `s_wait_kmcnt(2)`, `s_wait_storecnt_dscnt(5, 2)` (storecnt=vscnt=5, dscnt=lgkmcnt=2); assert no `s_wait_loadcnt` (vmcnt relaxed).
- **RDNA4 boundary (vscnt=0):** `encode_waitcnt({lgkmcnt=0, vscnt=0}, RDNA4)` emits `s_wait_loadcnt(0)` + `s_wait_kmcnt(0)` + `s_wait_storecnt_dscnt(0, 0)` — all three counters at zero (full fence).

---

### Phase 8.5: Legalization Tables and Encoding Translators (3–5 days) [parallel with Phases 1–8] ✅ DONE

**Depends on:** Nothing (extends the existing `amdisa` library; no new build dependencies)

**8.5a — Legalization tables:** Auto-generated `Action` classification for every instruction on every ISA pair.

| Task | New / Modified Files |
|---|---|
| `LegalizationGenerator`: union-find equivalence classes, mnemonic rename map (~50 entries), structural classification using `cross_isa.py`'s `_field_signature` / `_operand_signature` | `lib/python/amdisa/legalization.py` (new) |
| Domain-specific override rules (waitcnt model crossings, barrier split, flat memory, MFMA/AccVGPR expansion) | `lib/python/amdisa/legalization.py` (new) |
| C++ codegen: emit sorted `InstructionLegalization[]` tables per (src, dst) pair + shared `legalization_tables.h` header | `lib/python/amdisa/legalization_codegen.py` (new) |
| CLI: `--gen-legalization`, `--legalization-output`, `--legalization-pairs` | `lib/python/amdisa/__main__.py` (modified) |

**8.5b — Encoding translators:** Auto-generated decode/encode functions with ISA-neutral intermediate field structs, using the existing `machine_insts.h` typed bitfield structs.

| Task | New / Modified Files |
|---|---|
| `encoding_translator_codegen`: for each (src, dst) ISA pair, classify fields as COPY/REMAP/DROP/INSERT/COHERENCY and emit ISA-neutral field structs, per-ISA decode functions (src struct → neutral), per-ISA encode functions (neutral → dst struct + coherency remap), and a dispatch function composing decode + encode | `lib/python/amdisa/encoding_translator_codegen.py` (new) |
| Field rename map (~10 entries): `srsrc→rsrc`, `addr→vaddr`, `data→vsrc`, `op_sel→opsel`, `offset→ioffset` | `lib/python/amdisa/encoding_translator_codegen.py` (new) |
| Coherency remap helpers: `remap_gfx940_to_gfx12(sc0,sc1,nt)→{scope,th}` and `remap_gfx9_glc_to_gfx12(glc)→{scope,th}` | Generated inline in output header |
| Size-changing translations: 64-bit MUBUF → 96-bit VBUFFER, 64-bit FLAT → 96-bit VFLAT/VGLOBAL/VSCRATCH | Handled by encode functions returning `TranslationResult` with variable `word_count` |
| Coherency remap helpers and `TranslationResult` struct | `lib/rocjitsu/src/rocjitsu/code/dbt/encoding_translator.h` (hand-written, shared across all pairs) |
| Dispatch function: `translate_encoding_<src>_to_<dst>()` switching on encoding ID | Generated in per-pair header |
| CLI: `--gen-encoding-translators`, `--encoding-translator-output` | `lib/python/amdisa/__main__.py` (modified) |

One header is generated per (src, dst) pair: `code/dbt/generated/encoding_<src>_to_<dst>.h`. The shared header `code/dbt/encoding_translator.h` provides coherency remap helpers and `TranslationResult`. Adding a new pair requires only running the codegen with different XML specs. For the MVP, only `cdna4_to_rdna4` is generated.

**Invocation:**
```bash
python -m amdisa --multi cdna4:cdna4.xml rdna4:rdna4.xml --gen-encoding-translators
python -m amdisa --multi cdna4:cdna4.xml rdna4:rdna4.xml --gen-legalization \
    --legalization-output code/dbt/generated
```

**Tests:**
- Legalization golden test: CDNA3→CDNA4 pair — assert `S_WAITCNT` = IDENTITY (both use GFX9 monolithic encoding), `V_ADD_F32` (VOP2) = IDENTITY, `S_BARRIER` = LOWER (monolithic → signal/wait pair), `V_MFMA_F32_32X32X1_2B_F32` = IDENTITY.
- CDNA3→RDNA3 pair: assert `V_MFMA_*` = EXPAND, `V_ACCVGPR_*` = EXPAND, `V_ADD_F32` (VOP2) = IDENTITY.
- Rename map test: assert `S_LOAD_DWORD` (CDNA1) and `S_LOAD_B32` (CDNA4) are in the same equivalence class; assert CDNA1→CDNA4 classifies `S_LOAD_DWORD` as SUBSTITUTE (same encoding, different opcode) or LOWER (different encoding).
- Encoding translator field classification: for CDNA4→RDNA4, assert SMEM translator has COHERENCY remap for `glc→scope/th`; assert MUBUF→VBUFFER translator has REMAP for `srsrc→rsrc`; assert SOP1/SOP2/SOPP translators have all-COPY fields (identical layouts).
- Never-empty test: for every supported (src, dst) pair with all 9 ISAs loaded, assert the generated legalization table is non-empty and contains at least one IDENTITY entry.
- Round-trip consistency: compare legalization table instruction counts against `CrossIsaAnalyzer.analyze()` totals — universal instructions must all be IDENTITY or SUBSTITUTE, not LOWER or ILLEGAL.

---

### Phase 9: BinaryTranslator — CDNA4→RDNA4 MVP (1–2 weeks) ✅ DONE

**Depends on:** Phase 1, Phase 2, Phase 4, Phase 8, Phase 8.5

| Task | New Files |
|---|---|
| `BinaryTranslator`: ISA-agnostic translation loop using function pointers (`EncodingTranslateFn`, `LegalizationLookupFn`) selected by (guest, host) pair at construction; instruction classification via `InstFlags` (`is_waitcnt()`, `is_barrier()`, `is_mfma()`, `is_accvgpr()`), raw access via `raw_encoding()` | `dbt/binary_translator.h/.cpp` |
| CDNA4 → RDNA4 translation: `s_waitcnt` expansion (GFX9 monolithic → GFX12 split) via `WaitcntTranslator` with code cave; encoding translation via generated `translate_encoding_cdna4_to_rdna4()`; legalization lookup via generated `kLegalization_cdna4_to_rdna4` table | Inside `binary_translator.cpp` |
| `rj_code_translate()` C API: opaque handle interface via `rj_code_internal.h`; creates `BinaryTranslator`, returns translated code object | `dbt/rj_code_translate.cpp` |
| ELF flag patching: `elf_mach_for_arch()` maps `rj_code_arch_t` to `EF_AMDGPU_MACH` constants for all 9 ISAs | `amdgpu_elf.h` additions |

Note: `BranchFixup` is not needed for DBT — the code cave invariant (§3.3.2) ensures no instruction in `.text` ever changes its address, so all branch offsets remain valid. The `BranchReloc` infrastructure defined in §2.6 is used only by the DBI path (`CodeObjectPatcher::insert_at()`).

**Tests:** *(DBT milestone)* 21 tests covering: coherency remap (GFX940→GFX12, GFX9→GFX12), encoding field preservation (SOP1/SOP2/SOPP/SMEM/VOP3), decode-encode round-trip, legalization lookup, zero-ILLEGAL invariant across all 27 ISA pairs, waitcnt decode/encode (6 tests including vmcnt-only, lgkmcnt-only, full, zero-all, expcnt-only, multi-counter).

---

### Phase 9.5: KernelDescriptorTranslator — Resource-Level Translation (1 week) ✅ DONE

**Depends on:** Phase 3 (LivenessAnalysis), Phase 9 (BinaryTranslator skeleton)

| Task | New Files |
|---|---|
| `KernelDescriptorTranslator`: systematic KD field patching; VGPR granularity re-encoding; SGPR count adjustment; AccVGPR→VGPR remapping offset calculation; user SGPR layout shuffle generation | `dbt/kernel_descriptor_translator.h/.cpp` |
| `TargetResourceLimits`: per-ISA hardware resource limits (max VGPRs, SGPRs, LDS, waves/CU, granularity); populated from ISA profile constants or `hsa_agent_get_info` at runtime | `dbt/target_resource_limits.h/.cpp` |
| Future: VGPR tiered spill logic: tier 0 (direct), tier 1 (reduced occupancy), tier 2 (RegDem-style spill to LDS), tier 3 (spill to scratch) | `kernel_descriptor_translator.cpp` plus spill insertion support |
| Future: SGPR spill logic: detect overflow, select coldest SGPRs via liveness, emit `s_store_dword`/`s_load_dword` spill/fill pairs | `kernel_descriptor_translator.cpp` plus spill insertion support |
| Future: LDS overflow (Zorua-style): partition LDS address space, generate conditional `ds_*` → `global_load/store` redirect sequences | `dbt/lds_overflow.h/.cpp` |
| Occupancy computation: standard AMDGPU formula for source and target; diagnostic logging | Inside `kernel_descriptor_translator.cpp` |
| Wire `KdTranslation` into `BinaryTranslator::translate()`: hand KD patches/prologues to `CodeObjectPatcher`, redirect KD entries for prologues, apply spill/fill insertions | `dbt/binary_translator.cpp` modifications |

**Tests:**
- **VGPR tier 0:** Translate a CDNA3 kernel using 128 VGPRs to RDNA3 Wave64 (max 512). Assert KD's `GRANULATED_WORKITEM_VGPR_COUNT` uses AMDHSA GFX10-GFX12 Wave64 `/4` descriptor encoding. Assert no spill instructions inserted.
- **VGPR tier 1:** Translate a CDNA3 kernel using 384 VGPRs to RDNA3 Wave32 (max 256 at full occupancy, 512 at occupancy=1). Assert KD requests 384 VGPRs. Assert `target_occupancy < source_occupancy` in diagnostics. Assert no spill instructions.
- **VGPR tier 2:** Synthesize a kernel needing 600 VGPRs on a target with max 512. Assert 88 VGPRs spilled to LDS. Assert `ds_write_b32`/`ds_read_b32` pairs inserted at spill/fill points. Assert `group_segment_fixed_size` increased by `88 × wf_size × 4` bytes.
- **SGPR spill:** Translate an RDNA kernel using 105 SGPRs to CDNA (max 102). Assert 3 SGPRs spilled to scratch. Assert `s_store_dword`/`s_load_dword` pairs inserted.
- **AccVGPR remap:** Translate a CDNA3 kernel using 256 VGPRs + 128 AccVGPRs to RDNA3. Assert `accvgpr_base = 256`. Assert `v_accvgpr_write acc[0], v0` becomes `v_mov_b32 v256, v0`. Assert KD VGPR count = 384.
- **User SGPR shuffle:** Translate CDNA3→CDNA1 (flat scratch init changes from hardware to SGPR pair). Assert `ENABLE_SGPR_FLAT_SCRATCH_INIT` set in target KD. Assert prologue contains SGPR initialization sequence.
- **LDS overflow:** Synthesize a kernel requesting 96KB LDS on a target with 64KB max. Assert overflow buffer size = 32KB. Assert `ds_read_b32` instructions with dynamically-computed addresses gain the conditional overflow redirect sequence. Assert statically-known addresses below 64KB pass through unchanged.
- **Never-fail invariant:** For every supported (src, dst) pair, translate every kernel in the test suite. Assert zero `ILLEGAL` results from resource translation (instruction-level ILLEGAL is acceptable for genuinely missing opcodes; resource-level must always succeed).

---

### Phase 10: Wave Emulation — Kernel Descriptor Fixup and Exec-Masking Shims (1 week)

**Depends on:** Phase 9
**Note:** Tests for this phase reference CDNA3→RDNA3 and RDNA3→CDNA3 translation. RDNA decoder infrastructure is now available (Phase A); RDNA encoders (machine_insts.h, encodings.h bitfield structs) are also generated for all RDNA targets. Phase 10 can be implemented and tested against real RDNA decoders rather than stubs. The remaining Phase 16 work is wiring RDNA ↔ CDNA translation *pairs* into `BinaryTranslator`'s dispatch table.

| Task | New Files |
|---|---|
| CDNA→RDNA (Wave64 mode): clear `ENABLE_WAVEFRONT_SIZE32` (bit 10 of `kernel_code_properties`) in translated kernel descriptor; re-encode `GRANULATED_WORKITEM_VGPR_COUNT` using the AMDHSA descriptor formula (CDNA3/4 Wave64 `/8` → RDNA GFX10-GFX12 Wave64 `/4`) | Inside `binary_translator.cpp` |
| RDNA→CDNA (Wave32→Wave64 exec-masking shim): inject `s_mov_b32 exec_lo, 0xFFFFFFFF` + `s_mov_b32 exec_hi, 0` at kernel entry; insert `s_mov_b32 exec_hi, 0` after every `v_cmpx` (always writes EXEC) and after every `v_cmp` **whose destination is EXEC** (SDST field == 126); `v_cmp` targeting VCC or a regular SGPR pair does NOT receive the shim | `dbt/wave_shim.h/.cpp` |

**Tests:** Translate a CDNA3 kernel targeting RDNA3. Assert `ENABLE_WAVEFRONT_SIZE32=0` in output kernel descriptor. Assert the VGPR descriptor field uses AMDHSA GFX10-GFX12 Wave64 `/4` encoding, not the ISA manual's physical Wave64 allocation block size. Translate a simple RDNA3 kernel to CDNA3. Assert exec-mask prologue is present at offset 0 and decodes as exactly two instructions: `s_mov_b32 exec_lo, 0xFFFFFFFF` followed by `s_mov_b32 exec_hi, 0` — not `s_and_b64` or any other sequence. Assert every `v_cmpx` (always writes EXEC unconditionally) and every `v_cmp` whose destination operand is `exec` (SDST encoding value 126) is followed by the exec AND shim (`s_mov_b32 exec_hi, 0`); verify `v_cmp` targeting VCC or a regular SGPR pair does NOT get the shim.

---

### Phase 11a: Semantic Translator Framework + MFMA→WMMA Rules (1 week) ✅ DONE

**Depends on:** Phase 9 (BinaryTranslator skeleton)

**Implemented (differs from original plan):** The `SemanticRule` struct was implemented then replaced with a unified `TranslationRule` table keyed by `(encoding_id, opcode)`. All expansion rules (waitcnt, MFMA→WMMA, AccVGPR read/write, v_lshl_add_u64) are registered in one sorted table and dispatched via binary search in `try_lower_expand()`. No separate `SemanticAnchor`/`SemanticMatch` types — the unified `ExpandFn` signature handles all cases.

**MFMA→WMMA translation:** `v_mfma_f32_16x16x16_f16` → `v_wmma_f32_16x16x16_f16` with:
- Lane permutation derived from `LaneLayout` descriptors via `compute_lane_permutation()`
- ds_bpermute_b32 for XOR-48 lane remap at lanes 16-47
- Liveness-based VGPR and SGPR allocation (`find_free_run`, `find_free_sgpr_pair`)
- `HazardTracker` for automatic `s_delay_alu` insertion
- Hardware-verified: 256/256 elements correct, 10 fuzzing iterations with random FP16 inputs on GFX1201

**Key files:** `dbt/semantic_translator.h/.cpp`, `dbt/translation_rule.h`, `dbt/lane_permutation.cpp`, `dbt/hazard_tracker.h/.cpp`, `analysis/liveness.h/.cpp`

### Phase 11b: MFMA Software Fallback Stubs (1 week)

**Depends on:** Phase 7 (trampoline mechanism), Phase 9, Phase 11a
**Note:** Tests reference RDNA3 as a target. RDNA3 decoder and encoder infrastructure is available from Phase A; no stub decoders needed.

| Task | New Files |
|---|---|
| `dbt/mfma_stub/`: device-side GPU code object library; one stub function per MFMA variant; implementations derived from `shared/mfma_exec.h` reference code | `dbt/mfma_stub/` (separate GPU CMake target) |
| Fallback expansion: for MFMA shapes not covered by semantic rules (Tier 0), insert trampoline to the corresponding software stub; handle AccVGPR layout remapping via `mfma_exec.h` `input_loc()`/`output_loc_32/64()` | Inside `binary_translator.cpp` |

**Tests:** Translate a CDNA3 kernel using `v_mfma_f32_16x16x16f16` to RDNA3 target (no Tier 0 WMMA rule for RDNA3 since RDNA3 WMMA has different lane mapping). Run through the simulator. Assert matrix output matches the reference computed directly from `mfma_exec.h::exec_f32()`.

---

### Phase 12: TranslationCache (3–5 days) [parallel with Phases 10–11]

**Depends on:** Phase 9

| Task | New Files |
|---|---|
| `TranslationCache`: on-disk cache keyed by `(src_arch, dst_arch, xxHash64(elf_bytes))`; 16-byte header with magic, arch pair; atomic write (temp file + rename) | `dbt/translation_cache.h/.cpp` |
| Wire into `translate_elf()` as pre/post cache check | `dbt/binary_translator.cpp` |

**Tests:** Translate a CDNA3 kernel; verify `.rjcache/` entry is created. Translate the same kernel again; verify no translation work is performed (assert via a translation-call counter or log). Mutate one byte in the ELF; verify cache miss and new entry written. Set `RJ_DBT_DISABLE_CACHE=1`; verify cache is bypassed.

---

### Phase 13: HSA Tools Layer — Skeleton and PASSTHROUGH (3–5 days) [parallel with Phases 1–12]

**Depends on:** ROCR headers only; no dependency on Phases 1–12

| Task | New Files |
|---|---|
| `OnLoad`/`OnUnload` entry points; `RjHsaLayer` singleton with `CoreApiTable` version check | `hooks/rj_hsa_layer.h/.cpp` |
| `RjTranslationPolicy` env-var parsing; `detect_native_arch_from_agent` | `code/rj_translation_policy.h/.cpp`, `code/rj_arch_detect.h/.cpp` |
| `hsa_code_object_reader_create_from_memory_fn` override + `CodeObjectReaderRegistry` | `hooks/rj_hsa_layer.cpp`, `code/patch/code_object_reader_registry.h/.cpp` |
| `hsa_queue_create_fn` override → `hsa_amd_queue_intercept_create` + no-op packet handler + `KernelDescriptorRegistry` (empty) | `hooks/rj_hsa_layer.cpp`, `hooks/kernel_descriptor_registry.h/.cpp` |
| `hsa_queue_destroy_fn` override → `KernargPoolRegistry::erase()` + real destroy; `KernargPoolRegistry` singleton | `hooks/rj_hsa_layer.cpp`, `hooks/kernarg_pool_registry.h/.cpp` |
| CMake: `librocjitsu_hooks` shared library target | `CMakeLists.txt` |

**Tests:** `HSA_TOOLS_LIB=librocjitsu_hooks.so RJ_DBI_POLICY=passthrough ./hip_vector_add_test`. Assert: test completes with correct output; `OnLoad` was called (log line); every `hsa_code_object_reader_create_from_memory` call is visible in the registry; every queue creation goes through the intercept path (verify via log); ELF bytes passed to `hsa_executable_load_agent_code_object` are byte-identical to the original.

---

### Phase 14a: Source Attribution (1 week) [after Phase 9; parallel with Phase 13]

**Depends on:** Phase 4 (CodeObjectPatcher for PatchOffsetMap), Phase 9 (TranslationSiteMap from BinaryTranslator)

| Task | New Files |
|---|---|
| `DwarfLineTable::parse()` — self-contained DWARF v4/v5 LNP state-machine interpreter; no external libdwarf | `code/dwarf_line_table.h/.cpp` |
| Add `PatchEdit` accumulation and `original_offset_of()` to `CodeObjectPatcher` | `patch/code_object_patcher.h/.cpp` (additions) |
| `SourceAttributor::create()` — DWARF present path + symbol fallback | `code/source_attributor.h/.cpp` |
| `TranslationSiteMap` — cave body offset → original text offset mapping | `dbt/translation_site_map.h/.cpp` |
| Wire `SourceAttributor` into `LoggingBufferRegistry` and `TranslationCache` | `hooks/logging_buffer_registry.cpp`, `dbt/translation_cache.cpp` |

**Tests:** Compile a CDNA3 kernel with `-g`. Parse the resulting ELF. Assert `DwarfLineTable::lookup(0)` returns a valid source location. Instrument a kernel; assert `SourceAttributor::attribute(patched_offset)` returns the correct source line for a known instruction offset. DBT: assert cave-body offsets in `.rj_translations` attribute to the original instruction's source line via `TranslationSiteMap`.

---

### Phase 14: HSA Tools Layer — Pipeline Integration (1 week)

**Depends on:** Phase 7, Phase 9, Phase 12, Phase 13, Phase 14a

| Task | Modified Files |
|---|---|
| Wire `hsa_executable_load_agent_code_object_fn` wrapper to call `run_rj_pipeline()` (DBI and/or DBT based on policy) | `hooks/rj_hsa_layer.cpp` |
| Populate `KernelDescriptorRegistry` with original→patched KD VA mapping after successful patched load | `hooks/rj_hsa_layer.cpp` |
| CMake: link `rocjitsu_dbi`, `rocjitsu_dbt` into `librocjitsu_hooks` | `CMakeLists.txt` |

**Tests:** *(Full integration milestone)*

Happy-path integration:
- `HSA_TOOLS_LIB=librocjitsu_hooks.so RJ_DBI_POLICY=instrument ./hip_vector_add_test` — instrumentation fires; assert counter value equals `grid_size_x × ops_per_workitem` (the expected count must be computed from the kernel's source, not left vague).
- `HSA_TOOLS_LIB=librocjitsu_hooks.so RJ_DBT_TARGET_ISA=cdna4 ./hip_vector_add_test` — kernel is translated; output buffer is byte-identical to un-translated run.
- Cache hit on second run: assert translation is NOT re-executed (verify via log or counter increment; Phase 12 wired through).

Multi-kernel scenarios:
- Binary with 2+ kernels: assert each kernel is independently instrumented/translated; assert `KernelDescriptorRegistry` maps each original KD VA to its patched KD VA without cross-contamination.
- Mixed policy: one kernel instrumented, one passed through; assert only the expected kernel fires instrumentation.
- Same kernel dispatched twice: assert `KernelDescriptorRegistry` lookup succeeds on both dispatches (registry is not cleared between dispatches).

Error fallback paths:
- Simulate instrumentation failure (e.g., probe offset out of kernel bounds): assert rocjitsu falls back to original ELF with no crash; assert ROCR error code is success (silent fallback).
- Simulate translation failure (e.g., set `RJ_DBT_TARGET_ISA` to an unsupported pair): assert fallback to original with `stderr` warning; assert output is correct.
- Invalid policy string (`RJ_DBI_POLICY=garbage`): assert default (passthrough) behavior; no crash.

Resource cleanup:
- After `hsa_shut_down()`, assert `LogDrainThread` has terminated (no dangling threads).
- Assert all logging buffer GPU allocations are freed on shutdown (verify via allocation tracking or ASAN clean exit).
- Assert no orphaned `.rjcache` entries from failed translations (partial writes must be cleaned up).

---

### Post-MVP

### Phase 14b: Cross-ISA Validation Harness (1–2 weeks) [parallel with Phase 14]

**Depends on:** Phase 9 (BinaryTranslator), Phase 14 (HSA integration)

| Task | New Files |
|---|---|
| Reference kernel corpus: collect representative HIP kernels (vector_add, matmul, reduction, stencil, FFT, GEMM) compiled for each of the 9 ISAs. Store disassembly + expected outputs as golden files | `tests/corpus/` directory; `tools/corpus_capture.py` |
| Cross-ISA test runner: for each (src, dst) pair, translate the src kernel, run on dst hardware (or simulator), and compare output buffers against golden values | `tools/run_cross_isa.py`; `tests/dbt/test_cross_isa.cpp` |
| Waitcnt fuzz harness: generate random `s_waitcnt` simm16 values, round-trip through `decode_waitcnt()` → `encode_waitcnt()` for all ISA pairs, verify against LLVM MC as an oracle (assemble with `llvm-mc` and compare encodings) | `tests/fuzz/fuzz_waitcnt.cpp` |
| Differential simulator testing: run the same kernel on two rocjitsu simulator instances (source ISA and translated target ISA), compare per-instruction register state traces | Extension to existing `amdgpu_vm_test.cpp` |
| Occupancy regression tracking: for each translated kernel, compare `source_occupancy` vs `target_occupancy` from `KdTranslation` diagnostics; flag regressions >50% | `tools/occupancy_report.py` |

**Tests:**
- Translate `vector_add` from each CDNA ISA to each RDNA ISA and vice versa. Assert byte-identical output buffers.
- Translate `matmul_naive` (MFMA-using, CDNA3) to RDNA3 (WMMA path) and RDNA1 (software fallback). Assert output matches within 1 ULP tolerance.
- Waitcnt fuzz: 10,000 random round-trips per ISA pair with zero mismatches.
- No pair produces more than 5% EXPAND entries (EXPAND is expensive; most instructions should be IDENTITY/SUBSTITUTE/LOWER).

---

### Phase 15: Wave64→Wave32 Splitting (Fallback) (2–3 weeks)

**Depends on:** Phase 10, Phase 11

Only needed for a hypothetical Wave32-only host. All RDNA generations support Wave64 (`ENABLE_WAVEFRONT_SIZE32=0`), so this path is not required for RDNA targets.

| Task | New Files |
|---|---|
| Dual kernel entry emission (Wave64 kernel forks into two Wave32 sub-kernels) | `dbt/wave_shim.h/.cpp` |
| Exec-split translation: each sub-kernel activates its half of the original exec mask | `dbt/wave_shim.h/.cpp` |
| LDS exchange zone allocation for cross-half `v_readlane`/`ds_permute` | `dbt/wave_shim.h/.cpp` |
| 64-bit packing optimization pass (opt-in; element-wise loops only; no lane-indexed addresses) | `dbt/wave_pack.h/.cpp` |

**Tests:** Translate a CDNA3 Wave64 kernel to a forced Wave32-only target. Assert both sub-kernel entries exist. Assert exec-split at entry. Run on simulator; assert output matches Wave64 run.

---

### Phase 16: RDNA Translation Pairs (1–2 weeks)

**Depends on:** Phase 9 (BinaryTranslator extension points)

> **Phase A status:** The ISA infrastructure prerequisite for this phase is complete. `rj_code_arch_t` already enumerates all RDNA targets (RDNA1–4). All RDNA decoders (rdna1, rdna2, rdna3, rdna3_5, rdna4) and encoder structs (machine_insts.h, encodings.h) are generated and compile-clean. GFX10/GFX11 waitcnt is handled by `WaitcntTranslator` (Phase 8). What remains is wiring the translation pairs into `BinaryTranslator`.

| Task | Notes |
|---|---|
| ~~Add `ROCJITSU_CODE_ARCH_RDNA*` to `rj_code_arch_t`~~ | **Done (pre-existing)** |
| ~~Generate RDNA1/2/3/3.5/4 decoders and encoder structs~~ | **Done (Phase A)** |
| ~~GFX10/GFX11 waitcnt~~ | **Handled by `WaitcntTranslator` (Phase 8)** |
| RDNA ↔ CDNA translation pairs in `BinaryTranslator` | Extend dispatch table; ISA-pair kernel descriptor fixup |
| RDNA4 `VglobalMachineInst` / `VscratchMachineInst` addr_calc completion | Resolve Phase A stubs in `rdna4/addr_calc.cpp` (Phase D prerequisite) |

**Tests:** Decode an RDNA3 kernel using the Phase A decoder; verify basic liveness (Phase 3 infrastructure). Translate an RDNA3 kernel to CDNA3; run on simulator; verify output correctness.

---

### Dependency Graph

```
Phase 0: InstFlags
  ├─────────────────────────────────────────────┐
  │                                             │
Phase 1: RegisterRef + to_register_ref codegen   Phase 4: ELF Patcher [parallel]
  │
Phase 2: CFG edges in BasicBlock::build()
  │
Phase 3: LivenessAnalysis
  │
  │ (consumed by Phase 6+; Phase 5 does not require it)
  │
Phase 5: SpillManager ──────────────────── (needs Phase 1, Phase 4)
  │
Phase 6: TrampolineBuilder + BufferInjector
  │
Phase 7: Instrumentor + DBI C API          Phase 8: WaitcntTranslator [parallel]
  │ (DBI milestone)                          │
  │                           Phase 8.5: amdisa legalization codegen [parallel with 1–8]
  │                             │        (no dependencies; extends existing amdisa library)
  │                             │
  │                           Phase 9: BinaryTranslator + CDNA3→CDNA4
  │                             │        (needs 1+2+4+8; consumes 8.5 tables)
  │                           Phase 9.5: KernelDescriptorTranslator
  │                             │        (needs 3+9; resource-level translation)
  │                           Phase 10: Wave emulation
  │                             │
  │                           Phase 11: MFMA stub ────── (needs Phase 7)
  │                             │
  │                           Phase 12: TranslationCache [parallel with 10+11]
  │                             │
  │                           Phase 14a: Source Attribution [after Phase 9; parallel with 13]
  │                             │         (needs 4+9)
Phase 13: HSA Layer skeleton + PASSTHROUGH [parallel with all above]
  │
Phase 14: HSA Layer pipeline integration ── (needs 7+9+12+13+14a)
  │ (Full integration milestone)
  │
Phase 14b: Cross-ISA Validation Harness ── (needs 9+14)
  │
Phase 15: Wave splitting (post-MVP)
Phase 16: RDNA translation pairs (post-MVP; decoders done in Phase A)
  │
(Future) Stage 2: own hsa_executable_* → rocjitsu code/ loader
```

### Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| `CoreApiTable` major version mismatch (ROCR ABI break) | High | Check `table->core_->version.major_id` in `OnLoad`; fall back to PASSTHROUGH if unexpected; version is stable across minor releases |
| `HSA_TOOLS_LIB` not set → no interception at all | High | Document clearly; `rocjitsu`'s simulated driver path (`SimulatedDriver`) does not depend on this mechanism; only the real-hardware DBI/DBT path does |
| Multiple tools in `HSA_TOOLS_LIB` (space-separated) — ordering matters | Medium | rocjitsu's `OnLoad` saves the pointer it finds in `CoreApiTable` at call time; if another tool runs first and installs its own wrapper, rocjitsu calls the tool's wrapper (chain is preserved); document load order recommendation |
| `hsa_code_object_reader_t` freed before `hsa_executable_load_agent_code_object` is called | Medium | ROCR keeps the reader alive for the executable lifetime; standard usage pattern is safe; add assertion in registry lookup |
| GPU VA (kernel_object) not in `KernelDescriptorRegistry` at dispatch time | Low | Means load-time hook handled it (registry miss = VA already correct); handler is a no-op pass-through in this case |
| ELF patcher introduces misaligned sections | High | Validate emitted ELF with `readelf -a` in unit tests; enforce 256-byte section alignment |
| Spill slot overflow (more registers live than scratch allows) | Medium | Detect at analysis time and fall back to PASSTHROUGH with a warning |
| Branch offset overflow (simm16 range ±128KB) after large trampolines | Medium | Detect before patching; fall back to indirect branch through trampoline pool |
| Wave splitting LDS overflow (exchange area + kernel LDS > max) | High | Check at translation time; fall back to PASSTHROUGH with error if LDS budget exceeded |
| `v_readlane`/`ds_permute` with dynamic lane indices crossing wave halves | High | LDS exchange protocol required; detect dynamic cross-half indices via liveness + range analysis |
| Kernels that inspect wave size via `s_getreg_b32` / TTMP | Medium | Detect and stub; return 64 for Wave64 guest, 32 for Wave32 guest |
| 64-bit packing changes effective lane ID in address expressions | High | Only enable packing pass on loops with no memory ops that use lane ID as address offset |
| Wave32→Wave64: `v_cmp` activating upper 32 lanes | Medium | Insert `s_mov_b32 exec_hi, 0` after every `v_cmp` targeting EXEC (SDST==126) and every `v_cmpx`; 64-bit literal not directly encodable in `s_and_b64` — use `s_mov_b32 exec_hi, 0` instead |
| CDNA4 vmcnt clamping: source GFX9 vmcnt may exceed 4-bit CDNA4 max | Low | Clamp vmcnt to 15 when encoding for CDNA4 target; document that precision is lost |
| SimulatedDriver ioctl path races with pipeline | Low | Protect the ELF mutation with the existing `alloc_mutex_` |
| RDNA3+ reconvergence semantics differ from CDNA implicit model | High | RDNA3+ introduces hardware-assisted reconvergence; kernels that rely on implicit wavefront reconvergence at post-divergence join points may silently mis-execute when translated to CDNA targets (or vice versa). `enable_reconvergence_fixups` in `rj_code_dbt_options_t` gates insertion of `s_wait_event`/`s_barrier_signal` fixups. Detect divergent branch patterns during liveness analysis and emit fixups only at identified join points; validate with a known-divergent kernel golden test |
| Inline assembly with target-specific instructions | Medium | Classify unknown opcodes as LOWER (generic); if the lowering body encounters an unrecognizable instruction, fall back to PASSTHROUGH for the entire kernel with a diagnostic naming the instruction offset and opcode |
| Performance cliffs from excessive EXPAND (many MFMA→software fallback) | High | Log warnings when EXPAND ratio exceeds 5% of total instructions; provide `RJ_DBT_MAX_EXPAND_RATIO` env var to set a threshold; default behavior: translate anyway (never fail) but emit prominent performance warning |
| `s_getreg_b32` reading HW_REG_HW_ID field layout changes across GFX generations | Medium | Detect `s_getreg` instructions during translation; when the `hwreg` field references a register whose layout changed (HW_REG_HW_ID on GFX9 → HW_REG_HW_ID1/HW_REG_HW_ID2 on GFX10+), emit a lowering sequence that reads the target's register layout and packs the result into the source format |
| Self-modifying GPU code via `s_setpc_b64` to dynamically generated instructions | Low | GPU self-modification is extremely rare; `s_setpc_b64` targets are already classified as INDIRECT_BRANCH with empty successor lists; translated code at the jump target is valid if it was translated at load time; runtime-generated code is untranslatable by design — document as a known limitation |

---

### Critical Files for Implementation

**rocjitsu (new files / key existing files):**
- `lib/rocjitsu/include/rocjitsu/code/rj_code.h`
- `lib/rocjitsu/src/rocjitsu/code/basic_block.h`
- New: `lib/rocjitsu/src/rocjitsu/hooks/rj_hsa_layer.h/.cpp` — `OnLoad`/`OnUnload`, `RjHsaLayer`, CoreApiTable wrappers
- New: `lib/rocjitsu/src/rocjitsu/hooks/kernel_descriptor_registry.h/.cpp` — GPU VA → patched KD mapping
- New: `lib/rocjitsu/src/rocjitsu/code/patch/code_object_reader_registry.h/.cpp` — reader handle → ELF bytes
- `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand_types.h`
- `lib/rocjitsu/src/rocjitsu/code/dbt/binary_translator.h/.cpp` — ISA-agnostic translation loop using `EncodingTranslateFn`/`LegalizationLookupFn` function pointers; guest/host terminology
- `lib/rocjitsu/src/rocjitsu/code/dbt/semantic_translator.h/.cpp` — `SemanticRule`, `SemanticTranslator` class; waitcnt GFX9→GFX12 splitting; `try_lower_expand()` for instruction lowering (v_lshl_add_u64 etc.)
- `lib/rocjitsu/src/rocjitsu/code/dbt/kernel_descriptor_translator.h/.cpp` — kernel descriptor ABI/resource translation; kernel-entry prologue generation for guest-visible launch-state values such as CDNA workgroup-id SGPRs on RDNA4's `TTMP9`/packed `TTMP7` payload.
- `lib/rocjitsu/src/rocjitsu/code/dbt/rj_code_translate.cpp` — `rj_code_translate()` C API implementation
- `lib/rocjitsu/src/rocjitsu/code/patch/code_object_patcher.h/.cpp` — ELF read/modify/emit with code cave support
- `lib/rocjitsu/src/rocjitsu/code/patch/instruction_builder.h` — ISA-parameterized instruction encoding helpers (s_branch, s_nop)
- `lib/rocjitsu/src/rocjitsu/code/rj_code_internal.h` — shared internal header for opaque handle structs
- `lib/rocjitsu/src/rocjitsu/code/amdgpu_elf.h` — `EF_AMDGPU_MACH` constants for all 9 ISAs, `elf_mach_for_arch()` helper
- `lib/rocjitsu/src/rocjitsu/isa/instruction.h` — `WAITCNT`, `BARRIER`, `MFMA`, `ACCVGPR` flags with accessors; `raw_encoding()`
- `lib/rocjitsu/src/rocjitsu/code/dbt/encoding_translator.h` — coherency remap helpers and `TranslationResult`
- `lib/rocjitsu/src/rocjitsu/code/dbt/generated/legalization_types.h` — `Action` enum, `InstructionLegalization` struct, `lookup()`
- `lib/rocjitsu/src/rocjitsu/code/dbt/generated/legalization_<src>_to_<dst>.h` — per-pair legalization tables
- `lib/rocjitsu/src/rocjitsu/code/dbt/generated/encoding_fields.h` — ISA-neutral field structs (union across all 9 ISAs)
- `lib/rocjitsu/src/rocjitsu/code/dbt/generated/encoding_<src>_to_<dst>.h` — per-pair decode/encode functions
- `lib/python/amdisa/legalization.py` — legalization table generator
- `lib/python/amdisa/legalization_codegen.py` — legalization table codegen
- `lib/python/amdisa/encoding_translator_codegen.py` — encoding translator codegen
- `tests/dbt/translate_test.cpp` — encoding, coherency, waitcnt, and legalization tests

**ROCR (read-only reference headers, paths relative to the rocr-runtime project root):**
- `runtime/hsa-runtime/inc/hsa_api_trace.h` — `HsaApiTable`, `CoreApiTable`, `AmdExtTable`, `hsa_amd_queue_intercept_handler`, intercept marker packet structs
- `runtime/hsa-runtime/core/inc/hsa_api_trace_int.h` — internal `HsaApiTable` wrapper, `Init()`/`UpdateCore()` methods
- `runtime/hsa-runtime/core/inc/intercept_queue.h` — `InterceptQueue` class, `QueueProxy`, `AddInterceptor()`
- `runtime/hsa-runtime/core/runtime/intercept_queue.cpp` — packet batching, overflow, `StoreRelaxed()` dispatch flow
- `runtime/hsa-runtime/loader/AMDHSAKernelDescriptor.h` — kernel descriptor layout
