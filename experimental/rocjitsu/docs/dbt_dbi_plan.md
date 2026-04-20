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
4. [Pillar 2: Dynamic Binary Instrumentation (DBI)](#pillar-2)
5. [Pillar 3: Dynamic Binary Translation (DBT)](#pillar-3)
6. [Pillar 4: Def-Use and Dataflow Analysis](#pillar-4)
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
  │  TrampolineBuilder    │            │  InstructionMapper              │
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
| `BasicBlock` (`code/basic_block.h`) | Splits decoded instructions at branch terminators using `InstFlags::BRANCH`; provides `instructions()` iterator and `start_offset()` / `end_offset()` for byte-level position in `.text` |
| `AmdGpuCodeObject` (`code/amdgpu_code_object.h`) | ELF parser; `kernel_descriptor_offset()` looks up `.kd` symbols; `image_data()` / `image_size()` give the raw ELF bytes to the patcher |
| `amdgpu_elf.h` | `Elf64_Ehdr`, `Elf64_Shdr`, `Elf64_Sym`, `EF_AMDGPU_MACH_*` constants — used by `CodeObjectPatcher` to locate and rewrite sections and by `detect_arch_from_elf()` for ISA detection |
| `Executable` (`code/executable.h`) | HIP fat binary parser; `code_object(target, index)` extracts `AmdGpuCodeObject` per GFX target; used in Pillar 5 to parse application binaries before the pipeline runs |

### CFG Edges Belong Directly in `BasicBlock`

`BasicBlock::build_cfg()` currently returns a flat `vector<unique_ptr<BasicBlock>>` with no predecessor/successor links. The liveness analysis (Pillar 4) requires a connected graph.

**Design decision: embed edges in `BasicBlock` directly**, not in a separate `ControlFlowGraph` class.

```cpp
// Addition to BasicBlock (code/basic_block.h — hand-written, not autogenerated):
std::vector<BasicBlock *> successors_;    // raw non-owning pointers
std::vector<BasicBlock *> predecessors_;

const std::vector<BasicBlock *> &successors() const { return successors_; }
const std::vector<BasicBlock *> &predecessors() const { return predecessors_; }
```

`BasicBlock::build_cfg()` populates these in a second pass after all blocks are constructed: for each block's terminator, call `inst->branch_offset_bytes()` (a virtual method on `Instruction` overridden by ISA-specific subclasses) to get the signed byte delta to the branch target, compute the absolute target offset, look it up in an `offset → BasicBlock*` map built during the first pass, and link successor/predecessor pairs. Blocks with no terminator branch (fall-through or exit) get an empty successor list.

This gives an **implicit CFG** — the block list returned by `build_cfg()` *is* the CFG. No separate `ControlFlowGraph` class or object is needed. `LivenessAnalysis` takes `span<const BasicBlock *>` and traverses `bb->successors()` directly, with no auxiliary object to thread through call sites or manage lifetime for.

The `ControlFlowGraph` phase (Phase 2 in the roadmap) is therefore: add the two fields and the edge-building second pass to `BasicBlock::build_cfg()`. No new class file is required.

### Pillar-by-Pillar Usage Summary

**Pillar 1 (ELF foundation)** provides `CodeObjectPatcher` and `SpillManager` as the write layer; all pillars that produce modified binaries drive these two components. The `{isa}/machine_insts.h` and `{isa}/encodings.h` structs are the instruction encoding primitives shared across Pillars 2 and 3.

**Pillar 2 (DBI)** uses `{isa}/machine_insts.h` bitfield structs to emit new instructions (spill stores, restores, `s_swappc_b64`), uses `{isa}/addr_calc.h` formulas inside `SpillManager`, and consults `{isa}/mfma_exec.h`'s `output_loc_32/64()` to identify AccVGPR ranges clobbered by MFMA instructions at the patch point (CDNA ISAs only).

**Pillar 3 (DBT)** calls `{isa}::Decoder::decode()` to decode guest instructions, uses cross-ISA `OperandType` comparison to detect capability gaps (e.g., `OPR_ACCVGPR` present in CDNA `operand_types.h` but absent in RDNA), and reuses the guest ISA's `mfma_exec.h` `exec_f32()` / `exec_i32_i8()` / `exec_f64()` as the software fallback body when translating to a target that lacks MFMA.

**Pillar 4 (Liveness)** reads `inst.src_operands_` and `inst.dst_operands_` directly, classifies each operand by its `{isa}::OperandType` tag (not by string-parsing `name()`), uses `encoding_value_` as the register number, and sizes EXEC as `{isa}::Isa::WF_SIZE / 32` bitset words — automatically correct for both Wave64 (CDNA) and Wave32 (RDNA default).

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

### 1.1 Motivation and Design Rationale

The existing `rocjitsu` IR (the `Instruction` / `BasicBlock` / `InstructionList` hierarchy in `isa/instruction.h` and `code/basic_block.h`) provides decoded instructions with operand vectors and `InstFlags`, but exposes no register-level def-use information. Both DBI (spill-set calculation) and DBT (register renaming, waitcnt rewriting) require precise knowledge of which registers are live at every program point.

The key observation is that all necessary raw data already exists: every decoded instruction carries `src_operands_` and `dst_operands_` vectors of `Operand*`, each of which exposes an `encoding_value_` (the register number within its file) and an `OperandType` tag (from `{isa}/operand_types.h`) that classifies it as SGPR, VGPR, AccVGPR (CDNA only), VCC, EXEC, M0, etc. **No string parsing of `name()` is needed** — register classification is done entirely through the `OperandType` enum and value-range constants such as `OPR_SGPR_MIN/MAX`.

The analysis subsystem must:
- Work across the entire `IsaOperand<{isa}::Isa>` family (CDNA1–4, RDNA1–4) without requiring changes to the autogenerated ISA files.
- Provide O(|RegisterSet|/64) set operations (union, intersection, complement) via `std::bitset` word-parallel operations.
- Handle Wave64 EXEC (64-bit) vs Wave32 EXEC (32-bit) with no per-instruction ISA checks in generic code.
- Be fast enough to run at kernel-load time — total complexity O(D · N · |RegisterSet|/64), where N is instruction count and D is the number of dataflow iterations to convergence (typically 3–5 for structured GPU kernels).

### 1.2 New File Layout

```
lib/rocjitsu/src/rocjitsu/code/analysis/
  register_set.h         # RegClass enum, RegisterRef, RegisterSet, expand()
  def_use_chain.h/.cpp   # Per-instruction def and use sets
  liveness.h/.cpp        # Backwards dataflow: live-in/live-out per basic block

lib/rocjitsu/src/rocjitsu/isa/
  isa_traits.h              # (existing) — gains ISA_MAX_SGPRS, ISA_MAX_VGPRS,
                            # ISA_MAX_ACC_VGPRS as constexpr values computed from
                            # the GpuIsa concept members; RegisterSet includes this header

lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/{isa}/
  operand.h/.cpp         # (existing) — gains autogenerated to_register_ref() override body
  instruction.h/.cpp     # (existing) — gains autogenerated implicit_defs/uses() override bodies
```

### 1.3 `RegisterRef`, `RegisterSet`, and `RegClass`

`GpuIsa` already provides `MAX_SGPRS_PER_WF` and `MAX_VGPRS_PER_WF`; `MAX_ACC_VGPRS_PER_WF` is added to the concept (CDNA ISAs = 256, RDNA ISAs = 0). `RegisterSet` uses per-class `std::bitset` members sized by codegen-aggregated maxima — no flat namespace:

```cpp
// lib/rocjitsu/src/rocjitsu/code/analysis/register_set.h
#pragma once
#include <bitset>
#include <cstdint>
#include <optional>
#include "rocjitsu/isa/isa_traits.h"  // ISA_MAX_SGPRS, ISA_MAX_VGPRS, ISA_MAX_ACC_VGPRS

namespace rocjitsu {

/// @brief Register file class. ISA-agnostic; ISAs that lack a class never
/// return it from to_register_ref() — the corresponding bitset stays empty.
enum class RegClass : uint8_t {
  SGPR, VGPR, ACC_VGPR,
  EXEC, VCC, SCC, M0, FLAT_SCRATCH, TTMP, PC
};

/// @brief A single register reference returned by Operand::to_register_ref().
/// index is within the register file (0-based). width is dwords spanned.
struct RegisterRef {
  RegClass cls;
  uint16_t index;
  uint8_t  width = 1;
};

/// @brief Per-class register sets for dataflow analysis.
/// Used to represent gen, kill, live-in, live-out, def, and use sets.
/// All set operations (union, intersection, complement) apply member-wise;
/// each member is O(N/64) word operations.
/// Sizes come from isa_traits.h — the max across all compiled ISAs.
class RegisterSet {
public:
  std::bitset<ISA_MAX_SGPRS>     sgprs;
  std::bitset<ISA_MAX_VGPRS>     vgprs;
  std::bitset<ISA_MAX_ACC_VGPRS> acc_vgprs;  // zero-size on non-CDNA ISAs (see isa_traits.h note)
  std::bitset<ISA_MAX_SPECIAL>   special;     // EXEC, VCC, SCC, M0, FLAT_SCRATCH, TTMPs, PC

  RegisterSet  operator| (const RegisterSet &o) const;
  RegisterSet  operator& (const RegisterSet &o) const;
  RegisterSet  operator~ ()                     const;
  RegisterSet &operator|=(const RegisterSet &o);
  RegisterSet &operator&=(const RegisterSet &o);
  bool         none()                           const;

  /// @brief Set all bits covered by ref (index .. index+width-1 within its class).
  void expand(RegisterRef ref);
};

} // namespace rocjitsu
```

`isa_traits.h` gains the aggregate capacity constants alongside `GpuIsa`. The codegen computes each as the max of that member across all compiled ISAs when regenerating `isa_traits.h`. `GpuIsa` also gains `MAX_ACC_VGPRS_PER_WF` as a required concept member:

```cpp
// isa/isa_traits.h — extend GpuIsa concept and add capacity constants (autogenerated values)
template <typename Isa>
concept GpuIsa = requires {
  { Isa::WF_SIZE              } -> std::convertible_to<uint32_t>;
  { Isa::MAX_SGPRS_PER_WF     } -> std::convertible_to<uint32_t>;
  { Isa::MAX_VGPRS_PER_WF     } -> std::convertible_to<uint32_t>;
  { Isa::MAX_ACC_VGPRS_PER_WF } -> std::convertible_to<uint32_t>;
  { Isa::WAITCNT_LGKMCNT_MASK } -> std::convertible_to<uint32_t>;
  typename Isa::Context;
  typename Isa::OperandType;
  typename Isa::StatusReg;
};
// Derived concepts (Phase B):
template <typename Isa> concept HasAccVgpr = GpuIsa<Isa> && (Isa::MAX_ACC_VGPRS_PER_WF > 0);
template <typename Isa> concept HasMonolithicWaitcnt = GpuIsa<Isa> && (Isa::WAITCNT_LGKMCNT_MASK != 0);

// Aggregate maxima across all compiled ISAs — updated by codegen when a new ISA is added.
// RegisterSet is sized to these; each Isa's static_assert verifies it fits within.
inline constexpr size_t ISA_MAX_SGPRS     = /* max(Isa::MAX_SGPRS_PER_WF) */;
inline constexpr size_t ISA_MAX_VGPRS     = /* max(Isa::MAX_VGPRS_PER_WF) */;
inline constexpr size_t ISA_MAX_ACC_VGPRS = /* max(Isa::MAX_ACC_VGPRS_PER_WF) */;
// NOTE: ISA_MAX_ACC_VGPRS must be at least 1 even when all compiled ISAs have
// MAX_ACC_VGPRS_PER_WF == 0 (RDNA-only builds).  std::bitset<0> is technically
// well-formed in C++20 but misbehaves in some standard library implementations.
// The codegen sets ISA_MAX_ACC_VGPRS = max(max_acc_vgprs_per_wf_values, 1u).
inline constexpr size_t ISA_MAX_SPECIAL   = 32;  // EXEC+VCC+SCC+M0+FLAT_SCRATCH+TTMPs+PC
```

CDNA ISAs set `MAX_ACC_VGPRS_PER_WF = 256`; RDNA ISAs set it to `0`, leaving `acc_vgprs` as a zero-size bitset that compiles away.

### 1.4 Autogenerated `to_register_ref` in ISA Operand Classes

The codegen extends `{isa}/operand.cpp` with the `to_register_ref()` override. No new files; the override lives alongside the rest of the ISA-specific operand implementation:

```cpp
// isa/operand.h — add to Operand base class
/// @brief Map this operand to its register reference for analysis.
/// Returns nullopt for immediates, literals, labels, and other non-register operands.
/// @param wf_size  Wavefront width (32 or 64); determines width of VCC/EXEC.
virtual std::optional<RegisterRef> to_register_ref(uint8_t wf_size) const { return std::nullopt; }
```

```cpp
// isa/arch/amdgpu/cdna3/operand.cpp  (autogenerated section — do not edit)
std::optional<RegisterRef> cdna3::Operand::to_register_ref(uint8_t wf_size) const {
  switch (opr_type_) {
    case OperandType::OPR_SDST: case OperandType::OPR_SRC: case OperandType::OPR_SSRC:
      if (encoding_value_ >= OPR_SDST_SGPR_MIN && encoding_value_ <= OPR_SDST_SGPR_MAX)
        return RegisterRef{RegClass::SGPR, uint16_t(encoding_value_), 1};
      if (encoding_value_ == OPR_SDST_VCC_LO)
        return RegisterRef{RegClass::VCC, 0, uint8_t(wf_size == 64 ? 2 : 1)};
      if (encoding_value_ == OPR_SDST_EXEC_LO)
        return RegisterRef{RegClass::EXEC, 0, uint8_t(wf_size == 64 ? 2 : 1)};
      if (encoding_value_ == OPR_SDST_M0)
        return RegisterRef{RegClass::M0, 0, 1};
      // ... other special regs, TTMPs, FLAT_SCRATCH
      return std::nullopt;
    case OperandType::OPR_VGPR: case OperandType::OPR_SRC_VGPR:
      return RegisterRef{RegClass::VGPR, uint16_t(encoding_value_), 1};
    case OperandType::OPR_ACCVGPR: case OperandType::OPR_SRC_ACCVGPR:
      return RegisterRef{RegClass::ACC_VGPR, uint16_t(encoding_value_), 1};
    default:
      return std::nullopt;
  }
}
```

`encoding_value_` for SGPR and VGPR operands is already a 0-based register number — it maps directly to the per-class bitset index with no base offset needed. The ISA is resolved at decode time via the concrete subclass type; no runtime ISA lookup occurs.

`implicit_defs()` and `implicit_uses()` on `Instruction` are autogenerated the same way in `{isa}/instruction.cpp`, constructing `RegisterRef` values using the same `OperandType`-guided logic for well-known implicit registers.

Generic analysis code in `def_use_chain.h` calls only the virtuals:

```cpp
// lib/rocjitsu/src/rocjitsu/code/analysis/def_use_chain.h (excerpt)

namespace rocjitsu {

/// @brief Def and use sets for a single instruction.
///
/// Constructed directly from an instruction — no separate factory function needed.
/// Calls operand.to_register_ref(wf_size) and inst.implicit_defs/uses(wf_size)
/// with no ISA headers or OPR_* constants in this class.
class InstDefUse {
public:
  InstDefUse(const Instruction &inst, uint8_t wf_size);

  RegisterSet defs;             ///< Registers written by this instruction.
  RegisterSet uses;             ///< Registers read by this instruction.
  bool exec_masked_def = false; ///< True if VGPR defs are predicated by EXEC.
};

} // namespace rocjitsu
```

Whether an operand is a def or use comes from its position in `dst_operands_` vs `src_operands_`, not from `opr_type_`. The virtual `to_register_ref()` tells you *which register*; the vector membership tells you *def or use*.

**Implicit defs and uses** (registers not encoded as explicit operands but read or written by an instruction) are also ISA-specific and must not be hardcoded in generic analysis. They are handled by adding two virtual methods alongside `to_register_ref()` on the `Instruction` base class:

```cpp
// isa/instruction.h — add to Instruction base class

/// @brief Registers implicitly written by this instruction (not in dst_operands_).
/// @param wf_size  Wavefront width (32 or 64).
/// @param defs     Output accumulator; entries are appended, not replaced.
virtual void implicit_defs(uint8_t wf_size, std::vector<RegisterRef> &defs) const {}

/// @brief Registers implicitly read by this instruction (not in src_operands_).
/// @param wf_size  Wavefront width (32 or 64).
/// @param uses     Output accumulator; entries are appended, not replaced.
virtual void implicit_uses(uint8_t wf_size, std::vector<RegisterRef> &uses) const {}
```

ISA-specific instruction subclasses override these for instructions with implicit register effects. For example, `cdna3`'s override for `s_cbranch_vccnz` appends the ISA's VCC pair to `defs` via `cdna3::vcc_64()` (or `cdna3::vcc_lo()` for wave32). `InstDefUse(inst, wf_size)` in generic analysis code calls both virtuals alongside iterating `dst_operands_`/`src_operands_`, with no knowledge of which registers those are.

### 1.5 CFG Edges in `BasicBlock`

CFG edges are stored directly on `BasicBlock` (see §Pillar 1 / Phase 2). No separate `ControlFlowGraph` class is needed. Analysis code receives the block list from `BasicBlock::build_cfg()` and traverses `bb->successors()` / `bb->predecessors()` directly.

The edge-building second pass in `BasicBlock::build_cfg()` works as follows:

1. Build an `offset → BasicBlock*` map over `block->start_offset()` for all blocks.
2. For each block whose terminator has `InstFlags::BRANCH` or `InstFlags::COND_BRANCH`, call the virtual `inst->branch_offset_bytes()` to get the signed byte delta and compute:
   ```cpp
   uint64_t target = terminator_byte_offset + terminator->size() + terminator->branch_offset_bytes();
   ```
3. Look up the target in the offset map and call `link(this, target_block)` to append to both `successors_` and `predecessors_`.
4. For conditional branches, also link the fall-through block (next block in the flat list) as the second successor.
5. For indirect branches (`InstFlags::INDIRECT_BRANCH`): no outgoing edge; block treated as function-exit terminator (empty `successors_`).

RPO traversal (used by liveness convergence) is computed with a standard iterative DFS over `bb->successors()`, returning blocks in reverse finish order. It is a free function, not a class member:</p>

```cpp
// analysis/liveness.h (or basic_block.h)
std::vector<const BasicBlock *> reverse_post_order(
    const std::vector<std::unique_ptr<BasicBlock>> &blocks);
```

### 1.6 `LivenessAnalysis`

```cpp
// lib/rocjitsu/src/rocjitsu/code/analysis/liveness.h

namespace rocjitsu {

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
  /// @brief Run liveness analysis on a connected block list.
  /// @param blocks   Block list from BasicBlock::build_cfg() — edges already populated.
  /// @param wf_size  Wavefront width (32 or 64); forwarded to Operand::to_register_ref().
  explicit LivenessAnalysis(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                             uint8_t wf_size);

  const BlockLiveness &block_liveness(uint32_t node_index) const {
    return liveness_.at(node_index);
  }

  /// @brief Compute the live set at the program point immediately before @p inst.
  ///
  /// Applies the per-instruction transfer function
  ///   live = (live \ defs) ∪ uses
  /// backwards from live_out(block(inst)) to the program point before @p inst.
  /// (Standard compiler notation: kill = defs, gen = uses — same formula.)
  ///
  /// Complexity: O(k · |RegisterSet|/64), where k is the number of instructions
  /// following @p inst in its block.
  ///
  /// @pre @p inst must belong to a BasicBlock in the block list passed to this
  ///      LivenessAnalysis. All instructions must have parent_ set (done by
  ///      BasicBlock::build_cfg()); calling live_before() before build_cfg() has
  ///      run on the block list causes UB.
  /// @param inst  Instruction to query.
  /// @returns RegisterSet of registers live at the program point just before @p inst.
  RegisterSet live_before(const Instruction &inst) const;

  /// @brief Compute the live set at the point immediately after @p inst.
  RegisterSet live_after(const Instruction &inst) const;

  /// @brief Find the lowest-numbered contiguous run of @p count dead (not live-before) SGPRs
  /// at or after index @p from_sgpr.
  /// @param count      Number of consecutive SGPRs needed.
  /// @param from_sgpr  First index to consider.
  /// @param alignment  Required starting-index alignment (1 = any, 2 = even — required for
  ///                   64-bit ops like s_swappc_b64 which need an even-indexed SGPR pair).
  /// @returns The lowest aligned register index in the found run, or -1 if no such run exists.
  int find_free_sgprs(const Instruction &inst, int count, int from_sgpr = 0, int alignment = 1) const;

  /// @brief Find the lowest-numbered contiguous run of @p count VGPRs that are dead on
  /// ALL active lanes at the program point immediately before @p inst.
  /// (A VGPR that is live on any lane is considered live and must not be clobbered.)
  ///
  /// "All active lanes" is interpreted conservatively: liveness analysis tracks the
  /// worst-case EXEC (all-ones / 64 lanes) across the entire data-flow. A VGPR marked
  /// dead here is dead on every possible lane — even inside masked control-flow. Callers
  /// need not reason about dynamic EXEC values.
  /// @returns Lowest VGPR index in the found run, or -1 if no such run exists.
  int find_free_vgprs(const Instruction &inst, int count, int from_vgpr = 0) const;

private:
  /// Pointer (not reference) to allow the member to be stored after construction.
  /// **Precondition (caller invariant):** The pointed-to vector must outlive every
  /// LivenessAnalysis that references it. Passing a temporary or a vector that is
  /// destroyed before this object causes UB. Typical usage:
  ///   auto &blocks = bb_list_owned_by_calling_scope;
  ///   LivenessAnalysis la(blocks, wf_size);   // la must not outlive blocks
  const std::vector<std::unique_ptr<BasicBlock>> *blocks_;
  uint8_t wf_size_;
  std::vector<BlockLiveness> liveness_;

  /// @brief Maps BasicBlock* → index into blocks_ / liveness_; built in constructor.
  std::unordered_map<const BasicBlock *, uint32_t> block_index_;

  /// @brief Precomputed per-instruction def/use, indexed parallel to BasicBlock::instructions().
  std::unordered_map<const Instruction *, InstDefUse> inst_du_;

  void compute_gen_kill();
  bool iterate_once();
};

} // namespace rocjitsu
```

Implementation sketch for `live_before`:

```cpp
// Note: Instruction::parent_ must be declared in the Instruction base class
// (e.g., `BasicBlock *parent_ = nullptr;`) and set by BasicBlock::build_cfg()
// when instructions are added to blocks. This field is not present in the
// auto-generated ISA class hierarchy and must be added as part of Phase 2.

RegisterSet LivenessAnalysis::live_before(const Instruction &inst) const {
  // 1. Find the BasicBlock containing inst via inst.parent_.
  assert(inst.parent_ != nullptr &&
         "live_before: inst.parent_ is null — call BasicBlock::build_cfg() before LivenessAnalysis");
  const BasicBlock *bb = inst.parent_;
  // 2. Look up its index in the blocks_ list (index is stable after build_cfg()).
  uint32_t ni = block_index_.at(bb);   // block_index_ built in constructor: bb* → index
  RegisterSet live = liveness_[ni].live_out;
  // 3. Apply the per-instruction transfer function backwards through instructions
  //    following @p inst in the block: live = (live \ defs) ∪ uses.
  bool found = false;
  for (auto it = bb->instructions().rbegin(); it != bb->instructions().rend(); ++it) {
    const Instruction &cur = *it;
    if (&cur == &inst) { found = true; break; }
    const auto &du = inst_du_.at(&cur);
    live &= ~du.defs;  // kill
    live |= du.uses;   // gen
  }
  assert(found);
  // Apply inst's own transfer function: live = (live_after(inst) \ defs(inst)) ∪ uses(inst).
  // The loop above computes live_after(inst); this step converts it to live_before(inst).
  {
    const auto &du = inst_du_.at(&inst);
    live &= ~du.defs;
    live |= du.uses;
  }
  return live;
}
```

### 1.7 Public C API Extensions

The entire analysis pipeline (`RegisterSet`, `RegisterRef`, `InstDefUse`, `LivenessAnalysis`) is internal C++ — nothing in `rj_code.h` exposes it. The public API only needs to support the two end-user outcomes this library exists for: binary instrumentation and binary translation. Internal pipeline components (`SpillManager`, `Instrumentor`, `BinaryTranslator`) consume `LivenessAnalysis` and `InstDefUse` directly.

Analysis-derived features for external tools (e.g., register pressure queries for profilers) are deferred to a later extension point and will be designed when the internal pipeline is stable.

### 1.8 CDNA/RDNA ISA Differences for Analysis

| Property | CDNA1/2/3 (Wave64) | CDNA4 (Wave64) | RDNA1 (Wave32/64) | RDNA2/3/3.5 (Wave32/64) | RDNA4 (Wave32/64) |
|---|---|---|---|---|---|
| EXEC width | 64-bit (EXEC_LO+EXEC_HI) | 64-bit (EXEC_LO+EXEC_HI) | 32-bit or 64-bit per-kernel flag | 32-bit default; 64-bit in Wave64 mode | 32-bit default; 64-bit in Wave64 mode |
| VCC width | 64-bit (VCC_LO+VCC_HI) | 64-bit | 32-bit in Wave32 mode; 64-bit in Wave64 mode | 32-bit (Wave32 default); 64-bit in Wave64 mode | 32-bit (Wave32 default); 64-bit in Wave64 mode |
| AccVGPR | CDNA1: Absent; CDNA2/3: Present (256 regs) | Present (256 regs) | Absent | Absent | Absent |
| Max SGPR | 102 (CDNA) | 102 (CDNA4) | 106 (RDNA) | 106 (RDNA) | 106 (RDNA) |
| FLAT_SCRATCH | CDNA1/2: Via SGPR pair (ABI-defined); CDNA3: Dedicated hardware pair | Dedicated hardware pair | Via SGPR | Via SGPR | Via SGPR |
| s_waitcnt encoding | GFX9 classic: lgkmcnt[11:8], vmcnt[15:14,3:0], expcnt[6:4] | GFX9 classic (same as CDNA1-3; CDNA4 retains monolithic S_WAITCNT despite GFX11-gen hardware) | GFX10: same 16-bit simm16 + `s_waitcnt_vscnt` | GFX11: new layout expcnt[2:0], lgkmcnt[9:4], vmcnt[15:10] + named per-counter variants | GFX12-style split `s_wait_*`; no monolithic `s_waitcnt` |

The analysis code parameterizes on `wf_size` passed at construction (32 or 64). The caller derives this from both the arch and the kernel descriptor's `ENABLE_WAVEFRONT_SIZE32` bit — e.g., all CDNA ISAs are Wave64-only; all RDNA ISAs default to Wave32 but support Wave64 when `ENABLE_WAVEFRONT_SIZE32=0`. The derivation happens outside the generic analysis layer, typically in the per-ISA register info or the build_cfg() call site. No ISA-specific constants or `register_info.h` headers appear in generic analysis code:

```cpp
// Caller: derive wf_size from arch before constructing LivenessAnalysis.
// wf_size = 64 for CDNA1-4; 32 or 64 for RDNA depending on kernel EF flags.
LivenessAnalysis la(blocks, wf_size);
```

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

One source file, compiled eight times (once per GFX target). Uses:
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

The 8 targets and their ISA family:

| GFX ID | ISA family | Wave default |
|---|---|---|
| `gfx908` | CDNA1 | Wave64 |
| `gfx90a` | CDNA2 | Wave64 |
| `gfx942` | CDNA3 | Wave64 |
| `gfx950` | CDNA4 | Wave64 |
| `gfx1010` | RDNA1 | Wave32 |
| `gfx1030` | RDNA2 | Wave32 |
| `gfx1100` | RDNA3 | Wave32 |
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

#### 3.3.1 `InstructionMapper` — Legalization Rule Table

`InstructionMapper` is the **legalization rule table** for a specific (src_arch, dst_arch) pair. It answers one question per instruction: *is this instruction legal on the target, and if not, how should it be legalized?* It does not perform any transformation itself — that is `BinaryTranslator`'s job.

**Where rules are defined:**

- **Substitution rules** (IDENTITY, SUBSTITUTE) — stored as lookup tables inside each concrete `InstructionMapper` subclass. These are pure data: a source opcode maps to an action and, for SUBSTITUTE, a target opcode. Adding a new substitution rule means adding a row to the table in the relevant `*_mapper.cpp` file.
- **Lowering rules** (LOWER) — the *classification* of which instructions require lowering is stored in the mapper subclass (e.g., "all `s_waitcnt` on CDNA3→CDNA4 are LOWER"). The *lowering logic* is implemented as a private method on `BinaryTranslator` (e.g., `lower_waitcnt()`), because lowering requires context the mapper does not hold (decoded instruction fields, liveness, target encoding structs).
- **Expansion rules** (EXPAND) — same split: mapper classifies the instruction as needing expansion (e.g., "all MFMA ops on CDNA→RDNA are EXPAND"); the expansion logic lives in a `BinaryTranslator` private method (e.g., `expand_mfma()`).

**Where rules are applied:**

`BinaryTranslator::translate_instruction()` is the single rule application point. It queries the mapper for the legalization action and dispatches:

```cpp
// lib/rocjitsu/src/rocjitsu/code/dbt/instruction_mapper.h

namespace rocjitsu {

/// @brief Legalization action for a single source instruction.
///
/// Mirrors LLVM SelectionDAG legalization categories adapted for binary
/// translation: IDENTITY / SUBSTITUTE correspond to "legal"; LOWER and
/// EXPAND correspond to "custom lower" and "expand" respectively.
enum class LegalizationAction {
  /// Legal on target with identical encoding; copy instruction word(s) verbatim.
  IDENTITY,
  /// Legal on target with a direct opcode substitution; semantics unchanged.
  /// substitute_opcode field of InstructionMapping is valid.
  SUBSTITUTE,
  /// Must be lowered to a target-native instruction sequence.
  /// BinaryTranslator dispatches to the matching lower_*() method.
  LOWER,
  /// No target equivalent; must be expanded to a software emulation sequence.
  /// BinaryTranslator dispatches to the matching expand_*() method.
  EXPAND,
  /// No legalization rule exists for this instruction on this target pair.
  /// BinaryTranslator records a translation error and returns failure.
  ILLEGAL,
};

/// @brief Result of a single-instruction legalization lookup.
struct InstructionMapping {
  LegalizationAction action;
  /// Valid only when action == SUBSTITUTE. Holds the target ISA opcode bits
  /// to substitute into the otherwise-unchanged instruction word.
  uint32_t substitute_opcode = 0;
};

/// @brief Legalization rule table for a (src_arch, dst_arch) pair.
///
/// Stores IDENTITY and SUBSTITUTE rules as a direct opcode lookup table.
/// Instructions requiring more complex transformation are classified as
/// LOWER or EXPAND so BinaryTranslator can dispatch to the appropriate
/// lowering or expansion method that has the necessary context.
///
/// Concrete subclasses are the authoritative rule definitions for each pair:
///   dbt/cdna3_to_cdna4_mapper.cpp — SMEM rename substitutions, s_barrier → LOWER
///   dbt/cdna_to_rdna_mapper.cpp   — flat/scratch → LOWER, MFMA/ACC-VGPR → EXPAND
///   dbt/rdna_to_rdna_mapper.cpp   — s_waitcnt split → LOWER, WMMA substitutions
///
/// If no concrete subclass covers a (src, dst) pair, create() returns nullptr
/// and BinaryTranslator::translate() returns ROCJITSU_STATUS_UNSUPPORTED.
class InstructionMapper {
public:
  virtual ~InstructionMapper() = default;

  /// @brief Factory. Returns nullptr for unsupported (src, dst) pairs.
  static std::unique_ptr<InstructionMapper> create(rj_code_arch_t src_arch,
                                                    rj_code_arch_t dst_arch);

  /// @brief Query the legalization action for a decoded source instruction.
  virtual InstructionMapping legalize(const Instruction &inst) const = 0;
};

} // namespace rocjitsu
```

`translate_instruction()` applies the legalization rule and returns the words to write in-place at the original instruction's location. For cave-based rules it returns a same-size branch stub and appends the expansion body to `cave_body_`; for in-place rules it returns the replacement words directly. Either way, the returned word count always equals `inst.size() / 4`, so the surrounding layout is never disturbed.

```cpp
std::vector<uint32_t> BinaryTranslator::translate_instruction(
    const Instruction &inst, const InstDefUse &du) {
  const InstructionMapping m = mapper_->legalize(inst);
  switch (m.action) {

  case LegalizationAction::IDENTITY:
    // In-place: encoding-compatible — copy verbatim.
    return inst.raw_words();

  case LegalizationAction::SUBSTITUTE:
    // In-place: swap the opcode field; operand encoding and size unchanged.
    return inst.with_opcode(m.substitute_opcode).raw_words();

  case LegalizationAction::LOWER: {
    // Same-size lowering: rewrite in-place; no cave needed.
    if (inst.is_flat_memory())  return lower_flat_memory(inst);
    if (inst.is_exec_write())   return lower_wave_exec(inst, du);

    // Size-expanding lowering: emit body into cave, return branch stub in-place.
    std::vector<uint32_t> body;
    if (inst.is_waitcnt())  body = lower_waitcnt_body(inst);
    if (inst.is_barrier())  body = lower_barrier_body(inst);
    // Additional LOWER categories added alongside their lower_*_body() method.
    if (!body.empty()) {
      uint64_t cave_off = cave_byte_offset();  // current end of cave_body_
      append_cave_entry(body, inst.byte_offset() + inst.size());
      return emit_cave_stub(inst, cave_off);
    }
    return inst.raw_words();
  }

  case LegalizationAction::EXPAND: {
    // Expansion: always cave-based; no target equivalent exists.
    std::vector<uint32_t> body;
    if (inst.is_mfma())  body = expand_mfma_body(inst);
    // Additional EXPAND categories added alongside their expand_*_body() method.
    if (!body.empty()) {
      uint64_t cave_off = cave_byte_offset();
      append_cave_entry(body, inst.byte_offset() + inst.size());
      return emit_cave_stub(inst, cave_off);
    }
    // Mapper returned EXPAND but no expand_*_body() recognised the instruction.
    // Record the gap so callers can report the missing expansion rule.
    errors_.push_back("EXPAND: no expansion rule for " + inst.mnemonic() +
                      " at offset 0x" + hex_str(inst.byte_offset()));
    return inst.raw_words();  // emit original words as a safe fallback
  }

  case LegalizationAction::ILLEGAL:
    errors_.push_back("no legalization rule for '" + inst.mnemonic() +
                      "' on target arch");
    return {};
  }
  return inst.raw_words();
}
```

After all instructions are processed, `translate()` calls `CodeObjectPatcher::append_section(".rj_translations", ...)` to emit `cave_body_` as a new ELF section. The cave stubs written in-place already contain the correct `s_branch` offsets computed relative to the start of that section.

#### 3.3.2 Code Cave Strategy

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

**Far stub constraint.** For larger kernels where the per-stub range exceeds ±128 KB, a far stub requires 16 bytes (`s_getpc_b64` [4B] + `s_add_u32` [4B] + `s_addc_u32` [4B] + `s_setpc_b64` [4B]). This is incompatible with 4-byte source instructions (e.g., `s_barrier`, `s_waitcnt`) that only provide 4 bytes in-place. The solution is a **trampoline island**: a short landing pad allocated within near-branch reach of the 4-byte stub containing the full 16-byte far stub sequence. The 4-byte in-place stub branches to the island (near `s_branch`); the island jumps to the cave entry (far indirect jump).

Island allocation rules:
- One `.rj_islands` section is inserted for each contiguous 128 KB span of `.text` that contains at least one far stub.
- Section size = `num_far_stubs_in_span × 16 bytes`. Maximum 4096 entries per island (= 64 KB island, still reachable by the ±128 KB near branch from the span's far stub).
- If `num_far_stubs_in_span > 4096`, `BinaryTranslator::translate()` returns an error (non-empty `errors` field): the kernel is too large/dense for DBT and the caller falls back to PASSTHROUGH.
- **DBI does NOT use far stubs or islands.** DBI trampolines are inserted directly into `.text` via `CodeObjectPatcher::insert_at()`, growing the section in-place; there is no separate `.rj_translations` section and no branch-range constraint (the inserted call stub is adjacent to the patched instruction). Island logic is a DBT-only concern.
- For 8-byte source instructions the 16-byte far stub does not fit in-place either; these also use an island entry.

In practice this path is only needed for exceptionally large shaders (>128 KB `.text`).

**Effect on branch fixup.** Since no instruction in `.text` changes size, all branch offsets in the original `.text` remain valid after translation. The global `fixup_branch_targets()` pass that would otherwise scan the entire translated text is not needed and is not part of `BinaryTranslator`.

### 3.4 `s_waitcnt` Encoding Translation

`s_waitcnt` is a SOPP instruction where the `simm16` field packs counter thresholds differently across generations. This is one of the most common sources of incompatibility between GFX9-class and GFX10/11-class hardware.

#### GFX9 (CDNA1/2/3) `s_waitcnt` encoding:

```
simm16[15:14]  = vmcnt_hi[1:0]    (bits 15-14 are high 2 bits of vmcnt)
simm16[13]     = reserved (write 1 in GFX9 — assembler convention; must be 1)
simm16[12:8]   = lgkmcnt[4:0]
simm16[7]      = reserved (write 1 in GFX9 — assembler convention; must be 1)
simm16[6:4]    = expcnt[2:0]
simm16[3:0]    = vmcnt_lo[3:0]    (low 4 bits of vmcnt)
```

Full vmcnt = `{vmcnt_hi, vmcnt_lo}` = 6-bit value (max 63).

#### GFX10 (RDNA1/2) `s_waitcnt` encoding:

```
simm16[15:14]  = vmcnt_hi[1:0]
simm16[13:12]  = reserved (write 0 on GFX10 — distinct from GFX9 convention)
simm16[11:8]   = lgkmcnt[3:0]    (only 4 bits; max 15; note narrower than GFX9's 5-bit field at [12:8])
simm16[7]      = reserved (write 0 on GFX10)
simm16[6:4]    = expcnt[2:0]
simm16[3:0]    = vmcnt_lo[3:0]
```

#### GFX11+ (RDNA3, CDNA4) and GFX12 (RDNA4): Split wait instructions

GFX11 replaces `s_waitcnt` with individual per-counter instructions (there is no
monolithic `s_waitcnt` on GFX11+ — do not pass RDNA3/4 or CDNA4 as `src_arch` to
`decode_waitcnt()`):

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

**Note: CDNA4 (GFX950) does not have a single consolidated `s_waitcnt` instruction.** CDNA4 follows the same split-instruction model as RDNA3 (GFX11-style) — use `s_wait_loadcnt`, `s_wait_storecnt`, `s_wait_kmcnt`, `s_wait_expcnt`, `s_wait_dscnt`, `s_wait_samplecnt`, and `s_wait_bvhcnt` as separate instructions. RDNA4 (GFX12-style) uses `s_wait_storecnt_dscnt` (combined) in place of the separate `s_wait_kmcnt` + `s_wait_dscnt` pair. A GFX9-source `s_waitcnt` translating to any of these targets must be expanded to a sequence of `s_wait_*` instructions via a code cave (LOWER action, size-expanding). The `WaitcntTranslator` handles this via `encode_waitcnt()` returning a multi-word vector; RDNA4 is dispatched separately to emit the combined `s_wait_storecnt_dscnt`.

Translation function:

```cpp
// lib/rocjitsu/src/rocjitsu/code/dbt/waitcnt_translator.h

namespace rocjitsu {

/// @brief Decoded counter thresholds from s_waitcnt (and s_waitcnt_vscnt for RDNA1/2).
///
/// Default values represent "don't wait" (all counters relaxed).
///
/// Sentinel values use the GFX9 maximums (vmcnt=0x3F, lgkmcnt=0x1F, expcnt=0x07).
/// decode_waitcnt() normalizes GFX10/RDNA1/2 lgkmcnt 4-bit sentinel (0xF) up to 0x1F,
/// so encode_waitcnt() can use uniform != sentinel checks regardless of source ISA.
///
/// is_noop() compares all counters to their normalized sentinel values.
/// After decode_waitcnt() normalization, the sentinels are always 0x3F/0x1F/0x07/0x3F.
struct WaitcntValues {
  uint8_t vmcnt   = 0x3F;  ///< 6-bit normalized sentinel (GFX9/CDNA/RDNA1/2 native split field);
                           ///< encode_waitcnt() clamps to 4-bit for GFX11-style targets (max 15);
                           ///< 0x3F = "don't wait"
  uint8_t lgkmcnt = 0x1F;  ///< max 5-bit (GFX9/CDNA); 4-bit on RDNA1/2 (normalized to 0x1F); 0x1F = "don't wait"
  uint8_t expcnt  = 0x07;  ///< max 3-bit; 0x07 = "don't wait"
  /// GFX10/RDNA1/2 only: store counter from the separate `s_waitcnt_vscnt` instruction.
  /// Not part of the s_waitcnt simm16; encode_waitcnt() emits a second `s_waitcnt_vscnt`
  /// word when dst is RDNA1/2 and vscnt != 0x3F.  Always 0x3F (ignored) on other ISAs.
  uint8_t vscnt   = 0x3F;  ///< max 6-bit; 0x3F = "don't wait"

  /// True if all counters are at their "don't wait" values.
  bool is_noop() const {
    return vmcnt == 0x3F && lgkmcnt == 0x1F && expcnt == 0x07 && vscnt == 0x3F;
  }
};

/// @brief Decode s_waitcnt simm16 for the given source architecture.
///
/// @param src_arch Must be CDNA1/2/3 or RDNA1/2.  CDNA4 and RDNA3/3.5/4 use
///                 split s_wait_* instructions and never produce a single
///                 s_waitcnt simm16 — passing them here is a precondition
///                 violation.  GFX10 (RDNA1/2) "don't wait" values (0xF
///                 for lgkmcnt) are normalized to the GFX9 sentinels (0x1F)
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
///   - GFX11-style (CDNA4, RDNA3, RDNA3.5): up to four words (one per active counter);
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
  // CDNA1/2/3 and RDNA1/2 all encode vmcnt as a split 6-bit field (lo in [3:0], hi in [15:14]).
  // GFX11+ (RDNA3/4, CDNA4) are a precondition violation — they have no monolithic s_waitcnt.
  const bool has_split_vmcnt = (src == ROCJITSU_CODE_ARCH_CDNA1 ||
                                 src == ROCJITSU_CODE_ARCH_CDNA2 ||
                                 src == ROCJITSU_CODE_ARCH_CDNA3 ||
                                 src == ROCJITSU_CODE_ARCH_RDNA1 ||
                                 src == ROCJITSU_CODE_ARCH_RDNA2);
  // GFX9/CDNA1/2/3: lgkmcnt is 5-bit [12:8].
  // GFX10/RDNA1/2:  lgkmcnt is 4-bit [11:8]; sentinel is 0xF; normalize to 0x1F.
  const bool rdna1_or_2 = (src == ROCJITSU_CODE_ARCH_RDNA1 ||
                            src == ROCJITSU_CODE_ARCH_RDNA2);
  if (has_split_vmcnt) {
    uint8_t vmcnt_lo    = (s >> 0)  & 0xF;
    uint8_t expcnt      = (s >> 4)  & 0x7;
    uint8_t raw_lgkmcnt = rdna1_or_2 ? ((s >> 8) & 0xF) : ((s >> 8) & 0x1F);
    uint8_t vmcnt_hi    = (s >> 14) & 0x3;
    v.vmcnt   = (vmcnt_hi << 4) | vmcnt_lo;
    v.expcnt  = expcnt;
    v.lgkmcnt = (rdna1_or_2 && raw_lgkmcnt == 0xF) ? 0x1F : raw_lgkmcnt;
  } else {
    // Only reached if src_arch is invalid (GFX11+ or unknown) — precondition violation.
    assert(false && "decode_waitcnt: src_arch has no monolithic s_waitcnt encoding");
    (void)s;
  }
  return v;
}

std::vector<uint32_t> encode_waitcnt(WaitcntValues v, rj_code_arch_t dst) {
  // CDNA1/2/3: 6-bit split vmcnt field; 5-bit lgkmcnt.
  const bool has_split_vmcnt = (dst == ROCJITSU_CODE_ARCH_CDNA1 ||
                                 dst == ROCJITSU_CODE_ARCH_CDNA2 ||
                                 dst == ROCJITSU_CODE_ARCH_CDNA3);
  // GFX11+ (CDNA4, RDNA3, RDNA3.5, RDNA4): s_waitcnt expands to separate s_wait_* instructions.
  // encode_waitcnt() returns a multi-word sequence for these targets.
  // RDNA3.5 (GFX115x) uses the same GFX11-style split encoding as RDNA3.
  const bool use_split_wait = (dst == ROCJITSU_CODE_ARCH_CDNA4 ||
                                dst == ROCJITSU_CODE_ARCH_RDNA3 ||
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
    // GFX11-style (CDNA4, RDNA3) or GFX12-style (RDNA4): emit split s_wait_* instructions.
    // Each counter gets its own instruction; "don't wait" counters (sentinels 0x3F/0x1F/0x07)
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
    if (v.lgkmcnt != 0x1F) {
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
      const bool need_dc = (v.lgkmcnt != 0x1F);
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
      // GFX11-style (CDNA4, RDNA3, RDNA3.5): separate dscnt instruction.
      if (v.lgkmcnt != 0x1F)
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
    // lgkmcnt is 4-bit only (bits [11:8], max 15); clamp if source was GFX9 5-bit value.
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

All generations document Wave64-specific VGPR allocation blocks, Wave64-specific instruction skipping behavior, and treat Wave64 as a first-class execution mode. RDNA1 and RDNA2 VGPR allocation is in groups of 4 DWords for Wave64 and 8 for Wave32. RDNA3 and RDNA4 use groups of 8 DWords for Wave64 and 16 for Wave32. The underlying hardware issues Wave64 instructions in two 32-lane passes internally, but this is fully transparent to ISA-level code.

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

**VGPR allocation adjustment:**

RDNA allocates VGPRs in different granularities than CDNA. The translated kernel descriptor's `GRANULATED_WAVEFRONT_VGPR_COUNT` field must be recomputed using the RDNA target's block size:

```cpp
// CDNA3 (Wave64): VGPR granularity = 4 DWords → ceil(vgpr_count / 4) - 1
// RDNA3 (Wave64): VGPR granularity = 8 DWords → ceil(vgpr_count / 8) - 1
uint32_t actual_vgprs = (cdna_gran + 1) * 4;  // recover actual count from CDNA encoding
uint32_t rdna_gran    = (actual_vgprs + 7) / 8 - 1;  // re-encode for RDNA3 Wave64
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
| Wave32 RDNA → CDNA Wave64 | N/A (guest is Wave32) | exec masking: exec_hi=0 at entry | No KD change | s_and_b64 after every v_cmp; s_waitcnt encoding |

### 3.6 MFMA Expansion

On non-CDNA hardware (RDNA), MFMA instructions have no target equivalent and are classified `EXPAND` by the relevant `InstructionMapper` subclass. `BinaryTranslator::expand_mfma()` handles the expansion. The mapper detects these by checking whether the decoded `Instruction` subclass corresponds to a `Vop3pMfmaMachineInst` encoding (identifiable via `InstFlags` or mnemonic prefix `v_mfma_`).

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

| Term | `LegalizationAction` | Meaning |
|---|---|---|
| **Identity** | `IDENTITY` | Instruction encoding is identical on both ISAs; copy word(s) verbatim. |
| **Substitute** | `SUBSTITUTE` | Same semantics, different encoding; `BinaryTranslator` performs a direct opcode substitution. |
| **Lower** | `LOWER` | Instruction must be lowered to a target-native sequence (e.g., s_waitcnt re-encoding, flat→global). Lowering logic is in the matching `lower_*()` method on `BinaryTranslator`. |
| **Expand** | `EXPAND` | No target equivalent; instruction is expanded to a software emulation sequence (e.g., MFMA on RDNA). Expansion logic is in the matching `expand_*()` method on `BinaryTranslator`. See §3.6. |
| **N/A** | — | Instruction class does not exist on the source ISA; no action needed. |

#### 3.9.1 ISA Feature Summary

The table below summarizes which instruction categories are available on each supported ISA family. This drives which translation actions apply for a given guest→host pair.

| Feature / Category | CDNA1 GFX908 | CDNA2 GFX90a | CDNA3 GFX942 | CDNA4 GFX950 | RDNA1 GFX1010 | RDNA2 GFX1030 | RDNA3 GFX1100 | RDNA3.5 GFX1150 | RDNA4 GFX1200 |
|---|---|---|---|---|---|---|---|---|---|
| Default wave size | Wave64 | Wave64 | Wave64 | Wave64 | Wave32 | Wave32 | Wave32 | Wave32 | Wave32 |
| Wave64 capable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| `s_waitcnt` encoding | GFX9 classic | GFX9 classic | GFX9 classic | Split (GFX11-style) | GFX10 (+vscnt) | GFX10 (+vscnt) | Split (GFX11-style) | Split (GFX11-style) | Split (GFX12-style) |
| `s_barrier` | SOPP single | SOPP single | SOPP single | `signal`/`wait` pair | SOPP single | SOPP single | `signal`/`wait` pair | `signal`/`wait` pair | `signal`/`wait` pair |
| Memory addressing | `flat` + `seg` field | `flat` + `seg` field | Separate `global`/`scratch` | Separate `global`/`scratch` | Separate `global`/`scratch` | Separate `global`/`scratch` | Separate `global`/`scratch` | Separate `global`/`scratch` | Separate `global`/`scratch` |
| SMEM mnemonic family | `s_load_dword` | `s_load_dword` | `s_load_dword` | `s_load_b32` | `s_load_dword` | `s_load_dword` | `s_load_b32` | `s_load_b32` | `s_load_b32` |
| MFMA ops | Yes (v1, no AccVGPR) | Yes (v2, +XDLOPS, +AccVGPR) | Yes (v3, +sparse) | Yes (v4, +FP8/BF8) | No | No | No | No | No |
| WMMA ops | No | No | No | No | No | No | Yes (v1) | Yes (v1) | Yes (v2) |
| ACC VGPRs (`v_accvgpr_*`) | No (MFMA uses regular VGPRs) | Yes | Yes | Yes | No | No | No | No | No |
| Hardware reconvergence insts | No | No | No | No | No | No | Yes | Yes | Yes |
| `s_sleep` / `s_nop` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

**`s_waitcnt` encoding variants:**
- **GFX9 classic** — 16-bit imm: vmcnt\[3:0\]+\[15:14\], expcnt\[6:4\], lgkmcnt\[12:8\] (5-bit).
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
| `s_waitcnt` | `=` | `=` | `L` §3.4 | `L` §3.4 | `L` §3.4 | `L` §3.4 | `L` §3.4 | `L` §3.4 |
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
/// @param[in]  elf_bytes     Raw ELF image to translate.
/// @param[in]  elf_size      Size of elf_bytes in bytes.
/// @param[in]  options       Translation options (NULL = use defaults / env-var policy).
/// @param[out] translated_elf       Translated ELF (caller must free with free()).
/// @param[out] translated_elf_size  Size of @p translated_elf in bytes.
/// @returns ROCJITSU_STATUS_SUCCESS on success.
///          ROCJITSU_STATUS_UNSUPPORTED if the translation pair is not yet implemented.
///          ROCJITSU_STATUS_ERROR on fatal translation failure (translated_elf set to NULL).
[[nodiscard]] RJ_API_EXPORT rj_status_t
rj_code_translate(const uint8_t *elf_bytes,
                  size_t elf_size,
                  const rj_code_dbt_options_t *options,
                  uint8_t **translated_elf,
                  size_t *translated_elf_size);
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
static hsa_status_t rj_queue_destroy(hsa_queue_t **queue)
{
  if (queue && *queue)
    KernargPoolRegistry::instance().erase((*queue)->handle);  // frees KernargExtensionPool
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

Implementation uses the `EF_AMDGPU_MACH` field and constants already defined in `code/amdgpu_elf.h` (e.g., `EF_AMDGPU_MACH_AMDGCN_GFX942 = 0x49`, `EF_AMDGPU_MACH_AMDGCN_GFX950 = 0x4F`) — no new ELF parsing code is needed:

```cpp
rj_code_arch_t detect_arch_from_elf(const uint8_t *bytes, size_t size) {
  if (size < sizeof(Elf64_Ehdr)) return ROCJITSU_CODE_ARCH_INVALID;
  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(bytes);
  uint32_t mach = ehdr->e_flags & EF_AMDGPU_MACH;
  switch (mach) {
    case 0x30: return ROCJITSU_CODE_ARCH_CDNA1;  // GFX908 (MI100)
    case 0x3F: return ROCJITSU_CODE_ARCH_CDNA2;  // GFX90A (MI200)
    case 0x47: return ROCJITSU_CODE_ARCH_CDNA3;  // GFX940
    case 0x48: return ROCJITSU_CODE_ARCH_CDNA3;  // GFX941
    case 0x49: return ROCJITSU_CODE_ARCH_CDNA3;  // GFX942 (MI300X) — most common CDNA3 target
    case 0x4F: return ROCJITSU_CODE_ARCH_CDNA4;  // GFX950 (MI350)
    case 0x4D: return ROCJITSU_CODE_ARCH_RDNA4;  // GFX1200
    case 0x4E: return ROCJITSU_CODE_ARCH_RDNA4;  // GFX1201
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
| Add `EXEC_MODIFY` flag to `InstFlags`; annotate SOPP/SOP1/VOP* instructions that write EXEC | `instruction.h`, all 9 ISA files |
| Add `INDIRECT_BRANCH` flag to `InstFlags`; annotate `s_setpc_b64` / `s_swappc_b64` | `instruction.h`, relevant instruction subclasses |
| Add `rj_code_arch_t` parameter to `BasicBlock::build_cfg()` signature | `basic_block.h/.cpp` |

> **Design note:** `BasicBlock::build_cfg()` adds CFG edges in a second pass: after constructing all blocks it populates `successors_` and `predecessors_` on each block using the virtual `Instruction::branch_offset_bytes()` and an `offset → BasicBlock*` map. `basic_block.h` is hand-written and safe to modify. No separate `ControlFlowGraph` class is needed — the returned block list is the implicit CFG.

**Tests:** Existing CTest suite passes unchanged. Spot-check: decode a CDNA3 or RDNA3 kernel (decoders for all 9 ISAs are available as of Phase A); assert at least one instruction has `EXEC_MODIFY` set; assert `s_setpc_b64` has `INDIRECT_BRANCH` set.

---

### Phase 1: RegisterRef, RegisterSet, and `to_register_ref` (3–5 days)

| Task | Files |
|---|---|
| `RegClass` enum, `RegisterRef` struct, `RegisterSet` class with `expand()` | `analysis/register_set.h` (new) |
| Add `virtual to_register_ref(wf_size)` to `Operand` base | `isa/operand.h` |
| Extend codegen to emit `to_register_ref()` override body in `{isa}/operand.cpp` using `OpSel*` ranges from `operand_types.h` | `{isa}/operand.cpp` (autogenerated section) |
| Add `virtual implicit_defs/uses(wf_size, out)` to `Instruction` base; codegen emits override body in `{isa}/instruction.cpp` | `isa/instruction.h`, `{isa}/instruction.cpp` (autogenerated section) |
| `InstDefUse(inst, wf_size)` constructor iterating `dst_operands_`/`src_operands_` + calling both implicit virtuals | `analysis/def_use_chain.h/.cpp` (new) |

**Tests:** For all 9 ISAs (decoders available as of Phase A; prioritize CDNA3, CDNA4, RDNA3, RDNA4): call `to_register_ref()` on every `OperandType` variant and assert the returned `RegisterRef` is stable across identical calls; assert two operands referencing the same register return equal `RegisterRef`; assert VGPR and SGPR with the same `encoding_value` return `RegisterRef` with distinct `cls`; verify `RegisterSet` union/intersection; verify `InstDefUse(inst, wf_size)` produces non-empty def sets for VGPR-writing instructions and non-empty use sets for SGPR-reading instructions; verify `implicit_defs` for a branch-conditional instruction includes the expected condition register. For CDNA ISAs also verify AccVGPR operands map to a distinct `RegClass`. For RDNA ISAs verify `wf_size = 32` is used when `WF_SIZE == 32`.

---

### Phase 2: CFG Edge Population in `BasicBlock::build_cfg()` (3–5 days)

**Depends on:** Phase 0

| Task | Modified Files |
|---|---|
| Add `successors_` and `predecessors_` (`std::vector<BasicBlock*>`) to `BasicBlock` with accessors | `code/basic_block.h` |
| Add `virtual int64_t branch_offset_bytes() const` to `Instruction` base; codegen emits override body in `{isa}/instruction.cpp` for branch instructions | `isa/instruction.h`, `{isa}/instruction.cpp` (autogenerated section) |
| Add second pass to `BasicBlock::build_cfg()`: build `offset → BasicBlock*` map, call `terminator->branch_offset_bytes()` to compute target offset, link successor/predecessor pairs; mark `INDIRECT_BRANCH` blocks as exits | `code/basic_block.cpp` |

No new files. No `ControlFlowGraph` class — the block list returned by `build_cfg()` is the implicit CFG.

**Tests:** Decode a CDNA3 kernel containing a `for` loop and an `if`/`else`. Assert: correct block count; loop back-edge present (`bb->successors()` contains an earlier block); `if`-head has exactly two successors; exit block has no successors; every predecessor list is the inverse of the successor lists.

---

### Phase 3: LivenessAnalysis (1 week)

**Depends on:** Phase 1, Phase 2

| Task | New Files |
|---|---|
| `LivenessAnalysis` backward dataflow: `live_in(block)`, `live_out(block)`, `live_before(inst)` | `analysis/liveness.h/.cpp` |

No public C API additions — `LivenessAnalysis` is consumed internally by `SpillManager`, `Instrumentor`, and `BinaryTranslator`.

**Tests:** Decode a CDNA3 matmul kernel. Assert: EXEC is live-in to every block whose terminator is `s_cbranch_execz` or `s_cbranch_execnz` (blocks terminated by `s_cbranch_scc*` or `s_cbranch_vcc*` do not read EXEC and need not have it live-in); dead SGPRs after their last use are absent from `live_before()`. For an instruction that reads and writes the same register — e.g., `s_add_u32 s0, s0, s1` — assert `live_before(inst)` contains `s0` and `s1` (the uses), not just the definition; this validates that `live_before` applies `inst`'s own gen/kill: `live_before(inst) = (live_after(inst) \ defs(inst)) ∪ uses(inst)`.

---

### Phase 4: CodeObjectPatcher — ELF Rewriter (1 week) [parallel with Phases 1–3]

**Depends on:** `amdgpu_elf.h` only (no analysis dependency)

| Task | New Files |
|---|---|
| `CodeObjectPatcher`: insert/overwrite bytes in `.text`; fix all `s_branch`/`s_cbranch`/`s_setpc` offsets after insert; update `.symtab` symbol sizes; update section headers; re-emit valid ELF | `patch/code_object_patcher.h/.cpp` |

**Tests:** Load a compiled CDNA3 kernel ELF. Insert a 4-byte NOP at three distinct offsets (before a branch, after a branch target, at end of kernel). After each insertion: verify `readelf -a` reports no broken sections; verify all branch offsets in the patched text section are correct; verify symbol start addresses (`st_value`) in `.symtab` are correctly adjusted for symbols whose start is beyond the insertion point; verify the enclosing function symbol's `st_size` grows by 4 to reflect the inserted bytes.

---

### Phase 5: SpillManager (3–5 days)

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

### Phase 8: WaitcntTranslator (3–5 days) [parallel with any phase]

**Depends on:** nothing

| Task | New Files |
|---|---|
| `WaitcntTranslator`: encode/decode `s_waitcnt` for GFX9 (CDNA1/2/3), GFX10 (RDNA1/2; has `s_waitcnt_vscnt` for store counts), GFX11 (CDNA4, RDNA3/4 use split `s_wait_*` instructions); map flag semantics across formats; clamp vmcnt to 4-bit max when encoding for CDNA4/RDNA3/4 targets | `dbt/waitcnt_translator.h/.cpp` |

**Tests:**
- **GFX9 ↔ GFX11 round-trip:** Golden-value round-trip for `vmcnt(0)`, `lgkmcnt(0)`, `expcnt(0)`, combined `vmcnt(15) lgkmcnt(0) expcnt(0)`. Verify GFX9 encoding encodes/decodes for every case.
- **GFX9 vmcnt clamping:** GFX9-source `s_waitcnt` with `vmcnt=16` (valid 6-bit, exceeds GFX11's 4-bit max of 15) clamps to `vmcnt=15` when encoding for GFX11/GFX12 — emits `s_wait_loadcnt(15)`.
- **GFX9 sentinel omission:** `vmcnt=63` (GFX9 "don't wait") is omitted on GFX11+ targets (no `s_wait_loadcnt` emitted).
- **GFX10 lgkmcnt 4-bit normalization:** Decode a GFX10/RDNA1-source `s_waitcnt` with `lgkmcnt=0xF` (4-bit sentinel); assert returned `WaitcntValues.lgkmcnt == 0x1F` (normalized to GFX9 sentinel).
- **GFX10 vscnt emission:** Call `encode_waitcnt({vmcnt=0x3F, lgkmcnt=0x1F, expcnt=0x07, vscnt=3}, RDNA2)` — assert the returned vector contains exactly two words: one `s_waitcnt` (all other fields relaxed, i.e., `s_waitcnt 0xFFF7`) and one `s_waitcnt_vscnt 3`.
- **GFX10 no vscnt when relaxed:** `encode_waitcnt({..., vscnt=0x3F}, RDNA2)` returns exactly one word (no `s_waitcnt_vscnt`).
- **RDNA4 lgkmcnt → kmcnt + storecnt_dscnt (vscnt relaxed):** `encode_waitcnt({lgkmcnt=2, vscnt=0x3F}, RDNA4)` — vscnt is the "don't wait" sentinel, so storecnt is left at max (relaxed). Emits exactly two words: `s_wait_kmcnt(2)` AND `s_wait_storecnt_dscnt(storecnt_max, dscnt=2)`; assert no `s_wait_loadcnt` (vmcnt relaxed). Storecnt field rule: `storecnt = (vscnt == 0x3F) ? storecnt_max : min(vscnt, storecnt_max)`.
- **RDNA4 vscnt + lgkmcnt combined (vscnt active):** `encode_waitcnt({lgkmcnt=2, vscnt=5}, RDNA4)` emits three words: `s_wait_kmcnt(2)`, `s_wait_storecnt_dscnt(5, 2)` (storecnt=vscnt=5, dscnt=lgkmcnt=2); assert no `s_wait_loadcnt` (vmcnt relaxed).
- **RDNA4 boundary (vscnt=0):** `encode_waitcnt({lgkmcnt=0, vscnt=0}, RDNA4)` emits `s_wait_loadcnt(0)` + `s_wait_kmcnt(0)` + `s_wait_storecnt_dscnt(0, 0)` — all three counters at zero (full fence).

---

### Phase 9: BinaryTranslator Skeleton and CDNA3→CDNA4 (1–2 weeks)

**Depends on:** Phase 1, Phase 2, Phase 4, Phase 8

| Task | New Files |
|---|---|
| `BinaryTranslator`: instruction dispatch table; default identity passthrough for unrecognized instructions; per-ISA-pair translation hook registry | `dbt/binary_translator.h/.cpp` |
| CDNA3 → CDNA4 translation: `s_waitcnt` re-encoding (GFX9→GFX11) via `WaitcntTranslator`; re-encode `GRANULATED_WAVEFRONT_VGPR_COUNT` in kernel descriptor | Inside `binary_translator.cpp` |
| `translate_elf()` top-level function (ELF in → translated ELF out) | `dbt/binary_translator.cpp` |

Note: `BranchFixup` is not needed for DBT — the code cave invariant (§3.3.2) ensures no instruction in `.text` ever changes its address, so all branch offsets remain valid. The `BranchReloc` infrastructure defined in §2.6 is used only by the DBI path (`CodeObjectPatcher::insert_at()`).

**Tests:** *(DBT milestone)* Translate a CDNA3 `vector_add` kernel to CDNA4. Load and run through the rocjitsu simulator. Assert output matches un-translated run. Assert no GFX9-encoded `s_waitcnt` patterns remain in the translated text section. Assert VGPR granularity field in kernel descriptor reflects the CDNA4 encoding.

---

### Phase 10: Wave Emulation — Kernel Descriptor Fixup and Exec-Masking Shims (1 week)

**Depends on:** Phase 9
**Note:** Tests for this phase reference CDNA3→RDNA3 and RDNA3→CDNA3 translation. RDNA decoder infrastructure is now available (Phase A); RDNA encoders (machine_insts.h, encodings.h bitfield structs) are also generated for all RDNA targets. Phase 10 can be implemented and tested against real RDNA decoders rather than stubs. The remaining Phase 16 work is wiring RDNA ↔ CDNA translation *pairs* into `BinaryTranslator`'s dispatch table.

| Task | New Files |
|---|---|
| CDNA→RDNA (Wave64 mode): clear `ENABLE_WAVEFRONT_SIZE32` (bit 10 of `kernel_code_properties`) in translated kernel descriptor; re-encode VGPR granularity (CDNA3 groups-of-4 → RDNA3/4 groups-of-8) | Inside `binary_translator.cpp` |
| RDNA→CDNA (Wave32→Wave64 exec-masking shim): inject `s_mov_b32 exec_lo, 0xFFFFFFFF` + `s_mov_b32 exec_hi, 0` at kernel entry; insert `s_mov_b32 exec_hi, 0` after every `v_cmpx` (always writes EXEC) and after every `v_cmp` **whose destination is EXEC** (SDST field == 126); `v_cmp` targeting VCC or a regular SGPR pair does NOT receive the shim | `dbt/wave_shim.h/.cpp` |

**Tests:** Translate a CDNA3 kernel targeting RDNA3. Assert `ENABLE_WAVEFRONT_SIZE32=0` in output kernel descriptor. Assert VGPR granularity field uses groups-of-8 encoding. Translate a simple RDNA3 kernel to CDNA3. Assert exec-mask prologue is present at offset 0 and decodes as exactly two instructions: `s_mov_b32 exec_lo, 0xFFFFFFFF` followed by `s_mov_b32 exec_hi, 0` — not `s_and_b64` or any other sequence. Assert every `v_cmpx` (always writes EXEC unconditionally) and every `v_cmp` whose destination operand is `exec` (SDST encoding value 126) is followed by the exec AND shim (`s_mov_b32 exec_hi, 0`); verify `v_cmp` targeting VCC or a regular SGPR pair does NOT get the shim.

---

### Phase 11: MFMA Fallback Stub (1–2 weeks)

**Depends on:** Phase 7 (trampoline mechanism), Phase 9 (BinaryTranslator dispatch)
**Note:** Tests reference RDNA3 as a target. RDNA3 decoder and encoder infrastructure is available from Phase A; no stub decoders needed. Full integration tests (including Phase 16 translation-pair wiring) run after BinaryTranslator dispatch table extension.

| Task | New Files |
|---|---|
| `dbt/mfma_stub/`: device-side GPU code object library; one stub function per MFMA variant; implementations derived from `cdna3/mfma_exec.h` reference code | `dbt/mfma_stub/` (separate GPU CMake target) |
| MFMA translation pass: detect `v_mfma_*` in guest; insert trampoline to the corresponding stub function; handle AccVGPR layout remapping via `mfma_exec.h` `input_loc()`/`output_loc_32/64()` | Inside `binary_translator.cpp` |

**Tests:** Translate a CDNA3 kernel using `v_mfma_f32_16x16x16f16` to a CDNA4 or RDNA3 target. Run through the simulator. Assert matrix output matches the reference computed directly from `mfma_exec.h::exec_f32()`.

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
Phase 2: CFG edges in BasicBlock::build_cfg()
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
  │                           Phase 9: BinaryTranslator + CDNA3→CDNA4
  │                             │        (needs 1+2+4+8)
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

---

### Critical Files for Implementation

**rocjitsu (new files / key existing files):**
- `/home/agutierr/rocm-dev/rocm-systems/experimental/rocjitsu/lib/rocjitsu/include/rocjitsu/code/rj_code.h`
- `/home/agutierr/rocm-dev/rocm-systems/experimental/rocjitsu/lib/rocjitsu/src/rocjitsu/code/basic_block.h`
- New: `lib/rocjitsu/src/rocjitsu/hooks/rj_hsa_layer.h/.cpp` — `OnLoad`/`OnUnload`, `RjHsaLayer`, CoreApiTable wrappers
- New: `lib/rocjitsu/src/rocjitsu/hooks/kernel_descriptor_registry.h/.cpp` — GPU VA → patched KD mapping
- New: `lib/rocjitsu/src/rocjitsu/code/patch/code_object_reader_registry.h/.cpp` — reader handle → ELF bytes
- `/home/agutierr/rocm-dev/rocm-systems/experimental/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/cdna3/operand_types.h`

**ROCR (read-only reference headers):**
- `/home/agutierr/rocm-dev/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_api_trace.h` — `HsaApiTable`, `CoreApiTable`, `AmdExtTable`, `hsa_amd_queue_intercept_handler`, intercept marker packet structs
- `/home/agutierr/rocm-dev/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/inc/hsa_api_trace_int.h` — internal `HsaApiTable` wrapper, `Init()`/`UpdateCore()` methods
- `/home/agutierr/rocm-dev/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/inc/intercept_queue.h` — `InterceptQueue` class, `QueueProxy`, `AddInterceptor()`
- `/home/agutierr/rocm-dev/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/intercept_queue.cpp` — packet batching, overflow, `StoreRelaxed()` dispatch flow
- `/home/agutierr/rocm-dev/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/loader/AMDHSAKernelDescriptor.h` — kernel descriptor layout