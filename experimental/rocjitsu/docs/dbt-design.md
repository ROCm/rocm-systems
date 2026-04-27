# DBT Design Document

## Overview

The Dynamic Binary Translation (DBT) system translates AMDGPU code objects compiled for one ISA to execute on a different ISA. The initial target pair is CDNA4 (GFX950) → RDNA4 (GFX1200/1201), but the architecture is designed to support any directional ISA pair.

The translation pipeline is organized into three layers, each with a clear responsibility boundary. The binary translator orchestrates the process without containing ISA-specific logic. The encoding translator handles per-instruction binary format conversion. The semantic translator handles behavioral and ABI differences between ISA generations.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  BinaryTranslator                                            │
│  Orchestration: code objects, basic blocks, code caves, ELF  │
│                                                              │
│  ┌────────────────────────┐  ┌─────────────────────────────┐ │
│  │  SemanticTranslator    │  │  EncodingTranslator         │ │
│  │  Behavioral changes:   │  │  Binary format conversion:  │ │
│  │  - waitcnt splitting   │  │  - opcode remapping         │ │
│  │  - workgroup_id ABI    │  │  - field layout changes     │ │
│  │  - instruction lowering│  │  - coherency bit remap      │ │
│  │  - instruction expand  │  │  - null register sentinels  │ │
│  └────────────────────────┘  └─────────────────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  CodeObjectPatcher                                       ││
│  │  ELF mutation: kernel descriptors, e_flags, .text        ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

---

## Binary Translator

**Files:** `code/dbt/binary_translator.h`, `code/dbt/binary_translator.cpp`

The binary translator is the top-level orchestrator. It operates on binary and structural representations of code objects — ELF sections, basic blocks, kernel descriptors — and delegates ISA-specific decisions to the encoding and semantic translators.

### Responsibilities

- Load code objects via the `AmdGpuCodeObject` API
- Decode guest instructions into basic blocks via `Decoder::create(guest_arch)`
- Traverse each basic block and apply semantic rules first, then per-instruction encoding translation
- Manage code caves for expanded instructions (branch stubs in .text, bodies in NOP padding)
- Delegate ELF and kernel descriptor patching to `CodeObjectPatcher`

### Key design constraint

The binary translator's main loop is ISA-agnostic. It contains no conditional branches on `guest_arch_` or `host_arch_`, no ISA-specific register constants, no encoding format knowledge, and no instruction mnemonics. All ISA-specific logic lives in the encoding and semantic translators.

### Code cave mechanism

When a translated instruction sequence is larger than the source instruction, the binary translator creates a code cave: a branch stub replaces the original instruction slot, and the expanded sequence is placed in the NOP padding after `s_endpgm`. A return branch at the end of the cave jumps back to the next instruction. The `s_branch` target is computed as `(cave_offset - (branch_pc + 4)) / 4` per the AMDGPU branch encoding.

---

## Encoding Translator

**Files:** `code/dbt/encoding_translator.h` (shared types), `code/dbt/generated/encoding_*.h` (auto-generated per pair)

The encoding translator works at the instruction encoding layer. It uses the ISA specification (via the `amdisa` Python library) to translate one encoding format to another. This layer handles the majority of instructions — those where the semantics are identical between source and target but the binary layout differs.

### What it handles

- **Identity** instructions: same opcode, different encoding format (field widths, bit positions)
- **Substitute** instructions: different opcode on target, same semantics
- Coherency bit remapping (GFX940 sc0/sc1/nt → GFX12 scope/th)
- Null register sentinel remapping (CDNA4 `0x7F` → RDNA4 `0x7C` for saddr/soffset)
- SMEM soffset_en disabled state → null register on target
- FLAT/GLOBAL/SCRATCH segment disambiguation (extracted from instruction bits internally)

### What it does NOT handle

- Instructions marked `Action::Expand` (no equivalent on target — delegated to semantic translator)
- ABI differences (workgroup ID delivery, waitcnt counter models)
- Multi-instruction patterns (MFMA → WMMA decomposition)

### Data-driven generation

The encoding translator is auto-generated from ISA XML specifications by `encoding_translator_codegen.py`. For each ISA pair, it produces a single function:

```cpp
TranslationResult translate_encoding_<src>_to_<dst>(
    uint32_t encoding_id, uint32_t w0, uint32_t w1, uint32_t w2, uint16_t dst_op);
```

The function decodes the guest instruction into neutral field values using a typed bitfield struct, remaps the opcode, coherency bits, and register sentinels, then re-encodes into the target format. No hand-written code per instruction — the codegen covers all instructions in the ISA specification.

### Legalization tables

A companion table (auto-generated by `legalization_codegen.py`) classifies every (encoding_id, opcode) pair as Identity, Substitute, Expand, or Illegal. The binary translator consults this table to decide which layer handles each instruction.

---

## Semantic Translator

**Files:** `code/dbt/semantic_translator.h`, `code/dbt/semantic_translator.cpp`

The semantic translator handles instructions and ABI conventions whose behavior changes across ISA generations. This layer is responsible for lowering, expanding, and other semantic translations between instruction(s) in the source ISA to instruction(s) in the target ISA.

### What it handles

- **Waitcnt splitting:** GFX9 monolithic `s_waitcnt` → GFX12 split `s_wait_loadcnt` / `s_wait_storecnt_dscnt` / `s_wait_kmcnt` / `s_wait_expcnt`. Decode and encode functions live in `semantic_translator.cpp`.
- **Workgroup ID ABI:** CDNA4 delivers workgroup IDs via SGPRs; RDNA4 delivers them via TTMP registers (TTMP9 for X, TTMP7 for Y/Z). The semantic translator rewrites operand fields in affected instructions.
- **Instruction lowering:** One-to-many expansion for instructions that don't exist on the target ISA. Example: `v_lshl_add_u64` → `v_add_co_u32` + `s_wait_alu` + `v_add_co_ci_u32` carry chain.
- **MFMA → WMMA translation:** `v_mfma_f32_16x16x16_f16` → `v_wmma_f32_16x16x16_f16` with ds_bpermute lane remap (XOR-48 at lanes 16-47). Address VGPR selected via RegisterLiveness to avoid clobbering live state.
- **AccVGPR elimination:** `v_accvgpr_read/write` → `v_mov_b32` or NOP on the unified VGPR file.
- **Future:** Additional MFMA shapes, transpose load replacement.

### Rule-based translations

Semantic translations matched by instruction flags use the `SemanticRule` table:

```cpp
struct SemanticRule {
  const char *name;
  uint64_t anchor_flags;   // Match against Instruction::flags()
  TranslateFn translate;   // Produce SemanticReplacement on match
};
```

Context-dependent translations (workgroup ID rewrite) use dedicated methods on `SemanticTranslator` that receive kernel-level context from `CodeObjectPatcher::WorkGroupIdInfo`.

### Instruction lowering

Instruction lowering handles `Action::Expand` from the legalization table — instructions with no target ISA equivalent that must be decomposed into a sequence of target instructions. The `try_lower_expand()` method on `SemanticTranslator` dispatches by mnemonic to per-instruction lowering functions defined in `semantic_translator.cpp`. As more lowering targets are added (MFMA, AccVGPR, etc.), they are added as functions in the same file.

---

## Supporting Modules

### Register Liveness Analysis (`analysis/register_liveness.h`)

Per-basic-block backward liveness analysis over VGPR indices (0-511, covering both VGPR and AccVGPR ranges in the unified file). Computed per-block before semantic and encoding passes. Provides:

- `is_live(offset, vgpr_index)` — query whether a register is live at an instruction
- `find_free_run(offset, count)` — find consecutive free registers for operand expansion

Used by the semantic translator for safe register remapping (AccVGPR elimination, MFMA→WMMA expansion). Lives at the top level of rocjitsu (not in `code/`) because it's general analysis infrastructure shared by DBT, DBI, and the simulator.

### Code Object Patcher (`code/patch/code_object_patcher.h`)

Handles ELF-level mutations: kernel descriptor translation (`compute_pgm_rsrc1/2/3`, `kernel_code_properties`), ELF flag updates, `.text` overwrite, code cave management, and workgroup ID SGPR layout extraction from kernel descriptors. Documented separately.

### Instruction Builder (`code/patch/instruction_builder.h`)

Provides ISA-parameterized helpers for encoding common instructions (`s_branch`, `s_nop`). Used by both the DBT code cave mechanism and the semantic translator's lowering functions. Will also be used by the DBI layer. Documented separately.

---

## Translation Flow

For each code object:

1. **Decode:** Create a `Decoder` for the guest ISA and build basic blocks from the `.text` section.
2. **Analyze:** Compute `RegisterLiveness` per basic block (backward scan for VGPR gen/kill sets).
3. **Semantic pass:** For each basic block, run `SemanticTranslator::translate()` to handle waitcnt and other semantic rules. Apply `rewrite_workgroup_ids()` after encoding translation to fix ABI differences.
4. **Encoding pass:** For each instruction not consumed by the semantic pass:
   - Look up the legalization table for the (encoding_id, opcode) pair.
   - If Identity or Substitute: call the encoding translator.
   - If Expand: call `try_lower_expand()` for instruction lowering.
   - If the result is larger than the source, create a code cave.
5. **Patch:** Update ELF flags, translate kernel descriptors, write cave body into NOP padding.
6. **Emit:** Return the modified ELF bytes.

---

## Coverage

Across all ISA pairs, the encoding translator handles 60–99% of instructions depending on how similar the source and target are. Adjacent generations within the same family (CDNA3→CDNA4) are ~99% Identity. Cross-family pairs (CDNA4→RDNA4) are ~60% encoding-translatable, with ~40% marked Expand. Most Expand instructions are exotic (MFMA, buffer atomics, image ops) — common compute kernels translate with high coverage.

---

## Testing

- **Simulator tests (292):** Encoding correctness, legalization tables, coherency remapping, waitcnt translation, end-to-end translation with disassembly validation.
- **Hardware tests (2):**
  - `vector_add`: Translate CDNA4 → RDNA4, dispatch on GFX1201 via HSA, verify 1024 elements with random float inputs against CPU golden.
  - `matmul_mfma_16x16`: Translate MFMA 16×16×16 FP16 → WMMA with ds_bpermute lane remap, 10 fuzzing iterations with random FP16 inputs, verify 256 elements per iteration against CPU golden.
  - Run with `build/tests/hsa_translate_test`.
