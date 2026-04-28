# DBT Design Document

## Overview

The Dynamic Binary Translation (DBT) system translates AMDGPU code objects compiled for one ISA to execute on a different ISA. The initial target pair is CDNA4 (GFX950) → RDNA4 (GFX1200/1201), but the architecture is designed to support any directional ISA pair.

The translation pipeline is organized into layers with clear responsibility boundaries. The binary translator orchestrates the process without owning ISA-specific policy. The encoding translator handles per-instruction binary format conversion. The semantic translator handles instruction-level behavioral differences. The kernel descriptor translator handles descriptor ABI and resource differences.

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
│  │  - instruction lowering│  │  - field layout changes     │ │
│  │  - instruction expand  │  │  - coherency bit remap      │ │
│  └────────────────────────┘  └─────────────────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  KernelDescriptorTranslator                              ││
│  │  Descriptor ABI/resource policy and entry prologue words  ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  CodeObjectPatcher                                       ││
│  │  ELF mutation: descriptor bytes, entry redirects, .text  ││
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
- Manage code caves for expanded instructions (branch stubs in .text, bodies in `.rj_translations`)
- Delegate descriptor ABI/resource policy to `KernelDescriptorTranslator`
- Delegate byte-level ELF mutation and entry prologue redirects to `CodeObjectPatcher`

### Key design constraint

The binary translator's per-instruction loop is ISA-agnostic. It contains no conditional branches on `guest_arch_` or `host_arch_`, no ISA-specific register constants, no encoding format knowledge, and no instruction mnemonics. Instruction policy lives in the encoding and semantic translators; descriptor ABI/resource policy lives in the kernel descriptor translator.

### Code cave mechanism

When a translated instruction sequence is larger than the source instruction, the binary translator creates a code cave: a branch stub replaces the original instruction slot, and the expanded sequence is placed in the executable `.rj_translations` section immediately after `.text`. A return branch at the end of the cave jumps back to the next instruction. The `s_branch` target is computed as `(cave_offset - (branch_pc + 4)) / 4` per the AMDGPU branch encoding, where `cave_offset` is the `.text`-relative offset into the combined `.text` + `.rj_translations` code range. The patcher may add file padding after `.rj_translations` so later `PT_LOAD` segments keep their required `p_offset`/`p_vaddr` alignment.

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
- **Instruction lowering:** One-to-many expansion for instructions that don't exist on the target ISA. Example: `v_lshl_add_u64` → `v_add_co_u32` + `s_wait_alu` + `v_add_co_ci_u32` carry chain.
- **MFMA → WMMA translation:** `v_mfma_f32_16x16x16_f16` → `v_wmma_f32_16x16x16_f16` with ds_bpermute lane remap (XOR-48 at lanes 16-47). Address VGPR selected via `LivenessAnalysis` to avoid clobbering live state.
- **AccVGPR elimination:** `v_accvgpr_read/write` → `v_mov_b32` or NOP on the unified VGPR file.
- **Future:** Additional MFMA shapes, transpose load replacement.

### Unified rule dispatch

All expansion rules (waitcnt, MFMA→WMMA, AccVGPR, v_lshl_add_u64) are registered as `TranslationRule` entries in a single table keyed by `(encoding_id, opcode)`:

```cpp
struct TranslationRule {
  uint16_t src_encoding_id; // Encoding format (e.g., SOPP, VOP3P)
  uint16_t src_opcode;      // Opcode within the encoding
  RuleAction action;        // Expand, Substitute, FieldRemap, Identity
  ExpandFn expand_fn;       // Expansion generator function
  const LaneLayout *guest_layout; // Source matrix layout (for MFMA)
  const LaneLayout *host_layout;  // Target matrix layout (for WMMA)
};
```

The `try_lower_expand()` method performs binary search over the sorted rule table. For matrix instructions, the `guest_layout` and `host_layout` pointers are passed to the expansion function, which uses `compute_lane_permutation()` to derive the cross-lane shuffle rather than hardcoding it.

Context-dependent descriptor ABI translations are handled by `KernelDescriptorTranslator` and handed to `CodeObjectPatcher` as prebuilt kernel-entry prologue words. The original kernel entry stays untouched; when a prologue is actually required, the kernel descriptor is redirected to a prologue cave that branches into the original entry. For CDNA-to-RDNA4 translation today, CDNA workgroup-id SGPRs are materialized from RDNA4's `TTMP9` and packed `TTMP7` launch payload.

### Supporting infrastructure

- **HazardTracker** (`code/dbt/hazard_tracker.h`): Auto-inserts GFX12 `s_delay_alu` instructions based on pipeline class annotations (VALU, SALU, TRANS). Each expansion function emits instructions through the tracker, which manages a 2-deep producer history and computes dependency fields.
- **Lane permutation** (`code/dbt/lane_permutation.cpp`): Derives the XOR mask and lane range for MFMA→WMMA output remapping from `LaneLayout` descriptors. Adding new MFMA shapes requires only adding layout constants.
- **Temp register allocation**: `LivenessAnalysis` materializes live-before sets for each instruction and provides `find_free_run()` / `find_free_sgpr_pair()` / `find_free_sgpr()` for safe temporary register allocation in injected code.

---

## Supporting Modules

### Register Liveness Analysis (`analysis/liveness.h`)

Kernel-scoped backward liveness analysis over the CFG embedded in `BasicBlock`. DBT builds a `KernelBlockScope` from one kernel descriptor entry and ignores CFG edges that leave that scope. The analysis tracks ordinary SGPRs, VGPRs, and AccVGPRs through `RegisterSet`, then materializes live-before sets for instruction-level queries. Provides:

- `live_before(inst)` / `is_live_before(inst, ref)` — query the register set live before an instruction
- `find_free_run(inst, count)` — find consecutive dead VGPRs for operand expansion
- `find_free_sgpr_pair()` / `find_free_sgpr()` — find dead SGPR temporaries for injected code

`InstDefUse` includes explicit operands plus instruction-level implicit hooks. This currently models hidden FLAT `saddr` address dependencies when they map to ordinary SGPR uses. Special architectural state such as EXEC, VCC, SCC, M0, TTMP, and FLAT_SCRATCH is not tracked by liveness yet.

### Code Object Patcher (`code/patch/code_object_patcher.h`)

Handles ELF-level mutations: descriptor byte overwrite, entry prologue cave placement, kernel-entry descriptor redirects, ELF flag updates, `.text` overwrite, code cave storage, and load-segment alignment preservation. It does not decide descriptor translation policy.

### Kernel Descriptor Translator (`code/dbt/kernel_descriptor_translator.h`)

Handles descriptor-level ABI and resource policy: `compute_pgm_rsrc1/2/3`, `kernel_code_properties`, AccVGPR placement metadata, and kernel-entry prologue words for descriptor ABI shuffles when the target launch state cannot preserve the guest-visible register layout directly. It produces descriptor bytes and prologue words; `CodeObjectPatcher` places those words in the ELF and redirects the descriptor entry point when needed.

### Instruction Builder (`code/patch/instruction_builder.h`)

Provides ISA-parameterized helpers for encoding common instructions (`s_branch`, `s_nop`). Used by both the DBT code cave mechanism and the semantic translator's lowering functions. Will also be used by the DBI layer. Documented separately.

---

## Translation Flow

For each code object:

1. **Descriptor translate:** Parse kernel descriptors and compute descriptor byte patches plus any kernel-entry prologue words.
2. **Decode:** Create a `Decoder` for the guest ISA and build basic blocks from the `.text` section.
3. **Analyze:** Build per-kernel CFG scopes from kernel descriptor entry offsets and compute `LivenessAnalysis` for each scope.
4. **Per-instruction pass:** For each instruction in each basic block:
   - Call `try_lower_expand(inst, offset, liveness)` — binary search the expand rules table by `(encoding_id, opcode)`. If a rule matches, the `ExpandFn` generates replacement instruction words using the `HazardTracker` for automatic `s_delay_alu` insertion and liveness-based register allocation for temp VGPRs/SGPRs.
   - If no expand rule matched, look up the legalization table. If Identity or Substitute, call the encoding translator. If Expand with no handler, NOP-fill and emit a warning.
   - If the replacement is larger than the source instruction, create a code cave (branch to `.rj_translations`, return branch at end of cave).
5. **Entry prologues:** `CodeObjectPatcher` places descriptor-provided prologue words in a cave and redirects the kernel descriptor entry point to that cave.
6. **Patch:** Update ELF flags, write descriptor byte patches, and materialize cave body bytes in `.rj_translations`.
7. **Emit:** Return the modified ELF bytes.

**Code cave sizing:** The cave body is placed in `.rj_translations`, so translation no longer depends on compiler-emitted NOP padding after `s_endpgm`. Direct `s_branch` cave stubs still require the cave entry to be within the SOPP signed-16-bit branch range; trampoline islands remain future work for unusually large kernels.

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
