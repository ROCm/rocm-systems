# DBI Design Document

## Overview

The Dynamic Binary Instrumentation (DBI) system patches AMDGPU HSA code objects in-place, before they are loaded into device memory, to inject code at chosen anchor instructions. Patched code objects can be loaded by either the simulated KMD (`SimulatedDriver`) or — eventually — by real ROCR via the HSA tools layer (`HSA_TOOLS_LIB=librocjitsu_hooks.so`). DBI itself is target-agnostic at the layer boundary; per-ISA differences are confined to the instruction builder and decoder.

This document describes the DBI subsystem as currently implemented. The first end-to-end slice — an *inline-nop* trampoline at a single anchor — is in tree. The broader instrumentation framework (probe-call bodies, multi-site instrumentation with per-site failure tolerance, predicate-based anchor selection, EXEC-policy management, `AfterInst` / `BlockEntry` / `BlockExit` kinds, layout/negotiation between the builder and the orchestrator) is still future work and is tracked in `dbt_dbi_plan.md`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Instrumentor                                                    │
│  Orchestration: collect points, validate, plan, build, splice    │
│                                                                  │
│  ┌────────────────────────┐  ┌─────────────────────────────────┐ │
│  │  Validators            │  │  Trampoline planning            │ │
│  │  - is_relocatable_     │  │  - make_trampoline_plan()       │ │
│  │    anchor() (structural│  │  - validate_inline_nop_plan()   │ │
│  │  - validate_anchor()   │  │    (milestone guardrail)        │ │
│  │    (+ milestone rules) │  │                                 │ │
│  └────────────────────────┘  └─────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  TrampolineBuilder                                           ││
│  │  Plan → bytes: patched anchor word + trampoline body words   ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │  CodeObjectPatcher                                           ││
│  │  ELF mutation: splice + grow .text with trampoline cave, emit││
│  └──────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

The orchestrator owns the multi-stage pipeline. All per-site validation and builder output is *preflighted* before the patcher mutates anything — a late failure (e.g. branch-range overflow) cannot leak a half-built ELF.

---

## Instrumentor

**Files:** `code/patch/instrumentor.h`, `code/patch/instrumentor.cpp`

The Instrumentor is the top-level orchestrator. Callers queue `InstrumentationPoint`s (requests) and then invoke `patch()`. The orchestrator runs the pipeline: validate each anchor, plan a trampoline per site, build it, then mutate the ELF.

### Pipeline stages

```
InstrumentationPoint        -- request: "instrument here, with this body"
      |  validate_points()
      v
ResolvedInstrumentationSite -- one validated anchor + snapshot of the
      |                        original bytes (captured before mutation)
      |  make_trampoline_plan() + TrampolineBuilder::build()
      v
(preflight: per-site plan + built bytes, accumulated locally)
      |  splice anchors + append trampoline caves into .text + emit()
      v
InstrumentedCodeObject      -- patched ELF + diagnostics
```

### Responsibilities

- Lazy CFG construction: blocks are decoded on the first call that needs them via `Decoder::create(arch)` + `BasicBlock::build(obj, *decoder)`. Decoder creation failure (RV32I/RV64I/INVALID) surfaces as a structured `ValidationResult` / `InstrumentedCodeObject` error, not a crash.
- `validate_points()` looks up the decoded `Instruction` at each requested `anchor_offset` and runs `validate_anchor()`. All-or-nothing today: any per-site failure empties `sites` and reports diagnostics in `errors`.
- `patch()` runs validation, then per-site `make_trampoline_plan()` + `TrampolineBuilder::build()` as preflight. Only after every site succeeds does it splice patched anchor bytes into a local `.text` copy, `append_words()` each trampoline after the original bytes as a local code cave, and grow the section in one `replace_text()` call. Returns `InstrumentedCodeObject{elf_bytes, errors, warnings}`.
- `patch_with_debug_summaries()` is a test/debug entry point returning `InstrumentedCodeObjectDebug` (extends `InstrumentedCodeObject` with per-site `InstrumentationPatch` summaries — schema unstable; production callers should prefer `patch()` and recover per-site info from a fresh disassembly).
- Single-attempt: both entry points share one budget. After `patched_ = true`, subsequent calls return a fatal error. Recoverable errors require constructing a new Instrumentor.

### Current scope (inline-nop milestone)

- Exactly one queued `InstrumentationPoint` per `patch()` call (multi-site is fatal).
- Single `.text` section (multi-text is fatal).
- `BeforeInst` kind only (other kinds are fatal).
- Reserved `InstrumentationPoint` fields (`filter_flags`, `probe_obj`, `probe_symbol`, `force_full_exec`) must be default; non-default is fatal. These rejections are the milestone guardrail and disappear as each field gains a real consumer.

### Key design constraint

The Instrumentor knows about milestones; the TrampolineBuilder and CodeObjectPatcher do not. Milestone-scoped restrictions live in three places at the orchestrator boundary: reserved-field rejections in `validate_anchor()`, plan-shape checks in `validate_inline_nop_plan()`, and multi-text/multi-point rejections at the top of `patch()`. The builder accepts any well-formed plan and the patcher accepts any well-formed mutation request.

---

## TrampolineBuilder

**Files:** `code/patch/trampoline_builder.h`, `code/patch/trampoline_builder.cpp`

Generic byte emitter. Takes a `TrampolinePlan` and returns `TrampolineBytes{patched_anchor_bytes, trampoline_words}`. Knows nothing about `InstrumentationPoint`s, milestones, or probe bodies — ready to absorb richer plans (probe calls, clobber-bearing inline asm) without rewrites.

### What it handles

- Encoding the forward `s_branch` that goes into the anchor slot, pointing at the trampoline's first word.
- Emitting the trampoline body in order: `before_items` (each an `InlineAsmItem` containing one or more pre-encoded words), then the relocated original instruction words (when `emit_original`), then `after_items`, then the return `s_branch` back to `anchor_offset + original_size`.
- Plan well-formedness: `original_size` ∈ {4, 8}, `original_words.size() * sizeof(uint32_t) == original_size`, branch reach fits in SOPP `simm16`.
- Architecture awareness — SOPP encoding format is uniform across AMDGPU but opcodes differ (e.g. `s_branch` is opcode 2 on GFX9/CDNA, opcode 32 on GFX12/RDNA4). All opcode selection goes through `instruction_builder.h` helpers.

### What it does NOT handle

- Validating the *intent* of the plan (e.g. that it's a canonical inline-nop body). That's the orchestrator's `validate_inline_nop_plan()` job.
- Choosing the trampoline offset — the caller picks `trampoline_offset` in the plan.
- Mutating any ELF state — output is just bytes.

### Shared SOPP branch math

`compute_sopp_branch_simm16(branch_pc, target)` in `instruction_builder.h` is the single source of truth for the AMDGPU branch encoding `(target - (branch_pc + 4)) / 4`. Used by both the trampoline builder and the DBT code-cave path.

---

## Validators

Three validators sit at the orchestrator boundary, separated so each can evolve independently.

### `is_relocatable_anchor(anchor, anchor_offset, text_bytes, arch, error_out)`

Pure predicate over the anchor instruction and its position. Permanent structural checks only — no `InstrumentationPoint` involvement. Reusable by future predicate-based anchor selection (Instrumentor walks blocks and filters candidates) without inheriting milestone noise.

Rules enforced:
- `anchor_offset` is dword aligned.
- `anchor.size()` is 4 or 8 and fits inside `text_bytes` (subtraction-based bounds check; resists overflow when `anchor_offset` is huge).
- `anchor.raw_encoding()` is non-null.
- `anchor` is not a branch / cond branch / indirect branch / indirect call / program terminator, and `branch_offset_bytes()` is `nullopt`.
- `anchor.mnemonic()` is not on the small PC-relative denylist (`s_getpc_b64`, `s_call_b64`, `s_setpc_b64`, `s_swappc_b64`, `s_rfe_*`) — these may not surface as flag bits on every ISA.

`arch` is accepted now so a future ISA-specific denylist can grow without an API change.

### `validate_anchor(anchor, anchor_offset, text_bytes, pt, arch, error_out)`

Combines `is_relocatable_anchor()` with milestone-scoped policy checks against `pt`: `filter_flags` is zero, `kind` is `BeforeInst`, `probe_obj` is null, `probe_symbol` is empty, `force_full_exec` is false. On success returns a `ResolvedInstrumentationSite` with the captured anchor snapshot (offset, size, original bytes, mnemonic, kind).

### `validate_inline_nop_plan(plan, error_out)`

Defense-in-depth check that the orchestrator-produced `TrampolinePlan` matches the canonical inline-nop shape: exactly one `before_items` entry containing `s_nop 0`, empty `after_items`, `emit_original == true`. Lives at the orchestrator boundary rather than inside the builder so the builder stays generic. Deleted once arbitrary inline-asm bodies are supported.

---

## Data Types

| Type | Stage | Carries |
| --- | --- | --- |
| `InstrumentationPoint` | request | `anchor_offset`, `kind`, reserved fields |
| `ResolvedInstrumentationSite` | post-validation | `anchor_offset`, `original_size`, `original_bytes`, `mnemonic`, `kind` |
| `TrampolinePlan` | builder input | `arch`, `anchor_offset`, `original_size`, `original_words`, `trampoline_offset`, `return_target`, `before_items`, `after_items`, `emit_original` |
| `TrampolineBytes` | builder output | `patched_anchor_bytes`, `trampoline_words` |
| `InstrumentationPatch` | per-site summary (test/debug) | `anchor_offset`, `original_size`, `trampoline_offset`, `return_target`, `original_bytes`, `patched_anchor_bytes` |
| `InstrumentedCodeObject` | `patch()` output | `elf_bytes`, `errors`, `warnings` |
| `InstrumentedCodeObjectDebug` | `patch_with_debug_summaries()` output | `InstrumentedCodeObject` + `patches` |

The intermediate site/plan types are expected to thicken as the framework grows (e.g. `ResolvedInstrumentationSite` likely gains an ordered list of bodies once multi-point coalescing lands; a layout/negotiation stage will appear between planning and splicing).

---

## Supporting Modules

### Code Object Patcher (`code/patch/code_object_patcher.h`) [shared with DBT]

Owns ELF-level mutations. The DBI orchestrator reads the original payload via `text_bytes()`, assembles the new `.text` locally (each anchor spliced in place, then every trampoline appended after the original bytes with `append_words()`), and applies it with a single `replace_text()` before `emit()` returns the patched ELF buffer. The patcher accepts any well-formed mutation request; layout decisions stay in the orchestrator. Trampolines live inside `.text` as a local code cave (the same layout DBT uses) rather than a separate section, so cave offsets are plain `.text`-relative bytes in `[text_size, text_size + cave_bytes)`.

### Instruction Builder (`code/patch/instruction_builder.h`) [shared with DBT]

ISA-parameterized helpers for encoding common instructions (`s_branch`, `s_nop`, etc.) and the SOPP branch math (`compute_sopp_branch_simm16`). Used by both the trampoline builder and the DBT code-cave path. SOPP format is identical across AMDGPU generations but opcodes differ; always go through these helpers, never hardcode opcodes.

### Register Liveness Analysis [shared with DBT]

**Files:** `analysis/liveness.h`, `analysis/liveness.cpp`, `analysis/def_use_chain.h`, `analysis/def_use_chain.cpp`
**Used by:** DBT semantic translator (today); DBI probe-call passes (planned)

Kernel-scoped backward register liveness over the CFG embedded in `BasicBlock`. Callers construct a `LivenessAnalysis` from one `KernelBlockScope` (the blocks reachable from one kernel descriptor entry). Successor/predecessor edges that leave the scope are ignored, so one decoded code object containing N kernels yields N independent analyses.

#### What it tracks

Ordinary SGPRs, VGPRs, and AccVGPRs via `RegisterSet`. `InstDefUse` records explicit operand defs and uses plus instruction-level implicit hooks; the only implicit hook today is the FLAT `saddr` SGPR-pair use that does not appear as an explicit operand.

#### What it does NOT track

- EXEC, VCC, SCC, M0, FLAT_SCRATCH, TTMP — special architectural state. The `RegClass` enum names these for future use, but they are not in the dataflow set.
- Cross-kernel CFG. Edges that leave the kernel scope are silently dropped.
- Memory dependencies. Liveness is purely register-based.

### SpillManager

**Files:** `code/patch/spill_manager.h`, `code/patch/spill_manager.cpp`

Per-kernel scratch-layout planner for DBI spill/fill slots. Probe-call trampolines will use it to reserve byte offsets within per-lane scratch where saved SGPRs / VGPRs / AccVGPRs go before a probe runs and from which they are restored after. Not consumed by the inline-nop pipeline.

#### Responsibilities

- Reserve a "DBI spill zone" appended above the kernel's existing `private_segment_fixed_size`, aligned to 16 bytes.
- Hand out stable per-register byte offsets within that zone. Registers cannot get more than one offset.
- Enforce a hard per-lane scratch cap. Allocations that would push the bumped total past the cap fail; on failure the manager state is unchanged.
- Compute the bumped `private_segment_fixed_size` that the kernel descriptor patcher will write back.

#### What it is not

- Not a memory allocator. SpillManager only computes layout.
- Not the code generator. SpillManager hands out offsets; emitting the actual `scratch_store` / `scratch_load` (or equivalents) will be the trampoline builder's job.
- Not an MFMA-clobber tracker. AccVGPR clobbering by long-latency MFMA is deferred.

#### Public API

```cpp
class SpillManager final {
public:
  static constexpr uint32_t kSlotBytes = 4;         // one 32-bit lane per slot
  static constexpr uint32_t kDbiZoneAlignment = 16; // zone start alignment

  SpillManager(uint32_t original_private_bytes, uint32_t per_lane_scratch_limit);

  std::optional<uint32_t> allocate_slot(RegisterRef reg);
  std::optional<uint32_t> allocate_slots(RegisterRef reg, unsigned width);
  bool                    reserve(const RegisterSet &set);
  uint32_t                total_private_bytes() const;
  std::optional<uint32_t> offset_for(RegisterRef reg) const;
};
```

`allocate_slots` is the common multi-lane case (SGPR pair, 64-bit VGPR pair). `reserve` performs an upfront capacity check across the whole set so a partial allocation can never become visible. `offset_for` is the lookup used by code generators when emitting the matching `scratch_load` after a probe.

### RegisterRef / RegisterSet [shared with DBT]

**Files:** `isa/register_set.h`, `isa/register_set.cpp`
**Used by:** DBT semantic translator, DBI SpillManager and liveness

ISA-independent register-file model. `RegisterRef` is `(RegClass, uint16_t index, uint8_t width)` measured in 32-bit lanes. `RegisterSet` is three disjoint bitsets (SGPR / VGPR / ACC_VGPR) sized to the union of CDNA and RDNA hardware bounds (`REGISTER_SET_MAX_*`). For scratch selection across both families, `REGISTER_SET_ALLOCATABLE_SGPRS` gives the conservative `min(CDNA, RDNA)` bound.

`RegisterSet` exposes `expand` / `erase` / `contains` / `none` / `size` / `intersects`, the standard set operators (`|=`, `&=`, `-=`), and a `for_each` visitor that yields tracked single-lane `RegisterRef`s in (SGPR, VGPR, AccVGPR) ascending-index order.

---

## Instrumentation Flow (inline-nop)

For a code object with one queued `InstrumentationPoint` at `anchor_offset`:

1. **Lazy block decode:** On the first stage that needs the CFG, `Decoder::create(arch)` + `BasicBlock::build(obj, *decoder)` populates `blocks_`. Unsupported arch surfaces as a fatal error here.
2. **Validate:** `validate_points()` walks `points_`; for each, `find_instruction_at_offset()` locates the decoded `Instruction`, then `validate_anchor()` runs milestone-scoped + structural checks and produces a `ResolvedInstrumentationSite` capturing the original bytes.
3. **Plan + build (preflight):** For each site, `make_trampoline_plan(site, arch, trampoline_offset)` produces a canonical inline-nop plan (`before_items = {{ s_nop 0 }}`, `emit_original = true`); `validate_inline_nop_plan()` rechecks shape; `TrampolineBuilder::build(plan)` lowers it to `TrampolineBytes`. The trampoline cursor begins at `patcher.text_size()` (the first byte of the local cave) and advances by each built trampoline's size. Nothing is written to the patcher yet.
4. **Assemble + replace:** Once every site preflighted, copy `.text` into a local buffer, `memcpy` each `patched_anchor_bytes` into its `anchor_offset`, then `append_words()` every trampoline after the original bytes as a local code cave. A single `patcher.replace_text()` grows `.text` in place and fixes up the surrounding ELF (section/segment sizes, moved symbols, descriptor entries).
5. **Emit:** `patcher.emit()` returns the patched ELF.

The trampolines live inside `.text`, immediately after the original kernel bytes (like what DBT currently does). Keeping a single executable `.text` section avoids loaders that only treat `.text` as executable, and keeps cave offsets expressible as `.text`-relative bytes. Because the original bytes do not move, no branch offsets in the original code are relocated — only the in-place `s_branch` at each anchor is rewritten. One site lays out as:

```
.text:
  [original kernel bytes]
  ...
  @ anchor_offset:
    s_branch <trampoline>       <-- forward branch into the local cave
  ...
  [local cave, after the original bytes]
  @ trampoline_offset:
    s_nop 0                     <-- inline-nop placeholder (becomes probe call later)
    <relocated original word(s)> <-- 4 or 8 bytes, same encoding as the anchor
    s_branch <return>           <-- back to anchor_offset + original_size
```

Branch offsets are computed in SOPP `simm16` units; the trampoline must lie within `±32768 * 4` bytes of the anchor. For unusually large kernels this would require trampoline islands, which remain future work (same constraint as DBT code caves).

---

## Testing

- **Unit (`tests/patch/instrumentor_test.cpp`):** Validator coverage (each rejection path on synthetic anchors, including the bounds-overflow regression for `is_relocatable_anchor`), inline-nop plan guardrail, `make_trampoline_plan`, end-to-end `patch()` on a synthetic ELF (expected anchor splice + trampoline layout + reparse), and a decoded round-trip of every word in the emitted trampoline.
- **Unit (`tests/patch/trampoline_builder_test.cpp`):** Builder byte-layout contract, branch math, arch-honoring opcode selection, INT16 limit boundary cases.
- **Unit (`tests/patch/instruction_builder_test.cpp`):** `compute_sopp_branch_simm16` boundary / alignment / overflow / negative-unaligned-delta.
- **Static DBI smoke (`tests/dbi/hsa_dbi_smoke_test.cpp`, `HsaDbiSmokeStatic`):** Loads a real compiled gfx90a `vector_add` ELF, runs `Instrumentor::patch()`, asserts the patched ELF differs from the original, decodes the anchor as `s_branch`, and confirms `.text` grew to hold the appended trampoline cave. No GPU required.
- **Hardware DBI smoke (`HsaDbiSmokeHardware`, gated on `HAS_CDNA2_GPU`):** Three tests on a real gfx90a GPU — patched ELF loads + validates via HSA; dispatched kernel produces bit-identical output to the original (the inline-nop placeholder is a no-op); a *sabotage* test overwrites the trampoline's `s_nop 0` with `s_endpgm 0` and asserts the kernel actually fails, proving the GPU genuinely executes the trampoline path rather than silently bypassing the splice. `hsa_init` / `hsa_shut_down` and gfx90a agent enumeration run once per suite via `SetUpTestSuite` / `TearDownTestSuite`.
- Run with `build/tests/rocjitsu_tests` and `build/tests/hsa_dbi_smoke_test`.
