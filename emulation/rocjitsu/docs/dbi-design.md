# DBI Design Document

## Overview

The Dynamic Binary Instrumentation (DBI) system patches AMDGPU HSA code objects in-place, before they are loaded into device memory, to inject code at chosen anchor instructions. Patched code objects can be loaded by either the simulated KMD (`SimulatedDriver`) or — eventually — by real ROCR via the HSA tools layer (`HSA_TOOLS_LIB=librocjitsu_hooks.so`). DBI itself is target-agnostic at the layer boundary; most per-ISA differences are confined to the instruction/spill builders and decoder, but a few arch predicates (`max_scratch_offset_bytes`, `descriptor_vgpr_granularity_for_wavefront`, `arch_has_accvgpr`, `arch_has_unified_vgpr_allocation`) live in the orchestrator.

This document describes the DBI subsystem as currently implemented. Two end-to-end trampoline shapes are in tree: the original *inline-nop* trampoline, and a *probe call* that invokes a copied no-op probe body (`rj_nop_probe`) via `s_swappc_b64` before the relocated original. Multiple instrumentation points per code object are supported. **Register spilling is implemented**: registers that are both live at the anchor and clobbered by instrumentation are saved to a reserved per-lane scratch "DBI spill zone" before the probe call and restored after — VGPRs directly, SGPRs through a bridge VGPR (`v_writelane`/`v_readlane`), and AccVGPRs directly via the CDNA scratch `acc` bit. EXEC/VCC/M0 that a probe clobbers are preserved in dead SGPRs, and EXEC is forced to `-1` (full mask) around the spill store/load so all lanes round-trip. End-to-end scope is **CDNA3, CDNA4, and RDNA4** (sim-validated); RDNA2/3/3.5 and CDNA1/2 are deferred. Still future work: per-site failure tolerance, predicate-based anchor selection, `AfterInst` / `BlockEntry` / `BlockExit` kinds, layout/negotiation between the builder and the orchestrator, **automatic SGPR-count growth** so the probe's link pair is always granted (see [Probe-call register requirement](#instrumentation-flow-probe-call)), enabling scratch from zero, HWREG (MODE) preservation, and precise EXEC/VCC/M0 liveness (their clobbers are detected but their liveness is not tracked, so preservation is conservative — see [Register Liveness](#register-liveness-analysis-shared-with-dbt)).

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

### Current scope

- Multiple queued `InstrumentationPoint`s per `patch()` call are supported; sites sharing the same `(probe_obj, probe_symbol)` reuse one copied probe body. All-or-nothing: any per-site failure fails the whole patch.
- Single `.text` section (multi-text is fatal).
- `BeforeInst` kind only (other kinds are fatal).
- `probe_obj` + `probe_symbol` are **consumed**: set both to request a probe-call trampoline, or leave both empty for the inline nop. Setting only one is fatal.
- `filter_flags` and `force_full_exec` are still reserved milestone guardrails — non-default values are fatal until each gains a real consumer.
- Probe calls require the probe body to read no register its calling convention does not supply (`analyze_probe_live_ins`, see [Probe live-ins](#probe-live-ins)). Under `AmdGpuFuncNoArgsReturnS30S31` any such input is fatal.
- Probe calls require the probe's link pair (`s[30:31]` for `rj_nop_probe`) and any chosen scratch/special-state temps to be within the kernel's SGPR allocation (bounded by `kernel_sgpr_count`); auto-growing the SGPR count is not yet implemented (see [Probe-call register requirement](#instrumentation-flow-probe-call)).
- Register spilling is enabled for `spill_set = live_at_anchor ∩ (probe_clobbers ∪ builder_clobbers)`. The orchestrator scans the kernel descriptor (`scan_kernel_descriptors`), builds one `SpillManager` from its `private_segment_fixed_size`, splits the spill set by class, and calls `plan_vgpr_spills` / `plan_sgpr_spills` / `plan_acc_spills`. Spilling is gated per-arch by `max_scratch_offset_bytes()` (returns 0 ⇒ arch has no scratch emitter ⇒ unsupported) and by those fail-closed helpers; there is no `SpillPolicy` enum. A kernel with zero scratch, more than one kernel, an over-offset-cap slot, a probe that clobbers FLAT_SCRATCH, or (for SGPR spills) no dead bridge VGPR within the kernel's VGPR allocation fails closed.

### Key design constraint

The Instrumentor knows about milestones; the TrampolineBuilder and CodeObjectPatcher do not. Milestone-scoped restrictions live at the orchestrator boundary: reserved-field rejections in `validate_anchor()`, the `validate_inline_nop_plan()` shape check on the inline-nop path, the arch/scratch spill gating on the probe-call path (`max_scratch_offset_bytes` + the fail-closed `plan_*_spills` helpers, plus the single-kernel and nonzero-scratch checks), the probe live-in gate in `resolve_points()`, and the multi-text rejection at the top of `patch()`. The builder accepts any well-formed plan and the patcher accepts any well-formed mutation request.

---

## TrampolineBuilder

**Files:** `code/patch/trampoline_builder.h`, `code/patch/trampoline_builder.cpp`

Generic byte emitter. Takes a `TrampolinePlan` and returns `TrampolineBytes{patched_anchor_bytes, trampoline_words}`. Knows nothing about `InstrumentationPoint`s or milestones. It emits two body shapes: the inline-nop body, and the probe-call envelope wrapped around the relocated original. `plan_probe_call()` selects the link/target SGPR pairs, the SCC temp, and any special-state (EXEC/VCC/M0) and SGPR-bridge temps, and reports `builder_clobbers`; `emit_probe_call()` lowers the chosen plan to bytes.

The probe-call envelope, in emit order, is: an in-flight-load drain; the special-state saves (`s_mov` EXEC/VCC into dead SGPR pairs, M0 into a dead SGPR) and SCC save; the **spill prologue** (see below); the `s_getpc_b64` + 64-bit add chain that materializes the copied probe body's address; `s_swappc_b64` to it; then after the call a `build_wait_all_loads_complete()` drain (so the probe's own in-flight loads finish before the restores and host resume), the SCC restore, the **spill epilogue**, and the special-state restores, before the relocated original.

### Spill bracket

`build_spill_bracket()` produces the prologue (saves) and epilogue (restores) that wrap the call:
- **VGPRs** — a direct `build_scratch_store_dword` in the prologue, `build_scratch_load_dword` in the epilogue.
- **AccVGPRs** — the same builders with `acc=true` (CDNA scratch `acc` bit), addressing the accumulator file directly; no bridge. CDNA-only.
- **SGPRs** — bridged through one VGPR (`plan.spill_bridge_vgpr`): `v_writelane` then a scratch store in the prologue; a scratch load, load-wait, then `v_readlane` in the epilogue. The single bridge is reused, so each SGPR restore is its own load/wait/readlane.
- **Waits/drains** — `build_wait_stores_complete` drains the stores before the call (a WAR guard on the source registers, and on RDNA4 orders each store ahead of its reload, since RDNA4 tracks stores on STORECNT which `s_wait_loadcnt` misses); `build_wait_loads_complete` guards the reloads. The in-flight-load drains that bracket the whole envelope are emitted by `emit_probe_call` (so they also cover no-spill sites), not by the bracket.
- **EXEC full-mask** — when the site spills, `emit_probe_call` forces `EXEC = -1` around the spill store and load so lanes inactive at the anchor still round-trip, restoring the anchor mask before the `s_swappc` (so the probe runs under the real mask) and re-widening before the reloads.

Per-arch specifics (scratch encodings, waitcnt split) live in `code/builders/spill_builders.h`; the bracket logic itself is arch-generic.

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

Combines `is_relocatable_anchor()` with milestone-scoped policy checks against `pt`: `filter_flags` is zero, `kind` is `BeforeInst`, `force_full_exec` is false. The `probe_obj`/`probe_symbol` pair is consumed (not rejected): consistency (both set or both empty) and symbol resolution happen during point resolution. On success returns a `ResolvedInstrumentationSite` with the captured anchor snapshot (offset, size, original bytes, mnemonic, kind, and the resolved probe index when this is a probe call).

### `validate_inline_nop_plan(plan, error_out)`

Defense-in-depth check that the orchestrator-produced `TrampolinePlan` matches the canonical inline-nop shape: exactly one `before_items` entry containing `s_nop 0`, empty `after_items`, `emit_original == true`. Lives at the orchestrator boundary rather than inside the builder so the builder stays generic. Deleted once arbitrary inline-asm bodies are supported.

---

## Data Types

| Type | Stage | Carries |
| --- | --- | --- |
| `InstrumentationPoint` | request | `anchor_offset`, `kind`, `probe_obj`, `probe_symbol`, reserved fields |
| `ResolvedInstrumentationSite` | post-validation | `anchor_offset`, `original_size`, `original_bytes`, `mnemonic`, `kind`, `probe_index` (set for probe calls) |
| `ProbeCallable` | probe registry | resolved probe `symbol`, `arch`, calling convention, `body_words`, `output_text_offset` |
| `TrampolinePlan` | builder input | `arch`, `anchor_offset`, `original_size`, `original_words`, `trampoline_offset`, `return_target`, `before_items`, `after_items`, `emit_original`, `kernel_sgpr_count`; probe-call: `is_probe_call`, `probe_target_offset`, `link_pair_base`, `target_pair_base`, `scc_temp`, `preserve_scc`, `preserve_exec`, `preserve_vcc`, `preserve_m0`, `special_state_saves`, `vgpr_spills`, `sgpr_spills`, `acc_spills`, `spill_bridge_vgpr`, `before_word_count`, `builder_clobbers` |
| `TrampolineBytes` | builder output | `patched_anchor_bytes`, `trampoline_words` |
| `InstrumentationPatch` | per-site summary (test/debug) | `anchor_offset`, `original_size`, `trampoline_offset`, `return_target`, `original_bytes`, `patched_anchor_bytes`; probe-call: `is_probe_call`, `probe_symbol`, `probe_target_offset`, `link_pair_base`, `target_pair_base` |
| `InstrumentedCodeObject` | `patch()` output | `elf_bytes`, `errors`, `warnings` |
| `InstrumentedCodeObjectDebug` | `patch_with_debug_summaries()` output | `InstrumentedCodeObject` + `patches` |

The intermediate site/plan types are expected to thicken as the framework grows (e.g. `ResolvedInstrumentationSite` likely gains an ordered list of bodies once multi-point coalescing lands; a layout/negotiation stage will appear between planning and splicing).

---

## Supporting Modules

### Code Object Patcher (`code/patch/code_object_patcher.h`) [shared with DBT]

Owns ELF-level mutations. The DBI orchestrator reads the original payload via `text_bytes()`, assembles the new `.text` locally (each anchor spliced in place, then every trampoline appended after the original bytes with `append_words()`), and applies it with a single `replace_text()` before `emit()` returns the patched ELF buffer. When a site spills, the orchestrator also calls `set_private_segment_fixed_size(descriptor_file_offset, SpillManager::total_private_bytes())` to grow the kernel's scratch reservation to cover the DBI spill zone before `replace_text`. The patcher accepts any well-formed mutation request; layout decisions stay in the orchestrator. Trampolines live inside `.text` as a local code cave (the same layout DBT uses) rather than a separate section, so cave offsets are plain `.text`-relative bytes in `[text_size, text_size + cave_bytes)`.

### Instruction Builder (`code/builders/instruction_builder.h`) [shared with DBT]

ISA-parameterized helpers for encoding common instructions (`s_branch`, `s_nop`, `s_mov_b32`/`s_mov_b64`, `s_cselect_b32`, `s_getpc_b64`, `s_swappc_b64`, etc.) and the SOPP branch math (`compute_sopp_branch_simm16`). Used by both the trampoline builder and the DBT code-cave path. SOPP format is identical across AMDGPU generations but opcodes differ; always go through these helpers, never hardcode opcodes. (Moved from `code/patch/` to `code/builders/` alongside `spill_builders.h`.)

### Spill Builders (`code/builders/spill_builders.h`) [DBI-only]

Multi-word, generation-specific encoders for the spill bracket, split out from the scalar helpers because their prefixes/opcodes move by ISA and they return variable-length word lists. Not shared with DBT (which emits scratch through its own target-specific path):
- `build_scratch_store_dword` / `build_scratch_load_dword` — per-lane scratch store/load. CDNA3/CDNA4 use the gfx9 FLAT `seg=SCRATCH` encoding (2 words, 13-bit signed offset, `lds`=0); RDNA4 uses the dedicated VSCRATCH encoding (3 words, 24-bit offset, `sve`=0). Both take an `acc` flag that, on CDNA, sets the FLAT `acc` bit to address the AccVGPR file directly (RDNA throws — no acc file).
- `build_v_writelane_b32` / `build_v_readlane_b32` — the SGPR↔VGPR lane bridge (VOP3; CDNA prefix `0x34`, RDNA `0x35`).
- `build_wait_loads_complete` / `build_wait_stores_complete` — the async-access fences. CDNA uses a unified `s_waitcnt`; RDNA4 splits into `s_wait_loadcnt` (loads on LOADCNT) and `s_wait_storecnt` (stores on STORECNT). `build_wait_all_loads_complete` is the boundary drain used by `emit_probe_call`.

An unmodeled arch throws `UnimplementedInst`. The hard arch gates on spilling are these five builders, `max_scratch_offset_bytes`, and — in the orchestrator — `arch_has_accvgpr` (AccVGPR spills require an AGPR file) and `arch_has_unified_vgpr_allocation` (which selects the descriptor's ACCUM_OFFSET split used to size the AccVGPR window).

### Kernel Descriptor Scan (`code/kernel_descriptor_scan.h`) [shared with DBT]

Enumerates a code object's kernel descriptors and derives per-kernel allocation facts. `scan_kernel_descriptors(image, text_offset, text_size)` returns each kernel's descriptor file offset, entry, and `private_segment_fixed_size`, with overflow-safe extent checks and a descriptor-bounded-by-owning-section guard (rejects malformed ELFs). `kernel_wavefront_size` and `descriptor_vgpr_granularity_for_wavefront` decode the wave-size-dependent VGPR encoding granule (shared with DBT so the two cannot diverge); the orchestrator multiplies `(GRANULATED_WORKITEM_VGPR_COUNT + 1)` by that granule to get the kernel's VGPR count. The orchestrator currently rejects anything but a single kernel.

### Register Liveness Analysis [shared with DBT]

**Files:** `code/analysis/liveness.h`, `code/analysis/liveness.cpp`, `code/analysis/def_use_chain.h`, `code/analysis/def_use_chain.cpp`
**Used by:** DBT semantic translator; the DBI probe-call register planner (liveness at the anchor feeds dead-register selection for the link/target pairs and SCC temp, and the spill-set computation)

Kernel-scoped backward register liveness over the CFG embedded in `BasicBlock`. Callers construct a `LivenessAnalysis` from one `KernelBlockScope` (the blocks reachable from one kernel descriptor entry). Successor/predecessor edges that leave the scope are ignored, so one decoded code object containing N kernels yields N independent analyses.

#### What it tracks

Ordinary SGPRs, VGPRs, and AccVGPRs via `RegisterSet`. `InstDefUse` records explicit operand defs and uses plus instruction-level implicit hooks; the only implicit hook today is the FLAT `saddr` SGPR-pair use that does not appear as an explicit operand.

#### What it does NOT track

- EXEC, VCC, SCC, M0, FLAT_SCRATCH, TTMP — special architectural state. The `RegClass` enum names these, but they are not in the backward-liveness dataflow set (`RegisterSet` is SGPR/VGPR/AccVGPR only). See the special-state note below for how they are still preserved.
- Cross-kernel CFG. Edges that leave the kernel scope are silently dropped.
- Memory dependencies. Liveness is purely register-based.

#### Special state (EXEC / VCC / M0)

EXEC, VCC, and M0 *writes* are surfaced by the decoder as special-state operands (e.g. `v_cmp` → VCC, `v_cmpx` → EXEC), so a probe's `ProbeClobberSummary.touches_{exec,vcc,m0}` are set and drive preservation. They are **not** part of the backward-liveness `RegisterSet` dataflow, so preservation is *save-when-clobbered*, not save-when-live. Because implicit-def detection is not proven comprehensive (a truly operand-less implicit def could be missed), EXEC and VCC are preserved **unconditionally** as a safety net; M0 is preserved only when a write is detected. This is why the trampoline can reserve EXEC/VCC/M0 temps even though liveness never reports them.

The same `RegisterSet` limit bounds the probe live-in gate in the other direction: a probe that *reads* special state before writing it is not detected, because there are no bits to report it with. On GFX9/CDNA a `ds_*` instruction implicitly reads M0, which a kernel prologue initializes, so an LDS-using probe carries an input `analyze_probe_live_ins` is blind to.

### Probe Live-Ins

**Files:** `code/patch/probe_live_in.h`, `code/patch/probe_live_in.cpp`

The complement of `probe_clobber`: what does the probe body *read* before defining it? `analyze_probe_live_ins()` builds the probe object's CFG (probe entry passed as an `extra_split_point`), forms the block scope with `reachable_kernel_blocks()`, runs `LivenessAnalysis` over it, and returns the entry block's live-in set minus the registers the calling convention supplies. Today that subtraction is the `s[30:31]` return link alone.

A non-empty result is a probe expecting state that only exists at kernel entry — `workitem_id_x` in `v31`, written by the kernel's own prologue — which a trampoline at an arbitrary anchor cannot reproduce. `Instrumentor::resolve_points()` rejects the site. This is also the mechanism a convention that *declares* argument registers gets checked against, so it precedes any argument passing rather than competing with it.

CFG liveness rather than a linear scan of the body words: a def under a forward branch does not reach a use after the join, so a linear walk would report a smaller footprint than the body has.

The analysis sets `LivenessAnalysisOptions::exec_masked_defs_kill`, so a masked vector write counts as writing the whole register. Without it no masked def is ever a kill and every VGPR the probe reads — including ones it defines itself — is reported as an input. The probe owns its own EXEC; the option's doc states what that obliges.

Fail-closed, each with a distinct message: an unrecognized convention; gfx1250/CDNA5, whose vector operands resolve only from a block where `MODE.VGPR_MSB` is known zero, which an anchor does not guarantee; relative (`v_movrel*`) or GPR-indexed (`MODE.GPR_IDX_EN`) VGPR access, which displaces an encoded index at runtime and so loses live-ins rather than inventing them; a probe object without exactly one `.text`, or a body outside it, since `BasicBlock::build()` decodes nothing else and restarts its offsets per section; a body ahead of its section; no decoder for the arch; an undecodable `.text`; and an entry offset that starts no decoded block.

### SpillManager

**Files:** `code/patch/spill_manager.h`, `code/patch/spill_manager.cpp`

Per-kernel scratch-layout planner for DBI spill/fill slots. Probe-call trampolines use it to reserve byte offsets within per-lane scratch where saved SGPRs / VGPRs / AccVGPRs go before a probe runs and from which they are restored after; the orchestrator writes the bumped `total_private_bytes()` back into the descriptor (see [Code Object Patcher](#code-object-patcher-codepatchcode_object_patcherh-shared-with-dbt)). Not consumed by the inline-nop pipeline. Slots are laid out above the kernel's existing scratch via the shared `PrivateSegmentCursor` (an aligned, overflow-safe byte-range allocator, defined inline in `spill_manager.h`), so DBI slots can start above DBT's high-water mark when both passes run.

#### Responsibilities

- Reserve a "DBI spill zone" appended above the kernel's existing `private_segment_fixed_size`, aligned to 16 bytes.
- Hand out stable per-register byte offsets within that zone. Registers cannot get more than one offset.
- Enforce a hard per-lane scratch cap. Allocations that would push the bumped total past the cap fail; on failure the manager state is unchanged.
- Compute the bumped `private_segment_fixed_size` that the kernel descriptor patcher will write back.

#### What it is not

- Not a memory allocator. SpillManager only computes layout.
- Not the code generator. SpillManager hands out offsets; emitting the actual `scratch_store` / `scratch_load` (or the writelane/readlane bridge and `acc`-bit variants) is the trampoline builder's job (see [Spill Builders](#spill-builders-codebuildersspill_buildersh-dbi-only)).

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
    s_nop 0                     <-- inline-nop body (the probe-call variant is described below)
    <relocated original word(s)> <-- 4 or 8 bytes, same encoding as the anchor
    s_branch <return>           <-- back to anchor_offset + original_size
```

Branch offsets are computed in SOPP `simm16` units; the trampoline must lie within `±32768 * 4` bytes of the anchor. For unusually large kernels this would require trampoline islands, which remain future work (same constraint as DBT code caves).

---

## Instrumentation Flow (probe-call)

A probe-call point sets `probe_obj` + `probe_symbol`. Resolution copies the probe's self-contained body (`rj_nop_probe`: `s_waitcnt` then `s_setpc_b64 s[30:31]`) once into the cave; sites sharing a `(probe_obj, probe_symbol)` reuse that one copy. The cave is laid out as: original kernel bytes, then each distinct probe body, then the per-site trampolines. The forward `s_branch` at the anchor targets the trampoline (not the body); the trampoline *calls* the body via `s_swappc_b64`.

The trampoline envelope wraps the relocated original:

```
.text:
  [original kernel bytes]   @ anchor_offset: s_branch <trampoline>
  ...
  [copied probe body]       @ probe_target_offset:
    s_waitcnt ...
    s_setpc_b64 s[30:31]    <-- returns through the link pair
  [trampoline]              @ trampoline_offset:
    <in-flight-load drain>              <-- s_wait_loadcnt / s_waitcnt; also emitted on no-spill sites
    s_mov_b64    <exec_temp>, exec      <-- EXEC save (preserve_exec, or any spilling site)
    s_mov_b64    <vcc_temp>,  vcc       <-- VCC save  (preserve_vcc)
    s_mov_b32    <m0_temp>,   m0        <-- M0 save   (preserve_m0)
    s_mov_b64    exec, -1               <-- widen to full mask for the stores (spilling site)
    [spill prologue]                    <-- VGPR/acc direct stores; SGPR writelane + store; store wait
    s_mov_b64    exec, <exec_temp>      <-- restore anchor mask so the probe runs masked (spilling site)
    s_cselect_b32 <scc_temp>, 1, 0      <-- SCC save (preserve_scc)
    s_getpc_b64  s[target_pair]
    s_add_u32    s[target_lo], s[target_lo], (probe_target - pc)@lo
    s_addc_u32   s[target_hi], s[target_hi], (probe_target - pc)@hi
    s_swappc_b64 s[30:31], s[target_pair]   <-- call: PC=body, return->s[30:31]
    <in-flight-load drain>              <-- guards restores/host against the probe's own loads
    s_cmp_lg_u32 <scc_temp>, 0          <-- SCC restore
    s_mov_b64    exec, -1               <-- re-widen to full mask for the loads (spilling site)
    [spill epilogue]                    <-- VGPR/acc direct loads; SGPR load + wait + readlane; load wait
    s_mov_b64    exec, <exec_temp>      <-- EXEC restore
    s_mov_b64    vcc,  <vcc_temp>       <-- VCC restore
    s_mov_b32    m0,   <m0_temp>        <-- M0 restore
    <relocated original word(s)>
    s_branch <return>                   <-- back to anchor_offset + original_size
```

Lines above are conditional: the EXEC/VCC/M0 saves and restores appear only when that register is preserved (EXEC also rides in whenever the site spills); the four `exec, -1` / `exec, <exec_temp>` toggles appear only on a spilling site; the `[spill prologue]` / `[spill epilogue]` are empty when `spill_set` is empty (leaving the plain SCC-bracketed call). See [TrampolineBuilder → Spill bracket](#spill-bracket) for the bracket contents. The `s_getpc_b64` + 64-bit add chain is `.text`-relative, so the materialized target is load-base-independent; the `±simm16` branch range only constrains the forward/return `s_branch`es, not the call.

### Register requirement

The probe's calling convention fixes the **link pair** at `s[30:31]` (`AmdGpuFuncNoArgsReturnS30S31`): `s_swappc_b64` writes the return address there and the body's `s_setpc_b64 s[30:31]` reads it back. The planner additionally picks a dead, even-aligned **target pair** (holds the materialized address) and a dead **SCC temp** from the anchor's liveness. All of these must be *granted by the kernel's SGPR allocation*.

The planner also reserves, from the same dead-SGPR pool bounded by `plan.kernel_sgpr_count`, a temp per preserved special register (an even pair for EXEC/VCC, a single for M0) and — for SGPR spills — a bridge VGPR from the kernel's ordinary-VGPR range (`min(kernel_vgpr_count, accum_base)`, so it can never alias an AccVGPR). EXEC is reserved whenever the site spills, not just when the probe clobbers it, because the store/load run under a forced full mask.

The instrumentor does **not yet grow the kernel's SGPR count**, so the kernel must already allocate through `s31` (and through any special-state/bridge temps). Until auto-growth lands, the hardware smoke test instruments a register-padded fixture kernel (`vector_add_probe.hip`, `.sgpr_count` ≥ 32). Resource policy fails closed when the probe body reads a register its convention does not supply, when the link pair is live at the anchor, when no dead target pair or required temp is available within the kernel's allocation, when the probe clobbers FLAT_SCRATCH (the spill store/load depend on it), when the kernel has zero scratch or is one of several kernels, when a spill offset exceeds the arch's scratch-offset field, (for SGPR spills) when no dead bridge VGPR exists in the ordinary-VGPR range, or (for AccVGPR spills) when the target has no AccVGPR file (`arch_has_accvgpr` is false) or an AccVGPR index falls outside the descriptor-derived accumulator window (`kernel_vgpr_count − accum_base`). The spill set itself is `instrument_clobbers ∩ live_at_anchor` and is spilled, not rejected.

---

## Testing

- **Unit (`tests/patch/instrumentor_test.cpp`):** Validator coverage (each rejection path on synthetic anchors, including the bounds-overflow regression for `is_relocatable_anchor`), inline-nop plan guardrail, `make_trampoline_plan`, end-to-end `patch()` on a synthetic ELF (expected anchor splice + trampoline layout + reparse), a decoded round-trip of every word in the emitted trampoline, the probe-call path (probe body copied once and the trampoline call targets it), the spill formula, per-class spill planning (`plan_vgpr/sgpr/acc_spills` — ascending slots, class/arch/offset-cap rejections, the kernel-VGPR-count bridge bound), the drain ordering (`expect_drain_before_store` / `expect_drain_after_return`), EXEC/VCC/M0 preservation (incl. unconditional EXEC/VCC), and the fail-closed cases (FLAT_SCRATCH clobber, zero-scratch, SGPR temp past the kernel allocation).
- **Spill sim e2e (`tests/dbi/dbi_spill_sim_test.cpp`):** Runs the full `s_swappc` envelope + spill bracket through `DbiSim` and reads back registers after execution, each with a negative control that nops the restore. Fixtures cover VGPR, SGPR (VGPR-bridged), two-SGPR, reused-spilled-bridge, AccVGPR (CDNA), combined VGPR+SGPR+ACC, EXEC preserve (wave32 partial mask), full-mask EXEC-widen spill, and probe-runs-under-anchor-mask — parameterized across **CDNA3, CDNA4, and RDNA4** (AGPR is CDNA-only).
- **Unit (`tests/code/kernel_descriptor_scan_test.cpp`):** Single-kernel scan, and the malformed-ELF rejections (unterminated `.kd` name, section-header-table / symtab-range overflow, descriptor crossing its owning section).
- **Unit (`tests/patch/trampoline_builder_test.cpp`):** Builder byte-layout contract, branch math, arch-honoring opcode selection, INT16 limit boundary cases.
- **Unit (`tests/patch/instruction_builder_test.cpp`):** `compute_sopp_branch_simm16` boundary / alignment / overflow / negative-unaligned-delta.
- **Unit (`tests/patch/probe_symbol_test.cpp`, `probe_callable_test.cpp`, `probe_clobber_test.cpp`):** Probe symbol resolution (missing / duplicate / undefined / non-executable / zero-size rejection), `ProbeCallable` construction, and the `rj_nop_probe` clobber summary (empty ordinary clobbers, no special state).
- **Unit (`tests/patch/probe_live_in_test.cpp`):** Probe input footprint — `v31` read cold is reported, the same body defining `v31` first is not, a def under a forward branch (which a linear scan would miss) is reported, scalars both ways, the link pair excluded — plus every fail-closed path: unknown convention, gfx1250, relative / GPR-indexed VGPR access, multi-`.text` probe object, symbol outside `.text`, body ahead of its section, missing decoder, undecodable `.text`, and an entry that starts no block.
- **Probe fixture (`tests/dbi/probe_fixture_test.cpp`, gated on `HAS_PROBE_FIXTURES`):** Resolves `rj_nop_probe` in the real amdclang++-compiled gfx90a device ELF, confirms the body returns via `s_setpc_b64 s[30:31]`, builds the callable, and checks the clobber summary. No GPU required.
- **Static DBI smoke (`tests/dbi/hsa_dbi_nop_asm_test.cpp`, `HsaDbiNopAsmCdna2Static`):** Loads a real compiled gfx90a `vector_add` ELF, runs `Instrumentor::patch()`, asserts the patched ELF differs from the original, decodes the anchor as `s_branch`, and confirms `.text` grew to hold the appended trampoline cave. No GPU required.
- **Hardware DBI smoke (`HsaDbiNopAsmCdna2Hardware`, gated on `HAS_CDNA2_GPU`):** Three tests on a real gfx90a GPU — patched ELF loads + validates via HSA; dispatched kernel produces bit-identical output to the original (the inline-nop placeholder is a no-op); a *sabotage* test overwrites the trampoline's `s_nop 0` with `s_endpgm 0` and asserts the kernel actually fails, proving the GPU genuinely executes the trampoline path rather than silently bypassing the splice. `hsa_init` / `hsa_shut_down` and agent enumeration run once per suite via `SetUpTestSuite` / `TearDownTestSuite`; the agent is located by matching the target's `DbiTargetParams::isa_substring` against the HSA ISA name, so the same fixture binds to a real GPU or to whichever agent the CLI launcher supplies.
- **Static probe-call smoke (`tests/dbi/hsa_dbi_nop_probe_test.cpp`, `HsaDbiNopProbeCdna2Static`, gated on `HAS_PROBE_FIXTURES`):** Instruments the register-padded `vector_add_probe` kernel with a probe call to `rj_nop_probe`, then asserts the patch shape — anchor decodes as `s_branch`, `.text` grew, the copied probe body matches the resolved body and ends in `s_setpc_b64`, and the trampoline contains the `s_swappc_b64` to it. No GPU required.
- **Hardware probe-call smoke (`HsaDbiNopProbeCdna2Hardware`, gated on `HAS_CDNA2_GPU`):** Load + validate; dispatch the probe-call-patched kernel and confirm bit-identical output to the original (the no-op probe is transparent); a *sabotage* test overwrites the copied probe body's first word with `s_endpgm` and asserts the wave terminates, proving the `s_swappc_b64` genuinely transfers control into the body. **Preconditions:** these dispatching cases require the [register-padded fixture kernel](#instrumentation-flow-probe-call) *and* the SMEM SBASE operand decode fix in the branch's base; without both the probe-call wave hangs the GPU.
- Run with `build/tests/rocjitsu_tests`, `build/tests/hsa_dbi_nop_asm_test`, and `build/tests/hsa_dbi_nop_probe_test`.
