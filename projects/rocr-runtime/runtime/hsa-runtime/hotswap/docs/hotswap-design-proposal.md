# HotSwap Design Proposal: Binary Translation Architecture

## Status (2026-04-15)

This document proposes a tiered architecture for AMD GPU binary translation,
comparing the original HotSwap transpiler with an LLVM IR-based raiser
approach. It is intended as a reference for design discussions and
stakeholder review.

---

## 1. Problem Statement

Pre-compiled GPU binaries need to run on hardware they were not compiled for.
This requirement is driven by three forces:

1. **Library portability across GPU generations.** Pre-compiled code objects
   (`.co` / HSACO files) shipped in frameworks like AITER, CK, and hipBLAS-Lt
   target a specific ISA (e.g., gfx950). When users deploy on different
   hardware (gfx942, or future generations), those binaries must either be
   recompiled from source or translated at load time.

2. **Pre-compiled wheels and packages.** Python wheels and container images
   bundle GPU binaries for a single ISA. Source recompilation is not an option
   for end users who install via `pip install` or pull a Docker image.

3. **Forward compatibility.** New hardware should run existing binaries without
   waiting for library maintainers to ship updated packages.

Binary translation at load time solves all three cases transparently — no
source changes, no recompilation, no relinking.

---

## 2. Solution Architecture

The architecture is organized in two tiers, reflecting two fundamentally
different translation scenarios:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      HIP Application                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Code object loading (ROCR runtime)                                 │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Tier 1: Same-Family Retarget                                  │  │
│  │                                                               │  │
│  │ When source and target share binary encoding (e.g., gfx950    │  │
│  │ and gfx942 within CDNA):                                     │  │
│  │                                                               │  │
│  │   1. Pass through 98%+ of instructions unchanged              │  │
│  │   2. NOP/trampoline the delta instructions (1-2%)             │  │
│  │   3. Patch ELF metadata (e_flags, .note ISA)                  │  │
│  │                                                               │  │
│  │ Properties: O(delta_instructions), sub-millisecond,           │  │
│  │             zero performance overhead, production-deployed     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Tier 2: Cross-Family Translation                              │  │
│  │                                                               │  │
│  │ When source and target have different encodings (e.g.,        │  │
│  │ RDNA → CDNA, or future UDNA → legacy):                       │  │
│  │                                                               │  │
│  │   Full semantic translation of every instruction.             │  │
│  │   Two candidate architectures — see Section 3.                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Same-Family Retarget (Production)

The existing HotSwap retarget path. For ISA pairs that share binary encoding
(gfx942 ↔ gfx950, and historically most intra-CDNA and intra-RDNA pairs):

| Property | Value |
|----------|-------|
| Mechanism | Encoding-compatible pass-through + surgical NOP/trampoline |
| Code | ~500 LOC of instruction-specific handlers in `hotswap.cpp` |
| Coverage | 1,315/1,315 kernels load, 17/17 execution tests pass |
| Performance | 1.000x geomean (zero overhead) |
| Accuracy | 15/20 bit-identical; 5/20 approximate (FP4/bf16 emulation) |
| Constraint | Requires shared binary encoding between source and target |

**This tier is not under discussion.** It is the correct design for its scope.

### Tier 2: Cross-Family Translation (Under Discussion)

When source and target ISAs do not share binary encoding, every instruction
must be semantically translated. Two architectures are candidates:

| | Text-Level Transpiler | LLVM IR Raiser |
|---|---|---|
| **Implementation** | `transpiler.cpp` (~5,800 LOC) | `raiser.cpp` (~1,350 LOC) |
| **Representation** | Assembly text (string manipulation) | LLVM IR (typed, SSA, structured) |
| **Maturity** | 42/42 unit tests, 18/20 complex kernels | 27/27 raise rate on production kernels |
| **Target decoupling** | Hardcoded per ISA pair | `llc -mcpu=<target>` (any LLVM-supported ISA) |
| **Scaling model** | O(source x target) — new transpiler per pair | O(source + target) — new raiser per source ISA |
| **EXEC mask** | Explicit text-pattern widening | Scalar boolean (unprincipled for divergent flow) |
| **Wait counters** | Explicit cross-instruction tracking | Delegated to LLVM backend |
| **Register allocation** | Hardcoded temporaries | LLVM PromoteMemToReg + backend RA |
| **Known limitations** | LLVM MC state bug (1 code object/process); MFMA NOP-out; per-pair effort | EXEC divergence undetected; single-kernel assumption; not yet execution-validated cross-family |

---

## 3. Decision Matrix

The choice between architectures depends on which translation scenarios the
project must support. Each scenario favors a different approach:

### 3a. Scenario Analysis

| Scenario | Description | Transpiler | IR Raiser | Assessment |
|----------|-------------|------------|-----------|------------|
| **A: Same encoding, few new instructions** | Future CDNA adds ~20 new opcodes but keeps gfx950 encoding (e.g., CDNA5) | Add 20 NOP/trampoline entries. Low effort, high confidence. | Add 20 instruction handlers. Works but decompiles/recompiles 100% of instructions — overkill. | **Transpiler wins.** Tier 1 retarget handles this naturally. |
| **B: Same encoding family, many new instructions** | Encoding format unchanged but 10%+ new instructions with no equivalent on target | Moderate effort per instruction. Surgical approach still viable but trampoline count grows. | Each new instruction is a ~5-line handler. Batch approach handles volume naturally. | **Depends on volume.** Below ~50 new instructions, transpiler; above, IR raiser starts to pay off. |
| **C: Cross-family (RDNA ↔ CDNA)** | gfx1250 → gfx950, or any pair spanning ISA families | ~5,800 LOC per ISA pair. Proven for one pair. Each new pair is a fresh ~5,000 LOC effort. | ~1,350 LOC raiser for source ISA; target handled by LLVM backend. New pair = new raiser OR new `-mcpu`. | **IR raiser wins at scale.** For 1 pair the transpiler is cheaper; for N pairs the IR raiser dominates. |
| **D: Slightly different encodings within family** | Future CDNA where most encodings match but some formats shift (e.g., VOP3 encoding ID changes by 1 bit) | Batch assembly fallback exists but is blocked by LLVM MC state bug (1 code object/process). Needs MC fix. | Handles naturally — different MCSubtargetInfo for disassembler, `llc` for reassembly. | **IR raiser has structural advantage.** Transpiler viable if MC state bug is fixed. |
| **E: Radical hardware change** | Wave128, UDNA, new memory model, new compute primitives | Requires rewriting core transpiler logic. Multiplicative cost across all ISA-pair transpilers. | Semantic model needs updating in one place (raiser). Backend handles new target. | **IR raiser wins.** Single point of change vs. multiplicative rewrite. |
| **F: Completely new ISA family (UDNA)** | Every existing ISA becomes a cross-family target relative to UDNA | New ~5,800 LOC transpiler for EVERY (existing ISA, UDNA) pair. | One ~1,350 LOC raiser for UDNA source side; existing LLVM backends handle legacy targets. | **IR raiser wins decisively.** O(source + target) vs. O(source x target). |

### 3b. Effort Estimates

| Investment | Transpiler Path | IR Raiser Path |
|------------|-----------------|----------------|
| **Support next CDNA (encoding-compatible)** | ~1 week (Tier 1 retarget handles it) | ~1 week (same — Tier 1) |
| **Support one cross-family pair** | ~3–4 months (proven: transpiler took ~90 commits) | ~2–3 months to production quality (EXEC detection, multi-kernel, execution validation) |
| **Support N additional cross-family pairs** | ~2–3 months each (new transpiler per pair) | ~2–4 weeks each (new raiser per source ISA; target reused) |
| **Handle radical ISA change** | ~3–6 months (rewrite core logic across all transpilers) | ~1–2 months (update raiser semantic model; backend adapts) |
| **Fix LLVM MC state bug** | Required for batch fallback path | Not needed (uses `llc` subprocess) |

### 3c. Risk Assessment

| Risk | Transpiler | IR Raiser |
|------|-----------|-----------|
| **Encoding compatibility breaks** | HIGH — entire Tier 1 value proposition lost; MC state bug blocks fallback | LOW — different encodings are just different disassembler configs |
| **EXEC divergence in real kernels** | LOW — transpiler handles EXEC explicitly | HIGH — scalar boolean model silently wrong; needs conservative detection |
| **LLVM MC global state bug** | HIGH — limits to 1 code object per process for non-encoding-compatible pairs | LOW — not affected (subprocess isolation) |
| **Maintenance burden** | MEDIUM — 5,800 LOC per pair, text-level patterns fragile across LLVM versions | LOW — 1,350 LOC, metadata-driven, structurally correct by construction |
| **Performance regression from translation** | LOW — surgical patching preserves original schedule | MEDIUM — full recompilation may produce different scheduling/register allocation |

---

## 4. Recommendation

### Hybrid Architecture

Both design analysis documents independently arrive at the same conclusion:
a tiered hybrid is the strongest architecture.

**Tier 1 (keep as-is):** The surgical same-family retarget for
encoding-compatible ISA pairs. Fast, proven, zero-overhead. No changes
needed unless encoding compatibility breaks.

**Tier 2 (invest):** The LLVM IR raiser for cross-family translation.
The O(source + target) scaling property and ISA decoupling make it the
right long-term architecture. Key milestones to production quality:

| Priority | Work Item | Effort | Impact |
|----------|-----------|--------|--------|
| 1 | Conservative EXEC divergence detection | 2–3 weeks | Upgrades from silently-wrong to fail-loudly; makes the raiser honest |
| 2 | Multi-kernel code object support | 1 week | Fixes the single-kernel assumption (HIGH severity) |
| 3 | Cross-family validation (raise gfx950 → compile for gfx942, execute) | 2–3 weeks | Proves end-to-end correctness, not just raise rate |
| 4 | Instruction coverage to ~200 mnemonics | 2–4 weeks | Covers remaining production kernel gaps |
| 5 | ROCR integration (replace `std::system()` with in-process `llc`) | 2–3 weeks | Production deployment readiness |

**The transpiler remains available** as the proven fallback for cross-family
translation until the IR raiser reaches production quality. No existing code
is discarded.

### Decision Criteria

The recommendation above holds IF any of these are true:
- More than 2 cross-family ISA pairs will be needed in the next 2 years
- UDNA or a unified ISA is on the hardware roadmap
- Encoding compatibility within CDNA may break in future generations

The recommendation shifts to "extend transpiler only" IF all of these are true:
- Only 1 cross-family pair will ever be needed
- CDNA encoding compatibility is guaranteed for 3+ generations
- UDNA is not on the roadmap

---

## 5. Open Questions

These questions should be answered to finalize the investment decision:

| ID | Question | Why It Matters |
|----|----------|----------------|
| Q1 | What concrete ISA pairs must we support in the next 12 months? | Determines whether Tier 1 alone is sufficient |
| Q2 | Will future CDNA generations maintain encoding compatibility with gfx950? | If yes, Tier 1 scales indefinitely. If no, Tier 2 is mandatory. |
| Q3 | Is UDNA (unified RDNA+CDNA) on the roadmap? What timeframe? | UDNA makes every existing ISA a cross-family target — the O(source x target) transpiler model becomes untenable |
| Q4 | Are we optimizing for "runs somehow" or "production-quality translation"? | Determines acceptable correctness/performance thresholds |
| Q5 | What is the staffing budget for hot swap in the next 6 months? | Determines whether we can invest in IR raiser maturation or must stay with proven transpiler |
| Q6 | Is there a deadline for cross-family support? | Deadline pressure favors transpiler (proven now); no deadline favors IR raiser (better long-term) |
| Q7 | Must we support third-party pre-compiled code objects? | If only our own code: recompilation is an alternative to translation. If third-party: binary translation is the only option. |

---

## 6. Appendix: Evidence

### A. Same-Family Retarget (Tier 1)

- **Coverage:** 1,315/1,315 AITER kernels load successfully (100% load rate)
- **Performance:** 1.000x geomean across 20 AITER kernels (zero overhead)
- **Accuracy:** 15/20 bit-identical; 5/20 degraded (FP4/bf16 approximation)
- **Encoding delta:** 2,172 / 137,786 instructions (1.6%) need rewriting
- **Reference:** [ORIGINAL_HOTSWAP_DEEP_DIVE.md](ORIGINAL_HOTSWAP_DEEP_DIVE.md)

### B. Cross-Family Transpiler (Tier 2, Option A)

- **Coverage:** 42/42 unit tests, 18/20 complex transpiler kernels
- **Implementation:** ~5,800 LOC in `transpiler.cpp`, ~90 commits
- **Capabilities:** 500+ mnemonic mappings, wave32→wave64 widening, SALU float
  emulation, wait counter merging, scale-offset address lowering
- **Reference:** [gfx1250-on-gfx950-analysis.md](gfx1250-on-gfx950-analysis.md)

### C. LLVM IR Raiser (Tier 2, Option B)

- **Coverage:** 27/27 production gfx950 kernels raised (100% raise rate)
- **Corpus:** Flash Attention fwd/bwd, bf16 GEMM, FP8 block-scale GEMM, MoE,
  MLA, paged attention, topk-softmax — up to 10,173 instructions per kernel
- **Implementation:** ~1,350 LOC in `raiser.cpp`, metadata-driven
- **Strengths:** Typed IR, SSA, structural correctness (OpResolver, auto-SCC),
  standard LLVM backend integration, O(source + target) scaling
- **Gaps:** EXEC mask as scalar boolean (HIGH), single-kernel assumption (HIGH)
- **Reference:** [SHORTCUTS_AND_LIMITATIONS.md](../llvm_ir_proto/SHORTCUTS_AND_LIMITATIONS.md),
  [DESIGN_COMPARISON.md](../llvm_ir_proto/DESIGN_COMPARISON.md)

### D. Comparison Summary

| Dimension | Transpiler | IR Raiser |
|-----------|-----------|-----------|
| Same-family minor retarget | Excellent | Overkill |
| Cross-family principled-ness | Text manipulation, ad-hoc | Typed IR, metadata-driven |
| Cross-family maturity | 42/42 tests, 90 commits | 27/27 raise rate, ~10 commits |
| Scalability to new instructions | O(n), high constant | O(n), low constant |
| Scalability to new ISA pairs | O(source x target) | O(source + target) |
| EXEC mask handling | Explicit (text patterns) | Missing (scalar boolean) |
| Register allocation | Hardcoded temps | Alloca + PromoteMemToReg |
| Resilience to radical HW changes | New transpiler per pair | New raiser for source, reuse target |
| Production readiness | Deployed in ROCR | Research prototype |
