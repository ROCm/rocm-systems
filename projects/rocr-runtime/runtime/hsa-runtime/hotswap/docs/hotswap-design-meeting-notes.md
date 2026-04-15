# HotSwap Design Discussion — Meeting Notes

**Date:** 2026-04-15
**Context:** Decide on the binary translation architecture for cross-family
and future ISA support. Reference: [hotswap-design-proposal.md](hotswap-design-proposal.md)

---

## Agenda

| Time | Section | Goal |
|------|---------|------|
| 10 min | Problem statement alignment | Agree on which ISA pairs matter and when |
| 20 min | Architecture comparison | Walk through scenarios, identify where each approach wins |
| 15 min | Investment decision | Agree on path forward and next steps |

---

## Section 1: Problem Statement Alignment (10 min)

### What we have today

- **Tier 1 (shipping):** gfx950 → gfx942 same-family retarget. 1,315/1,315
  kernels load, zero perf overhead, 15/20 bit-identical. Exploits 98.4%
  encoding compatibility — only 1.6% of instructions need rewriting.

- **Tier 2 prototype A:** Cross-family text transpiler (gfx1250 → gfx950).
  5,800 LOC, 18/20 complex kernels passing. Built over ~90 commits against
  real AITER workloads.

- **Tier 2 prototype B:** LLVM IR raiser. 1,350 LOC, 27/27 production
  gfx950 kernels raised (100% raise rate). Metadata-driven, typed IR,
  standard LLVM backend integration. Two HIGH-severity gaps (EXEC mask,
  single-kernel).

### Questions to answer in this section

> **Q1: What concrete ISA pairs must we support in the next 12 months?**
>
> - gfx950 → gfx942 (done)
> - gfx1250 → gfx950?
> - Future CDNA (CDNA5) → gfx950?
> - Anything else?
>
> _This determines whether Tier 1 alone is sufficient._

> **Q2: Will future CDNA generations maintain encoding compatibility?**
>
> Historical pattern: GCN → CDNA encoding evolution has been incremental.
> gfx942 and gfx950 share identical encodings for 98.4% of instructions.
>
> If this holds for CDNA5+, the Tier 1 surgical retarget scales indefinitely.
> If it breaks, we need Tier 2 even for same-family pairs.

> **Q3: Is UDNA on the roadmap? What timeframe?**
>
> A unified RDNA+CDNA ISA would make every existing ISA a "cross-family"
> target. The O(source x target) transpiler model becomes N^2 work.
> The IR raiser's O(source + target) property would be essential.

> **Q4: Correctness bar — "runs somehow" or "production quality"?**
>
> - "Runs somehow": accept approximate results, 50% perf hit, some kernels fail
> - "Production quality": bit-accurate numerics, no perf regression, 100% kernels

---

## Section 2: Architecture Comparison (20 min)

### Walk through each scenario

For each scenario below, discuss: (a) which approach is better, (b) what
the effort would be, (c) what the risks are.

#### Scenario A: Same encoding, few new instructions

_Example: CDNA5 adds ~20 new opcodes, keeps gfx950 binary encoding._

- Tier 1 retarget handles this. Add NOP/trampoline entries for the new
  opcodes. ~1 week of work.
- Neither Tier 2 approach is needed.
- **Talking point:** This is the sweet spot for the existing design. If AMD
  continues incremental ISA evolution within families, we are covered.

#### Scenario B: Same encoding family, many new instructions

_Example: CDNA5 adds 100+ new instructions but encoding format is unchanged._

- Tier 1 still works but trampoline count grows. Each new instruction needs
  an emulation sequence.
- At some threshold (~50+ instructions needing emulation), the surgical
  approach becomes brittle — too many trampolines, NOP sleds may not have
  enough space, ELF growth becomes the norm.
- **Talking point:** Where is the breakpoint where surgical retarget stops
  being practical? Is there a historical precedent for large intra-family
  ISA jumps?

#### Scenario C: Cross-family (RDNA ↔ CDNA)

_Example: gfx1250 → gfx950 (the case we have prototyped)._

| | Transpiler | IR Raiser |
|---|---|---|
| LOC per pair | ~5,800 | ~1,350 (raiser) + reuse LLVM backend |
| Time to first pair | ~3-4 months (proven) | ~2-3 months to production quality |
| Time per additional pair | ~2-3 months | ~2-4 weeks (new raiser, target reused) |
| Maturity | 18/20 kernels | 27/27 raise rate (not execution-validated cross-family) |

- **Talking point:** How many cross-family pairs do we expect? If just
  gfx1250 ↔ gfx950, the transpiler's per-pair cost is acceptable. If we
  need N pairs, the IR raiser's scaling dominates at N >= 3.

#### Scenario D: Slightly different encodings within family

_Example: Future CDNA where VOP3 encoding ID shifts by 1 bit (like GFX12 did)._

- This breaks the Tier 1 "pass through 98.4% unchanged" property.
- The original HotSwap has a batch assembly fallback, but it is limited
  by the LLVM MC global state bug (1 code object per process).
- The IR raiser handles this naturally (different MCSubtargetInfo config).
- **Talking point:** Should we invest in fixing the LLVM MC state management
  issue? That would extend Tier 1's reach to this scenario without needing
  a new architecture. Estimated effort: unknown (LLVM internals).

#### Scenario E: Radical hardware change / UDNA

_Example: Wave128, unified ISA, new memory model._

- Transpiler: rewrite core logic. Multiplicative cost across all ISA-pair
  transpiler instances.
- IR raiser: update the raiser's semantic model in one place. LLVM backend
  handles new target.
- **Talking point:** This is where the architectural difference is starkest.
  The transpiler's per-pair architecture means every radical change is paid
  N times. The raiser pays once.

---

## Section 3: Investment Decision (15 min)

### Three options

| Option | Description | Risk | Effort |
|--------|-------------|------|--------|
| **1: Extend transpiler only** | Fix MC state bug, write new transpilers per pair | Low (proven tech) | High marginal cost per pair |
| **2: Invest in IR raiser** | Solve EXEC, multi-kernel, cross-family validation | Medium (newer, less proven) | High upfront, low marginal |
| **3: Hybrid** | Keep Tier 1; develop IR raiser for Tier 2 | Low (best of both) | Moderate total |

### Decision criteria

**Go with Option 1 (transpiler only) if ALL are true:**
- Only 1 cross-family pair needed, ever
- CDNA encoding compatibility guaranteed for 3+ generations
- UDNA not on the roadmap

**Go with Option 2 or 3 (invest in IR raiser) if ANY are true:**
- More than 2 cross-family pairs needed in next 2 years
- UDNA or unified ISA is on the hardware roadmap
- Encoding compatibility may break in future CDNA

### If we invest in the IR raiser, what is the path to production?

| Step | Work | Effort | Outcome |
|------|------|--------|---------|
| 1 | Conservative EXEC divergence detection | 2-3 weeks | Raiser refuses divergent kernels instead of producing wrong results |
| 2 | Multi-kernel code object support | 1 week | Handles all kernels in a `.co` file, not just the first |
| 3 | Cross-family execution validation | 2-3 weeks | Raise gfx950 → compile for gfx942 → run on MI300X → verify results |
| 4 | Instruction coverage expansion | 2-4 weeks | Cover remaining production kernel gaps (~200 mnemonics) |
| 5 | ROCR integration | 2-3 weeks | Replace subprocess `llc` with in-process compilation |

Total: ~10-14 weeks to production-quality Tier 2 via IR raiser.

---

## Key Points to Raise

### 1. The scalability cliff

The transpiler is 5,800 lines for ONE ISA pair. UDNA would require a
transpiler for every (existing ISA, UDNA) pair. The IR raiser's
O(source + target) property avoids this N^2 problem.

### 2. The encoding stability bet

The Tier 1 retarget's zero-overhead property depends entirely on encoding
compatibility. If this breaks — even slightly — the entire fast path
becomes the slow path. The batch assembly fallback has the MC state bug.

### 3. The "slightly expanded ISA" aligned path

If the team has strong signal that future ISAs will keep the same encoding
with only incremental instruction additions, we can potentially define an
aligned design:
- Tier 1 retarget for the common encodings (as today)
- Small, targeted emulation library for new instructions (NOP/trampoline)
- Avoid full cross-family translation entirely

This is the lowest-cost path but only works if the encoding stability
assumption holds. It should be explicitly discussed whether hardware
architects can commit to this property.

### 4. Correctness framing

Be honest about both approaches:
- Transpiler: production-proven coverage, but text-level manipulation is
  fragile. Each new instruction is a string transformation with no formal
  verification. 5,800 lines of implicit semantic knowledge.
- IR raiser: formal semantic model with structural correctness guarantees,
  but EXEC mask and single-kernel gaps mean it is not production-ready today.
  100% raise rate is not 100% execution correctness rate.

### 5. The LLVM MC state management ceiling

This affects the transpiler path more than the raiser path. The transpiler
relies on LLVM MC for both disassembly and reassembly. The AMDGPU backend's
global state only survives one MCContext lifecycle, limiting the batch
fallback to one code object per process. The raiser sidesteps this by
using `llc` as a subprocess (and would use in-process compilation with a
fresh module per kernel in the ROCR integration).

---

## Action Items Template

_Fill in during/after the meeting:_

| # | Action | Owner | Due |
|---|--------|-------|-----|
| 1 | Answer Q1-Q3 (ISA pairs, encoding stability, UDNA) | _hardware team_ | |
| 2 | Answer Q4 (correctness bar) | _product/PM_ | |
| 3 | Answer Q5-Q7 (staffing, deadlines, third-party support) | _management_ | |
| 4 | Based on Q1-Q7 answers, finalize Tier 2 investment decision | _design leads_ | |
| 5 | If IR raiser chosen: begin EXEC divergence detection work | _raiser team_ | |
| 6 | If transpiler chosen: investigate LLVM MC state bug fix | _transpiler team_ | |
