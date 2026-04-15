# Design Comparison: Original HotSwap vs. LLVM IR Raiser

A deep analysis of the two binary translation approaches, their resilience to
future hardware changes, and the question of whether intermediate IRs would
strengthen the design.

---

## 1. The Three Designs Inside Original HotSwap

The original HotSwap contains three fundamentally different translation
strategies, each for a different scenario:

| Strategy | Example | Mechanism | Complexity |
|----------|---------|-----------|------------|
| **Same-family retarget** | gfx950 → gfx942 | Encoding-compatible pass-through + surgical NOP/trampoline | ~500 LOC of instruction-specific code |
| **JSON rule-based rewrite** | Any ISA pair | Per-instruction match/replace via config | ~400 LOC parser + rule engine |
| **Cross-family transpiler** | gfx1250 → gfx950 | Full disassemble → text-level translate → reassemble | ~5,800 LOC |

The same-family retarget and the cross-family transpiler are the interesting
ones for comparison. The JSON rule engine is a deployment mechanism, not a
translation architecture.

---

## 2. Same-Family Retarget: Where HotSwap Excels

The gfx950 → gfx942 retarget exploits a key hardware property: **98.4% of
instructions share identical binary encoding across the two ISAs**. This
enables a design that is fast, correct, and surgical:

- Skip 98.4% of instructions entirely (no decode, no encode, no touch)
- NOP-out or trampoline the 1.6% that need changes
- Patch ELF metadata to match target ISA

**This is the right design for this problem.** Our LLVM IR raiser would be
overkill here — full decompilation and recompilation for a 1.6% delta is
wasteful and introduces unnecessary risk (what if `llc` produces different
scheduling? different register allocation? different performance?).

**Resilience to new hardware**: This approach works *only* when encodings are
shared. If AMD releases a new CDNA generation that changes the binary
encoding format, this path breaks entirely. It's a bet on encoding stability
within a family — a bet that has been true historically (GCN → CDNA encoding
evolution has been incremental) but isn't guaranteed.

**Assessment**: Highly principled for its narrow scope. Not a general
translation architecture.

---

## 3. Cross-Family Transpiler: The Core Comparison Target

The transpiler (5,780 lines in `transpiler.cpp`) is the design most
comparable to our LLVM IR raiser. Both attempt *semantic* translation —
understanding what an instruction does and re-expressing it for a different
ISA. But they approach this very differently.

### 3a. Architecture Comparison

| Aspect | Transpiler | LLVM IR Raiser |
|--------|-----------|----------------|
| **Representation** | Assembly text (strings) | LLVM IR (typed, SSA, structured) |
| **Semantic model** | Implicit in string transforms | Explicit in IR operations |
| **Operand parsing** | Split-on-comma + regex | MCInstrDesc metadata + srcMap |
| **Instruction dispatch** | Mnemonic string matching | Format-based switch + mnemonic |
| **Register model** | Operand text manipulation | AllocaInst + PromoteMemToReg (SSA) |
| **Control flow** | Linear text emission | BasicBlocks + branches + PHI nodes |
| **EXEC mask** | String rewriting (exec_lo → exec) | Scalar boolean (i1) |
| **Wait counters** | Manual cross-instruction tracking | Delegated to `llc` backend |
| **Lowering target** | Assembly text for LLVM MC | LLVM IR for `llc` |
| **Lines of code** | ~5,800 | ~1,350 |

### 3b. Principled-ness

**Transpiler — fundamentally text-level**:

The transpiler's core operation is *string manipulation with semantic intent*.
For example, SALU float emulation (lines 1936–2008):

```
s_mul_f32 sdst, ssrc0, ssrc1
→  v_mov_b32_e32 v255, ssrc0
   v_mul_f32_e32 v255, ssrc1, v255
   s_nop 0
   v_readfirstlane_b32 sdst, v255
```

This is *correct* — it captures the semantics of a scalar float multiply by
bouncing through a VGPR. But the implementation parses operands by splitting
strings on commas (`std::istringstream` + `std::getline`), hardcodes
temporary register names (`v255`, `vt0`, `vt1`), and emits replacement text.
There is no formal semantic model — the developer's understanding of the
instruction semantics is encoded directly in the string transformations.

The EXEC widening logic (lines 364–560) is similarly text-based: pattern-match
on "exec_lo" in the operand string, insert `s_mov_b32 exec_hi, 0` after
exec-modifying instructions. The correctness depends on exhaustively matching
all patterns that modify EXEC — a new EXEC-modifying instruction form would
need a new pattern.

**LLVM IR Raiser — formally structured**:

Our raiser expresses the same semantics through typed IR operations. An
instruction like `s_add_u32 dst, src0, src1` becomes:

```llvm
%src0 = load i32, ptr %sgpr_N
%src1 = load i32, ptr %sgpr_M
%res = add i32 %src0, %src1
%ov = call {i32, i1} @llvm.uadd.with.overflow.i32(%src0, %src1)
%carry = extractvalue {i32, i1} %ov, 1
store i32 %res, ptr %sgpr_D
store i1 %carry, ptr %scc
```

The operands are resolved through MCInstrDesc metadata (srcMap), the SCC
writeback is driven by `implicit_defs()`, and the types are explicit. The
correctness is verifiable by LLVM's IR verifier.

**Verdict**: The transpiler is *pragmatic but unprincipled*. It works because
it was debugged against real workloads (90 commits). The IR raiser is
*principled but less mature*. It works because the representation is
self-checking.

### 3c. Scalability to New Instructions

**Transpiler**: Adding a new instruction requires:
1. Add mnemonic mapping entry (if it's a rename): 1 line
2. If it needs emulation: write a new text-manipulation block that parses
   operands, emits replacement text, handles edge cases. 20–100 lines.
3. If it changes EXEC/VCC semantics: update the widening patterns. Error-prone.
4. If it changes wait counter semantics: update the tracking logic. Error-prone.

Scale: O(new_instructions) for renames, O(new_instructions × complexity)
for emulation. The 500+ mnemonic entries for gfx1250→gfx950 show the scale.

**LLVM IR Raiser**: Adding a new instruction requires:
1. If it maps to LLVM IR operations (add, mul, load, etc.): 3 lines in the
   format handler. OpResolver handles operands. Auto-SCC handles flags.
2. If it needs an intrinsic: add the intrinsic mapping. ~5 lines.
3. If it's a new format: add a format case to the switch. ~20 lines.

Scale: O(new_instructions) with a very small constant factor for most ALU
instructions.

**Critical difference**: The transpiler must be written *per ISA pair* (the
gfx1250→gfx950 transpiler cannot be reused for gfx1250→gfx1100 or
gfx1350→gfx950). Our raiser decouples source and target: the raiser handles
the source ISA, and `llc` handles the target ISA. Adding a new target is
just changing the `-mcpu` flag.

### 3d. Resilience to "Crazy New Hardware"

Consider concrete scenarios where new hardware introduces radical changes:

**Scenario 1: New wave width (e.g., wave128)**

- Transpiler: Must rewrite all EXEC widening logic. Every pattern that touches
  exec_lo/exec_hi needs updating. Wave32→wave128 is a different expansion than
  wave32→wave64. Estimated: 200+ lines of new widening code.
- IR Raiser: The EXEC mask model needs updating regardless (it's currently
  a scalar boolean). But if we model it properly as `iN`, changing the width
  is a constant change, not proportional to instruction count.

**Scenario 2: New memory model (e.g., scale-offset addressing)**

- Transpiler: Already handles this for gfx1250 (lines 2200+). Each new
  addressing mode requires a new expansion pattern that synthesizes address
  computation from constituent parts. Text-level operand parsing for each
  memory instruction class.
- IR Raiser: Memory operations are already raised to `load`/`store` with
  computed addresses. A new addressing mode means a new address computation
  pattern in the raiser. The backend handles lowering to the target's
  addressing modes.

**Scenario 3: New compute primitive (e.g., 2:4 sparse matrix)**

- Transpiler: If the target has no equivalent, must NOP-out or write a
  multi-instruction emulation sequence. All done in text.
- IR Raiser: If the source instruction maps to an LLVM intrinsic on the
  target, it's a table entry. If it needs emulation, the emulation is
  expressed in LLVM IR (typed, verifiable). If there's no reasonable
  emulation, fail loudly.

**Scenario 4: Completely new ISA family (e.g., UDNA)**

- Transpiler: A new 5,800-line transpiler must be written from scratch for
  every (source, target) pair involving the new family.
- IR Raiser: A new raiser (~1,350 lines, structured) for the source side.
  Target side is handled by LLVM's existing backend (or a new one if the
  ISA is truly new).

---

## 4. What Does Each Design Get Right?

### What the transpiler gets right

1. **EXEC widening is handled explicitly.** The transpiler knows that
   wave32→wave64 requires `exec_hi = 0` after every exec-modifying
   instruction. Our raiser doesn't model EXEC at all — it's a boolean.

2. **Wait counter translation is handled explicitly.** GFX12 split counters
   → GFX9 unified counters. Our raiser delegates this entirely to `llc`,
   which is correct for same-ISA recompilation but wouldn't work for
   cross-family translation where counter semantics differ.

3. **No semantic gap for "rename" instructions.** When `global_load_b32` is
   just `global_load_dword` with a different name, a string rename is the
   simplest correct translation. Our raiser decompiles and recompiles,
   which is correct but wasteful for pure renames.

4. **Production-proven at scale.** 42/42 tests, 18/20 complex kernels, real
   AITER workloads. Battle-tested through 90 commits.

### What our raiser gets right

1. **Formal semantic model.** LLVM IR is typed, SSA, and verifiable. When we
   say `add i32`, it *means* 32-bit addition. The transpiler's string
   `"s_add_u32"` only means addition because a developer verified it.

2. **ISA decoupling.** Source and target are independent. The raiser handles
   the source ISA. `llc` handles the target. This is O(source + target),
   not O(source × target).

3. **Metadata-driven operand resolution.** MCInstrDesc tells us where the
   operands are, what type they are, whether modifiers are present. The
   transpiler splits strings on commas.

4. **Standard backend integration.** Register allocation, wait counter
   insertion, instruction scheduling, kernel descriptor generation — all
   handled by LLVM's production-quality AMDGPU backend. The transpiler
   manages registers by hand and patches kernel descriptors manually.

5. **Structural correctness.** OpResolver makes operand-index bugs
   impossible. Auto-SCC makes forgotten flag writes impossible. The
   transpiler has no such guarantees — each string transformation must
   be manually verified.

---

## 5. The Case for (and Against) Intermediate IRs

The question: should we add intermediate representations between MCInst and
LLVM IR to model incomplete information during lifting?

### What intermediate IRs could help with

**Problem 1: EXEC mask modeling.** Currently we jump from "64-bit hardware
mask" to "i1 boolean" in one step, losing per-lane semantics. An
intermediate "GPU IR" could model EXEC as a first-class bitmask, with
operations like `exec_and(mask, cond)` and `predicated_store(exec, addr, val)`.
This would let us:
- Verify EXEC manipulation correctness before lowering
- Detect divergent regions (where exec ≠ all-ones) and either handle them
  or refuse them
- Lower uniformity-proven regions to simple branches (what we do now)

**Problem 2: Partial type recovery.** We infer types from mnemonics
(`_f32` → float, `_u32` → i32). Some instructions are ambiguous (bit
manipulation doesn't care about signedness). An intermediate with
"partially typed" values could carry what we know and what we don't,
resolving types progressively.

**Problem 3: Cross-family semantic gaps.** When translating between ISA
families, some operations exist on one side but not the other (SALU floats,
scale-offset addressing, split wait counters). An intermediate that
represents *intent* rather than *machine operations* would let us separate
"what to compute" from "how to compute it on this ISA."

**Problem 4: Register liveness.** Both the transpiler and our raiser have
register allocation problems (the transpiler uses hardcoded temps, we
allocate all 618 registers). An intermediate with explicit liveness would
let us find free registers for emulation sequences.

### What intermediate IRs would NOT help with

**Instruction coverage.** Whether we have 50 handlers or 500, each
instruction still needs a semantic mapping. An intermediate IR doesn't
reduce this work — it changes *where* the mapping lives.

**Correctness of individual mappings.** Whether `s_add_u32` maps to
`add i32` or to `gpu.scalar_add i32`, the mapping must still be correct.
The IR doesn't verify correctness of the mapping, only consistency of the
result.

**Performance of the translation.** Our current single-jump approach
(MCInst → LLVM IR) compiles in ~5ms for a typical kernel. Adding
intermediate passes would increase this. The transpiler's text-level
approach is also fast.

### MLIR as the intermediate

MLIR is the natural framework for multiple intermediate representations. A
hypothetical stack:

```
MCInst stream
    │
    ▼
"gpu.machine" dialect   ← typed MCInst, explicit register refs, no SSA
    │  annotate: format, implicit defs/uses, EXEC state
    ▼
"gpu.raised" dialect    ← SSA, typed, EXEC as bitmask, divergence-tracked
    │  type propagation, EXEC pattern recognition
    ▼
LLVM dialect / arith+memref+scf  ← standard MLIR
    │  lower to target
    ▼
LLVM IR → llc → HSACO
```

**Advantages of MLIR-based approach:**
- Each stage can be verified independently
- Custom operations model GPU semantics precisely (EXEC masks, LDS, barriers)
- Progressive lowering lets each pass focus on one concern
- MLIR's verifier catches type mismatches and structural errors
- Reusable across GPU architectures (CDNA, RDNA, even NVIDIA if desired)

**Disadvantages:**
- Large engineering surface area: 3 dialects × type systems × verifiers × passes
- MLIR dependency is heavy (much larger than LLVM MC)
- We don't yet know exactly what the intermediate should look like — the
  design space is large
- The current single-jump approach works for the kernels we handle

### Assessment: When to add intermediate IRs

**Not yet.** The unsolved problems in our raiser (EXEC mask, instruction
coverage, SCC carry) are *semantic* problems, not *representational* ones.
Adding an intermediate IR before we understand the semantics we need to
model would result in an IR that models the wrong things.

**The right sequence is:**
1. Solve the EXEC mask problem concretely for specific kernels
2. Solve cross-family translation for a specific ISA pair
3. *Then* extract the common patterns into an intermediate IR

This is the "make it work, make it right, make it general" principle. We're
still in "make it work" for EXEC and cross-family. An intermediate IR is
a "make it general" step.

**The exception**: If we find ourselves writing the same analysis in
multiple places (e.g., divergence detection in the raiser AND in a
hypothetical cross-family translator), that's the signal to extract an IR.
The IR should emerge from concrete need, not from architectural aesthetics.

---

## 6. Recommended Hybrid Architecture

Based on this analysis, the strongest design combines elements of both
approaches:

### Tier 1: Same-family retarget (keep original HotSwap)

For ISA pairs that share encoding (gfx950 ↔ gfx942, future CDNA pairs),
the surgical NOP/trampoline approach is optimal. Fast, correct, minimal
risk.

### Tier 2: LLVM IR raising for cross-family translation (our prototype)

For ISA pairs that DON'T share encoding, raise to LLVM IR and lower through
`llc`. This gives ISA decoupling, formal semantics, and standard backend
integration. The key work remaining:

| Priority | Work Item | Why |
|----------|-----------|-----|
| 1 | Conservative EXEC divergence detection | Upgrades from silently-wrong to fail-loudly |
| 2 | LDS + barrier support | Required for real-world tiled GEMM |
| 3 | Instruction coverage to ~200 mnemonics | Covers most production kernels |
| 4 | Cross-family validation (RDNA → CDNA) | Proves the ISA-decoupling claim |

### Tier 3: Intermediate IR (future, when needed)

Add a "GPU semantic" intermediate when:
- EXEC mask modeling is understood and we need it in multiple contexts
- Cross-family translation reveals patterns that need structured lowering
- The single-jump (MCInst → LLVM IR) becomes a bottleneck for information
  recovery

MLIR is the right framework when this time comes. The `gpu.raised` dialect
would model EXEC masks, divergence, and partially-typed values.

---

## 7. Summary

| Dimension | Transpiler | IR Raiser | Winner |
|-----------|-----------|-----------|--------|
| **Same-family minor retarget** | Excellent (encoding-compatible) | Overkill | Transpiler |
| **Cross-family principled-ness** | Text manipulation, ad-hoc | Typed IR, metadata-driven | IR Raiser |
| **Cross-family maturity** | 42/42 tests, 90 commits | 2 test kernels, ~10 commits | Transpiler |
| **Scalability to new instructions** | O(n) with high constant | O(n) with low constant | IR Raiser |
| **Scalability to new ISA pairs** | O(source × target) | O(source + target) | IR Raiser |
| **EXEC mask handling** | Explicit (text patterns) | Missing (scalar boolean) | Transpiler |
| **Wait counter handling** | Explicit (cross-instruction tracking) | Delegated to llc | Tie (different strategies) |
| **Register allocation** | Hardcoded temps | All-regs alloca + PromoteMemToReg | IR Raiser |
| **Resilience to radical HW changes** | New transpiler per pair | New raiser for source, reuse target | IR Raiser |
| **Production readiness** | Deployed in ROCR | Research prototype | Transpiler |

**The transpiler is a faster horse. The IR raiser is a car being built.**
The transpiler solves today's problems with today's tools. The IR raiser
solves tomorrow's problems — cross-family translation that scales to
arbitrary ISA pairs — but needs more engineering to reach production quality.

The recommended path forward is to keep both: use the original HotSwap's
surgical retarget for same-family pairs, and develop the IR raiser for
cross-family translation. Add intermediate IRs when concrete need demands
them, not before.
