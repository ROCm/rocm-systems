# Challenge Questions for the Transpiler Architecture

Pointed questions to surface the structural limitations of the text-level
transpiler approach. Each question is phrased as "how would you handle X?"
where the answer reveals a fundamental architectural constraint.

Each question includes a **"How the IR raiser handles this"** note — but
only where the raiser genuinely does better. Where it doesn't, we say so.

Reference documents:
- [ORIGINAL_HOTSWAP_DEEP_DIVE.md](ORIGINAL_HOTSWAP_DEEP_DIVE.md)
- [hotswap-design-proposal.md](hotswap-design-proposal.md)
- [DESIGN_COMPARISON.md](../llvm_ir_proto/DESIGN_COMPARISON.md)
- [SHORTCUTS_AND_LIMITATIONS.md](../llvm_ir_proto/SHORTCUTS_AND_LIMITATIONS.md)

---

## 1. Scaling to New ISA Pairs

**Q1a: If we need gfx1350 → gfx950 support next year, how much of
`transpiler.cpp` can we reuse?**

Almost none. The 5,800 lines and 500+ mnemonic mappings are hardcoded for
the gfx1250 → gfx950 pair. A new source ISA means a new transpiler from
scratch — another multi-month, ~90-commit effort.

> **IR raiser:** The raiser decouples source and target. A gfx1350 raiser
> would be ~1,350 LOC (similar to the gfx950 raiser) and would use the same
> `llc -mcpu=<target>` backend for any target ISA. Adding a new target is
> changing one flag, not writing 5,800 lines.

**Q1b: If UDNA lands and we need to translate gfx950, gfx942, gfx1250, and
gfx1100 binaries to UDNA, how many transpilers do we write?**

One per (source, target) pair. That's 4 transpilers at ~5,800 lines each.
The IR raiser writes one ~1,350-line raiser per source ISA and reuses the
LLVM backend for every target.

> **IR raiser:** O(source + target) scaling. 4 raisers + 1 shared backend
> vs. 4 transpilers each with their own target-specific logic. ~5,400 LOC
> vs. ~23,200 LOC.

**Q1c: If the gfx1250 → gfx950 transpiler is 5,800 lines for one pair, what
is the projected total LOC when we support 3 cross-family pairs? 5?**

Forces a concrete cost projection: 17,400 LOC for 3 pairs vs. ~4,050 for
the raiser approach (3 raisers + shared backend).

> **IR raiser:** Same argument as Q1b. The scaling advantage compounds with
> each additional pair.

---

## 2. The LLVM MC Global State Bug

**Q2a: The AMDGPU backend's global state only survives one MCContext
lifecycle, limiting the batch assembly fallback to one code object per
process. What is the plan to fix this?**

Unknown effort in LLVM internals, and this is not something we control. The
raiser sidesteps it entirely by using `llc` as a subprocess (or in-process
with a fresh module per kernel).

> **IR raiser:** Not affected. Each kernel is raised into a fresh LLVM
> Module. The `llc` subprocess (or in-process compilation) creates a new
> MCContext per invocation. No global state leaks between kernels.

**Q2b: If a user loads two fat binaries that both need cross-family
translation, the second one silently fails because of the MCContext limit.
How do you handle that?**

Right now, you can't. The `s_retarget_count` guard literally stops after one
code object. This is a showstopper for real-world applications that load
multiple libraries.

> **IR raiser:** Handles arbitrarily many code objects. Each raise is
> independent.

**Q2c: The retarget engine's "batch assembly fallback" is what you'd need if
encoding compatibility ever breaks within a family. But that fallback is
blocked by this same MCContext bug. What's the mitigation?**

Forces acknowledgment that the safety net for encoding changes is broken.

> **IR raiser:** Does not depend on encoding compatibility at all.
> Different encodings just mean a different `MCSubtargetInfo` for the
> disassembler — the raise-and-recompile path works regardless.

---

## 3. Text-Level Fragility

**Q3a: The EXEC widening logic pattern-matches on the string `"exec_lo"` in
operand text. If LLVM changes the disassembler's text output format (e.g.,
`exec_lo` becomes `exec.lo` or `EXEC_LO`), how would we detect the
breakage?**

Silent miscompilation. There's no type system or metadata driving these
matches — just string comparison.

> **IR raiser:** Detects EXEC-modifying instructions via
> `MCInstrDesc::implicit_defs()` — hardware metadata, not string matching.
> If LLVM renames the text output, the raiser is unaffected because it
> never parses the text. **However**, the raiser's EXEC *model* has its own
> limitation: it uses a scalar `i64` alloca, which is correct for uniform
> control flow but cannot represent per-lane divergence. The detection is
> more principled; the modeling is not yet complete.

**Q3b: The SALU float emulation hardcodes `v255` as a temporary VGPR. How
do you know `v255` isn't live in the kernel?**

You don't. There's no liveness analysis. The kernel descriptor is patched
to increase the VGPR allocation, but if the compiler already allocated v255
in the original kernel, the emulation silently clobbers a live value.

> **IR raiser:** Every register becomes an `AllocaInst`, promoted to SSA
> via `PromoteMemToReg`. LLVM's register allocator assigns physical
> registers with full liveness analysis, spilling when needed. No hardcoded
> temporaries, no clobber risk by construction.

**Q3c: Operand parsing is done by splitting strings on commas
(`std::istringstream` + `std::getline`). If an operand contains a comma
(e.g., a complex modifier syntax in a future ISA), what happens?**

The parser breaks. There's no formal grammar — just a hope that LLVM's
assembly syntax stays comma-separated.

> **IR raiser:** Operand resolution uses `MCInstrDesc` metadata —
> `srcMap[]` and `modMap[]` are built by iterating the instruction
> descriptor's operand list. No string splitting. Operand positions are
> determined by the hardware description, not by text parsing.

**Q3d: When you add a new instruction to the transpiler, what verifies that
the emitted replacement text is semantically correct? Is there a formal
model, or is it developer review of string transformations?**

Developer review only. The IR raiser's output is verified by LLVM's IR
verifier and then compiled by the production AMDGPU backend — two layers of
formal checking.

> **IR raiser:** The LLVM IR verifier checks type consistency, SSA
> dominance, and structural well-formedness. Then `llc` runs instruction
> selection, register allocation, and the AMDGPU-specific verifiers. This
> doesn't prove semantic equivalence to the original kernel (no tool does),
> but it does catch entire categories of bugs (wrong types, broken SSA,
> invalid operands) that string-level replacement can't.

---

## 4. Encoding Compatibility as a Single Point of Failure

**Q4a: The entire Tier 1 value proposition rests on the fact that 98.4% of
gfx950 instructions have identical encoding to gfx942. If a future CDNA
generation changes the VOP3 encoding format by even 1 bit, what percentage
of instructions now need full re-encoding?**

Potentially 100%. A format-level change (not an opcode change) affects every
instruction that uses that format, not just new instructions.

> **IR raiser:** Not applicable to this question — this is about Tier 1,
> which we agree is the right design for same-family retarget. The raiser
> is a Tier 2 tool. But the relevant point is: the raiser doesn't depend
> on encoding compatibility at all, so it serves as a hedge if the Tier 1
> assumption breaks.

**Q4b: When encoding compatibility breaks, your fallback is the batch
assembly path — the one blocked by the MCContext bug. So the system goes
from "handles 100% of kernels" to "handles 0%" in one hardware generation?**

Forces acknowledgment of the cliff edge. There's no graceful degradation
path.

> **IR raiser:** Would provide the graceful degradation path. If encoding
> compatibility breaks, the raiser can handle the translation without
> relying on LLVM MC's batch assembly.

**Q4c: Has the hardware team committed to maintaining binary encoding
compatibility across future CDNA generations? If not, what is the plan?**

If the answer is "no commitment," the entire Tier 1 architecture is a bet
with no hedge.

> **IR raiser:** Not relevant to this question directly — this is a
> hardware roadmap question. But the raiser's existence means having a
> hedge regardless of the answer.

---

## 5. Concrete Gaps That Expose Architectural Limits

**Q5a: `v_mfma_f32_16x16x128_f8f6f4` (128 elements/instruction) needs 4:1
expansion to gfx942's 32-element MFMA. Currently it's NOPed out. When will
this produce correct results?**

A 4:1 expansion in the trampoline model means 4 MFMA instructions + setup +
branch, all fitting in a NOP sled that was designed for 256-byte alignment
padding. This is an architectural mismatch, not a missing handler.

> **IR raiser:** Maps MFMA instructions to LLVM intrinsics (35+ shapes
> supported, including scaled f8f6f4 via `llvm.amdgcn.mfma.scale`). If
> the target ISA has a smaller MFMA, the expansion would be expressed in
> LLVM IR and the backend handles instruction selection. **However**, the
> raiser doesn't currently have the 4:1 MFMA expansion logic either — this
> is an unsolved semantic problem in both approaches. The difference is
> that the raiser would express the expansion in typed IR with proper
> register allocation, while the transpiler must fit it into a NOP sled
> with hardcoded registers.

**Q5b: The trampoline branch (`s_branch`) has a ±128KB range. What happens
when a kernel exceeds this? You mention `s_setpc_b64` as an alternative but
say "not implemented." What is the plan?**

Forces acknowledgment that the trampoline model has hard distance limits.

> **IR raiser:** No trampolines. Full recompilation through `llc` produces
> a new `.text` section with no distance constraints between emitted code
> regions.

**Q5c: When NOP sleds are exhausted (all post-`s_endpgm` padding consumed),
the fallback is ELF growth via `RewriteCodeObjectGrow()`. This reallocates
the buffer and patches all section headers. What testing exists for this
path with large numbers of trampolines?**

Limited. The NOP-sled path works for the ~1.6% instruction delta. At higher
delta percentages, you're in largely untested territory.

> **IR raiser:** No NOP sleds, no ELF growth. The raiser builds a complete
> new code object via `llc` + `llvm-mc` + `ld.lld`. The ELF is generated
> fresh, not patched.

---

## 6. Wait Counter and Control Flow Complexity

**Q6a: The transpiler manually tracks split wait counters across
instructions (GFX12 `s_wait_loadcnt` → GFX9 `s_waitcnt vmcnt`). This
requires cross-instruction state. If a new ISA introduces a third counter
model, you rewrite this logic for every transpiler. How does this scale?**

It doesn't. The IR raiser delegates wait counter insertion entirely to
`llc`'s backend, which already knows the target ISA's counter model.

> **IR raiser:** Wait counters are completely absent from the raised LLVM
> IR. The AMDGPU backend's `SIInsertWaitcnts` pass inserts the correct
> wait counter instructions for whatever target ISA is selected. A new
> counter model requires zero changes in the raiser — only the LLVM
> backend needs updating (which AMD maintains anyway for the compiler).

**Q6b: The wave32 → wave64 EXEC widening inserts `s_mov_b32 exec_hi, 0`
after every exec-modifying instruction. How do you verify that you've caught
every exec-modifying instruction pattern? Is there a systematic check, or is
it pattern-by-pattern?**

Pattern-by-pattern. If a new instruction modifies EXEC and the pattern list
isn't updated, the widening silently breaks.

> **IR raiser:** Detects EXEC-modifying instructions via
> `MCInstrDesc::implicit_defs()` — this is the hardware's own instruction
> descriptor, which is exhaustive by construction. If LLVM adds a new
> EXEC-modifying instruction, it will appear in `implicit_defs()`
> automatically. **However**, the raiser's wave-width handling is not yet
> implemented for cross-family translation — EXEC is modeled as a scalar
> `i64`, which is correct for same-width translation but the wave32→wave64
> widening problem is not solved yet.

---

## 7. Maintenance and Long-Term Cost

**Q7a: The transpiler was built over ~90 commits of incremental debugging
against real AITER kernels. If the AITER workload changes significantly (new
kernels, new instruction patterns), how much re-debugging is needed?**

Forces acknowledgment that the transpiler's correctness is empirically
proven against a specific corpus, not structurally guaranteed.

> **IR raiser:** The raiser also has corpus-dependent testing (27 kernels).
> The structural difference is that unrecognized instructions cause an
> immediate hard failure with a diagnostic (format + mnemonic + offset),
> rather than silent miscompilation. New instruction patterns require new
> handlers, but missing handlers are caught at raise time, not at runtime
> on the GPU.

**Q7b: When LLVM's AMDGPU disassembler changes its text output format
(mnemonic naming, operand ordering, modifier syntax) — which happens across
LLVM major versions — how much of the transpiler breaks?**

Potentially all 500+ mnemonic mappings and all regex-based pattern matchers.
The raiser uses MCInstrDesc metadata, which evolves with the LLVM API and
breaks at compile time (not silently at runtime).

> **IR raiser:** Uses `MCInstrDesc` metadata, TSFlags, `implicit_defs()`,
> and operand type enums — all of which are C++ API surfaces. When LLVM
> changes them, the raiser fails to compile (linker error, type mismatch),
> not silently at runtime. The one exception: `OPERAND_INPUT_MODS` (value
> 45) is a copied constant that could drift silently — this is a known
> LOW-severity coupling.

**Q7c: You have a hand-written JSON parser (~400 lines) to avoid a
dependency on `nlohmann/json`. If the rule format needs to grow (e.g., to
support per-kernel configuration, conditional rules, version constraints),
who maintains this parser?**

Illustrates the "zero dependencies" philosophy creating maintenance
surfaces.

> **IR raiser:** No equivalent concern — the raiser doesn't use a rule
> engine or configuration files. Translation is driven by code, not
> configuration. (This is not inherently better — a data-driven approach
> has its own advantages — but it eliminates the custom-parser maintenance
> surface.)

---

## 8. Trampoline Register Safety

**Q8a: Trampoline emulation sequences use temporary registers (e.g., `v255`
for SALU float emulation). The approach is: read the kernel descriptor's
declared VGPR count, pick temporaries above that count, then patch the
descriptor to allocate more. What happens when the kernel already uses the
maximum 256 VGPRs?**

There are no spare registers. The hardcoded `v255` clobbers a live value.
There is no check for this case.

> **IR raiser:** Every value is an SSA variable (alloca → `PromoteMemToReg`).
> LLVM's register allocator has full liveness information and spills to
> scratch memory when register pressure exceeds available registers. A
> kernel that uses all 256 VGPRs will get correct spill code, not silent
> clobber.

**Q8b: Patching the kernel descriptor to allocate more VGPRs reduces GPU
occupancy (fewer wavefronts per SIMD). A kernel tuned for 4 wavefronts at
200 VGPRs might drop to 2-3 wavefronts at 256 VGPRs. Is this occupancy
impact measured or analyzed anywhere?**

It is not. The performance benchmark shows 1.000x geomean, but this only
holds because the 1.6% delta on AITER kernels happens not to push register
counts past an occupancy boundary. There is no analysis for the general case.

> **IR raiser:** LLVM's register allocator is occupancy-aware — it
> considers the target's register budget when making allocation decisions.
> **However**, the raiser has not been benchmarked for occupancy impact
> either. The structural advantage is that the backend *can* make informed
> tradeoffs; the transpiler's approach has no mechanism to even consider
> occupancy.

**Q8c: If two trampolines in the same kernel both use `v255`, and execution
flows through both sequentially, the second trampoline sees whatever the
first one left in `v255`. Is there any analysis of cross-trampoline register
interference?**

No. Each trampoline is written in isolation. There is no whole-kernel
analysis of how trampolines interact with each other's temporary registers.

> **IR raiser:** No trampolines exist. All emulation code is expressed
> inline in the LLVM IR function body. The register allocator sees the
> entire kernel at once and handles all register lifetimes globally. Cross-
> region interference is impossible by construction.

**Q8d: The IR raiser's approach is to raise everything to LLVM IR and let
the standard LLVM register allocator assign registers from scratch — with
full liveness analysis, spilling when needed, and occupancy-aware
allocation. Given that the trampoline approach has no liveness analysis at
all, how confident are we that the hardcoded-temporary strategy is safe
beyond the current AITER corpus?**

Forces acknowledgment that the register safety of the trampoline model is
an empirical bet on spare capacity, not a structural guarantee.

> _(This question already contains the contrast — no additional note needed.)_

---

## 9. Correctness Bar

**Q9a: 5/20 AITER kernels produce approximate results due to FP4/bf16
emulation. For a user who installs a Python wheel expecting correct
numerics, is "approximate" acceptable? What is the plan to close the
accuracy gap?**

Forces a conversation about the correctness bar. The surgical approach chose
speed over fidelity for these instructions.

> **IR raiser:** No clear advantage here. The raiser faces the same
> fundamental problem: if the target ISA doesn't have FP4/bf16 hardware
> instructions, emulation is needed regardless of the translation
> architecture. The raiser would express the emulation in LLVM IR rather
> than in assembly text, which is more verifiable, but the numerical
> accuracy of the emulation itself is the same challenge.

**Q9b: The transpiler NOP-outs unsupported MFMA variants. A NOP in a matrix
multiply means the kernel produces wrong results. Is there a mechanism to
detect this and warn the user, or does it fail silently?**

Depends on the specific path. Some are logged, some are silent. This is a
correctness hole.

> **IR raiser:** The raiser maps MFMA to LLVM intrinsics. If the target
> backend supports the intrinsic, `llc` emits correct code. If the
> intrinsic doesn't exist for the target, `llc` fails with a hard error
> during instruction selection — never a silent NOP. **However**, if the
> semantic gap requires multi-instruction emulation (e.g., 4:1 MFMA
> expansion), that emulation must still be written in the raiser. The
> advantage is fail-loudly vs. fail-silently, not automatic correctness.

---

## Summary: The Pattern

Each question follows the same structure:

1. **Pose a concrete future scenario** (new ISA pair, encoding break, UDNA,
   large kernel, LLVM version change)
2. **Ask how the current transpiler handles it** (the answer is: it doesn't,
   or it requires proportional re-work)
3. **The implicit contrast** is that the IR raiser handles it structurally
   (ISA decoupling, formal semantics, standard backend, metadata-driven
   dispatch)

### Where the IR raiser is genuinely stronger

- **ISA pair scaling** (Q1a–c): O(source + target) vs. O(source x target)
- **LLVM MC state bug** (Q2a–c): not affected at all
- **Operand resolution** (Q3b–c): metadata-driven, not string-based
- **Formal verification** (Q3d): IR verifier + backend vs. developer review
- **No trampolines** (Q5b–c, Q8a–d): full recompilation with proper RA
- **Wait counters** (Q6a): delegated to backend, zero raiser-side work
- **EXEC detection** (Q6b): `implicit_defs()` metadata, exhaustive by construction
- **LLVM version resilience** (Q7b): compile-time breaks vs. silent runtime breaks
- **Register allocation** (Q8a–c): full liveness, spilling, occupancy-aware

### Where the IR raiser is NOT stronger (honest gaps)

- **EXEC per-lane divergence** (Q3a, Q6b): The raiser detects EXEC-modifying
  instructions more principally, but models EXEC as a scalar `i64` — it
  cannot represent per-lane divergent behavior. The transpiler's text-level
  EXEC widening, while fragile, at least *handles* the wave32→wave64 case.
- **FP emulation accuracy** (Q9a): Same fundamental challenge — if the target
  lacks the instruction, emulation accuracy depends on the algorithm, not
  the translation architecture.
- **MFMA expansion** (Q5a): Neither approach has solved 4:1 MFMA expansion.
  The raiser would express it more cleanly but doesn't have the logic yet.
- **Cross-family maturity**: The transpiler has 42/42 unit tests and 18/20
  complex kernels across 90 commits. The raiser has 27/27 raise rate but
  is not execution-validated cross-family.
