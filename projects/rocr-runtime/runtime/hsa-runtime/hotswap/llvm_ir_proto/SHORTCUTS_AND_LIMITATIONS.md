# LLVM IR Prototype — Shortcuts, Limitations, and Design Assessment

Systematic analysis of the LLVM IR binary translation prototype's design
decisions, shortcuts, and limitations. Each item is assessed for whether the
approach is *principled* (sound by construction) or *unprincipled* (known to
be wrong, relying on luck or limited test coverage).

Updated after: table-driven raiser refactor, srcMap-based OpResolver fix.

## Severity Legend
- **HIGH** — Would cause incorrect results or crashes on non-trivial kernels
- **MEDIUM** — Limits applicability but doesn't affect correctness for tested kernels
- **LOW** — Engineering debt; straightforward to fix

---

## Design Principles

The prototype aspires to these principles:

1. **Fail loudly**: Never silently produce wrong results. If we can't handle
   something, abort with a diagnostic.
2. **Metadata over strings**: Use `MCInstrDesc` metadata (TSFlags, operand
   info, implicit defs) instead of mnemonic string parsing where possible.
3. **Structural correctness**: Make bug classes impossible by construction
   (e.g., OpResolver makes operand-index bugs impossible).
4. **Standard backend**: Feed raised IR into LLVM's unmodified AMDGPU backend
   — no manual assembly patching, no custom metadata.

The assessment below grades each component against these principles.

---

## 1. Semantic Model

How the raiser models hardware-level concepts (EXEC mask, condition codes,
FP modes) in LLVM IR.

### 1a. EXEC mask modeled as scalar boolean [HIGH — UNPRINCIPLED]

The 64-bit EXEC and VCC masks are modeled as a single `i1` boolean. EXEC
always reads as all-ones; writing EXEC is a no-op. VCC is an `i1` alloca
widened via `sext i1 → i64` when read as 64-bit.

**Why this is unprincipled**: It violates fail-loudly. A kernel with
divergent control flow (some lanes take one path, others take another) will
produce wrong results *silently*. The raiser has no mechanism to detect that
it's operating outside its validity envelope.

**What principled looks like**: At minimum, *conservative soundness* — track
when `s_and_saveexec_b64` narrows EXEC and flag that we're inside a
potentially divergent region. If a memory operation occurs in that region,
fail loudly. This doesn't solve divergence but makes the raiser honest: it
either handles the kernel correctly or refuses it.

Full principled fix: model EXEC as `i64`, emit predicated stores, or use
LLVM's divergence analysis to prove uniformity. Much larger effort.

**Impact**: Correct only when all 64 lanes take the same branch path (full
wavefronts or uniform branches). This covers our test kernels (vecadd, MFMA
GEMM) but would fail silently on production kernels with partial wavefronts.

### 1b. SCC semantics for `s_add_i32` / `s_sub_i32` [LOW — UNPRINCIPLED]

`s_add_i32` sets `sccResult = result`, which the auto-writeback mechanism
emits as `SCC = (result != 0)`. The hardware sets SCC to the carry-out bit.
The unsigned variant `s_add_u32` correctly uses `llvm.uadd.with.overflow`
with `sccHandled = true`.

**Why this is unprincipled**: The auto-writeback pattern (`SCC = result != 0`)
is correct for bitwise/logical SOP2 ops (AND, OR, SHL, etc.) but wrong for
add/sub where SCC means carry. The handler *should* set `sccHandled = true`
and compute carry explicitly, exactly like `s_add_u32` already does. This is
a straightforward fix that would make the SCC model fully principled.

**Impact**: If an `s_add_i32` feeds an `s_addc_u32` carry chain, the carry
propagation is wrong. Not triggered in current test kernels.

### 1c. FP mode register silently ignored [LOW — UNPRINCIPLED]

The MODE register is parsed but writes are silently ignored. The translated
kernel uses LLVM's default FP semantics (round-to-nearest-even, flush
denormals to zero on CDNA).

**Why this is unprincipled**: Violates fail-loudly. A kernel that sets
non-default rounding or denorm behavior will produce subtly different
numerical results with no diagnostic.

**What principled looks like**: Detect writes to MODE. If the written value
differs from the default, fail loudly. This is a one-line check.

### 1d. `v_mad_u64_u32` carry output is zeroed [MEDIUM — UNPRINCIPLED]

The 64-bit carry (SDST) is written as 0. The 64-bit result (VDST) is
correct.

**Why this is unprincipled**: We write a concrete wrong value (0) to a
register. If downstream code reads SDST for carry arithmetic, the result is
silently wrong. We cannot even detect this at raise time without dataflow
analysis.

**What principled looks like**: Compute the actual carry (high 64 bits of a
96-bit multiply-add). Alternatively, mark SDST as "unknown" and fail if it's
subsequently read before being overwritten.

**Mitigating observation**: In observed kernels, SDST is clobbered by a
subsequent comparison before being read.

---

## 2. Operand Resolution

How the raiser accesses instruction operands from the decoded MCInst.

### 2a. srcMap-based OpResolver [STRENGTH — PRINCIPLED]

During instruction decode, a `srcMap[]` is built by iterating
`MCInstrDesc::operands()` and skipping entries where
`OperandType == OPERAND_INPUT_MODS` (VOP3 source modifiers).
`OpResolver::src(i)` reads through `srcMap[i]`, which correctly resolves
source operands regardless of encoding:

- **VOP2 e32** (no modifiers): srcMap = [1, 2]
- **VOP3 e64** (modifiers at indices 1, 3): srcMap = [2, 4, 5]
- **Multi-def** `v_mad_u64_u32` (2 defs): srcMap = [2, 3, 4]
- **MFMA** (register sources + control immediates): srcMap = [1, 2, 3, 4, 5, 6]

This is principled because:
- It uses MCInstrDesc metadata to determine operand roles
- It makes the operand-index bug class *structurally impossible*
- No handler uses raw `readOp32(di, N)` with hardcoded indices
- All handlers use the same `op.src(i)` / `op.srcReg(i)` / `op.srcImm(i)` API

**Residual coupling**: The `OPERAND_INPUT_MODS` constant (value 45) is
copied from AMDGPU's `SIDefines.h`. If the value shifts across LLVM
versions, modifier operands would not be skipped and `op.src()` would read
modifier values (typically 0) as source operands. This coupling is **less
fail-safe** than the SIInstrFlags coupling — it would produce silent wrong
values rather than an immediate failure.

### 2b. `v_lshl_add_u64` and `v_lshlrev_b64` shift assumed immediate [LOW — UNPRINCIPLED]

Both handlers call `op.srcImm(N)` unconditionally without checking
`di.isImm(op.srcIdx(N))`. If the shift amount were in a register, this would
crash with an assertion failure in `MCInst::getImm()`.

**Why this is unprincipled**: Violates fail-loudly in spirit — the
assertion fires in the MCInst accessor rather than with a meaningful
diagnostic.

**What principled looks like**: Check `di.isImm(op.srcIdx(N))`. If true,
use `srcImm`. If register, use `src()` and emit a dynamic shift. This is a
few-line fix per handler.

---

## 3. Instruction Dispatch

How the raiser classifies instructions and routes them to handlers.

### 3a. Format-based dispatch from MCInstrDesc TSFlags [STRENGTH — PRINCIPLED]

A `switch (di.format)` on the format classification (derived from
`MCInstrDesc::TSFlags` via `classifyFormat()`) routes each instruction to its
format-specific handler block (SOPP, SMEM, SOPC, SOP1, SOP2, VALU, FLAT,
MFMA).

This is principled because:
- Format classification uses hardware metadata, not string parsing
- New instructions enter the correct format bucket automatically
- Error diagnostics include the format name (`[format=VOP3]`)
- VOP1/VOP2/VOP3/VOPC/VOP3P cases are unified, handling encoding promotion
  transparently

### 3b. Auto SCC writeback from implicit_defs [STRENGTH — PRINCIPLED]

After each handler, the dispatch loop checks `di.defsSCC` (derived from
`MCInstrDesc::implicit_defs()`) and, if the handler set `sccResult`, emits
`SCC = (sccResult != 0)`. Handlers with special SCC semantics (carry,
compare) set `sccHandled = true` to bypass auto-writeback.

This is principled because:
- It uses hardware metadata to determine *when* SCC should be written
- It eliminates the bug class where a handler forgets to write SCC
- The override mechanism (`sccHandled`) is explicit

**Caveat**: The auto-writeback assumes `SCC = (result != 0)` which is
correct for bitwise/logical ops but wrong for add/sub (see item 1b). The
auto-writeback mechanism itself is principled; the *use* of it for
`s_add_i32` is not.

### 3c. Mnemonic-based dispatch within format cases [LOW — PRAGMATIC]

Within each format case, dispatch uses string comparison on the stripped
mnemonic (`mn == "s_add_u32"`). This is O(n) per format (n ≈ 5–15), not
O(n) total (n ≈ 50+).

**Why this is pragmatic, not principled**: The canonical instruction identity
at the MC layer is the opcode integer, not the mnemonic string. But LLVM's
opcodes are encoding-specific (`V_ADD_F32_e32` ≠ `V_ADD_F32_e64`), so
we'd need an encoding-canonical mapping. The mnemonic stripping
(`stripEncoding`) serves this role today.

**What more principled looks like**: Key handlers on `MCInstrInfo::getName(opcode)`
in a hash map, avoiding the strip step. Or build a mapping from opcode to a
canonical "operation ID" during initialization. Marginal benefit since
per-format n is small.

### 3d. SIInstrFlags and OPERAND_INPUT_MODS copied from LLVM internals [LOW — PRAGMATIC]

The `amdgpu_formats.hpp` header copies ~20 `SIInstrFlags` bit constants
and the `OPERAND_INPUT_MODS` operand type value from LLVM's `SIDefines.h`.
These are target-internal, not part of LLVM's public API.

**Mitigation for SIInstrFlags**: If bits shift, `classifyFormat()` returns
`Unknown` for everything, triggering the fail-loudly path. This is **safe**.

**Mitigation for OPERAND_INPUT_MODS**: If the value shifts, modifiers won't
be skipped in srcMap. Modifier values are typically 0, so `readOp32` returns
0, producing silent zero-source bugs. This is **NOT safe**.

**What principled looks like**: A compile-time or startup-time sanity check —
decode a known instruction and verify the format / operand layout matches
expectations. This would detect constant drift immediately.

---

## 4. Register Model

### 4a. AllocaInst-based register file with PromoteMemToReg [STRENGTH — PRINCIPLED]

All registers (106 SGPRs, 256 VGPRs, 256 AGPRs, VCC, SCC) are modeled as
`AllocaInst` in the entry block. After raising, `PromoteMemToReg` converts
to SSA with PHI nodes.

This is principled because:
- It correctly handles arbitrary control flow including loops
- The MFMA GEMM loop carries accumulator state through PHIs automatically
- No manual PHI insertion or dominance computation required
- The standard LLVM pass handles all corner cases

### 4b. All 618 registers allocated unconditionally [LOW]

Unused allocas become dead after PromoteMemToReg. LLVM's optimizer removes
them. Adds compile-time overhead for small kernels. Not a correctness issue.

---

## 5. Coverage and Scaling

### 5a. ~50 instruction mnemonics + 25 MFMA shapes [MEDIUM]

The raiser handles the instruction set listed below. Any unrecognized
instruction causes immediate failure with format + mnemonic diagnostic.

| Category | Instructions |
|---|---|
| Scalar load | `s_load_dword{,x2,x4,x8}` |
| Scalar ALU | `s_add_u32`, `s_add_i32`, `s_sub_i32`, `s_addc_u32`, `s_mul_i32`, `s_mul_hi_u32`, `s_and_b32`, `s_or_b32`, `s_lshl_b32`, `s_lshl_b64`, `s_lshr_b32`, `s_ashr_i32`, `s_mov_b32`, `s_mov_b64`, `s_cselect_b32`, `s_cselect_b64` |
| Scalar 64-bit | `s_and_b64`, `s_or_b64`, `s_andn2_b64`, `s_and_saveexec_b64` |
| Scalar compare | `s_cmp_{gt,lt,ge,le}_i32`, `s_cmp_{eq,lg,ge,gt}_u32` |
| Vector ALU | `v_add_u32`, `v_add3_u32`, `v_or_b32`, `v_and_b32`, `v_mov_b32`, `v_lshrrev_b32`, `v_lshlrev_b32`, `v_ashrrev_i32`, `v_mul_lo_u32`, `v_mad_u64_u32`, `v_lshl_add_u32`, `v_lshl_add_u64`, `v_lshl_or_b32`, `v_lshlrev_b64`, `v_perm_b32`, `v_cndmask_b32`, `v_add_f32`, `v_fmac_f32` |
| Vector compare | `v_cmp_{gt,le,lt,ge}_i32`, `v_cmp_{ne,gt,eq}_u32` (e32 + e64) |
| Memory | `global_load_dword{,x2,x4}`, `global_load_{ushort,sshort,ubyte,sbyte,short_d16_hi}`, `global_store_dword` |
| MFMA | 25 `v_mfma_*` shapes (f16, f32, i8, bf16, xf32) |
| Branch | `s_branch`, `s_cbranch_scc{0,1}`, `s_cbranch_vcc{nz,z}`, `s_cbranch_exec{z,nz}` |
| Control | `s_endpgm`, `s_waitcnt`, `s_nop` |

**Not supported** (would require new handlers):
- LDS (`ds_read_*`, `ds_write_*`) — needed for tiled GEMM
- Buffer memory (`buffer_load_*`, `buffer_store_*`)
- Atomics (`global_atomic_*`, `ds_atomic_*`)
- Type conversions (`v_cvt_*`)
- Packing/dot (`v_pack_*`, `v_dot*`)
- Cross-lane (`v_readlane_b32`, `ds_bpermute_b32`)
- Multi-dword stores (`global_store_dwordx{2,4}`)
- Barriers (`s_barrier`)

**Scaling assessment**: The format dispatch + OpResolver + auto-SCC pattern
makes adding new handlers mechanical. A new SOP2 integer op requires ~3
lines (read sources, emit IR op, set sccResult). The architecture scales to
200+ instructions. The coverage gap is *effort*, not *design*.

### 5b. Single-kernel assumption [HIGH — UNPRINCIPLED]

The raiser stops at the first `s_endpgm`. Multi-kernel code objects silently
skip all subsequent kernels.

**Why this is unprincipled**: Violates fail-loudly. Should detect multiple
`s_endpgm` markers and either raise all kernels or fail with a diagnostic.

### 5c. Branch offset range ±32K instructions [LOW]

Branch target sign-extension uses `(int16_t)(uint16_t)(raw & 0xFFFF)`.
Kernels larger than 128 KB would compute wrong branch targets. No current
test kernel is this large.

---

## 6. Pipeline (IR → HSACO)

### 6a. Full recompilation through `llc` [MEDIUM — PRINCIPLED]

The raised IR is fed into `llc` for full instruction selection, register
allocation, and scheduling. The output will use different registers and
scheduling than the original binary. This is by design — it demonstrates
*semantic recovery*.

### 6b. External tools via `std::system()` [MEDIUM]

`llc`, `llvm-mc`, `ld.lld` are invoked as subprocesses. Fragile but
functional.

### 6c. Temporary file I/O without cleanup [LOW]

Files in `/tmp/ir_proto_<pid>/`. No cleanup on success or failure.

### 6d. Implicit arg offset is ABI-version-specific [MEDIUM]

Implicit argument base from `.note` metadata assumes COV6 layout. Different
code object versions may have different layouts.

---

## 7. Validation

### 7a. Standard backend integration [STRENGTH]

This is the only prototype where generated IR feeds into LLVM's unmodified
AMDGPU backend. The backend handles register allocation, wait counter
insertion, kernel descriptor generation, and metadata emission. No manual
assembly patching required.

### 7b. MFMA GEMM bit-identical on GPU [STRENGTH]

The raiser translates a `v_mfma_f32_16x16x16_f16` GEMM kernel and produces
bit-identical results across three matrix sizes (16×16, 32×32, 64×64). This
validates wide vector register packing, MFMA intrinsic mapping, loop-carried
accumulator state via PHI nodes, and conditional store patterns.

### 7c. Dynamic kernel signature from ELF metadata [STRENGTH]

Kernel function signatures are built from `.note` MsgPack metadata. Pointer
args, scalar args, and hidden runtime args are all handled dynamically.

---

## 8. Principled Design Assessment

### What IS principled

| Component | Why |
|-----------|-----|
| **Operand resolution** (srcMap + OpResolver) | Uses MCInstrDesc operand metadata; bug class structurally impossible |
| **Format dispatch** (TSFlags → FormatKind) | Uses hardware metadata, not strings; O(1) classification |
| **Auto SCC writeback** (implicit_defs → sccResult) | Uses hardware metadata to determine when to write; override mechanism explicit |
| **Register model** (AllocaInst + PromoteMemToReg) | Standard LLVM pass handles all SSA construction including loops |
| **Standard backend** (llc pipeline) | No manual patching; backend handles all code generation |
| **Fail-loudly on unknown instructions** | Unrecognized mnemonics abort with format + offset diagnostic |

### What is NOT principled

| Component | Failure mode | Fix complexity |
|-----------|-------------|----------------|
| **EXEC mask as scalar boolean** | Silent wrong results for divergent control flow | High (per-lane modeling) or Medium (conservative refuse-on-divergence) |
| **`s_add_i32`/`s_sub_i32` SCC = result != 0** | Wrong carry propagation if feeding `s_addc_u32` | Low (use `sccHandled + uadd.with.overflow` like `s_add_u32`) |
| **`v_mad_u64_u32` carry zeroed** | Silent wrong carry if SDST is read | Medium (compute 96-bit product upper half) |
| **FP MODE writes silently ignored** | Subtle numerical differences | Low (fail-loudly on non-default MODE write) |
| **Single-kernel assumption** | Silently skips subsequent kernels | Low (detect multiple s_endpgm) |
| **`v_lshl_add_u64`/`v_lshlrev_b64` shift assumed immediate** | Crash on register shift operand | Low (check isImm, handle register case) |
| **OPERAND_INPUT_MODS constant coupling** | Silent zero-source bugs if value drifts | Low (startup sanity check on known instruction) |

### Priority-ordered action items

1. **Conservative divergence detection** for EXEC mask — upgrades from
   "silently wrong" to "fail-loudly wrong". Moderate effort, high impact on
   soundness.
2. **Fix `s_add_i32`/`s_sub_i32` SCC** to use carry like `s_add_u32` —
   trivial fix, completes the SCC model.
3. **Fail-loudly on MODE writes** — one-line check, aligns with design
   principle.
4. **Check `isImm` before `srcImm`** for shift handlers — few-line fix,
   prevents assertion crash.
5. **Detect multi-kernel code objects** — straightforward scan.
6. **OPERAND_INPUT_MODS sanity check** — decode a known VOP3 instruction at
   startup and verify the srcMap layout.

---

## Summary

| Category | Count |
|----------|-------|
| HIGH | 2 (exec mask as scalar, single kernel) |
| MEDIUM | 4 (instruction coverage, mad carry, full recompilation, implicit ABI) |
| LOW | 8 (SCC semantics, shift assumptions, branch range, constants coupling, all-regs allocated, FP mode, IR quality differences, temp files) |
| STRENGTHS | 7 (standard backend, MFMA validated, dynamic signature, alloca SSA, format dispatch, srcMap OpResolver, auto SCC writeback) |

**The architecture is principled for operand resolution, instruction
dispatch, and register modeling.** These three components use MCInstrDesc
metadata by construction and make entire bug classes structurally impossible.
The MFMA GEMM validation proves they work end-to-end on non-trivial kernels.

**The architecture is NOT principled for semantic modeling of EXEC, SCC
carry, and FP modes.** The EXEC mask is the most critical gap — it produces
silently wrong results for divergent kernels, violating the fail-loudly
principle. Adding conservative divergence detection (refuse-on-divergence)
would be the single highest-impact improvement to design soundness.

**The remaining gaps are engineering, not architecture.** Instruction
coverage is limited by effort, not by design — the dispatch infrastructure
scales cleanly. The unprincipled items (SCC carry, FP mode, shift
assumptions) each have known, small fixes that would bring them in line with
the design principles.
