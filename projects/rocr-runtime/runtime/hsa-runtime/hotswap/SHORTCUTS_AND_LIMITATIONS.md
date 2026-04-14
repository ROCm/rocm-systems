# HotSwap MLIR Pipeline – Shortcuts & Limitations

Known unsound shortcuts and scaling limitations in the current implementation.
Items are ranked by severity within each category.

## High – Correctness

### 1. `s_waitcnt` operands are destroyed
The lifter discards the packed counter operand. The emitter always emits
`vmcnt(0) lgkmcnt(0) expcnt(0)` — a full pipeline drain. This is conservatively
correct but kills IPC. For cross-ISA (GFX12 split counters → GFX9 packed), the
situation is worse: all five GFX12 split waits (`s_wait_loadcnt`,
`s_wait_storecnt`, `s_wait_kmcnt`, `s_wait_dscnt`, `s_wait_expcnt`) collapse to
a single `s_waitcnt` with no record of which counter was being waited on.
Performance-sensitive code (overlapping loads with ALU) will stall completely.

### 2. Implicit register defs/uses are not modeled
The lifter only captures explicit MCInst operands. Many instructions have
architecturally significant implicit defs/uses (VCC from `v_cmp_*_e32`, EXEC
from `s_*_saveexec_*`, SCC from most SALU ops). The emitter papers over this
with mnemonic-specific hacks (hardcoded `vcc` prefix for VOPC), but the IR
itself is missing these edges. Any analysis or transform pass will see
incomplete def-use chains.

### 3. `RegKind::Other` silently becomes immediate 0
Unrecognized registers (FLAT_SCRATCH, MODE, NULL_REG, hardware-internal regs)
are classified as `ImmType(0)`. The instruction looks successfully lifted, but
one operand is silently replaced with zero. Violates the no-silent-failures
policy.

### 4. `physRegCache` doesn't track register redefinitions
The lifter caches the first `precolored.*` op per MC register number and reuses
it for every subsequent reference. If a register is written, then read again
later, both reads alias the same MLIR Value even though the second should see
the new definition. Harmless for the direct-emit path (physical register names
are printed), but produces wrong SSA chains.

### 5. `scale_offset` lowering uses hardcoded scratch VGPRs
The lowering always uses `v3` as scratch and `v[4:5]` as the address pair. This
is safe for the vecadd kernel where `next_free_vgpr = 8`, but will silently
clobber live values in any kernel that uses `v3`–`v5` for other purposes. Needs
free-register analysis or a spill mechanism. Same class of bug as item 6.

### 6. Wave width pass uses hardcoded temp SGPR `s16`
The `v_cmpx` → `v_cmp` expansion saves/restores VCC through `s16`. If the
kernel has a live value in `s16`, it gets silently clobbered. Needs
free-register analysis or a spill slot.

### 7. `scale_offset` lowering doesn't handle `dwordx3` (12-byte) loads
When the mnemonic contains `dwordx3` / `b96`, the code sets `shiftAmount = -1`.
This value is then passed to `v_lshlrev_b32_e32` as the shift amount, producing
a garbage address. 12-byte elements require a multiply (not a shift). Should
either emit `v_mul_lo_u32 v3, 12, v_index` or fail loudly.

### 8. `scale_offset` detection relies on operand-type heuristics
The pattern detector scans operands for "first single-VGPR" (assumed to be the
index) and "first 64-bit SGPR pair with index < 100" (assumed to be the base).
This can misfire in several ways:
- A `global_store_byte` where the first single-VGPR is the data, not the index
- A global load that already uses 64-bit VGPR addressing (no SGPR pair → the
  heuristic correctly skips it, but this is coincidental, not principled)
- The `< 100` threshold is arbitrary — intended to exclude VCC (106) and
  EXEC (126) but could exclude legitimate high-numbered SGPR bases

The correct approach is to check for `scale_offset` as an encoding flag in the
MCInst or track it as an attribute through the lifter.

### 9. No validation of kernel descriptor against translated code
The gfx942 template's kernel descriptor metadata (`.amdhsa_next_free_vgpr`,
`.amdhsa_next_free_sgpr`, `.amdhsa_reserve_vcc`) comes from the original gfx942
compilation. If the translated core uses more registers than the template
expected (e.g., a complex kernel where scale_offset lowering adds v3–v5 on top
of an already-full VGPR budget), the descriptor would be wrong — the GPU would
under-allocate register file space, causing silent data corruption or hangs.

## Medium – Scaling

### 10. Emitter uses mnemonic-prefix heuristics for operand layout
Each instruction format (SMEM, VMEM, VOPC, VOP3, MUBUF, DS, FLAT, …) has
unique operand ordering and trailing flags. The emitter matches by
`mnemonic.starts_with(...)` and hardcodes operand counts. Will break on any
format not yet handled (MUBUF, scratch, buffer, DS, VOP3/VOP3P, DPP, SDWA).
The correct fix is to use `MCInstrDesc` format flags or TableGen-generated
format metadata.

### 11. Core extraction/splicing is fragile text-based pattern matching
`extractCoreFromPipelineOutput` identifies the preamble/core boundary by
looking for `s_cbranch_execz` as text. `spliceCoreInstructions` identifies
the core-end by looking for a label starting with `.` and ending with `:`.
Failure modes:
- Kernels without a bounds-check branch (no `s_cbranch_execz`) → empty
  extraction, no core found
- Kernels with multiple early exits or nested branches
- Labels from debug info or data sections matching the `.LBB*:` pattern
- Multi-kernel code objects (only the first kernel's structure is considered)

### 12. Assembly splicing searches for `s_endpgm` as text
`spliceInstructions` (the full-kernel version) finds the kernel label and
replaces everything up to the first `s_endpgm`. Breaks on kernels with
multiple `s_endpgm` (early returns), post-`s_endpgm` constant pools, and
multi-kernel code objects.

### 13. SSA construction is linear — no CFG awareness
The SSA pass walks the block linearly with a single "latest def" map. Branches
and labels create implicit control flow that the pass ignores. Registers
defined on one path and used on another get incorrect wiring. Blocks real SSA
passes (register allocation, scheduling) which need dominance frontiers and phi
nodes.

### 14. `rebuildCodeObject` shells out via `system()` with `/tmp/` paths
Hardcoded temp file paths (`/tmp/hotswap_mve_rebuild.*`) — not thread-safe.
Uses `system()` for `llvm-mc` and `ld.lld` — shell injection risk and
portability issue. Should use the LLVM MC / LLD library APIs in-process.

### 15. GFX12 hint dropping is incomplete
Only `s_clause`, `s_delay_alu`, and `s_code_end` are erased when targeting
GFX9. Other GFX12-specific instructions that have no GFX9 equivalent are not
handled: `s_wait_event`, `s_set_inst_prefetch_distance`,
`s_singleuse_vdst`, `s_wait_idle`, etc. These will pass through to the emitter
and produce invalid assembly that fails at `llvm-mc` time. This at least fails
loudly, but would be better caught at the MLIR level.

### 16. Mnemonic mapping is purely 1:1 — no semantic adjustments
Some renames are not semantically neutral. For example, `s_add_co_u32` (GFX12)
and `s_add_u32` (GFX9) both write SCC on overflow, but GFX12 may define
additional implicit defs that don't exist on GFX9. The pipeline assumes all
1:1 renames are drop-in replacements. For the current vecadd kernel this is
safe, but more complex code may have subtle correctness differences.

### 17. Preamble-core split assumes a specific ABI contract
The cross-ISA strategy uses the gfx942 template's preamble and only translates
the "core" (instructions after `s_cbranch_execz`). This implicitly assumes:
- The template preamble computes the same thread index (`v0`) and loads the
  same ABI registers (`s[0:1]` for kernarg pointer, `s2` for workgroup ID)
- The core only reads from these registers
- Both ISA compilations produce the same preamble structure

This works for simple kernels where the compiler always puts kernarg in
`s[0:1]` and thread ID in `v0`, but will break for kernels that use multiple
workgroup dimensions, dispatch pointer, flat scratch init, or any other ABI
register in the core body.

### 18. `readFile` / `readFileAsString` fail silently
Both return empty on I/O failure without printing an error message. Callers
must remember to check for empty results. Violates the error handling policy
— should `report_fatal_error` or at least print a diagnostic.

### 19. `assembleToBytes` silently skipped for cross-ISA
The pipeline returns `success = true` with empty `bytes` when `sourceISA !=
targetISA`. This semantic change is not documented in the `PipelineResult`
contract — callers checking `result.bytes.empty()` would wrongly interpret a
successful cross-ISA run as a failure.

## Low – Test Coverage

### 20. Narrow test coverage
Both MVE tests use a single `vecadd` kernel (one branch, no shared memory, no
atomics, no matrix ops, no divergent reconvergence, no multi-dimensional
dispatch). The gfx942→gfx942 round-trip test's 292 "lifted" instructions are
~24 real + ~268 `s_nop` padding. The cross-ISA test's gfx1250 kernel has only
~20 real instructions in the core. Functional correctness is verified for one
input pattern (A[i]=i, B[i]=2i) on one GPU — does not catch:
- Timing-dependent bugs from wrong wait counts causing races
- Edge cases in register allocation (all temp registers happen to be free)
- Larger-than-dword memory operations (dwordx2/x3/x4 with scale_offset)
- Negative or non-contiguous index patterns
- Kernels that use the full VGPR/SGPR budget
