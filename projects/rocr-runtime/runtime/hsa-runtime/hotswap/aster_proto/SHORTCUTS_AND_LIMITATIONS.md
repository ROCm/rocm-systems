# Aster Proto Pipeline – Shortcuts & Limitations

Known unsound shortcuts and scaling limitations in the current implementation.
Items are ranked by severity within each category.

## High – Correctness

### 1. Exec-mask → VCC conversion loses per-lane predication
The `s_and_saveexec_b64` + `s_cbranch_execz` pattern is converted to
`cbranch s_cbranch_vccz`, which is a wavefront-level branch on VCC — not
exec-mask predication. In the original binary, EXEC masks off inactive lanes
so that memory ops (loads/stores) only execute for threads where `i < N`. The
VCC-branch conversion only skips the entire wavefront if ALL lanes fail the
compare. If even one lane passes, ALL lanes execute the body — including those
with `i >= N`. This is correct only when the grid dispatch is aligned to the
wavefront size (64 for GFX9). For the last partial wavefront in an unaligned
dispatch, out-of-bounds threads will execute loads and stores to invalid
addresses, causing memory corruption or GPU faults.

### 2. Exec-mask pattern recognizer assumes a single linear pattern
The converter tracks `lastVCC` (the most recent compare result) and expects
exactly one `s_and_saveexec_b64` → `s_cbranch_execz` sequence per kernel.
Will fail on:
- Nested exec-mask manipulation (`if/else` with multiple `saveexec`/restore)
- Loops that modify EXEC
- Multiple independent guarded regions
- `s_and_saveexec` whose source is not the most recent VCC (e.g., an
  intervening compare for a different purpose)
- `s_or_saveexec`, `s_andn2_saveexec`, and other variants

### 3. Kernel arguments are hardcoded for vecadd
The lifter hardcodes exactly 3 `BufferArgAttr` (read, read, write) + 1
`ByValueArgAttr<i32>`. Any kernel with a different argument signature — more
args, different types, different access patterns, pointer-to-pointer, structs,
images, samplers — will produce incorrect kernel metadata. The args must be
extracted from the original code object's metadata, not assumed.

### 4. Hidden kernel args are hardcoded for the HIP ABI
The pipeline injects a fixed set of 13 hidden arg entries at fixed offsets
(32–96) and hardcodes `kernarg_segment_size: 288`. These offsets and the set
of hidden args come from the original vecadd binary's metadata. Other kernels
may need different hidden args (e.g., `hidden_queue_ptr`, `hidden_private_base`,
`hidden_completion_action`, `hidden_multigrid_sync_arg`), different offsets, or
different sizes. The injected metadata must be extracted from the original code
object rather than hardcoded.

### 5. `RegClass::Other` silently drops the instruction
Unrecognized registers (FLAT_SCRATCH, M0, MODE, NULL_REG, TTMP registers)
cause `getOrCreateSingleReg` to return `{}`. This makes the operand check
(`if (dst && src0 && ...)`) fail, and the instruction is tracked as
"unsupported" — but the reason it's unsupported (unrecognized register class)
is not reported. The lifter should fail loudly with a diagnostic identifying
the specific register.

### 6. All immediates are truncated to i32
`getImmVal` always creates `arith.constant` with 32-bit type. Some instructions
use 16-bit immediates (SOPP branches, DPP modifiers) or 64-bit immediates
(literal constants in VOP3). Truncation to 32 bits may silently corrupt the
value. Particularly dangerous for negative 16-bit sign-extended values.

### 7. Disassembly failure silently skips 4 bytes
When `getInstruction` fails (line 327-329), the lifter advances by 4 bytes and
continues without reporting an error. This can silently skip valid instructions
in unusual encodings or misaligned data sections, producing an incomplete IR
that appears successful.

### 8. Target is hardcoded to GFX942 / CDNA3
The `amdgcn.module` is created with `Target::GFX942, ISAVersion::CDNA3`
regardless of the `targetISA` string parameter. Passing `"gfx950"` or any
other ISA would produce MLIR targeting GFX942 while the disassembler decodes
for the requested ISA — a silent mismatch.

### 9. `s_waitcnt` counter decoding is GFX9-specific
The packed encoding (`vmcnt = bits[3:0]`, `expcnt = bits[6:4]`,
`lgkmcnt = bits[13:8]`) is specific to GFX9. GFX10+ uses a different bit
layout (vmcnt extends to 6 bits with high bits at [15:14]). GFX12 replaces
`s_waitcnt` entirely with split counters (`s_wait_loadcnt`, etc.). Using the
GFX9 decoder on a different generation will silently produce wrong counter
values.

## Medium – Scaling

### 10. Only ~20 instruction mnemonics are supported
The lifter explicitly handles: `v_add_f32_e32`, `v_add_u32_e32`,
`v_lshlrev_b32_e32`, `v_ashrrev_i32_e32`, `v_mov_b32_e32`, `s_and_b32`,
`s_or_b32`, `s_mul_i32`, `s_lshl_b32`, `s_mov_b32`, `v_cmp_gt_i32_*`,
`s_load_dword{,x2,x4}`, `global_load_dword`, `global_store_dword`,
`v_lshlrev_b64`, `v_lshl_add_u64`, `s_waitcnt`, `s_nop`, `s_endpgm`.
Missing: all MUBUF/scratch/DS/FLAT variants, VOP3P/SDWA/DPP, atomics, matrix
ops (MFMA), buffer loads, dwordx2/x3/x4 global loads, all other compare ops,
all other SOP1/SOP2/SOPC ops, all branch ops besides `s_cbranch_execz`, etc.
Every new kernel will require adding instructions.

### 11. Only `v_cmp_gt_i32` is handled as a comparison
Other VOPC opcodes (`v_cmp_lt_i32`, `v_cmp_eq_i32`, `v_cmp_ge_u32`,
`v_cmp_*_f32`, `v_cmp_*_f64`, etc.) are not recognized. The exec-mask
conversion depends on a preceding `v_cmp_gt_i32` setting `lastVCC`; any
other compare opcode will leave `lastVCC` empty, causing the subsequent
`s_and_saveexec` to fall through as unsupported.

### 12. Assembly metadata patching is fragile string manipulation
The pipeline modifies the generated assembly via `find()`/`substr()` to:
- Replace `.amdhsa_kernarg_size 28` → `288`
- Replace `.kernarg_segment_size: 28` → `288`
- Find `.value_kind: by_value` to locate injection point
These string patterns are sensitive to Aster's output formatting. Changes in
whitespace, field ordering, or value formatting will silently break the
patching. The substring search for `".kernarg_segment_size: 28"` could also
match `".kernarg_segment_size: 280"` or `"288"`, producing corrupt YAML.

### 13. Hardcoded temp file paths — not thread-safe
The pipeline writes to `/tmp/aster_proto.{s,o,hsaco}`. Multiple concurrent
pipeline invocations will race on these files. Should use `mkstemp()` or
`llvm::sys::fs::createTemporaryFile`.

### 14. Shells out via `system()` for assembly and linking
Uses `system()` to invoke `llvm-mc` and `ld.lld` as subprocesses. This is:
- Slow (process creation overhead per kernel)
- A shell injection risk if `targetISA` contained special characters
- Fragile (depends on external tool availability at hardcoded paths)
Should use the LLVM MC and LLD library APIs in-process.

### 15. `llvm-mc` and `ld.lld` come from different LLVM versions
`llvm-mc` is invoked from `/opt/rocm/llvm/bin/` (system ROCm) while `ld.lld`
is invoked from Aster's LLVM build. Object files produced by one LLVM version
and linked by another could have subtle format incompatibilities.

### 16. Single-kernel code objects only
The lifter starts disassembling at offset 0 in `.text` and stops at the first
`s_endpgm`. Multi-kernel code objects (common in production HIP binaries) have
kernels at different offsets identified through the ELF symbol table. The lifter
needs to look up kernel symbols to find the correct start offset.

### 17. No cross-ISA support
Unlike the waveasm prototype (gfx1250 → gfx942), the Aster prototype only
handles same-ISA round-tripping (gfx942 → gfx942). Cross-ISA would require
mnemonic mapping, encoding translation, and different kernel descriptor
generation — all of which are future work.

### 18. Mnemonic identification is string-based rather than opcode-based
Instructions are identified by printing the `MCInst` and parsing the mnemonic
string. This is fragile because:
- The same instruction can print with different suffixes (`_e32`, `_e64`,
  `_sdwa`, `_dpp`)
- Printing roundtrips through string formatting, which is slow
- The `MCInstrDesc::getOpcode()` provides a reliable numeric ID that should be
  used instead

### 19. LLVM static/dynamic symbol conflict is papered over
The `--exclude-libs,ALL` linker flag hides Aster/LLVM static symbols to prevent
clashes with the HIP runtime's dynamically loaded `libLLVM.so`. This is a
workaround — if both versions register conflicting global constructors or
pass plugins, silent corruption can occur.

### 20. Pipeline header comment is stale
`pipeline.hpp` documents "no subprocess calls" but the implementation uses
`system()` for both `llvm-mc` and `ld.lld`.

## Low – Test Coverage

### 21. Single kernel, single data pattern
The test uses one `vecadd` kernel with `A[i]=i`, `B[i]=2*i`, `N=1024`. Does
not exercise: NaN/Inf, negative values, denormals, f64, integer overflow,
large grids, partial wavefronts, shared memory, atomics, matrix ops, divergent
control flow, multi-dimensional dispatch, dynamic LDS.

### 22. Dispatch is exactly aligned to wavefront size
`N=1024` with block size 256 produces 4 full blocks of 4 full wavefronts.
No thread has `i >= N`. This means the exec-mask → VCC conversion shortcut
(item 1) is never actually tested — the comparison is always true for all
lanes, so the branch is never taken regardless of whether EXEC or VCC is used.
A test with `N=1000` (partial last wavefront) would expose the unsoundness.

### 23. Original kernel name is lost
The lifter uses `"vecadd"` as the kernel name. The original binary has the
mangled name `_Z6vecaddPfS_S_i`. If the host code looked up the kernel by its
mangled name, the lookup would fail. The kernel name should be extracted from
the original code object's symbol table or metadata.
