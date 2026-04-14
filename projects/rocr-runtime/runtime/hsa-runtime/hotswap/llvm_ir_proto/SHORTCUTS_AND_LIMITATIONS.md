# LLVM IR Prototype — Shortcuts and Limitations

Systematic analysis of shortcuts, limitations, and assumptions in the LLVM IR
binary translation prototype.

## Severity Legend
- **HIGH** — Would cause incorrect results or crashes on non-trivial kernels
- **MEDIUM** — Limits applicability but doesn't affect correctness for vecadd
- **LOW** — Engineering debt; straightforward to fix

---

## Raiser (Binary → LLVM IR)

### 1. Per-instruction raising with limited opcode coverage [MEDIUM]
The raiser walks the decoded instruction stream and translates each
instruction individually using a register value tracking system (`RegFile`).
It handles ~15 instruction mnemonics covering scalar loads, scalar/vector
ALU, comparisons, exec-mask branches, global memory, and control flow. Any
unrecognized instruction causes an immediate failure with a diagnostic.

**Impact**: The raiser works for kernels using only the supported instruction
set (which covers vecadd and similar simple kernels). Extending to more
complex kernels requires adding handlers for additional instruction classes
(MUBUF, DS, atomics, MFMA, additional VOP3, etc.).

### 2. Hardcoded kernel signature [HIGH]
The kernel function signature (`ptr addrspace(1), ptr addrspace(1),
ptr addrspace(1), i32`) is hardcoded. A production system would parse the
original ELF's `.note` metadata to recover the kernarg layout and construct
the signature dynamically.

### 3. No MIR intermediate — raises directly from MCInst [MEDIUM]
The raiser lifts directly from the decoded MCInst stream to LLVM IR, bypassing
the LLVM MIR layer. It uses `MCInstrDesc` flags (`isBranch`,
`isConditionalBranch`) for branch detection but relies on mnemonic strings
(with encoding suffixes stripped) for instruction classification.

**When this matters**: Using the full MIR layer (with `MCInstrDesc::mayLoad`,
`mayStore`, implicit defs/uses) would make the instruction classifier more
robust and eliminate mnemonic-based matching.

### 4. Exec-mask to branch conversion assumes wavefront alignment [HIGH]
The `s_and_saveexec_b64` + `s_cbranch_execz` exec-mask pattern is converted
to `icmp slt` + `br i1`. This is correct only when the dispatch is aligned
to wavefront boundaries (all lanes active or all inactive per branch). For
partial wavefronts (e.g., N=1000 with 64-wide waves), the per-lane
predication semantics are lost.

**Impact**: Same limitation as the Aster prototype. The GPU test passes because
N=1024 is a multiple of 256 (4 full wavefronts of 64).

### 5. Implicit arg offset is ABI-version-specific [MEDIUM]
The raiser hardcodes that `hidden_group_size_x` is at implicit arg offset 12
(kernarg byte offset 44 for vecadd). This depends on the HIP implicit argument
ABI layout, which varies by code object version. The prototype assumes COV6.

---

## IR Quality

### 6. IR is semantically correct but may differ from compiler output [LOW]
The generated IR is valid and produces correct results, but may differ from
`hipcc -emit-llvm` output in:
- Instruction ordering (we emit in a fixed order, the compiler may reorder)
- Attribute sets (we only set `amdgpu-flat-work-group-size`)
- Alignment annotations on loads/stores

These differences don't affect correctness — `llc` handles them — but they
mean a direct `diff` against compiler output won't be clean.

### 7. No `nsw`/`nuw` flags on arithmetic [LOW]
The compiler emits `add nsw`, `mul nsw` etc. with no-signed-wrap flags that
enable optimizations. Our lifted IR uses plain `add`/`mul`. This may produce
slightly different register allocation or scheduling but not incorrect results.

### 8. No `noundef` or `range` metadata [LOW]
The compiler annotates intrinsic results with `noundef` and range metadata
(e.g., `workitem.id.x` returns `range(i32 0, 1024)`). Our lifted IR omits
these. No correctness impact.

---

## Pipeline (IR → HSACO)

### 9. `llc` fully recompiles the IR [MEDIUM]
Unlike the MIR prototype (which round-trips the original bytes), the LLVM IR
prototype feeds the raised IR into `llc`, which performs full instruction
selection, register allocation, and scheduling. The output assembly will use
**different registers and instruction ordering** than the original binary.
This is by design — it demonstrates semantic recovery — but means:
- Debug output is harder to correlate with the original binary
- Performance characteristics may differ

### 10. External tools dependency [MEDIUM]
Assembly → HSACO uses external `llc`, `llvm-mc`, and `ld.lld` via
`std::system()`. Same limitation as all other prototypes.

### 11. Temporary file I/O [LOW]
IR, assembly, object, and HSACO files are written to `/tmp/ir_proto_<pid>/`.
No cleanup on success or failure.

### 12. Single-kernel assumption [HIGH]
The raiser assumes the entire `.text` section is one kernel, stopping at
`s_endpgm`. Multi-kernel code objects would be truncated.

---

## Compared to Other Prototypes

### 13. The only prototype with standard backend integration [STRENGTH]
This is the only prototype where the generated IR feeds directly into LLVM's
standard AMDGPU backend (`llc`). The backend handles:
- Register allocation (no manual metadata needed)
- Wait counter insertion (`SIInsertWaitcnts`)
- Kernel descriptor generation (`.amdhsa_kernel`)
- Metadata emission (`.amdgpu_metadata` YAML)

No manual assembly patching is required — a major improvement over waveasm,
Aster, and LLVM MIR prototypes.

### 14. Limited instruction coverage [MEDIUM]
The raiser handles ~15 instruction mnemonics (scalar loads, scalar/vector ALU,
compare, exec-mask, global memory, branches). Kernels using unsupported
instructions (MUBUF, DS, atomics, MFMA, additional VOP3 variants) will fail
loudly. The other prototypes have broader coverage:
- waveasm: `waveasm.raw` escape hatch (100% coverage by definition)
- Aster: per-format-class ops (extensible but manual)
- LLVM MIR: per-opcode `BuildMI` (all opcodes supported)

Adding new instruction handlers is straightforward — each is a self-contained
block in the instruction loop.

---

## Summary

| Category | Count by Severity |
|----------|-------------------|
| HIGH | 2 (hardcoded signature, single kernel) |
| MEDIUM | 6 (limited opcodes, no MIR intermediate, ABI-specific offsets, exec-mask, full recompilation, external tools) |
| LOW | 5 (IR differences, no nsw/nuw, no metadata, temp files, attribute gaps) |

**Key achievement**: This is the only prototype that uses LLVM's standard
backend pipeline. The IR → `llc` → `llvm-mc` → `ld.lld` path requires **no
manual kernel metadata or assembly patching** — everything is generated by
the AMDGPU backend from the lifted IR. This validates the approach: if we
can raise the IR correctly, the backend handles everything else.

**Key achievement (v2)**: The raiser now performs **per-instruction translation**
using a register value tracking system (`RegFile`). Each decoded instruction
reads source operands from the register file, creates IR via `IRBuilder`, and
writes results back. Register pairs (64-bit values) are automatically combined
from individual 32-bit halves when needed. Pointer values flow correctly
through the register file, enabling `getelementptr` emission for address
computation.

**Key limitation**: The kernel signature and kernarg layout are still hardcoded
for vecadd. Extending to other kernels requires parsing the `.note` metadata
from the ELF for kernel argument types and sizes.
