; RUN: %raise_cli %s_bitset0_b32_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=s_bitset0_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for s_bitset0_b32. Pins that the SOP1 bit-clear lowers
; to an LLVM `and` against `~(1 << (bit_pos & 0x1F))`, with the
; *prior* dst value (here, the 0xDEADBEEF sentinel) on the
; left-hand side of the AND.  The handler lives in
; transpiler/handle_sop1.cpp under the
; `S_BITSET0_B32 || S_BITSET1_B32 || S_BITSET0_B64 || S_BITSET1_B64`
; block; the SemOp lives in transpiler/semop.hpp under SOP1.
;
; Why this fixture matters: TableGen declares S_BITSET0_B32 with a
; tied `$sdst_in` operand (`SOP1_32` with `tied_in=1`,
; `Constraints = "$sdst = $sdst_in"`), but the AMDGPU disassembler
; collapses the tied slot and emits a 2-operand MCInst (`sdst`,
; `src0`).  Earlier handler revisions asserted `op.nSrcs() >= 2`
; expecting `sdst_in` in srcMap and aborted SIGABRT on every real
; s_bitset corpus site (triton f16 GEMM TDM-pipelined kernels and
; the compare_correctness s_bitset0_b{32,64} fixtures themselves).
; The fix reads the prior dst value via
; `ctx.regs.readReg32(op.dst())` — same pattern as S_CMOV_B32 — and
; this fixture pins that read-modify-write shape so the regression
; cannot return.
;
; Invariants:
;
;   1. The lifted IR contains exactly one `and i32 ...` whose
;      left-hand operand carries the 0xDEADBEEF sentinel (decimal
;      `-559038737` signed, or `3735928559` unsigned — accept
;      either rendering).  This pins "prior dst correctly read".
;   2. The right-hand operand of that AND is a `not (shl 1, bit_pos
;      & 0x1F)` chain — the standard bit-clear mask.  Pinning the
;      `xor ... -1` (LLVM's canonical `not`) and the `0x1F` mask
;      is what catches a regression that silently dropped the bit-
;      index masking and let `shl 1, N` become poison for N >= 32.
;   3. NO assertion / abort in stderr — exit 0, not the legacy
;      SIGABRT path.
;
; What we don't pin: the exact SSA names (they depend on the
; kernarg-lowering pipeline upstream of the bitset), nor the
; ordering of the AND operands (LLVM canonicalisation may swap
; them), nor whether the lifted result IR uses `i32` or a wider
; type for the bit-position computation.

; CHECK-LABEL: define amdgpu_kernel void @s_bitset0_b32_kernel(

; The bit-clear AND.  The handler names this instruction `bitset0`
; (see handle_sop1.cpp's `CreateAnd(..., "bitset0")`); pinning the
; SSA name is greppable downstream and is also a back-reference
; from the test to the handler that emits it.
;
; LHS of the AND must carry the 0xDEADBEEF sentinel — either as
; a constant operand or via an SSA copy; the prior-dst read goes
; through `regs.readReg32(op.dst())` which returns a load of the
; sdst alloca, and that load's defining store is the s_mov_b32
; that materialised the sentinel.  We pin on the sentinel
; appearing somewhere in the IR (constant materialisation may be
; folded/CSE'd by IRBuilder before the bitset AND lands).
; CHECK-DAG: {{(-559038737|3735928559)}}
; CHECK-DAG: %bitset0 = and i32

; The bit-index mask: hardware only consumes `src0[4:0]`, so the
; handler emits an `and i32 %src0, 31` before the shl.  A
; regression that dropped this mask would let `shl 1, N` produce
; poison for N >= 32 (legitimate inputs the hardware silently
; truncates).  We pin the mask constant and the shl as separate
; checks because IRBuilder may name them different things across
; LLVM versions.
; CHECK-DAG: and i32 %{{[^,]+}}, 31
; CHECK-DAG: shl i32 1, %{{[^,]+}}

; Negative pin: no `unreachable` or assertion in the lifted body —
; the kernel must complete its lift, not refuse mid-instruction.
; CHECK-NOT: unreachable
