; RUN: %raise_cli %s_bitcmp_co --isa=gfx1250 --target-isa=gfx942 \
; RUN:     --emit-ir=s_bitcmp_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for the SOPC bit-test family
; (s_bitcmp0_b32 / s_bitcmp1_b32 / s_bitcmp0_b64 / s_bitcmp1_b64).
; See SemOp::S_BITCMP{0,1}_{B32,B64} in transpiler/semop.hpp; the
; shared handler block in transpiler/handle_sopc.cpp under
; `if (is64 || isB32) { ... }`; and the SOPC mapping in
; transpiler/opcode_map.cpp.  Bidirectional handler <-> test
; back-reference is non-negotiable per the transpiler test policy.
;
; Closes the kerneldex corpus blocker on scope_discovery `_attn_fwd`
; (one `s_bitcmp0_b32 s68, 1` that the pre-fix handler left
; unhandled — the compare silently dropped, the downstream
; SCC-dependent branch lifted to an undef-driven CFG).
;
; INVARIANTS PINNED:
;
;   1. For B32 variants, the shift amount is masked to 5 bits
;      (`and i32 _, 31`) before the `shl i32 1, _` bit-construction.
;      Mirrors the hardware's (src1 & 0x1F) shift-amount truncation
;      and keeps the LLVM IR `shl i32` in well-defined range.
;
;   2. For B64 variants, the shift amount is masked to 6 bits
;      (`and i32 _, 63`), then widened to i64 before the `shl i64`.
;      Same hardware invariant (src1 & 0x3F) and same IR-legality
;      argument, but over the 64-bit source width.
;
;   3. The final predicate differs per variant only:
;        _B32 / _B64 with the `0` suffix -> `icmp eq _, 0`
;        _B32 / _B64 with the `1` suffix -> `icmp ne _, 0`
;      A regression that flipped the 0/1 polarity would break every
;      SCC-branch that depends on bit presence.
;
;   4. The B64 sources are read as i64 (the SGPR pair is kept intact
;      through the `op.src64` reader) rather than narrowed to i32 —
;      a pair-of-i32 lift would only test the low 32 bits.

; CHECK-LABEL: define amdgpu_kernel void @s_bitcmp_kernel(

; --- s_bitcmp0_b32: shamt mask=0x1F, shift i32, compare eq 0 ------------
; Uses CHECK (not CHECK-NEXT) at the block boundary because each
; bitcmp block is separated from the next by emitted SPE /
; writeReg32 traffic for the surrounding store. The four lines
; within a single bitcmp block are consecutive in the IR though,
; so CHECK-NEXT is used inside each block.
; CHECK: %bitcmp_shamt = and i32 %{{[^,]+}}, 31
; CHECK-NEXT: %bitcmp_bit = shl i32 1, %bitcmp_shamt
; CHECK-NEXT: %bitcmp_mask = and i32 %{{[^,]+}}, %bitcmp_bit
; CHECK-NEXT: %bitcmp0 = icmp eq i32 %bitcmp_mask, 0

; --- s_bitcmp1_b32: shamt mask=0x1F, shift i32, compare ne 0 ------------
; Each block-entry pattern anchors on the `bitcmp_shamt` value name
; (with a numeric-suffix wildcard) rather than on a bare
; `and i32 _, 31` — the SPE lane-projection code introduces
; unrelated `and i32 %lane_id, 31` expressions between bitcmp
; blocks, and without this anchor the block-entry match would
; straddle the SPE region and misalign the follow-on adjacency
; pins.  See SPE lane detection in handle_common.cpp.
; CHECK: %bitcmp_shamt{{[0-9]*}} = and i32 %{{[^,]+}}, 31
; CHECK-NEXT: %bitcmp_bit{{[0-9]*}} = shl i32 1, %bitcmp_shamt{{[0-9]*}}
; CHECK-NEXT: %bitcmp_mask{{[0-9]*}} = and i32 %{{[^,]+}}, %bitcmp_bit{{[0-9]*}}
; CHECK-NEXT: %bitcmp1 = icmp ne i32 %{{[^,]+}}, 0

; --- s_bitcmp0_b64: shamt mask=0x3F, zext to i64, shift i64, eq 0 -------
; CHECK: %bitcmp_shamt{{[0-9]*}} = and i32 %{{[^,]+}}, 63
; CHECK-NEXT: %bitcmp_shamt64{{[0-9]*}} = zext i32 %{{[^,]+}} to i64
; CHECK-NEXT: %bitcmp_bit{{[0-9]*}} = shl i64 1, %bitcmp_shamt64{{[0-9]*}}
; CHECK-NEXT: %bitcmp_mask{{[0-9]*}} = and i64 %{{[^,]+}}, %bitcmp_bit{{[0-9]*}}
; CHECK-NEXT: %bitcmp0{{[0-9]*}} = icmp eq i64 %{{[^,]+}}, 0

; --- s_bitcmp1_b64: shamt mask=0x3F, zext to i64, shift i64, ne 0 -------
; CHECK: %bitcmp_shamt{{[0-9]*}} = and i32 %{{[^,]+}}, 63
; CHECK-NEXT: %bitcmp_shamt64{{[0-9]*}} = zext i32 %{{[^,]+}} to i64
; CHECK-NEXT: %bitcmp_bit{{[0-9]*}} = shl i64 1, %bitcmp_shamt64{{[0-9]*}}
; CHECK-NEXT: %bitcmp_mask{{[0-9]*}} = and i64 %{{[^,]+}}, %bitcmp_bit{{[0-9]*}}
; CHECK-NEXT: %bitcmp1{{[0-9]*}} = icmp ne i64 %{{[^,]+}}, 0

; Negative pin: no B64 variant emits a 32-bit shift against the
; 64-bit source. The literal token `%bitcmp_shamt64` is only
; introduced by the B64 lowering, so catching `shl i32 1, %bitcmp_shamt64`
; would expose a regression that dropped the zext step while keeping
; the value-name.
; CHECK-NOT: shl i32 1, %bitcmp_shamt64

; Negative pin: no i32 source operand fed into a B64-shaped icmp.
; A pair-of-i32 regression for the B64 variants would replace the
; `i64` source width with `i32`.
; CHECK-NOT: %bitcmp_mask{{.*}} = and i32 %{{.*}}, %bitcmp_bit20
; CHECK-NOT: %bitcmp_mask{{.*}} = and i32 %{{.*}}, %bitcmp_bit26
