// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_INTEGER_H_
#define RJ_ARITH_INTEGER_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Select each output byte using the corresponding byte of C. Selectors 0..7
// select bytes of the concatenation A:B (B supplies the low four bytes).
// Selectors 8..11 replicate the sign bit of its four 16-bit halves; 12 selects
// zero, and 13..255 select 0xff. Each operand uses one VGPR. Supports DPP8/DPP16
// on A. CLAMP is accepted and ignored; other low flags are invalid.
// All supported semantics produce exact integer results; host FP state is preserved.
int rj_arith_v_perm_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

// Bitwise equivalence: D = ~(A ^ B). Each operand uses one VGPR. Supports
// DPP8/DPP16 on A. CLAMP is accepted and ignored; other low flags are invalid.
// Both semantics are bit-exact; host FP state is preserved.
int rj_arith_v_xnor_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

// Extract the low 32 bits of the concatenation A:B shifted right by C & 31
// bits (ALIGNBIT), or (C & 3) bytes (ALIGNBYTE). Each operand uses one VGPR.
// Supports DPP8/DPP16 on A. CLAMP is accepted and ignored; other low flags
// are invalid. All supported semantics produce exact integer results; host FP state
// is preserved.
int rj_arith_v_alignbit_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_alignbyte_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

// Multiply two 32-bit lanes and add a 64-bit accumulator. A/B use one VGPR;
// C/D use low/high pairs. The scalar output contains bit 64 of the full sum:
// unsigned carry for U64, or the sign of the mathematical 65-bit sum for I64.
// CLAMP saturates to the unsigned/signed 64-bit range without changing that mask.
// Supports CLAMP only, loose and empirical exact semantics, every EXEC mask and
// whole-register alias. All inputs are read before D0, then D1, then carry are
// written; D1 wins if D0/D1 alias. Inactive scalar bits are cleared, even for zero
// EXEC. carry must always be writable; zero EXEC permits null VGPR pointers.
// Errors leave all destinations unchanged. Host FP state is preserved and
// RJ_ARITH_FP16_OVFL has no effect.
int rj_arith_v_mad_co_u64_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mad_co_i64_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c);

// RDNA3 spelling of unsigned 32x32+64 multiply-add. Same carry, CLAMP,
// aliasing and execution-mask contract as rj_arith_v_mad_co_u64_u32_w32.
int rj_arith_v_mad_u64_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

// RDNA3 signed spelling. The scalar result is bit 64 of the full signed sum,
// with the same contract as rj_arith_v_mad_co_i64_i32_w32 (not a signed-overflow bit).
int rj_arith_v_mad_i64_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

// Unsigned add/subtract with a scalar carry/borrow output. CI forms also consume
// one input carry/borrow bit per lane. SUBREV computes B-A-input_borrow.
// CLAMP saturates overflow to UINT32_MAX for addition and underflow to zero for
// subtraction; carry/borrow bits still describe the unsaturated result.
// Supports CLAMP and DPP8/DPP16: ADD/SUB permute A, SUBREV permutes B. Input
// carry bits stay in their original lanes. Supports loose and empirical exact
// semantics, all EXEC masks and whole-register aliases. Inactive and DPP-filtered
// scalar output bits are cleared, even at zero
// EXEC. The scalar output must always be writable and is written after VGPR D;
// zero EXEC permits null VGPR pointers. Errors leave all destinations unchanged.
// Host FP state is preserved; RJ_ARITH_FP16_OVFL has no effect.
int rj_arith_v_add_co_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_sub_co_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_subrev_co_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                                 const uint32_t *const *b);

int rj_arith_v_add_co_ci_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t input_carry);

int rj_arith_v_sub_co_ci_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                                 const uint32_t *const *b, uint32_t input_carry);

int rj_arith_v_subrev_co_ci_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, uint32_t *carry, const uint32_t *const *a,
                                    const uint32_t *const *b, uint32_t input_carry);

// Combined three-input integer operations, one VGPR per operand. Shift
// counts wrap modulo 32; all addition and shifting wrap to 32 bits. Supports
// DPP8/DPP16 on A. CLAMP is accepted and ignored; other low instruction flags
// are invalid. Both semantics are exact; preserves all host FP state.

// (A << (B & 31)) + C.
int rj_arith_v_lshl_add_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

// (A + B) << (C & 31).
int rj_arith_v_add_lshl_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

// (A << (B & 31)) | C.
int rj_arith_v_lshl_or_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

// (A & B) | C.
int rj_arith_v_and_or_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c);

// A | B | C.
int rj_arith_v_or3_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

// A ^ B ^ C.
int rj_arith_v_xor3_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

// (A ^ B) + C.
int rj_arith_v_xad_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

// Each result byte is floor((A_byte + B_byte + (C_byte & 1)) / 2).
// Higher C-byte bits are ignored. Carries do not cross byte boundaries.
int rj_arith_v_lerp_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

// Sum of unsigned absolute differences between packed fields of A and B, plus
// C. SAD_U8 sums four byte differences, SAD_U16 two halfword differences, and
// SAD_U32 one full-word difference. SAD_HI_U8 shifts the byte sum left by 16
// before accumulation. MSAD_U8 omits differences where the corresponding B
// byte is zero. These forms use one VGPR for each operand.
// RJ_ARITH_ALU_CLAMP saturates the final unsigned accumulation. DPP8/DPP16 permute
// A before arithmetic; B and C stay in their original lanes. Other instruction
// flags are invalid. Both semantics are exact; preserves all host FP state.
int rj_arith_v_sad_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c);
int rj_arith_v_sad_hi_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c);
int rj_arith_v_sad_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);
int rj_arith_v_sad_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);
int rj_arith_v_msad_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

// Quad SAD compares four overlapping four-byte windows of A (starting at
// byte offsets 0, 1, 2 and 3) against B. A has two VGPRs, low word first; B has
// one. QSAD uses all B bytes; MQSAD omits zero B bytes. PK_U16 packs four
// independent 16-bit accumulations into two C/D VGPRs. U32 uses four C/D VGPRs.
// RJ_ARITH_ALU_CLAMP saturates each accumulation to its destination width; otherwise
// each wraps independently. The high byte of A's second VGPR is unused.
// All other instruction flags are invalid. Both semantics are exact; preserves
// all host floating-point state.
int rj_arith_v_qsad_pk_u16_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mqsad_pk_u16_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mqsad_u32_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

// Wave64 SAD forms have the same VGPR layouts and arithmetic as above.
// EXEC and DPP row enables cover all 64 lanes.
int rj_arith_v_sad_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c);
int rj_arith_v_sad_hi_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c);
int rj_arith_v_sad_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);
int rj_arith_v_sad_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);
int rj_arith_v_msad_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);
int rj_arith_v_qsad_pk_u16_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mqsad_pk_u16_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c);
int rj_arith_v_mqsad_u32_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

// Reverse shifts: shift B by the count in A. A is one VGPR. B and D are one
// VGPR for 32-bit forms or two VGPRs (low word first) for 64-bit forms. Counts
// wrap modulo 32 or 64. ASHR replicates the sign bit; logical shifts insert
// zero bits. CLAMP is accepted and ignored; other low flags are invalid.
// The 32-bit forms support DPP8/DPP16 on A (the shift count); the 64-bit forms
// do not support DPP. Both semantics are exact; all host FP state is preserved.
int rj_arith_v_lshlrev_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);
int rj_arith_v_lshrrev_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);
int rj_arith_v_ashrrev_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);
int rj_arith_v_lshlrev_b64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);
int rj_arith_v_lshrrev_b64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);
int rj_arith_v_ashrrev_i64_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

// Ordinary 16-bit integer binary arithmetic: one VGPR per operand. HIGH_A/B/D
// select the input and output halves; the other destination half is preserved.
// ADD/SUB accept RJ_ARITH_ALU_CLAMP for signed/unsigned saturation; otherwise they
// wrap. Other forms accept and ignore CLAMP. Multiply retains its low 16 bits.
// Shifts use B as the value and A modulo 16 as the count. All forms support
// DPP8/DPP16 on A before half selection. Other instruction flags are invalid.
// Both semantics are exact; independent of host FP state.
int rj_arith_v_add_nc_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_sub_nc_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_add_nc_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_sub_nc_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_min_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_max_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_min_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_max_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_mul_lo_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_lshlrev_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_lshrrev_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_ashrrev_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

// Integer multiply-add with 32-bit C/D: one VGPR per operand. The 16-bit
// forms support HIGH_A/B; the 24-bit forms discard A/B's upper byte. Signed
// forms sign-extend the selected factors and interpret C as signed 32-bit.
// CLAMP saturates the full product plus C to the result's 32-bit range;
// otherwise results wrap. Supports DPP8/DPP16 on A before half selection. Other
// instruction flags are invalid. Both semantics are exact; independent of host FP
// state. Whole-register aliases are supported.
int rj_arith_v_mad_u32_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mad_i32_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mad_u32_u24_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mad_i32_i24_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

// Ordinary 16-bit ternary arithmetic: one VGPR per operand. HIGH_A/B/C/D
// select source and destination halves; the other D half is preserved.
// MAD supports CLAMP after full-precision A * B + C, otherwise wrapping.
// MIN3/MAX3 select the smallest/largest of all three signed/unsigned inputs.
// MED3 selects the middle value. MIN3/MAX3/MED3 accept and ignore CLAMP.
// Supports DPP8/DPP16 on A before half selection; other instruction flags are
// invalid. Both semantics are exact; independent of host floating-point state.
int rj_arith_v_mad_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int rj_arith_v_mad_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int rj_arith_v_min3_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_min3_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_max3_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_max3_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_med3_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_med3_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

// Packed integer binary arithmetic: one VGPR per operand, two 16-bit results
// per lane. Supports RJ_ARITH_PK_* half selectors for A/B and RJ_ARITH_PK_CLAMP. CLAMP
// saturates ADD/SUB to the signed/unsigned 16-bit range; min/max and multiply
// ignore it. Without CLAMP, ADD/SUB wrap. Multiply keeps the low 16 bits.
// Negation and C flags are invalid. Both results read original inputs even
// when D aliases A/B. Both semantics are exact; independent of host FP state.
int rj_arith_v_pk_add_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_sub_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_add_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_sub_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_min_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_max_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_min_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_max_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_mul_lo_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b);

// Packed 16-bit shifts: A supplies counts modulo 16; B supplies values.
// Supports RJ_ARITH_PK_* half selectors for A/B. CLAMP is accepted and ignored;
// negation and C flags are invalid. ASHR sign-extends each selected B half.
// Two results per lane, one VGPR per operand. Both semantics are exact.
int rj_arith_v_pk_lshlrev_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_pk_lshrrev_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_pk_ashrrev_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

// Packed integer multiply-add: two results per lane, one VGPR per operand.
// All RJ_ARITH_PK_* half selectors for A/B/C are supported; negation is invalid.
// CLAMP saturates the full A * B + C result to the signed/unsigned 16-bit range;
// without it, results wrap. Both halves read original sources before D is written.
// Both semantics are exact; independent of host floating-point state.
int rj_arith_v_pk_mad_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_pk_mad_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c);

// Signed/unsigned 32-bit integer selection. Each operand uses one VGPR;
// D may alias any whole source VGPR. CLAMP is accepted without changing the
// selected integer. Supports DPP8/DPP16 on A; other low instruction flags are
// invalid. Both semantics are exact.
// MINMAX computes max(min(A, B), C); MAXMIN computes min(max(A, B), C).
int rj_arith_v_min_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_max_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_min3_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_max3_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_minmax_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c);

int rj_arith_v_maxmin_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c);

int rj_arith_v_med3_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_min_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_max_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_min3_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_max3_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_minmax_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c);

int rj_arith_v_maxmin_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c);

int rj_arith_v_med3_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

// Integer multiplication into one 32-bit VGPR. Each source uses one VGPR;
// D may alias A or B. The 24-bit forms discard the upper byte of each input,
// then sign-extend signed inputs. High forms select bits 63:32 of the product.
// MUL_I32_I24 and MUL_U32_U24 support RJ_ARITH_ALU_CLAMP to saturate to the signed
// or unsigned 32-bit range; otherwise low results wrap. All four 24-bit forms
// support DPP8/DPP16 on A; other instruction flags are invalid. The 32-bit forms
// require zero instruction_flags. Both semantics are exact; preserves the host FP environment.
int rj_arith_v_mul_lo_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_mul_hi_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_mul_hi_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_mul_i32_i24_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_mul_hi_i32_i24_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_mul_u32_u24_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_mul_hi_u32_u24_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

// Non-carry 32-bit integer addition/subtraction. Each operand uses one VGPR;
// D may alias any whole source VGPR. The two-input forms accept RJ_ARITH_ALU_CLAMP
// to saturate in the signed/unsigned result domain; otherwise results wrap.
// ADD3 wraps modulo 2^32 and accepts but ignores CLAMP. All forms
// support DPP8/DPP16 on A, except SUBREV_NC_U32 which permutes B. Both semantics
// are exact and preserve the host FP environment. No carry mask is produced.
int rj_arith_v_add_nc_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_sub_nc_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_subrev_nc_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b);

int rj_arith_v_add_nc_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_sub_nc_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_add3_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

// Wave64 integer MAD. EXEC has 64 bits; 16/24-bit factors produce one VGPR
// using a one-VGPR addend. The 64-bit-result forms take one VGPR per factor and
// two each for addend/destination, low word first. Their required scalar carry
// output has 64 bits and is stored last, including zero for zero EXEC. Signed
// carry is bit 64 of the extended sum, before saturation. Modifiers and whole-
// VGPR aliases match Wave32; the 64-bit-result forms reject DPP. Both semantics
// are exact and preserve host FP state.
int rj_arith_v_mad_u32_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mad_i32_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mad_u32_u24_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mad_i32_i24_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mad_co_u64_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mad_co_i64_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mad_u64_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mad_i64_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

// Wave64 ordinary and packed 16-bit integer operations. EXEC has 64 bits;
// each operand uses one 64-lane VGPR. Ordinary forms preserve the unselected
// destination half and support DPP on A. Packed forms compute both halves and
// reject DPP. Half selectors, saturation and whole-VGPR aliases match Wave32.
// Both semantics are exact and independent of host FP state.
int rj_arith_v_pk_add_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_sub_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_add_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_sub_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_min_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_max_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_min_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_max_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_pk_mul_lo_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b);

int rj_arith_v_pk_lshlrev_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_pk_lshrrev_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_pk_ashrrev_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_add_nc_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_sub_nc_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_add_nc_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_sub_nc_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_min_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_max_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_min_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_max_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_mul_lo_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_lshlrev_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_lshrrev_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_ashrrev_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_pk_mad_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_pk_mad_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_mad_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int rj_arith_v_mad_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int rj_arith_v_min3_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_min3_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_max3_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_max3_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_med3_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_med3_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

// Wave64 integer min/max/median and ternary operations. EXEC has 64 bits; all
// operand VGPRs contain 64 lanes. Ordering, wrapping, byte rounding, DPP on A
// and whole-VGPR aliases match Wave32. Both semantics are exact. No arithmetic
// modifiers are accepted, and inactive destination lanes remain unchanged.
int rj_arith_v_min_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_max_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_min3_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_max3_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_minmax_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c);

int rj_arith_v_maxmin_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c);

int rj_arith_v_med3_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_min_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_max_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_min3_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_max3_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_minmax_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c);

int rj_arith_v_maxmin_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c);

int rj_arith_v_med3_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_lshl_add_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_add_lshl_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_lshl_or_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_and_or_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c);

int rj_arith_v_or3_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int rj_arith_v_xor3_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_xad_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int rj_arith_v_lerp_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

// Wave64 non-carry integer addition/subtraction and multiplication. EXEC has
// 64 bits; each operand uses one VGPR. Saturation, signed extension, truncation,
// modifiers, DPP restrictions and whole-VGPR aliases match Wave32. Both
// semantics are exact. No carry mask or floating exception state is produced.
int rj_arith_v_add_nc_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_sub_nc_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_subrev_nc_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b);

int rj_arith_v_add_nc_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_sub_nc_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_add3_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

int rj_arith_v_mul_lo_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_mul_hi_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_mul_hi_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_mul_i32_i24_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_mul_hi_i32_i24_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

int rj_arith_v_mul_u32_u24_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_mul_hi_u32_u24_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b);

// Wave64 shifts and bitfields. EXEC has 64 bits; operand VGPR counts, count
// wrapping, modifiers and whole-VGPR aliases match Wave32. Shifts of 64-bit
// elements still take one count VGPR and two value/destination VGPRs. Only
// 32-bit shifts support DPP; all bitfield forms support DPP on A. Both
// semantics are exact, with inactive lanes preserved and no host FP changes.
int rj_arith_v_lshlrev_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_lshrrev_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_ashrrev_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_lshlrev_b64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_lshrrev_b64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_ashrrev_i64_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b);

int rj_arith_v_bfe_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int rj_arith_v_bfe_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int rj_arith_v_bfi_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int rj_arith_v_bfm_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_bfrev_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_alignbit_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_alignbyte_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);

int rj_arith_v_perm_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

// Wave64 carry/borrow operations. EXEC, input_carry and the required carry
// output have 64 bits. Inactive/DPP-filtered carry bits are zero. The scalar
// output is stored after all VGPR reads/writes; zero EXEC permits null VGPR
// pointers but still writes carry=0. Modifiers and whole-VGPR aliases match
// Wave32, including DPP on B for SUBREV. Both semantics are exact.
int rj_arith_v_add_co_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_sub_co_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                              uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                              const uint32_t *const *b);

int rj_arith_v_subrev_co_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                                 const uint32_t *const *b);

int rj_arith_v_add_co_ci_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                                 const uint32_t *const *b, uint64_t input_carry);

int rj_arith_v_sub_co_ci_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                                 const uint32_t *const *b, uint64_t input_carry);

int rj_arith_v_subrev_co_ci_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, uint64_t *carry, const uint32_t *const *a,
                                    const uint32_t *const *b, uint64_t input_carry);

// Wave64 Boolean operations and conditional selects. EXEC and the CNDMASK
// condition have one bit per lane. Modifiers, DPP and whole-VGPR aliasing match
// Wave32; inactive lanes and unselected destination halves are preserved.
// Both semantics produce exact bitwise results. No FP state is accessed.
int rj_arith_v_and_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_or_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_xor_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_not_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_and_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_or_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_xor_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_not_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_xnor_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

// Bit counts: every operand holds one VGPR. CLZ/CTZ return 0xffffffff for zero;
// CLS counts leading sign bits (including the sign bit) and returns 0xffffffff
// if all bits match. BCNT returns popcount(A) + B, wrapping modulo 2^32.
// MBCNT_LO counts A bits below min(lane, 32); MBCNT_HI counts A bits below
// max(lane - 32, 0). Both add B with wrapping. The lane index is physical and
// independent of EXEC. Wave64 forms use 64 lane words per VGPR.
// Both wave sizes support DPP8/DPP16 on A. CLAMP is accepted and ignored;
// other low flags are invalid.
// DPP changes the source value, not the physical destination lane used for counting.
// Both semantics are exact; all host FP state is preserved.
int rj_arith_v_clz_i32_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_clz_i32_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_ctz_i32_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_ctz_i32_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cls_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_cls_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_bcnt_u32_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

int rj_arith_v_bcnt_u32_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b);

int rj_arith_v_mbcnt_lo_u32_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

int rj_arith_v_mbcnt_hi_u32_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

int rj_arith_v_mbcnt_lo_u32_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

int rj_arith_v_mbcnt_hi_u32_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b);

// Wave32 Boolean operations: each operand holds one VGPR. All forms support
// DPP8/DPP16 on A. CLAMP is accepted and ignored. B16 forms select A/B/D
// halves with HIGH_A/B/D after DPP and leave the other D half unchanged. Other
// modifiers are invalid; NOT has no B.
// Both semantics are bit-exact. All host FP state is preserved.
int rj_arith_v_and_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_or_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_xor_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_not_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_and_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_or_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                          uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_xor_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_not_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a);

// Wave32 bit operations: every operand holds one VGPR. Supports DPP8/DPP16 on A;
// CLAMP is accepted and ignored; other low flags are invalid. Both semantics are exact.
// BFE offsets and widths, and BFM widths and offsets, use their low five bits;
// a zero width produces zero.
// Signed BFE sign-extends A before extraction and sign-extends the extracted field.
// BFI selects B where A has a set bit, C otherwise. BFREV reverses all 32 bits.
int rj_arith_v_bfe_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int rj_arith_v_bfe_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int rj_arith_v_bfi_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

int rj_arith_v_bfm_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_bfrev_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                             uint32_t *const *d, const uint32_t *const *a);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
