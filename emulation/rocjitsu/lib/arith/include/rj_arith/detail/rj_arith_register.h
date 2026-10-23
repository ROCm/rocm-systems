// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_REGISTER_H_
#define RJ_ARITH_REGISTER_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bit-preserving Wave32/Wave64 moves, shared by RDNA3 and RDNA4. One VGPR per operand.
// Supports loose and exact semantics, EXEC masking, and DPP8/DPP16 source routing.
// ABS_A clears and NEG_A toggles bit 31, in that order, after DPP routing.
// Other low instruction flags, including OMOD and CLAMP, are unsupported.
// No FP interpretation or exceptions; inactive destinations remain unchanged.
int rj_arith_v_mov_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_mov_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a);

// Wave32/Wave64 raw half moves. HIGH_A/D select source/destination halves; the other
// destination half is preserved. ABS_A clears and NEG_A toggles the selected
// sign bit (in that order), with no FP conversion or NaN canonicalization.
// OMOD and CLAMP are accepted but ignored, matching hardware. DPP8/DPP16 route
// A before half/sign selection. Both semantics are exact; whole VGPRs may alias.
// Inactive destinations are preserved; zero EXEC permits null pointers.
int rj_arith_v_mov_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a);

int rj_arith_v_mov_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a);

// Exchange selected halves in active lanes. HIGH_A/D are the only valid flags.
// Both operands are read/write; unselected halves and inactive lanes survive.
// Supports exchanging two halves of the same VGPR. Both semantics are exact.
// Zero EXEC permits null pointers. Shared RDNA3/RDNA4 Wave32/Wave64 operations.
int rj_arith_v_swap_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, uint32_t *const *a);

int rj_arith_v_swap_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, uint32_t *const *a);

// Exchange two read/write VGPRs in active lanes. Both operands use their
// original values; whole-register aliasing is allowed. Inactive lanes are
// unchanged. No instruction flags are supported. Both semantics are exact.
// Zero EXEC permits null pointers. Shared by RDNA3 and RDNA4.
int rj_arith_v_swap_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, uint32_t *const *a);

int rj_arith_v_swap_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                            uint32_t *const *d, uint32_t *const *a);

// Wave32 PERMLANE64 is a no-op: no EXEC input and no operand access; null
// pointers are allowed. Wave64 copies A[lane ^ 32] into active D lanes, reading
// inactive source lanes too. D may alias A; inactive destinations are unchanged.
// No instruction flags are supported. Both semantics are exact. Shared by
// RDNA3 and RDNA4; neither operation interprets bits as floating-point values.
int rj_arith_v_permlane64_b32_w32(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                  const uint32_t *const *a);

int rj_arith_v_permlane64_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a);

// Register-only DS permutations; no LDS storage or global-state updates.
// ADDR and DATA each use one VGPR. Add the 16-bit byte offset to ADDR, then
// select address bits [6:2] within each 32-lane group. Wave32 is shared;
// RDNA3 Wave64 operates independently in each half. RDNA4 Wave64 instead uses
// bits [7:2] to route across all 64 lanes. BPERMUTE gathers DATA from the selected
// source; PERMUTE scatters DATA to the selected destination. EXEC applies to
// both reads and writes: inactive sources contribute zero, unfilled active
// destinations receive zero, and inactive destinations are preserved.
// For PERMUTE collisions rj-arith chooses the highest active source lane, following
// the ISA pseudocode; hardware prose does not guarantee a particular winner.
// All inputs are read before output stores; whole-register aliases are allowed.
// Both semantics are exact within this collision rule. No instruction flags
// are supported; errors preserve D. Zero EXEC permits null VGPR pointers.
int rj_arith_ds_permute_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *addr,
                                const uint32_t *const *data, uint16_t offset);

int rj_arith_ds_permute_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *addr,
                                const uint32_t *const *data, uint16_t offset);

int rj_arith_ds_permute_b32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *addr, const uint32_t *const *data,
                                      uint16_t offset);

int rj_arith_ds_bpermute_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *addr,
                                 const uint32_t *const *data, uint16_t offset);

int rj_arith_ds_bpermute_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *addr,
                                 const uint32_t *const *data, uint16_t offset);

int rj_arith_ds_bpermute_b32_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *addr, const uint32_t *const *data,
                                       uint16_t offset);

// RDNA4 FI backward permutation: same byte addressing as BPERMUTE, but reads
// inactive source lanes too. EXEC only controls destination writes. Wave64
// routes across all 64 lanes. The other DS permutation contracts apply.
int rj_arith_ds_bpermute_fi_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *addr,
                                    const uint32_t *const *data, uint16_t offset);

int rj_arith_ds_bpermute_fi_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *addr,
                                    const uint32_t *const *data, uint16_t offset);

// DS_SWIZZLE routes raw bits within each 32-lane row without LDS storage.
// offset selects bit-mask (<0x8000), quad (<0xc000), masked rotate (<0xe000),
// or FFT routing. Rotations wrap within the row. Shared RDNA3/RDNA4 ISA
// pseudocode semantics; rotate/FFT empirical validation is RDNA4-only.
// Inactive sources supply zero; inactive destinations are preserved. All source
// values precede output writes, permitting aliases. Both semantics are exact;
// no instruction flags are supported. Zero EXEC permits null VGPR pointers.
int rj_arith_ds_swizzle_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a, uint16_t offset);

int rj_arith_ds_swizzle_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a, uint16_t offset);

// PERMLANE flags are a bit field. FI permits reading inactive source lanes.
// Otherwise BOUND_CTRL selects zero for an inactive source; without it the
// destination is preserved. Neither flag enables inactive destination lanes.
constexpr uint32_t RJ_ARITH_PERMLANE_FI = 1U << 0;
constexpr uint32_t RJ_ARITH_PERMLANE_BOUND_CTRL = 1U << 1;

// Wave32 lane permutations support both semantics and preserve all host FP state.
// Guest FP flags have no effect. Each source index is its low four bits; the
// ordinary forms select within each group of 16 lanes, and X forms select from
// the other group. Whole-register aliasing, including d == a or b, is supported.

// Indices are successive nibbles of lo (lanes 0..7) and hi (8..15), repeated.
int rj_arith_v_permlane16_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t lo,
                                  uint32_t hi);

// Indices are successive nibbles of lo (lanes 0..7) and hi (8..15), repeated.
int rj_arith_v_permlanex16_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t lo,
                                   uint32_t hi);

// Indices come from each destination lane of b.
int rj_arith_v_permlane16_var_b32_w32(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b);

// Indices come from each destination lane of b.
int rj_arith_v_permlanex16_var_b32_w32(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);

// Wave64 variants use the same selectors and inactive-source rules. X forms
// exchange rows 0/1 and 2/3 independently; no source crosses a 32-lane half.
int rj_arith_v_permlane16_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t lo,
                                  uint32_t hi);

int rj_arith_v_permlane16_var_b32_w64(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b);

int rj_arith_v_permlanex16_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a, uint32_t lo,
                                   uint32_t hi);

int rj_arith_v_permlanex16_var_b32_w64(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b);

// Scalar comparisons execute once per wave and ignore EXEC, including zero.
// scc is required and receives 0 or 1. Inputs are passed by value and may come
// from output storage, including a word within uint64_t storage. Both loose and
// empirical exact semantics are supported. instruction_flags must be zero.
// Preserves all host FP state independently of rounding; errors leave scc unchanged.
// FP16 reads low halves. Floating comparisons equate signed zeros and apply guest
// input flushing. Output flushing and FP16_OVFL have no effect; integer/bit tests
// also ignore input flushing. Ordered predicates are false for either NaN;
// negated predicates are their logical complements, including for NaNs.
// In exact mode with non-NULL excp_flag_user, signaling NaNs accumulate INVALID.
// Finite operand pairs accumulate INPUT_DENORM if either input is subnormal and
// input flushing is disabled. Loose mode leaves excp_flag_user unchanged.

// Wave32/Wave64 relative VGPR moves shared by RDNA3/RDNA4. d/a are full register-file
// views starting at VGPR0, bounded by d_count/a_count. d_base/a_base are encoded
// register indices. RELS adds M0 to source; RELD to destination; RELSD to both.
// RELSD_2 uses M0[9:0] for source and M0[25:16] for destination. Per ISA section
// 3.3.2.2, offsets above 255 or resolved indices outside the allocation are out
// of range: sources redirect to VGPR0, and destination writes are discarded.
// A source allocation of zero returns zero. Zero counts permit null arrays.
// No indexing wraps; addition uses widened arithmetic. Whole VGPRs may alias.
// Both semantics are exact; DPP routes the selected source with MOV_B32 flags.
// Invalid flags leave all outputs unchanged.
int rj_arith_v_movrels_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                               uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

int rj_arith_v_movrels_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                               uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

int rj_arith_v_movreld_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                               uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

int rj_arith_v_movreld_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                               uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

int rj_arith_v_movrelsd_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                                uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

int rj_arith_v_movrelsd_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                                uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

int rj_arith_v_movrelsd_2_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                                  uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

int rj_arith_v_movrelsd_2_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t d_count,
                                  uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

// Relative VGPR swaps shared by RDNA3/RDNA4. d/a are mutable full register-file
// views starting at VGPR0, bounded by d_count/a_count. M0[9:0] offsets a_base;
// M0[25:16] offsets d_base. Other M0 bits are ignored. If either offset exceeds
// 255 or either resolved index is outside its allocation, neither operand is
// written. Zero counts permit NULL arrays. Index arithmetic does not wrap.
// Active lanes exchange raw words; inactive lanes are preserved. Whole VGPRs
// may alias. Both semantics are exact; instruction_flags must be zero.
// Invalid flags leave all outputs unchanged, including when EXEC is zero.
int rj_arith_v_swaprel_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint32_t *const *a, uint32_t d_count,
                               uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

int rj_arith_v_swaprel_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, uint32_t *const *a, uint32_t d_count,
                               uint32_t a_count, uint32_t d_base, uint32_t a_base, uint32_t m0);

// Scalar lane transfers shared by RDNA3/RDNA4; one VGPR input/output.
// READFIRSTLANE reads the first active lane, or lane 0 for zero EXEC.
// READLANE/WRITELANE ignore EXEC; lane indices wrap modulo 32 or 64.
// WRITELANE preserves every other lane. All copy raw bits, support both
// semantics, require zero instruction_flags, and preserve host FP state.
// Scalar destinations may overlap source lane storage; values are read first.
int rj_arith_v_readfirstlane_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a);

int rj_arith_v_readlane_b32_w32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                const uint32_t *const *a, uint32_t lane);

int rj_arith_v_writelane_b32_w32(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                 uint32_t a, uint32_t lane);

int rj_arith_v_readfirstlane_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                     uint32_t *d, const uint32_t *const *a);

int rj_arith_v_readlane_b32_w64(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                const uint32_t *const *a, uint32_t lane);

int rj_arith_v_writelane_b32_w64(uint64_t flags, uint64_t instruction_flags, uint32_t *const *d,
                                 uint32_t a, uint32_t lane);

// Select B where the corresponding condition bit is set, A otherwise. Each
// operand uses one VGPR. ABS_A/B clear source sign bits, then NEG_A/B toggle
// them; all other payload bits, including signaling NaNs, are preserved.
// The B16 form also supports HIGH_A/B/D: selected source halves are written to
// the selected destination half, preserving the other half. CLAMP is accepted
// and ignored; OMOD is unsupported.
// DPP8/DPP16 permutes A before source modifiers and half selection; B and
// the condition mask remain in their original lanes.
// Both semantics are bit-exact; all host FP state is preserved.
int rj_arith_v_cndmask_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t condition);
int rj_arith_v_cndmask_b16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint32_t condition);

int rj_arith_v_cndmask_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint64_t condition);

int rj_arith_v_cndmask_b16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, uint64_t condition);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
