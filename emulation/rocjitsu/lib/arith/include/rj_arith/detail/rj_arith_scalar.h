// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_SCALAR_H_
#define RJ_ARITH_SCALAR_H_

#include "rj_arith_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Scalar packing/selection ignores EXEC, including zero EXEC; output pointers
// are required. Inputs are passed by value and may come from destination storage.
// Both loose and empirical exact semantics are supported; instruction_flags must
// be zero. Guest FP flags have no effect and all host FP state is preserved.
// Errors leave outputs unchanged. SCC outputs, where present, are written after
// d and win on overlap, including within a 64-bit destination. Other operations
// preserve SCC. Conditional selection reads only input_scc bit 0.

// Pack low A and low B into low/high destination halves.
int rj_arith_s_pack_ll_b32_b16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b);

// Pack low A and high B into low/high destination halves.
int rj_arith_s_pack_lh_b32_b16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b);

// Pack high A and low B into low/high destination halves.
int rj_arith_s_pack_hl_b32_b16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b);

// Pack high A and high B into low/high destination halves.
int rj_arith_s_pack_hh_b32_b16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t b);

// Replicate each input bit i into output bits 2*i and 2*i+1.
int rj_arith_s_bitreplicate_b64_b32(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                    uint32_t a);

// Select a when input_scc bit 0 is 1, otherwise b.
int rj_arith_s_cselect_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t input_scc);

// Select a when input_scc bit 0 is 1, otherwise b.
int rj_arith_s_cselect_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                           uint64_t b, uint32_t input_scc);

// Set output bit i when input nibble i is nonzero; upper 24 bits are zero. SCC is result != 0.
int rj_arith_s_quadmask_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                            uint32_t *scc);

// Set output bit i when input nibble i is nonzero; upper 48 bits are zero. SCC is result != 0.
int rj_arith_s_quadmask_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                            uint32_t *scc);

// Replace each nonzero input nibble with 0xf. SCC is result != 0.
int rj_arith_s_wqm_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t *scc);

// Replace each nonzero input nibble with 0xf. SCC is result != 0.
int rj_arith_s_wqm_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint32_t *scc);

// S_GETREG reads a field from caller-supplied, wave-visible native register
// values. hwreg encodes ID[5:0], offset[10:6], width-minus-one[15:11]. Required
// output D is right-justified and zero-extended. Both semantics use exact integer
// extraction; fields extending beyond bit 31 are clipped. No EXEC or instruction
// flags. Unknown/reserved IDs (including RDNA3 write-only FLUSH_IB) return
// RJ_ARITH_ERROR_INVALID_FLAGS without touching D. Inputs are snapshots by value:
// clocks, PC, allocation/identity and snapshot visibility are supplied by the
// caller, not synthesized from host state. In particular, supply zero snapshot
// values for a non-sampled wave and zero scratch for a wave without allocation.
// D may alias storage from which a by-value input was loaded.
// RDNA3 has no specified snapshot read-update modeled by this interface.
int rj_arith_s_getreg_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t hwreg,
                          uint32_t mode, uint32_t status, uint32_t trapsts, uint32_t gpr_alloc,
                          uint32_t lds_alloc, uint32_t ib_sts, uint64_t pc, uint32_t ib_dbg1,
                          uint32_t sh_mem_bases, uint64_t shader_tba, uint64_t perf_snapshot_pc,
                          uint64_t flat_scratch, uint32_t hw_id1, uint32_t hw_id2,
                          uint32_t pops_packer, uint32_t sched_mode, uint32_t perf_snapshot_data,
                          uint32_t ib_sts2, uint64_t shader_cycles);

// RDNA4 has a different register-ID map. Reading ID 12 (snapshot PC_HI) unlocks
// the snapshot registers. Its updates are not yet implemented: non-loose
// semantics with any snapshot output non-NULL return UNSUPPORTED_REGISTER_STATE
// before other validation or D access. NULL opts out. Loose mode leaves snapshot
// state unchanged even with non-NULL outputs. Other IDs do not update snapshots.
// Snapshot outputs must not overlap D. Caller-provided snapshot *_in values
// determine the required read result independently of optional output pointers.
int rj_arith_s_getreg_b32_gfx12(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                uint16_t hwreg, uint32_t mode, uint32_t status, uint32_t state_priv,
                                uint32_t gpr_alloc, uint32_t lds_alloc,
                                uint32_t perf_snapshot_data_in, uint32_t perf_snapshot_pc_lo_in,
                                uint32_t perf_snapshot_pc_hi_in, uint32_t perf_snapshot_data1_in,
                                uint32_t perf_snapshot_data2_in, uint32_t excp_flag_priv,
                                uint32_t excp_flag_user, uint32_t trap_ctrl, uint64_t scratch_base,
                                uint32_t hw_id1, uint32_t hw_id2, uint32_t ib_sts2,
                                uint64_t shader_cycles, uint32_t *perf_snapshot_data,
                                uint32_t *perf_snapshot_pc_lo, uint32_t *perf_snapshot_pc_hi,
                                uint32_t *perf_snapshot_data1, uint32_t *perf_snapshot_data2);

// RDNA3 S_SETREG: the encoded selector and scalar/literal source precede the
// implicit STATUS input. status_in supplies PRIV (bit 5), independently of the
// nullable STATUS output. Other outputs are also nullable native registers.
// Both semantics update documented MODE/STATUS/TRAPSTS fields, preserving
// unselected/reserved bits. Exception fields are replaced, not accumulated.
// STATUS writes require privilege; MODE and TRAPSTS do not. MODE bit 20 is
// reserved. Read-only writes are no-ops; unknown IDs return INVALID_FLAGS.
// Exact reporting of FLUSH_IB is unavailable. Non-loose semantics with flush_ib
// non-NULL and ID 14 return UNSUPPORTED_REGISTER_STATE. Privileged scratch writes
// likewise return that error if flat_scratch or status is non-NULL. These errors
// precede other validation. Loose mode skips these unavailable updates; null
// opts out. Outputs must not overlap. No EXEC, instruction flags or host FP-state
// changes; scheduling, trap delivery and instruction-buffer management are external.
int rj_arith_s_setreg_b32(uint64_t flags, uint64_t instruction_flags, uint16_t hwreg, uint32_t a,
                          uint32_t status_in, uint32_t *mode, uint32_t *status, uint32_t *trapsts,
                          uint32_t *flush_ib, uint64_t *flat_scratch);
int rj_arith_s_setreg_imm32_b32(uint64_t flags, uint64_t instruction_flags, uint16_t hwreg,
                                uint32_t immediate, uint32_t status_in, uint32_t *mode,
                                uint32_t *status, uint32_t *trapsts, uint32_t *flush_ib,
                                uint64_t *flat_scratch);

// RDNA4 S_SETREG: hwreg encodes ID[5:0], offset[10:6], width-minus-one[15:11].
// Supports loose and exact semantics. instruction_flags must be zero. STATUS.PRIV
// (bit 5) controls privileged writes. Each nullable output holds the old native
// register and receives its masked replacement; only the selected register is
// accessed. Distinct non-null outputs must not overlap. SCRATCH_BASE is one
// 64-bit register, selected by IDs 20/21. Privileged scratch writes are currently
// unsupported: non-loose semantics with scratch_base or state_priv non-NULL
// return RJ_ARITH_ERROR_UNSUPPORTED_REGISTER_STATE before other validation; loose mode
// skips these writes. Unprivileged scratch writes remain no-ops.
// Read-only IDs and privilege-denied fields are no-ops. Unknown/reserved IDs
// return RJ_ARITH_ERROR_INVALID_FLAGS. Unselected and reserved bits are preserved.
// Exception-register writes replace bits (and can clear them), rather than OR.
// Guest MODE writes never change host FP controls. These functions update raw
// register state; scheduling, trap delivery and scratch memory remain external.
int rj_arith_s_setreg_b32_gfx12(uint64_t flags, uint64_t instruction_flags, uint16_t hwreg,
                                uint32_t a, uint32_t status, uint32_t *mode, uint32_t *state_priv,
                                uint32_t *excp_flag_priv, uint32_t *excp_flag_user,
                                uint32_t *trap_ctrl, uint64_t *scratch_base);
int rj_arith_s_setreg_imm32_b32_gfx12(uint64_t flags, uint64_t instruction_flags, uint16_t hwreg,
                                      uint32_t immediate, uint32_t status, uint32_t *mode,
                                      uint32_t *state_priv, uint32_t *excp_flag_priv,
                                      uint32_t *excp_flag_user, uint32_t *trap_ctrl,
                                      uint64_t *scratch_base);

// Set guest MODE rounding bits [3:0] or denormal bits [7:4] to immediate[3:0].
// Higher immediate bits are ignored and every other MODE bit is preserved.
// mode is an optional read/write architectural register pointer; NULL opts out.
// Both loose and exact semantics perform this inexpensive update when non-NULL.
// No EXEC input, instruction flags, or host FP-environment changes. Errors leave
// MODE unchanged. Shared by RDNA3 and RDNA4.
int rj_arith_s_round_mode(uint64_t flags, uint64_t instruction_flags, uint16_t immediate,
                          uint32_t *mode);

int rj_arith_s_denorm_mode(uint64_t flags, uint64_t instruction_flags, uint16_t immediate,
                           uint32_t *mode);

// Store the next instruction address, pc + 4, in the required scalar pair D.
// pc is the byte address of this instruction; addition wraps modulo 2^64.
// No EXEC input or instruction flags. Both semantics are exact. Does not modify
// the guest PC or host FP state; errors preserve D. Shared RDNA3/RDNA4 operation.
int rj_arith_s_getpc_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t pc);

// EXEC Boolean updates shared by RDNA3/RDNA4. SAVEEXEC writes old exec_mask
// to d; WREXEC writes the new EXEC value to d. EXEC is replaced, not accumulated.
// NOT0 complements a, NOT1 complements exec_mask. SCC is new EXEC != 0.
// These execute even when exec_mask is zero. Both semantics are exact;
// instruction_flags must be zero. d and scc are required ordinary results;
// exec may be null to omit the implicit register update. Writes occur in order
// d, exec, scc; outputs may alias and later writes win. Errors preserve outputs.
int rj_arith_s_and_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                                uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int rj_arith_s_or_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                               uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int rj_arith_s_xor_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                                uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int rj_arith_s_nand_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int rj_arith_s_nor_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                                uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int rj_arith_s_xnor_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                 uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int rj_arith_s_and_not0_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                     uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int rj_arith_s_or_not0_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                    uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int rj_arith_s_and_not1_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                     uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int rj_arith_s_or_not1_saveexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                    uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int rj_arith_s_and_not0_wrexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int rj_arith_s_and_not1_wrexec_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                   uint32_t a, uint32_t exec_mask, uint32_t *exec, uint32_t *scc);

int rj_arith_s_and_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                                uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int rj_arith_s_or_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                               uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int rj_arith_s_xor_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                                uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int rj_arith_s_nand_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                 uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int rj_arith_s_nor_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                                uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int rj_arith_s_xnor_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                 uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int rj_arith_s_and_not0_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                     uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int rj_arith_s_or_not0_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                    uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int rj_arith_s_and_not1_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                     uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int rj_arith_s_or_not1_saveexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                    uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int rj_arith_s_and_not0_wrexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                   uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

int rj_arith_s_and_not1_wrexec_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                                   uint64_t a, uint64_t exec_mask, uint64_t *exec, uint32_t *scc);

// Scalar relative moves shared by RDNA3/RDNA4. Indexed arrays start at SGPR0;
// indices 0..107 cover SGPRs/VCC and 108..123 cover TTMP0..15. Counts bound
// accessible storage. For destinations, d_count must also exclude inaccessible
// TTMPs (use at most 108 outside the trap handler). Zero counts permit NULL arrays.
// Base indices and M0 offsets count 32-bit registers, including in B64 forms.
// Indexing cannot cross regions; out-of-range reads redirect to SGPR0 (missing
// words read as zero), and out-of-range writes are discarded. No index wraps.
// B64 base indices and M0 must be even. A B64 value's low word occupies the lower
// register index. RELSD_2 uses M0[9:0] for source and M0[25:16] for destination.
// Non-indexed inputs are values; non-indexed outputs are ordinary scalar pointers.
// Storage may overlap when the C/C++ types permit it; inputs are read before writes.
// Both semantics are exact, instruction_flags must be zero, and EXEC is ignored.
// Errors preserve all outputs. No implicit global register is modified.
int rj_arith_s_movrels_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                           const uint32_t *a, uint32_t a_count, uint32_t a_base, uint32_t m0);

int rj_arith_s_movrels_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d,
                           const uint32_t *a, uint32_t a_count, uint32_t a_base, uint32_t m0);

int rj_arith_s_movreld_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t d_count, uint32_t d_base, uint32_t m0);

int rj_arith_s_movreld_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a,
                           uint32_t d_count, uint32_t d_base, uint32_t m0);

int rj_arith_s_movrelsd_2_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                              const uint32_t *a, uint32_t d_count, uint32_t a_count,
                              uint32_t d_base, uint32_t a_base, uint32_t m0);

// Scalar moves shared by RDNA3/RDNA4. Copy raw bits; MOVK/CMOVK sign-extend
// their 16-bit immediate. Conditional forms write only when input_scc bit 0
// is set, otherwise d is unchanged. SCC and EXEC are not modified; EXEC is
// ignored. Both semantics are exact and instruction_flags must be zero.
// Inputs passed by value may come from d. Errors preserve d and host FP state.
int rj_arith_s_mov_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);
int rj_arith_s_mov_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a);
int rj_arith_s_cmov_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t input_scc);
int rj_arith_s_cmov_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint32_t input_scc);
int rj_arith_s_movk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                        uint16_t immediate);
int rj_arith_s_cmovk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                         uint16_t immediate, uint32_t input_scc);

// RDNA3 immediate comparisons set SCC to 0 or 1. I32 forms sign-extend the
// 16-bit immediate; U32 forms zero-extend it. Both semantics are exact, EXEC
// is ignored, instruction_flags must be zero, and errors preserve SCC.
int rj_arith_s_cmpk_eq_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate);

int rj_arith_s_cmpk_lg_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate);

int rj_arith_s_cmpk_gt_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate);

int rj_arith_s_cmpk_ge_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate);

int rj_arith_s_cmpk_lt_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate);

int rj_arith_s_cmpk_le_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate);

int rj_arith_s_cmpk_eq_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate);

int rj_arith_s_cmpk_lg_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate);

int rj_arith_s_cmpk_gt_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate);

int rj_arith_s_cmpk_ge_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate);

int rj_arith_s_cmpk_lt_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate);

int rj_arith_s_cmpk_le_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint16_t immediate);

// SCC is a == b (signed).
int rj_arith_s_cmp_eq_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b);

// SCC is a != b (signed).
int rj_arith_s_cmp_lg_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b);

// SCC is a > b (signed).
int rj_arith_s_cmp_gt_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b);

// SCC is a >= b (signed).
int rj_arith_s_cmp_ge_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b);

// SCC is a < b (signed).
int rj_arith_s_cmp_lt_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b);

// SCC is a <= b (signed).
int rj_arith_s_cmp_le_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b);

// SCC is a == b (unsigned).
int rj_arith_s_cmp_eq_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b);

// SCC is a != b (unsigned).
int rj_arith_s_cmp_lg_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b);

// SCC is a > b (unsigned).
int rj_arith_s_cmp_gt_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b);

// SCC is a >= b (unsigned).
int rj_arith_s_cmp_ge_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b);

// SCC is a < b (unsigned).
int rj_arith_s_cmp_lt_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b);

// SCC is a <= b (unsigned).
int rj_arith_s_cmp_le_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b);

// SCC is 1 when bit (b modulo 32) of a equals 0.
int rj_arith_s_bitcmp0_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b);

// SCC is 1 when bit (b modulo 32) of a equals 1.
int rj_arith_s_bitcmp1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b);

// SCC is 1 when bit (b modulo 64) of a equals 0.
int rj_arith_s_bitcmp0_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint64_t a,
                           uint32_t b);

// SCC is 1 when bit (b modulo 64) of a equals 1.
int rj_arith_s_bitcmp1_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint64_t a,
                           uint32_t b);

// SCC is a == b (unsigned).
int rj_arith_s_cmp_eq_u64(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint64_t a,
                          uint64_t b);

// SCC is a != b (unsigned).
int rj_arith_s_cmp_lg_u64(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint64_t a,
                          uint64_t b);

// SCC is a < b.
int rj_arith_s_cmp_lt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user);

// SCC is a < b.
int rj_arith_s_cmp_lt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user);

// SCC is a == b.
int rj_arith_s_cmp_eq_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user);

// SCC is a == b.
int rj_arith_s_cmp_eq_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user);

// SCC is a <= b.
int rj_arith_s_cmp_le_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user);

// SCC is a <= b.
int rj_arith_s_cmp_le_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user);

// SCC is a > b.
int rj_arith_s_cmp_gt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user);

// SCC is a > b.
int rj_arith_s_cmp_gt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user);

// SCC is both operands are numbers and a != b.
int rj_arith_s_cmp_lg_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user);

// SCC is both operands are numbers and a != b.
int rj_arith_s_cmp_lg_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user);

// SCC is a >= b.
int rj_arith_s_cmp_ge_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user);

// SCC is a >= b.
int rj_arith_s_cmp_ge_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                          uint32_t b, uint32_t *excp_flag_user);

// SCC is both operands are numbers.
int rj_arith_s_cmp_o_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                         uint32_t b, uint32_t *excp_flag_user);

// SCC is both operands are numbers.
int rj_arith_s_cmp_o_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                         uint32_t b, uint32_t *excp_flag_user);

// SCC is either operand is NaN.
int rj_arith_s_cmp_u_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                         uint32_t b, uint32_t *excp_flag_user);

// SCC is either operand is NaN.
int rj_arith_s_cmp_u_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                         uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a >= b).
int rj_arith_s_cmp_nge_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a >= b).
int rj_arith_s_cmp_nge_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// SCC is 1 if either operand is NaN or a == b.
int rj_arith_s_cmp_nlg_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// SCC is 1 if either operand is NaN or a == b.
int rj_arith_s_cmp_nlg_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a > b).
int rj_arith_s_cmp_ngt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a > b).
int rj_arith_s_cmp_ngt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a <= b).
int rj_arith_s_cmp_nle_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a <= b).
int rj_arith_s_cmp_nle_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a == b).
int rj_arith_s_cmp_neq_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a == b).
int rj_arith_s_cmp_neq_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a < b).
int rj_arith_s_cmp_nlt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// SCC is not (a < b).
int rj_arith_s_cmp_nlt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *scc, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// Scalar conversions execute once per wave and ignore EXEC, including zero;
// d is required and SCC is unchanged. Operands are raw register bits. Sources
// may originate from d storage. instruction_flags must be zero; loose semantics
// only. Errors leave d unchanged. Requires host nearest-even rounding and enabled
// denormals; preserves host rounding but may change exception flags.
// Guest input flushing applies to floating inputs. Output flushing applies to
// FP16 outputs, including tininess detection before subnormal rounding for RNE.
// FP16_OVFL saturates finite RNE narrowing overflow; it does not affect RTZ.
// All other guest FP-state flags accepted here have no effect on integer results
// or integer inputs. Floating conversions quiet signaling NaNs.

// Signed 32-bit integer to FP32, nearest-even.
int rj_arith_s_cvt_f32_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t *excp_flag_user);

// Unsigned 32-bit integer to FP32, nearest-even.
int rj_arith_s_cvt_f32_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t *excp_flag_user);

// FP32 to signed 32-bit integer, truncating and saturating; NaNs yield zero.
int rj_arith_s_cvt_i32_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t *excp_flag_user);

// FP32 to unsigned 32-bit integer, truncating and saturating; NaNs/negatives yield zero.
int rj_arith_s_cvt_u32_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t *excp_flag_user);

// FP32 to FP16, nearest-even; writes a zero upper half.
int rj_arith_s_cvt_f16_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t *excp_flag_user);

// Low FP16 half to FP32.
int rj_arith_s_cvt_f32_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t *excp_flag_user);

// High FP16 half to FP32.
int rj_arith_s_cvt_hi_f32_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                              uint32_t *excp_flag_user);

// Two FP32 values to low/high FP16 halves, truncating toward zero.
// Finite overflow saturates to the largest finite half; infinities remain infinite.
int rj_arith_s_cvt_pk_rtz_f16_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                                  uint32_t a, uint32_t b, uint32_t *excp_flag_user);

// Scalar rounding ignores EXEC, including zero EXEC; d is required. Inputs and
// outputs are raw IEEE bits. FP16 reads the low half and writes a zero upper
// half. Signed zero and NaN payload/sign bits survive; signaling NaNs are quieted.
// Both loose and empirical exact semantics are supported. instruction_flags must
// be zero. Guest input flushing applies; output flushing and FP16_OVFL have no
// effect. Preserves all host FP state and SCC, independently of host rounding.
// Inputs may originate from d storage. Errors leave d unchanged.
// Exact mode with non-NULL excp_flag_user accumulates INVALID for signaling NaNs
// and INPUT_DENORM for preserved subnormals. Discarding a fractional part does
// not raise INEXACT. Loose mode leaves excp_flag_user unchanged.

// Round to an integral value toward positive infinity.
int rj_arith_s_ceil_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t *excp_flag_user);

// Round to an integral value toward positive infinity.
int rj_arith_s_ceil_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t *excp_flag_user);

// Round to an integral value toward negative infinity.
int rj_arith_s_floor_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);

// Round to an integral value toward negative infinity.
int rj_arith_s_floor_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);

// Round to an integral value toward zero.
int rj_arith_s_trunc_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);

// Round to an integral value toward zero.
int rj_arith_s_trunc_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);

// Round to an integral value to nearest, with ties to even.
int rj_arith_s_rndne_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);

// Round to an integral value to nearest, with ties to even.
int rj_arith_s_rndne_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);

// Scalar fused multiply-add ignores EXEC, including zero EXEC; d is required.
// Operands and literals are raw IEEE bits. FP16 reads low halves and writes a
// zero upper half. Rounds once to the destination format. SCC is unchanged.
// instruction_flags must be zero. FP32 supports both semantics; FP16 remains
// loose-only. Loose NaN signs/payloads are unspecified. Exact FP32 follows the
// borrowed FMA NaN ordering. Mode 3 FP status reporting is unsupported.
// Guest input/output flushing and FP16_OVFL are supported, with
// tininess detected before destination subnormal rounding. Loose mode requires
// host nearest-even rounding and enabled denormals; exact FP32 uses integer
// arithmetic and honors RJ_ARITH_FP_ROUND_MASK. Loose FP32 may ignore that selector;
// FP16 rejects it. Errors leave d unchanged. Inputs may come from d storage.

// Fused a * b + old *d.
int rj_arith_s_fmac_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *excp_flag_user);

// Fused a * b + old *d.
int rj_arith_s_fmac_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *excp_flag_user);

// Fused a * b + literal.
int rj_arith_s_fmaak_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t b, uint32_t literal, uint32_t *excp_flag_user);

// Fused a * literal + c.
int rj_arith_s_fmamk_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t literal, uint32_t c, uint32_t *excp_flag_user);

// Scalar FP16/FP32 binary arithmetic executes once per wave and ignores EXEC,
// including zero EXEC; d is required. Inputs are raw IEEE bits. FP16 reads the
// low halves and writes a zero upper half. instruction_flags must be zero.
// Loose semantics only; NaN signs/payloads are unspecified. Requires host
// nearest-even rounding and enabled denormals. Preserves host rounding but may
// change exception flags. SCC is unchanged. Errors leave d unchanged.
// Guest input flushing applies to every operation. Guest output flushing applies
// to add/subtract/multiply; min/max ignore it. FP16_OVFL saturates finite FP16
// overflow. Multiplication flushes tiny results before destination subnormal
// rounding, so a value that would round up to the smallest normal may flush.

// Sum of a and b.
int rj_arith_s_add_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *excp_flag_user);

// Sum of a and b.
int rj_arith_s_add_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *excp_flag_user);

// Difference a - b.
int rj_arith_s_sub_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *excp_flag_user);

// Difference a - b.
int rj_arith_s_sub_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *excp_flag_user);

// Product of a and b.
int rj_arith_s_mul_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *excp_flag_user);

// Product of a and b.
int rj_arith_s_mul_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *excp_flag_user);

// Minimum; a single NaN selects the numeric operand; -0 sorts below +0.
int rj_arith_s_min_num_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// Minimum; a single NaN selects the numeric operand; -0 sorts below +0.
int rj_arith_s_min_num_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// Maximum; a single NaN selects the numeric operand; -0 sorts below +0.
int rj_arith_s_max_num_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// Maximum; a single NaN selects the numeric operand; -0 sorts below +0.
int rj_arith_s_max_num_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// Minimum; either NaN yields NaN; -0 sorts below +0.
int rj_arith_s_minimum_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// Minimum; either NaN yields NaN; -0 sorts below +0.
int rj_arith_s_minimum_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// Maximum; either NaN yields NaN; -0 sorts below +0.
int rj_arith_s_maximum_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// Maximum; either NaN yields NaN; -0 sorts below +0.
int rj_arith_s_maximum_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *excp_flag_user);

// Scalar bitfields and counts ignore EXEC, including zero EXEC; output pointers
// are required. Both loose and empirical exact semantics are supported, and
// instruction_flags must be zero. All host FP state is preserved. SCC outputs
// are written after d and win on overlap, including within a 64-bit destination.
// Errors leave all outputs unchanged. Instructions without SCC outputs preserve SCC.

// Extract unsigned field: offset=b[4:0], width=b[22:16].
// Width clips at bit 32; zero width yields zero. SCC is result != 0.
int rj_arith_s_bfe_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

// Extract signed field: offset=b[4:0], width=b[22:16].
// Width clips at bit 32; zero width yields zero. SCC is result != 0.
int rj_arith_s_bfe_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

// Extract unsigned field: offset=b[5:0], width=b[22:16].
// Width clips at bit 64; zero width yields zero. SCC is result != 0.
int rj_arith_s_bfe_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint32_t b, uint32_t *scc);

// Extract signed field: offset=b[5:0], width=b[22:16].
// Width clips at bit 64; zero width yields zero. SCC is result != 0.
int rj_arith_s_bfe_i64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint32_t b, uint32_t *scc);

// Low 32 bits of ((1 << (a modulo 32)) - 1) << (b modulo 32).
int rj_arith_s_bfm_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b);

// Low 64 bits of ((1 << (a modulo 64)) - 1) << (b modulo 64).
int rj_arith_s_bfm_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint32_t a,
                       uint32_t b);

// Count zero bits in 32-bit a; SCC is result != 0.
int rj_arith_s_bcnt0_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                             uint32_t *scc);

// Count zero bits in 64-bit a; SCC is result != 0.
int rj_arith_s_bcnt0_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a,
                             uint32_t *scc);

// Count one bits in 32-bit a; SCC is result != 0.
int rj_arith_s_bcnt1_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                             uint32_t *scc);

// Count one bits in 64-bit a; SCC is result != 0.
int rj_arith_s_bcnt1_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a,
                             uint32_t *scc);

// Count trailing zero bits in 32-bit a; return UINT32_MAX for zero.
int rj_arith_s_ctz_i32_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);

// Count trailing zero bits in 64-bit a; return UINT32_MAX for zero.
int rj_arith_s_ctz_i32_b64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a);

// Count leading zero bits in 32-bit a; return UINT32_MAX for zero.
int rj_arith_s_clz_i32_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);

// Count leading zero bits in 64-bit a; return UINT32_MAX for zero.
int rj_arith_s_clz_i32_u64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a);

// Count leading sign bits in 32-bit a, including the sign bit;
// return UINT32_MAX for zero or all-ones input.
int rj_arith_s_cls_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);

// Count leading sign bits in 64-bit a, including the sign bit;
// return UINT32_MAX for zero or all-ones input.
int rj_arith_s_cls_i32_i64(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint64_t a);

// Clear bit b modulo 32 in old *d.
int rj_arith_s_bitset0_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t b);

// Clear bit b modulo 64 in old *d.
int rj_arith_s_bitset0_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint32_t b);

// Set bit b modulo 32 in old *d.
int rj_arith_s_bitset1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t b);

// Set bit b modulo 64 in old *d.
int rj_arith_s_bitset1_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint32_t b);

// Scalar bitwise and shift instructions ignore EXEC, including zero EXEC.
// Output pointers are required; all host FP state is preserved. Both loose and
// empirical exact semantics are supported. instruction_flags must be zero.
// SCC, when present, is written after d and wins on overlapping storage, including
// a 32-bit word within a 64-bit destination. Errors leave all outputs unchanged.

// 32-bit and; SCC is result != 0.
int rj_arith_s_and_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

// 64-bit and; SCC is result != 0.
int rj_arith_s_and_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint64_t b, uint32_t *scc);

// 32-bit or; SCC is result != 0.
int rj_arith_s_or_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                      uint32_t b, uint32_t *scc);

// 64-bit or; SCC is result != 0.
int rj_arith_s_or_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                      uint64_t b, uint32_t *scc);

// 32-bit xor; SCC is result != 0.
int rj_arith_s_xor_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

// 64-bit xor; SCC is result != 0.
int rj_arith_s_xor_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint64_t b, uint32_t *scc);

// 32-bit nand; SCC is result != 0.
int rj_arith_s_nand_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc);

// 64-bit nand; SCC is result != 0.
int rj_arith_s_nand_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint64_t b, uint32_t *scc);

// 32-bit nor; SCC is result != 0.
int rj_arith_s_nor_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

// 64-bit nor; SCC is result != 0.
int rj_arith_s_nor_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint64_t b, uint32_t *scc);

// 32-bit xnor; SCC is result != 0.
int rj_arith_s_xnor_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc);

// 64-bit xnor; SCC is result != 0.
int rj_arith_s_xnor_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint64_t b, uint32_t *scc);

// 32-bit and_not1; SCC is result != 0.
int rj_arith_s_and_not1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                            uint32_t b, uint32_t *scc);

// 64-bit and_not1; SCC is result != 0.
int rj_arith_s_and_not1_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                            uint64_t b, uint32_t *scc);

// 32-bit or_not1; SCC is result != 0.
int rj_arith_s_or_not1_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *scc);

// 64-bit or_not1; SCC is result != 0.
int rj_arith_s_or_not1_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                           uint64_t b, uint32_t *scc);

// 32-bit not; SCC is result != 0.
int rj_arith_s_not_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t *scc);

// 64-bit not; SCC is result != 0.
int rj_arith_s_not_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint32_t *scc);

// 32-bit bit reversal; leaves SCC unchanged.
int rj_arith_s_brev_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);

// 64-bit bit reversal; leaves SCC unchanged.
int rj_arith_s_brev_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a);

// 32-bit left shift by b modulo 32; SCC is result != 0.
int rj_arith_s_lshl_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc);

// 64-bit left shift by b modulo 64; SCC is result != 0.
int rj_arith_s_lshl_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint32_t b, uint32_t *scc);

// 32-bit logical right shift by b modulo 32; SCC is result != 0.
int rj_arith_s_lshr_b32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc);

// 64-bit logical right shift by b modulo 64; SCC is result != 0.
int rj_arith_s_lshr_b64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint32_t b, uint32_t *scc);

// 32-bit arithmetic right shift by b modulo 32; SCC is result != 0.
int rj_arith_s_ashr_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t *scc);

// 64-bit arithmetic right shift by b modulo 64; SCC is result != 0.
int rj_arith_s_ashr_i64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                        uint32_t b, uint32_t *scc);

// Low 32 bits of (a << 1) + b; SCC indicates that the full sum exceeds UINT32_MAX.
int rj_arith_s_lshl1_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                             uint32_t b, uint32_t *scc);

// Low 32 bits of (a << 2) + b; SCC indicates that the full sum exceeds UINT32_MAX.
int rj_arith_s_lshl2_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                             uint32_t b, uint32_t *scc);

// Low 32 bits of (a << 3) + b; SCC indicates that the full sum exceeds UINT32_MAX.
int rj_arith_s_lshl3_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                             uint32_t b, uint32_t *scc);

// Low 32 bits of (a << 4) + b; SCC indicates that the full sum exceeds UINT32_MAX.
int rj_arith_s_lshl4_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                             uint32_t b, uint32_t *scc);

// Scalar-register integer arithmetic executes once per wave and ignores EXEC,
// including zero EXEC. Scalar output pointers are required. Both loose and
// empirical exact semantics are supported; instruction_flags must be zero.
// All host FP state is preserved. Results wrap to the destination width.
// SCC outputs contain 0 or 1 and are written after d; if they alias, SCC wins.
// Errors leave all outputs unchanged. Inputs passed by value may originate
// from destination storage. Instructions without SCC outputs leave SCC alone.

// Unsigned sum; SCC is carry-out.
int rj_arith_s_add_co_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b, uint32_t *scc);

// Unsigned difference; SCC is borrow-out.
int rj_arith_s_sub_co_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b, uint32_t *scc);

// Signed sum; SCC is signed overflow.
int rj_arith_s_add_co_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b, uint32_t *scc);

// Signed difference; SCC is signed overflow.
int rj_arith_s_sub_co_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b, uint32_t *scc);

// Unsigned sum with input_scc bit 0; SCC is carry-out.
int rj_arith_s_add_co_ci_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                             uint32_t b, uint32_t *scc, uint32_t input_scc);

// Unsigned difference minus input_scc bit 0; SCC is borrow-out.
int rj_arith_s_sub_co_ci_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                             uint32_t b, uint32_t *scc, uint32_t input_scc);

// RDNA3 scalar add/sub spellings. Unsigned SCC is carry/borrow; signed SCC
// is overflow. ADDC/SUBB consume input_scc bit 0. ADDK sign-extends immediate
// and adds to old *d. Both semantics are exact; flags and alias rules above apply.
int rj_arith_s_add_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

int rj_arith_s_sub_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

int rj_arith_s_add_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

int rj_arith_s_sub_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

int rj_arith_s_addc_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t input_scc, uint32_t *scc);

int rj_arith_s_subb_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                        uint32_t b, uint32_t input_scc, uint32_t *scc);

int rj_arith_s_addk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint16_t immediate,
                        uint32_t *scc);

// Sign-extend the low 8 bits to 32 bits; preserves SCC.
int rj_arith_s_sext_i32_i8(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);

// Sign-extend the low 16 bits to 32 bits; preserves SCC.
int rj_arith_s_sext_i32_i16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a);

// Absolute value modulo 2^32; SCC is result != 0.
int rj_arith_s_abs_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t *scc);

// Absolute value of the wrapped 32-bit difference; SCC is result != 0.
int rj_arith_s_absdiff_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                           uint32_t b, uint32_t *scc);

// Signed minimum; SCC is a < b.
int rj_arith_s_min_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

// Unsigned minimum; SCC is a < b.
int rj_arith_s_min_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

// Signed maximum; SCC is a > b.
int rj_arith_s_max_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

// Unsigned maximum; SCC is a > b.
int rj_arith_s_max_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b, uint32_t *scc);

// Low 32 bits of the product.
int rj_arith_s_mul_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                       uint32_t b);

// High 32 bits of the unsigned product.
int rj_arith_s_mul_hi_u32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b);

// High 32 bits of the signed product.
int rj_arith_s_mul_hi_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t b);

// Sum modulo 2^64.
int rj_arith_s_add_nc_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                          uint64_t b);

// Difference modulo 2^64.
int rj_arith_s_sub_nc_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                          uint64_t b);

// Product modulo 2^64.
int rj_arith_s_mul_u64(uint64_t flags, uint64_t instruction_flags, uint64_t *d, uint64_t a,
                       uint64_t b);

// Add sign-extended immediate to old *d; SCC is signed overflow.
int rj_arith_s_addk_co_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                           uint16_t immediate, uint32_t *scc);

// Multiply old *d by sign-extended immediate, modulo 2^32.
int rj_arith_s_mulk_i32(uint64_t flags, uint64_t instruction_flags, uint32_t *d,
                        uint16_t immediate);

// Pseudo-scalar transcendental math on raw SGPR values. Executes once regardless
// of EXEC, including zero. a is a scalar value; d must always be writable.
// FP16 reads a's low half, ignores its high half, and zeros d's upper half.
// Supports ABS_A, NEG_A, OMOD and CLAMP, with ABS before NEG. No half selectors.
// All FP32 operations and FP16 RCP/RSQ/SQRT support empirical-exact semantics.
// FP16 EXP/LOG remain loose-only.
// Mode 3 FP status reporting is unsupported. Loose calls leave state unchanged.
// FP32 always flushes input/output denormals. FP16 follows the guest input/output
// flush flags and FP16_OVFL. Rounding precedes OMOD; nonzero OMOD flushes tiny
// values before and after scaling. A zero created by negative underflow retains
// its sign; pre-existing zero becomes positive with OMOD. CLAMP maps NaN and
// negative results to zero. Loose semantics require host nearest-even rounding
// and enabled denormals; exact paths use integer arithmetic. Host rounding is
// preserved; ordinary arithmetic exception flags may change.
int rj_arith_v_s_exp_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);
int rj_arith_v_s_exp_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);
int rj_arith_v_s_log_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);
int rj_arith_v_s_log_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);
int rj_arith_v_s_rcp_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);
int rj_arith_v_s_rcp_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);
int rj_arith_v_s_rsq_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);
int rj_arith_v_s_rsq_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                         uint32_t *excp_flag_user);
int rj_arith_v_s_sqrt_f32(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t *excp_flag_user);
int rj_arith_v_s_sqrt_f16(uint64_t flags, uint64_t instruction_flags, uint32_t *d, uint32_t a,
                          uint32_t *excp_flag_user);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
