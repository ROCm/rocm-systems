// SPDX-License-Identifier: MIT

// IWYU pragma: private, include "rj_arith/rj_arith.h"

#ifndef RJ_ARITH_COMMON_H_
#define RJ_ARITH_COMMON_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bits 0..3 select a CPU level: an enumeration, not OR-able feature bits.
// Levels are architecture-specific; other host architectures may reuse the
// numeric values. Prefix their constants with the host architecture name.
// Baseline is always zero and shared by all architectures.
static const uint64_t RJ_ARITH_CPU_MASK = 0xfULL;
static const uint64_t RJ_ARITH_CPU_BASELINE = 0ULL;
static const uint64_t RJ_ARITH_CPU_X86_64_V3 = 1ULL;
static const uint64_t RJ_ARITH_CPU_X86_64_V4 = 2ULL;
static const uint64_t RJ_ARITH_CPU_X86_64_ZEN4 = 3ULL;

// Bits 4..15 are reserved for independent CPU features on top of the level.
// This is a bit-field, with architecture-specific feature names and meanings.
// No features are defined yet: callers and detection must leave these bits zero.
static const uint64_t RJ_ARITH_CPU_FEATURES_MASK = 0xfff0ULL;

// Semantics form an ordered enumeration, not independently OR-able bits.
// Mode 0 is loose. Mode 1 prefers exact results but permits loose fallback.
// Mode 2 requires exact results. Mode 3 additionally requires FP status reporting.
// Modes 0 through 2 never access FP status outputs. Values 4 through 6 are reserved.
// Temporary migration mode 7 is outside this ordering: each instruction resolves
// it to the mode closest to old rocjitsu behavior. FP status storage must be
// valid if that resolution selects mode 3; otherwise it remains unused.
static const uint64_t RJ_ARITH_SEMANTICS_MASK = 7ULL << 16;
static const uint64_t RJ_ARITH_SEMANTICS_LOOSE = 0ULL;
static const uint64_t RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE = 1ULL << 16;
static const uint64_t RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS = 2ULL << 16;
static const uint64_t RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL = 3ULL << 16;
static const uint64_t RJ_ARITH_SEMANTICS_COMPATIBILITY = 7ULL << 16;

// GPU MODE.FP16_OVFL: saturate finite overflow to the largest finite value for
// supporting FP16/FP8/BF8 instructions. Otherwise use the format's overflow encoding.
// Does not change input-infinity handling or BF16/FP32 results, and is independent
// of the host floating-point environment.
static const uint64_t RJ_ARITH_FP16_OVFL = 1ULL << 19;

// Flush guest input subnormals to signed zero, independently of host FP settings.
// Availability and mandatory-flush behavior are specified by each instruction's
// contract. Zero preserves inputs where the policy is configurable.
static const uint64_t RJ_ARITH_FP_FLUSH_INPUT_DENORMALS = 1ULL << 20;

// Flush guest output subnormals to signed zero, independently of host FTZ.
// Availability and effects are specified by each instruction's contract. Zero
// preserves outputs where configurable; nonzero OMOD may flush independently.
static const uint64_t RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS = 1ULL << 21;

// Guest MODE.IEEE. Availability and effects on OMOD and NaN handling are
// instruction- and architecture-specific; see each entry-point contract.
// This flag never changes the host FP environment.
static const uint64_t RJ_ARITH_FP_IEEE_MODE = 1ULL << 22;

// Disable RDNA3 MODE.DX10_CLAMP for supporting instructions: CLAMP preserves
// NaNs instead of replacing them with +0. Zero retains the default enabled
// policy. Availability is instruction-specific; RDNA4 does not expose this mode.
static const uint64_t RJ_ARITH_FP_DX10_CLAMP_OFF = 1ULL << 23;

// Guest rounding for supporting arithmetic APIs. This is an
// enumeration: nearest-even (zero), toward +infinity, toward -infinity, toward
// zero. Loose paths may ignore it and retain SIMD dispatch. No host FP changes.
static const uint64_t RJ_ARITH_FP_ROUND_MASK = 3ULL << 24;
static const uint64_t RJ_ARITH_FP_ROUND_UP = 1ULL << 24;
static const uint64_t RJ_ARITH_FP_ROUND_DOWN = 2ULL << 24;
static const uint64_t RJ_ARITH_FP_ROUND_ZERO = 3ULL << 24;

// FP16 input-denormal override for supporting mixed-format instructions. This
// is an enumeration: zero inherits RJ_ARITH_FP_FLUSH_INPUT_DENORMALS; one flushes;
// two preserves. Three is invalid. FP32 inputs retain the common policy.
static const uint64_t RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK = 3ULL << 26;
static const uint64_t RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16 = 1ULL << 26;
static const uint64_t RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16 = 2ULL << 26;

// Status codes returned by instruction emulation functions.
static const int RJ_ARITH_SUCCESS = 0;
static const int RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS = 1;
static const int RJ_ARITH_ERROR_INVALID_FLAGS = 2;

// Required faithful FP status reporting is unavailable.
static const int RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS = 3;

// A non-FP architectural side effect requested by a register operation is unavailable.
static const int RJ_ARITH_ERROR_UNSUPPORTED_REGISTER_STATE = 4;

// Names follow rj_arith_<mnemonic>[_w32|_w64][_gfx12]: wave width precedes the
// optional architecture suffix. Scalar and pseudo-scalar APIs omit wave width.
// The _gfx12 suffix distinguishes changed RDNA4 semantics; shared semantics
// use the same API across architectures.

// EXEC-dependent Wave32 entry points use a 32-bit exec_mask (Wave64 uses 64 bits).
// Scalar, pseudo-scalar, WMMA and SWMMAC instructions ignore EXEC and omit exec_mask.
// Matrix instructions read and write all lanes. Each Wave32 VGPR pointer names 32
// contiguous uint32_t lane words. Pointer arrays and backing storage must be valid.
// Whole VGPRs may alias; distinct VGPR addresses must not overlap. Sources are
// conceptually read before writes. Inactive destination lanes and all destinations
// on error are unchanged. Arithmetic assumes the default host FP environment:
// nearest-even rounding, denormals enabled, and exceptions masked. Implementations
// never explicitly modify host FP state; ordinary arithmetic may raise sticky flags.

// FP status outputs are appended to the operand list. Their use is controlled
// solely by the semantics field. Modes 0 through 2 do not access these pointers:
// null is allowed and supplied storage remains unchanged. Mode 3 requires valid
// storage as a caller precondition; entry points do not check pointer nullness.
// Temporary compatibility mode 7 follows the resolved mode's pointer contract.
// Each entry resolves it to mode 0, 2 or 3 using the old rocjitsu instruction policy.
// trapsts (RDNA3) or excp_flag_user (RDNA4) accumulates participating-lane FP
// exception causes with bitwise OR, preserving all existing bits. Status storage
// must not overlap operands unless an instruction explicitly permits it.
// Errors leave status and all other outputs unchanged, including for empty EXEC.
// Required numeric semantics that are unavailable return UNSUPPORTED_SEMANTICS;
// required FP status reporting that is unavailable returns UNSUPPORTED_FP_STATUS.
// Reserved semantics values are invalid and rejected before operand access.
// Ordinary results, including comparison masks and SCC, are required in every
// mode. Explicit register writes and non-FP side effects have separate contracts.

// DPP8 permutes source A within each group of eight lanes before arithmetic
// modifiers. Pack eight 3-bit lane indices into bits 40..63, index 0 first, and
// set RJ_ARITH_DPP8. Without FI, an inactive source supplies positive zero; FI reads
// its stored value. Inactive destinations remain unchanged. Supported by FP32
// FMA/FMAC, ADD/SUB/SUBREV/MUL/MUL_DX9_ZERO, binary min/max, and three-input
// min/max/median, and FP32 unary math (rounding, SQRT/RCP/RSQ/EXP/LOG, FRACT,
// FREXP mantissa), and wave32 CLZ/CTZ/CLS/BCNT/MBCNT, 32-bit shifts, and 32-bit Boolean
// AND/OR/XOR/XNOR/NOT, and 32-bit integer min/max/median (two or three inputs),
// plus LSHL_ADD/ADD_LSHL/LSHL_OR/AND_OR/OR3/XOR3/XAD/LERP. Each supports its
// applicable modifiers and whole-VGPR aliases. BFE/BFI/BFM/BFREV, ALIGNBIT,
// ALIGNBYTE, and PERM also support DPP, as do 32-bit ADD_NC/SUB_NC/SUBREV_NC
// (including CLAMP) and ADD3. SUBREV_NC_U32 permutes B instead of A.
// Signed/unsigned 24-bit MUL and MUL_HI support DPP, with CLAMP for low results.
// The four integer MAD forms support DPP with CLAMP and their 16-bit selectors.
// The 16-bit AND/OR/XOR/NOT and non-packed integer arithmetic forms support
// DPP with their source/destination selectors and applicable CLAMP modifiers.
// This includes non-packed 16-bit MAD, MIN3, MAX3, and MED3.
// FP16 ADD/SUB/SUBREV/MUL and binary/ternary min/max/median support DPP
// with all modifiers. FP16 FMA/FMAC support DPP in loose and exact semantics.
// FP16 unary math also supports DPP with all applicable modifiers, as do FP16
// and FP32 LDEXP/FREXP exponent, all six FP32/integer conversions, and all six
// FP16 conversions, byte/nibble-to-FP32 conversions, and FP32-to-byte packing.
static const uint64_t RJ_ARITH_DPP8 = 1ULL << 32;
static const uint64_t RJ_ARITH_DPP_FI = 1ULL << 33;
static const uint32_t RJ_ARITH_DPP8_SELECT_SHIFT = 40;
static const uint64_t RJ_ARITH_DPP8_SELECT_MASK = 0xffffffULL << 40;

// DPP16 permutes the same source as DPP8 within 16-lane rows before arithmetic
// modifiers (B for SUBREV_NC_U32, A otherwise). Select
// exactly one of RJ_ARITH_DPP8/RJ_ARITH_DPP16. DPP16 control is the architectural 9-bit
// encoding: quad_perm 0x000..0x0ff; row_shl 0x101..0x10f; row_shr 0x111..0x11f;
// row_ror 0x121..0x12f; row_mirror 0x140; row_half_mirror 0x141;
// row_share 0x150..0x15f; row_xmask 0x160..0x16f. Other controls are invalid.
// Row/bank fields enable destination rows of 16 lanes and banks of four lanes.
// Use both full MASK constants to enable all destinations; zero fields disable
// writes. FI permits inactive in-range sources. Without a readable source,
// BOUND_CTRL supplies positive zero; otherwise the destination is preserved.
// FI does not permit out-of-range sources. Supports the same instructions as DPP8.
static const uint64_t RJ_ARITH_DPP16 = 1ULL << 34;
static const uint64_t RJ_ARITH_DPP_BOUND_CTRL = 1ULL << 35;
static const uint32_t RJ_ARITH_DPP_CTRL_SHIFT = 40;
static const uint64_t RJ_ARITH_DPP_CTRL_MASK = 0x1ffULL << 40;
static const uint32_t RJ_ARITH_DPP_ROW_SHIFT = 49;
static const uint64_t RJ_ARITH_DPP_ROW_MASK = 0xfULL << 49;
static const uint32_t RJ_ARITH_DPP_BANK_SHIFT = 53;
static const uint64_t RJ_ARITH_DPP_BANK_MASK = 0xfULL << 53;

// Arithmetic exception bits shared by RDNA3 TRAPSTS.EXCP[6:0] and RDNA4
// WAVE_EXCP_FLAG_USER[6:0]. Other fields of these registers are not interchangeable.
static const uint32_t RJ_ARITH_EXCEPTION_INVALID = 1U << 0;
static const uint32_t RJ_ARITH_EXCEPTION_INPUT_DENORM = 1U << 1;
static const uint32_t RJ_ARITH_EXCEPTION_FLOAT_DIV0 = 1U << 2;
static const uint32_t RJ_ARITH_EXCEPTION_OVERFLOW = 1U << 3;
static const uint32_t RJ_ARITH_EXCEPTION_UNDERFLOW = 1U << 4;
static const uint32_t RJ_ARITH_EXCEPTION_INEXACT = 1U << 5;
static const uint32_t RJ_ARITH_EXCEPTION_INT_DIV0 = 1U << 6;

// Packed source negation: independent for the low and high result calculations.
static const uint32_t RJ_ARITH_PK_NEG_LO_A = 1U << 0;
static const uint32_t RJ_ARITH_PK_NEG_LO_B = 1U << 1;
static const uint32_t RJ_ARITH_PK_NEG_LO_C = 1U << 2;
static const uint32_t RJ_ARITH_PK_NEG_HI_A = 1U << 3;
static const uint32_t RJ_ARITH_PK_NEG_HI_B = 1U << 4;
static const uint32_t RJ_ARITH_PK_NEG_HI_C = 1U << 5;

// Packed output clamp: floats to [0, 1], with NaNs and -0 to +0;
// integer ADD/SUB/MAD saturate to their result range.
static const uint32_t RJ_ARITH_PK_CLAMP = 1U << 6;

// Packed half selectors flip the default choice for each result calculation.
// Zero flags use low inputs for the low result and high inputs for the high one.
// These flags can swap or replicate halves independently for each source.
static const uint32_t RJ_ARITH_PK_LO_A_HIGH = 1U << 7;
static const uint32_t RJ_ARITH_PK_LO_B_HIGH = 1U << 8;
static const uint32_t RJ_ARITH_PK_LO_C_HIGH = 1U << 9;
static const uint32_t RJ_ARITH_PK_HI_A_LOW = 1U << 10;
static const uint32_t RJ_ARITH_PK_HI_B_LOW = 1U << 11;
static const uint32_t RJ_ARITH_PK_HI_C_LOW = 1U << 12;

// Floating ALU source modifiers: ABS precedes NEG.
static const uint32_t RJ_ARITH_ALU_NEG_A = 1U << 0;
static const uint32_t RJ_ARITH_ALU_NEG_B = 1U << 1;
static const uint32_t RJ_ARITH_ALU_NEG_C = 1U << 2;
static const uint32_t RJ_ARITH_ALU_ABS_A = 1U << 3;
static const uint32_t RJ_ARITH_ALU_ABS_B = 1U << 4;
static const uint32_t RJ_ARITH_ALU_ABS_C = 1U << 5;

// Floating ALU output scaling precedes CLAMP. OMOD is a two-bit enumeration:
// none, multiply by 2, multiply by 4, divide by 2.
static const uint32_t RJ_ARITH_ALU_OMOD_2 = 1U << 6;
static const uint32_t RJ_ARITH_ALU_OMOD_4 = 2U << 6;
static const uint32_t RJ_ARITH_ALU_OMOD_HALF = 3U << 6;

// Result clamping where supported: floating results to [0,1] (NaNs become zero),
// integer results to the representable signed/unsigned destination range.
static const uint32_t RJ_ARITH_ALU_CLAMP = 1U << 8;

// True16 source/destination half selectors. Zero selects the low half.
// DOT2 consumes both A/B halves and accepts only the C and D selectors.
static const uint32_t RJ_ARITH_ALU_HIGH_A = 1U << 9;
static const uint32_t RJ_ARITH_ALU_HIGH_B = 1U << 10;
static const uint32_t RJ_ARITH_ALU_HIGH_C = 1U << 11;
static const uint32_t RJ_ARITH_ALU_HIGH_D = 1U << 12;

// Returns CPU capability flags for features usable on this machine, accounting
// for CPU and operating-system support. All non-CPU flag bits are zero.
uint64_t rj_arith_init_cpu_flags(void);

// Returns a static CPU-level name, ignoring bits outside RJ_ARITH_CPU_MASK.
// Unknown levels return "unknown". The returned string must not be modified.
const char *rj_arith_cpu_name(uint64_t flags);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
