// Copyright (c) 2026 Advanced Micro Devices, Inc.
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
// Callers must select the level returned by rj_arith_init_cpu_flags() or a lower
// supported level. Instruction calls do not detect host features; accepting a
// defined level does not establish that the host supports it.
constexpr uint64_t RJ_ARITH_CPU_MASK = 0xfULL;
constexpr uint64_t RJ_ARITH_CPU_BASELINE = 0ULL;
constexpr uint64_t RJ_ARITH_CPU_X86_64_V3 = 1ULL;
constexpr uint64_t RJ_ARITH_CPU_X86_64_V4 = 2ULL;
constexpr uint64_t RJ_ARITH_CPU_X86_64_ZEN4 = 3ULL;

// Bits 4..15 are reserved for independent CPU features on top of the level.
// This is a bit-field, with architecture-specific feature names and meanings.
// No features are defined yet: callers and detection must leave these bits zero.
constexpr uint64_t RJ_ARITH_CPU_FEATURES_MASK = 0xfff0ULL;

// Semantics form an ordered enumeration, not independently OR-able bits.
// Mode 0 is loose. Mode 1 prefers exact results but permits loose fallback.
// Mode 2 requires exact results. Mode 3 additionally requires FP status reporting.
// Modes 0 through 2 never access FP status outputs. Values 4 through 6 are reserved.
// Mode 7 is reserved for later compatibility support and is currently rejected.
constexpr uint64_t RJ_ARITH_SEMANTICS_MASK = 7ULL << 16;
constexpr uint64_t RJ_ARITH_SEMANTICS_LOOSE = 0ULL;
constexpr uint64_t RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE = 1ULL << 16;
constexpr uint64_t RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS = 2ULL << 16;
constexpr uint64_t RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL = 3ULL << 16;
constexpr uint64_t RJ_ARITH_SEMANTICS_COMPATIBILITY = 7ULL << 16;

// Guest MODE.FP16_OVFL. Availability and effects, including saturation of
// conversion destinations, are specified by each instruction contract.
constexpr uint64_t RJ_ARITH_FP16_OVFL = 1ULL << 19;

// Flush guest input subnormals to signed zero, independently of host FP settings.
// Availability and mandatory-flush behavior are specified by each instruction's
// contract. Zero preserves inputs where the policy is configurable.
constexpr uint64_t RJ_ARITH_FP_FLUSH_INPUT_DENORMALS = 1ULL << 20;

// Flush guest output subnormals to signed zero, independently of host FTZ.
// Availability and effects are specified by each instruction's contract. Zero
// preserves outputs where configurable; nonzero OMOD may flush independently.
constexpr uint64_t RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS = 1ULL << 21;

// Guest MODE.IEEE. Availability and effects on OMOD and NaN handling are
// instruction- and architecture-specific; see each entry-point contract.
// This flag never changes the host FP environment.
constexpr uint64_t RJ_ARITH_FP_IEEE_MODE = 1ULL << 22;

// Disable RDNA3 MODE.DX10_CLAMP for supporting instructions: CLAMP preserves
// NaNs instead of replacing them with +0. Zero retains the default enabled
// policy. Availability is instruction-specific; RDNA4 does not expose this mode.
constexpr uint64_t RJ_ARITH_FP_DX10_CLAMP_OFF = 1ULL << 23;

// Guest rounding for supporting arithmetic APIs. This is an
// enumeration: nearest-even (zero), toward +infinity, toward -infinity, toward
// zero. Loose paths may ignore it and retain SIMD dispatch. No host FP changes.
constexpr uint64_t RJ_ARITH_FP_ROUND_MASK = 3ULL << 24;
constexpr uint64_t RJ_ARITH_FP_ROUND_UP = 1ULL << 24;
constexpr uint64_t RJ_ARITH_FP_ROUND_DOWN = 2ULL << 24;
constexpr uint64_t RJ_ARITH_FP_ROUND_ZERO = 3ULL << 24;

// FP16 input-denormal override for supporting mixed-format instructions. This
// is an enumeration: zero inherits RJ_ARITH_FP_FLUSH_INPUT_DENORMALS; one flushes;
// two preserves. Three is invalid. FP32 inputs retain the common policy.
constexpr uint64_t RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK = 3ULL << 26;
constexpr uint64_t RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16 = 1ULL << 26;
constexpr uint64_t RJ_ARITH_FP_PRESERVE_INPUT_DENORMALS_F16 = 2ULL << 26;

// Status codes returned by instruction emulation functions.
constexpr int RJ_ARITH_SUCCESS = 0;
constexpr int RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS = 1;
constexpr int RJ_ARITH_ERROR_INVALID_FLAGS = 2;

// Required faithful FP status reporting is unavailable.
constexpr int RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS = 3;

// A non-FP architectural side effect requested by a register operation is unavailable.
constexpr int RJ_ARITH_ERROR_UNSUPPORTED_REGISTER_STATE = 4;

// Names retain ISA mnemonics. Wave and architecture suffixes follow corresponding
// Clang AMDGPU builtins where available, with wave width before architecture.
// Vector APIs always identify wave width; scalar APIs omit it. Shared instruction
// contracts use the same API across architectures.

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
// trapsts (RDNA3) or excp_flag_user (RDNA4) accumulates participating-lane FP
// exception causes with bitwise OR, preserving all existing bits. Status storage
// must not overlap operands unless an instruction explicitly permits it.
// Errors leave status and all other outputs unchanged, including for empty EXEC.
// Required numeric semantics that are unavailable return UNSUPPORTED_SEMANTICS;
// required FP status reporting that is unavailable returns UNSUPPORTED_FP_STATUS.
// Reserved semantics values are invalid and rejected before operand access.
// Ordinary results, including comparison masks and SCC, are required in every
// mode. Explicit register writes and non-FP side effects have separate contracts.

// Arithmetic exception bits shared by RDNA3 TRAPSTS.EXCP[6:0] and RDNA4
// WAVE_EXCP_FLAG_USER[6:0]. Other fields of these registers are not interchangeable.
constexpr uint32_t RJ_ARITH_EXCEPTION_INVALID = 1U << 0;
constexpr uint32_t RJ_ARITH_EXCEPTION_INPUT_DENORM = 1U << 1;
constexpr uint32_t RJ_ARITH_EXCEPTION_FLOAT_DIV0 = 1U << 2;
constexpr uint32_t RJ_ARITH_EXCEPTION_OVERFLOW = 1U << 3;
constexpr uint32_t RJ_ARITH_EXCEPTION_UNDERFLOW = 1U << 4;
constexpr uint32_t RJ_ARITH_EXCEPTION_INEXACT = 1U << 5;
constexpr uint32_t RJ_ARITH_EXCEPTION_INT_DIV0 = 1U << 6;

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
