// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_waitcheck.h
/// @brief C API for synchronous wait-hazard analysis of in-memory AMDGPU code objects.

#ifndef ROCJITSU_ANALYSIS_RJ_WAITCHECK_H_
#define ROCJITSU_ANALYSIS_RJ_WAITCHECK_H_

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/base/rj_status.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup analysis
/// @{

/// @brief Analyze every kernel in the supplied code object.
#define ROCJITSU_WAITCHECK_ALL_KERNELS UINT64_MAX

/// @brief Hardware wait counter associated with a diagnostic.
typedef enum rj_waitcheck_counter_e {
  ROCJITSU_WAITCHECK_COUNTER_LOAD = 0,
  ROCJITSU_WAITCHECK_COUNTER_STORE,
  ROCJITSU_WAITCHECK_COUNTER_DS,
  ROCJITSU_WAITCHECK_COUNTER_KM,
  ROCJITSU_WAITCHECK_COUNTER_SAMPLE,
  ROCJITSU_WAITCHECK_COUNTER_BVH,
  ROCJITSU_WAITCHECK_COUNTER_EXP,
  ROCJITSU_WAITCHECK_COUNTER_X,
  ROCJITSU_WAITCHECK_COUNTER_ASYNC,
  ROCJITSU_WAITCHECK_COUNTER_TENSOR,
  ROCJITSU_WAITCHECK_COUNTER_VM_VSRC,
  ROCJITSU_WAITCHECK_COUNTER_VA_VDST,
  ROCJITSU_WAITCHECK_COUNTER_DEPCTR,
  ROCJITSU_WAITCHECK_COUNTER_INVALID
} rj_waitcheck_counter_t;

/// @brief How an instruction conflicts with an outstanding operation.
typedef enum rj_waitcheck_access_e {
  ROCJITSU_WAITCHECK_ACCESS_USE = 0,
  ROCJITSU_WAITCHECK_ACCESS_DEF,
  ROCJITSU_WAITCHECK_ACCESS_MEMORY_ORDER,
  ROCJITSU_WAITCHECK_ACCESS_PROGRAM_END,
  ROCJITSU_WAITCHECK_ACCESS_INVALID
} rj_waitcheck_access_t;

/// @brief ISA register file associated with a diagnostic.
typedef enum rj_waitcheck_register_class_e {
  ROCJITSU_WAITCHECK_REGISTER_SGPR = 0,
  ROCJITSU_WAITCHECK_REGISTER_VGPR,
  ROCJITSU_WAITCHECK_REGISTER_ACC_VGPR,
  ROCJITSU_WAITCHECK_REGISTER_EXEC,
  ROCJITSU_WAITCHECK_REGISTER_VCC,
  ROCJITSU_WAITCHECK_REGISTER_SCC,
  ROCJITSU_WAITCHECK_REGISTER_M0,
  ROCJITSU_WAITCHECK_REGISTER_FLAT_SCRATCH,
  ROCJITSU_WAITCHECK_REGISTER_TTMP,
  ROCJITSU_WAITCHECK_REGISTER_PC,
  ROCJITSU_WAITCHECK_REGISTER_INVALID
} rj_waitcheck_register_class_t;

/// @brief Contiguous register range implicated in a diagnostic.
typedef struct rj_waitcheck_register_s {
  rj_waitcheck_register_class_t register_class;
  uint16_t index;
  /// @brief Width in 32-bit register lanes.
  uint8_t width;
} rj_waitcheck_register_t;

/// @brief One missing or too-weak wait diagnostic.
///
/// @details String pointers are borrowed and remain valid only for the duration
/// of the diagnostic callback.
typedef struct rj_waitcheck_diagnostic_s {
  rj_waitcheck_counter_t counter;
  rj_waitcheck_access_t access;
  rj_waitcheck_register_t reg;
  const char *section_name;
  uint64_t section_offset;
  uint64_t file_offset;
  const char *instruction;
  uint64_t producer_section_offset;
  uint64_t producer_file_offset;
  const char *producer_instruction;
  uint32_t required_count;
  const char *message;
} rj_waitcheck_diagnostic_t;

/// @brief Receives one wait-hazard diagnostic during a synchronous analysis call.
typedef void (*rj_waitcheck_diagnostic_callback_t)(const rj_waitcheck_diagnostic_t *diagnostic,
                                                   void *user_data);

/// @brief Receives a human-readable explanation when analysis cannot complete.
///
/// @details The message pointer is borrowed and remains valid only for the
/// duration of the callback.
typedef void (*rj_waitcheck_error_callback_t)(const char *message, void *user_data);

/// @brief Options for one synchronous waitcheck analysis.
typedef struct rj_waitcheck_options_s {
  /// @brief `.text` byte offset of one kernel entry, or
  /// ROCJITSU_WAITCHECK_ALL_KERNELS to analyze the entire code object.
  uint64_t kernel_entry_offset;
  /// @brief Maximum diagnostic callbacks. Zero means unlimited.
  size_t max_diagnostics;
  /// @brief Reachability-cache budget in bytes. Zero selects the library default.
  size_t max_reachability_cache_bytes;
  /// @brief Stop analysis after the first observed hazard when nonzero.
  uint32_t stop_after_first_diagnostic;
  /// @brief Optional diagnostic receiver.
  rj_waitcheck_diagnostic_callback_t diagnostic_callback;
  /// @brief Optional receiver for an analysis-error explanation.
  rj_waitcheck_error_callback_t error_callback;
  /// @brief Opaque value passed to both callbacks.
  void *user_data;
} rj_waitcheck_options_t;

/// @brief Aggregate result of one successful waitcheck analysis.
typedef struct rj_waitcheck_result_s {
  size_t instructions_analyzed;
  size_t memory_events_tracked;
  size_t kernels_discovered;
  size_t kernels_analyzed;
  size_t diagnostics_observed;
  /// @brief Number of times diagnostic_callback was invoked.
  size_t diagnostics_reported;
  /// @brief Nonzero when no wait hazards were observed.
  uint32_t passed;
  /// @brief Nonzero when diagnostics_observed is a lower bound because
  /// callbacks were disabled or limited, or analysis stopped early.
  uint32_t diagnostics_truncated;
  /// @brief Nonzero when stop_after_first_diagnostic ended analysis early.
  uint32_t stopped_early;
} rj_waitcheck_result_t;

/// @brief Initialize waitcheck options to their defaults.
///
/// @details The defaults analyze every kernel, retain an implementation-defined
/// reachability cache, report every diagnostic, and do not stop early. Passing
/// NULL is a no-op.
RJ_API_EXPORT void rj_waitcheck_options_init(rj_waitcheck_options_t *options);

/// @brief Synchronously analyze an in-memory AMDGPU HSA code object.
///
/// @details The target is inferred from the ELF header. The function copies any
/// bytes it needs before returning, invokes callbacks on the calling thread, and
/// does not create worker threads. The input buffer need only remain valid until
/// the function returns. A reported wait hazard is a successful analysis with
/// result->passed set to zero, not an API error.
///
/// @param[in] code_object AMDGPU HSA ELF image in memory.
/// @param[in] code_object_size Size of @p code_object in bytes.
/// @param[in] options Analysis options, or NULL for defaults.
/// @param[out] result Aggregate analysis result. Its contents are unspecified on error.
/// @retval ROCJITSU_STATUS_SUCCESS Analysis completed, with or without hazards.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT A required argument is NULL or the
/// selected kernel entry offset is not present.
/// @retval ROCJITSU_STATUS_INVALID_CODE_OBJECT The buffer is malformed, is not a
/// final AMDGPU HSA code object, targets an unsupported architecture, or cannot
/// be decoded completely.
/// @retval ROCJITSU_STATUS_OUT_OF_RESOURCES Analysis allocation failed.
/// @retval ROCJITSU_STATUS_ERROR An unexpected analysis error occurred.
RJ_API_EXPORT rj_status_t rj_waitcheck_analyze(const void *code_object, size_t code_object_size,
                                               const rj_waitcheck_options_t *options,
                                               rj_waitcheck_result_t *result);

/// @}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_ANALYSIS_RJ_WAITCHECK_H_
