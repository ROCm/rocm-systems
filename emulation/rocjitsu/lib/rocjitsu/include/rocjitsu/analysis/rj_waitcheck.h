// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_waitcheck.h
/// @brief C API for synchronous wait-hazard analysis of in-memory AMDGPU code objects.
///
/// @details The dedicated `librocjitsu_waitcheck.so.1` implements this API
/// without initializing ROCR or rocJITsu's VM, registering atexit handlers, or
/// creating background threads. Once all calls and callbacks have returned,
/// callers may safely discard borrowed pointers and unload the library. The
/// library retains no process or thread-local analysis state across calls or
/// after `dlclose()`. The full `librocjitsu.so` also exports these symbols for
/// compatibility, but only the dedicated library carries this unload contract.

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

/// @brief Current ABI version for the append-only waitcheck C structures.
#define ROCJITSU_WAITCHECK_ABI_VERSION 1U

/// @brief AMDGPU target inferred from a code object's ELF machine flags.
///
/// @details Values are stable ABI constants. A target may use a compatible
/// waitcheck model while retaining its distinct identity here.
typedef enum rj_waitcheck_target_e {
  ROCJITSU_WAITCHECK_TARGET_UNKNOWN = 0,
  ROCJITSU_WAITCHECK_TARGET_GFX90A = 1,
  ROCJITSU_WAITCHECK_TARGET_GFX942 = 2,
  ROCJITSU_WAITCHECK_TARGET_GFX950 = 3,
  ROCJITSU_WAITCHECK_TARGET_GFX1100 = 4,
  ROCJITSU_WAITCHECK_TARGET_GFX1200 = 5,
  ROCJITSU_WAITCHECK_TARGET_GFX1201 = 6,
  ROCJITSU_WAITCHECK_TARGET_GFX1250 = 7,
  ROCJITSU_WAITCHECK_TARGET_GFX1150 = 8,
  ROCJITSU_WAITCHECK_TARGET_GFX1151 = 9
} rj_waitcheck_target_t;

/// @brief Stable machine-readable waitcheck diagnostic code.
///
/// @details Values are stable ABI constants. The counter, access, and register
/// fields provide the instruction-specific detail for each code.
typedef enum rj_waitcheck_diagnostic_code_e {
  ROCJITSU_WAITCHECK_DIAGNOSTIC_UNKNOWN = 0,
  /// @brief An ordinary wait counter was missing or too weak.
  ROCJITSU_WAITCHECK_DIAGNOSTIC_WAIT_COUNTER = 1,
  /// @brief An SGPR dependency requires an s_wait_alu depctr wait.
  ROCJITSU_WAITCHECK_DIAGNOSTIC_SGPR_DEPCTR = 2,
  /// @brief depctr_vm_vsrc(0) is required immediately before async barrier arrival.
  ROCJITSU_WAITCHECK_DIAGNOSTIC_ASYNC_BARRIER_PRE_WAIT = 3,
  /// @brief depctr_vm_vsrc(0) is required immediately after async barrier arrival.
  ROCJITSU_WAITCHECK_DIAGNOSTIC_ASYNC_BARRIER_POST_WAIT = 4,
  /// @brief The encoded wait_va_vdst value is too weak.
  ROCJITSU_WAITCHECK_DIAGNOSTIC_VA_VDST = 5
} rj_waitcheck_diagnostic_code_t;

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
  ROCJITSU_WAITCHECK_ACCESS_INVALID,
  /// @brief An outstanding SGPR dependency reaches a call, return, or indirect branch.
  ROCJITSU_WAITCHECK_ACCESS_CONTROL_TRANSFER
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
  /// @brief Size of this structure supplied by the library.
  size_t struct_size;
  /// @brief ABI version used to populate this structure.
  uint32_t abi_version;
  /// @brief Nonzero when this diagnostic is attributed to a kernel descriptor.
  uint32_t has_kernel;
  /// @brief Kernel symbol name, or NULL when no kernel identity is available.
  const char *kernel_name;
  /// @brief Kernel entry-point byte offset in `.text`.
  uint64_t kernel_entry_offset;
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
  /// @brief Stable machine-readable diagnostic code.
  rj_waitcheck_diagnostic_code_t code;
} rj_waitcheck_diagnostic_t;

/// @brief Receives one wait-hazard diagnostic during a synchronous analysis call.
/// @details Exceptions thrown by a C++ callback are suppressed; delivery is
/// best-effort and does not change the analysis status.
typedef void (*rj_waitcheck_diagnostic_callback_t)(const rj_waitcheck_diagnostic_t *diagnostic,
                                                   void *user_data);

/// @brief Stable classification for emitted-wait counter-parity findings.
typedef enum rj_waitcheck_counter_parity_kind_e {
  ROCJITSU_WAITCHECK_COUNTER_PARITY_UNKNOWN = 0,
  /// @brief Waitcheck modeled a dependency, but its required N was greater
  /// than the compiler-emitted M.
  ROCJITSU_WAITCHECK_COUNTER_PARITY_MODELED_UNDERACCOUNTING = 1,
  /// @brief The compiler emitted a non-sentinel wait, but waitcheck modeled no
  /// dependency for that counter before the guarded instruction.
  ROCJITSU_WAITCHECK_COUNTER_PARITY_UNMODELED_EMITTED_WAIT = 2
} rj_waitcheck_counter_parity_kind_t;

/// @brief One compiler-emitted wait field stronger than waitcheck's requirement.
///
/// @details Lower counter values are stronger. Therefore required_count greater
/// than emitted_count is a false-negative candidate. String pointers are
/// borrowed and remain valid only for the duration of the callback.
typedef struct rj_waitcheck_counter_parity_diagnostic_s {
  size_t struct_size;
  uint32_t abi_version;
  rj_waitcheck_counter_parity_kind_t kind;
  uint32_t has_kernel;
  const char *kernel_name;
  uint64_t kernel_entry_offset;
  rj_waitcheck_counter_t counter;
  uint32_t emitted_count;
  uint32_t required_count;
  uint32_t has_required_dependency;
  rj_waitcheck_access_t access;
  rj_waitcheck_register_t reg;
  const char *section_name;
  uint64_t wait_section_offset;
  uint64_t wait_file_offset;
  const char *wait_instruction;
  uint64_t consumer_section_offset;
  uint64_t consumer_file_offset;
  const char *consumer_instruction;
  uint64_t producer_section_offset;
  uint64_t producer_file_offset;
  const char *producer_instruction;
  const char *message;
} rj_waitcheck_counter_parity_diagnostic_t;

/// @brief Receives one counter-parity finding during synchronous analysis.
/// @details Exceptions thrown by a C++ callback are suppressed; delivery is
/// best-effort and does not change the analysis status.
typedef void (*rj_waitcheck_counter_parity_callback_t)(
    const rj_waitcheck_counter_parity_diagnostic_t *diagnostic, void *user_data);

/// @brief Receives a human-readable explanation when analysis cannot complete.
///
/// @details The message pointer is borrowed and remains valid only for the
/// duration of the callback.
typedef void (*rj_waitcheck_error_callback_t)(const char *message, void *user_data);

/// @brief Options for one synchronous waitcheck analysis.
typedef struct rj_waitcheck_options_s {
  /// @brief Caller allocation size, initialized by rj_waitcheck_options_init().
  size_t struct_size;
  /// @brief ABI version, initialized by rj_waitcheck_options_init().
  uint32_t abi_version;
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
  /// @brief Opaque value passed to all callbacks.
  void *user_data;
  /// @brief Enable intact emitted-wait counter-parity auditing when nonzero.
  ///
  /// @details The initial implementation supports gfx950 legacy vmcnt,
  /// lgkmcnt, and expcnt fields.
  uint32_t check_counter_parity;
  /// @brief Maximum counter-parity callbacks. Zero means unlimited.
  size_t max_counter_parity_diagnostics;
  /// @brief Optional receiver for counter-parity findings.
  rj_waitcheck_counter_parity_callback_t counter_parity_callback;
} rj_waitcheck_options_t;

/// @brief Aggregate result of one waitcheck analysis attempt.
typedef struct rj_waitcheck_result_s {
  /// @brief Caller allocation size, initialized by rj_waitcheck_result_init().
  size_t struct_size;
  /// @brief ABI version used to populate this structure.
  uint32_t abi_version;
  /// @brief Target inferred from the code object, including on later analysis errors.
  rj_waitcheck_target_t target;
  size_t instructions_analyzed;
  size_t memory_events_tracked;
  size_t kernels_discovered;
  size_t kernels_analyzed;
  size_t diagnostics_observed;
  /// @brief Number of times diagnostic_callback was invoked.
  size_t diagnostics_reported;
  /// @brief Nonzero when analysis completed and no wait hazards were observed.
  uint32_t passed;
  /// @brief Nonzero when diagnostics_observed is a lower bound because
  /// callbacks were disabled or limited, or analysis stopped early.
  uint32_t diagnostics_truncated;
  /// @brief Nonzero when stop_after_first_diagnostic ended analysis early.
  uint32_t stopped_early;
  size_t counter_parity_wait_groups;
  size_t counter_parity_fields_checked;
  size_t counter_parity_exact;
  size_t counter_underaccounting_observed;
  size_t counter_unmodeled_wait_observed;
  size_t counter_parity_indeterminate_groups;
  /// @brief Number of times counter_parity_callback was invoked.
  size_t counter_parity_diagnostics_reported;
  /// @brief Nonzero when counter-parity detail callbacks were omitted or limited.
  uint32_t counter_parity_diagnostics_truncated;
  /// @brief Nonzero when the requested counter-parity model was available and ran.
  uint32_t counter_parity_evaluated;
  /// @brief Nonzero when every encountered wait dependency was covered by the active model.
  uint32_t analysis_complete;
  /// @brief Number of observations that prevented complete analysis.
  size_t incomplete_observations;
} rj_waitcheck_result_t;

/// @brief Initialize waitcheck options to their defaults.
///
/// @details The defaults retain an implementation-defined reachability cache,
/// report every diagnostic, and do not stop early. Structures are append-only
/// within an ABI version. Callers must pass their allocation size and call this
/// initializer before analysis.
///
/// @retval ROCJITSU_STATUS_SUCCESS The structure was initialized.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT @p options is NULL or
/// @p options_size is too small for this ABI version.
RJ_API_EXPORT rj_status_t rj_waitcheck_options_init(rj_waitcheck_options_t *options,
                                                    size_t options_size);

/// @brief Initialize a caller-owned result before analysis.
///
/// @details Callers must pass their allocation size and call this initializer
/// before each result is first used. The analysis functions reset the result
/// while preserving its caller-supplied allocation size.
///
/// @retval ROCJITSU_STATUS_SUCCESS The structure was initialized.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT @p result is NULL or
/// @p result_size is too small for this ABI version.
RJ_API_EXPORT rj_status_t rj_waitcheck_result_init(rj_waitcheck_result_t *result,
                                                   size_t result_size);

/// @brief Return the stable lowercase spelling of a waitcheck target.
///
/// @details The returned static string is owned by the library. Unknown enum
/// values return `"unknown"`. Copy it before unloading the library if it must
/// outlive the call sequence.
RJ_API_EXPORT const char *rj_waitcheck_target_name(rj_waitcheck_target_t target);

/// @brief Return the stable lowercase spelling of a diagnostic code.
///
/// @details The returned static string is owned by the library. Unknown enum
/// values return `"unknown"`. Copy it before unloading the library if it must
/// outlive the call sequence.
RJ_API_EXPORT const char *rj_waitcheck_diagnostic_code_name(rj_waitcheck_diagnostic_code_t code);

/// @brief Return the stable lowercase spelling of a counter-parity kind.
///
/// @details The returned static string is owned by the library. Unknown enum
/// values return `"unknown"`.
RJ_API_EXPORT const char *
rj_waitcheck_counter_parity_kind_name(rj_waitcheck_counter_parity_kind_t kind);

/// @brief Synchronously analyze every kernel in an in-memory AMDGPU HSA code object.
///
/// @details The target is inferred from the ELF header. The function copies any
/// bytes it needs before returning, invokes callbacks on the calling thread, and
/// does not create worker threads. Independent calls are reentrant and may run
/// concurrently on different threads; each call invokes its callbacks on its
/// calling thread. Callers must externally synchronize any state shared between
/// calls. The input buffer need only remain valid until the function returns. A
/// reported wait hazard is a successful analysis with result->passed set to
/// zero, not an API error.
///
/// @param[in] code_object AMDGPU HSA ELF image in memory.
/// @param[in] code_object_size Size of @p code_object in bytes.
/// @param[in] options Initialized analysis options, or NULL for defaults.
/// @param[in,out] result Initialized aggregate analysis result. The inferred
/// target is retained when parsing succeeds but later analysis fails.
/// @retval ROCJITSU_STATUS_SUCCESS Analysis completed, with or without hazards.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT A required argument is NULL or a
/// caller-owned structure was not initialized for this ABI version.
/// @retval ROCJITSU_STATUS_INVALID_CODE_OBJECT The buffer is malformed, is not a
/// final AMDGPU HSA code object, targets an unsupported architecture, cannot be
/// decoded completely, or exceeds a configured analysis limit.
/// @retval ROCJITSU_STATUS_UNSUPPORTED The code object was recognized, but one
/// or more wait dependencies could not be analyzed completely. The result and
/// any definite diagnostic callbacks are still populated.
/// @retval ROCJITSU_STATUS_OUT_OF_RESOURCES Analysis allocation failed.
/// @retval ROCJITSU_STATUS_ERROR An unexpected analysis error occurred.
RJ_API_EXPORT rj_status_t rj_waitcheck_analyze(const void *code_object, size_t code_object_size,
                                               const rj_waitcheck_options_t *options,
                                               rj_waitcheck_result_t *result);

/// @brief Synchronously analyze one kernel in an in-memory AMDGPU HSA code object.
///
/// @details This has the same ownership, callback, target-inference, result, and
/// status semantics as rj_waitcheck_analyze(), but restricts analysis to the
/// kernel whose entry point has the supplied `.text` byte offset.
///
/// @param[in] code_object AMDGPU HSA ELF image in memory.
/// @param[in] code_object_size Size of @p code_object in bytes.
/// @param[in] kernel_entry_offset Kernel entry-point byte offset in `.text`.
/// Offset zero is valid.
/// @param[in] options Initialized analysis options, or NULL for defaults.
/// @param[in,out] result Initialized aggregate analysis result. The inferred
/// target is retained when parsing succeeds but later analysis fails.
/// @retval ROCJITSU_STATUS_SUCCESS Analysis completed, with or without hazards.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT A required argument is NULL or
/// @p kernel_entry_offset is not present.
/// @retval ROCJITSU_STATUS_INVALID_CODE_OBJECT The buffer is malformed, is not a
/// final AMDGPU HSA code object, targets an unsupported architecture, cannot be
/// decoded completely, or exceeds a configured analysis limit.
/// @retval ROCJITSU_STATUS_UNSUPPORTED The code object was recognized, but one
/// or more wait dependencies could not be analyzed completely. The result and
/// any definite diagnostic callbacks are still populated.
/// @retval ROCJITSU_STATUS_OUT_OF_RESOURCES Analysis allocation failed.
/// @retval ROCJITSU_STATUS_ERROR An unexpected analysis error occurred.
RJ_API_EXPORT rj_status_t rj_waitcheck_analyze_kernel(const void *code_object,
                                                      size_t code_object_size,
                                                      uint64_t kernel_entry_offset,
                                                      const rj_waitcheck_options_t *options,
                                                      rj_waitcheck_result_t *result);

/// @}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_ANALYSIS_RJ_WAITCHECK_H_
