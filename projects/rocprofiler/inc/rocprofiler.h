/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
//
// ROC Profiler API
//
// The goal of the implementation is to provide a HW specific low-level
// performance analysis interface for profiling of GPU compute applications.
// The profiling includes HW performance counters (PMC) with complex
// performance metrics and traces.
//
// The library can be used by a tool library loaded by HSA runtime or by
// higher level HW independent performance analysis API like PAPI.
//
// The library is written on C and will be based on AQLprofile AMD specific
// HSA extension. The library implementation requires HSA API intercepting and
// a profiling queue supporting a submit callback interface.
//
//
/** \mainpage ROCProfiler API Specification
 *
 * \section introduction Introduction
 *
 * ROCProfiler library, GPU Applications Profiling/Tracing APIs.
 * The API provides functionality for profiling GPU applications in kernel and
 * application and user mode and also with no replay mode at all and it
 * provides the records pool support with an easy sequence of calls, so the
 * user can be able to profile and trace in easy small steps, our samples code
 * can give good examples of how to use the API calls for both profiling and
 * tracing
 *
 * This document is going to discuss the following:
 * 1. @ref symbol_versions_group
 * 2. @ref versioning_group
 * 3. @ref status_codes_group
 * 4. @ref rocprofiler_general_group
 * 5. @ref timestamp_group
 * 6. @ref generic_record_group
 *      - @ref record_agents_group
 *      - @ref record_queues_group
 *      - @ref record_kernels_group
 * 7. @ref profiling_api_group
 *      - @ref profiling_api_counters_group
 * 8. @ref tracing_api_group
 *      - @ref roctx_tracer_api_data_group
 *      - @ref hsa_tracer_api_data_group
 *      - @ref hip_tracer_api_data_group
 * 9. @ref memory_storage_buffer_group
 * 10. @ref sessions_handling_group
 *      - @ref session_filter_group
 *      - @ref session_range_group
 *      - @ref session_user_replay_pass_group
 * 11. @ref device_profiling
 * 12. @ref rocprofiler_plugins
 */
//
/**
 * \file
 * ROCPROFILER API interface.
 */
////////////////////////////////////////////////////////////////////////////////

#ifndef INC_ROCPROFILER_H_
#define INC_ROCPROFILER_H_

/* Placeholder for calling convention and import/export macros */
#if !defined(ROCPROFILER_CALL)
#define ROCPROFILER_CALL
#endif /* !defined (ROCPROFILER_CALL) */

#if !defined(ROCPROFILER_EXPORT_DECORATOR)
#if defined(__GNUC__)
#define ROCPROFILER_EXPORT_DECORATOR __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define ROCPROFILER_EXPORT_DECORATOR __declspec(dllexport)
#endif /* defined (_MSC_VER) */
#endif /* !defined (ROCPROFILER_EXPORT_DECORATOR) */

#if !defined(ROCPROFILER_IMPORT_DECORATOR)
#if defined(__GNUC__)
#define ROCPROFILER_IMPORT_DECORATOR
#elif defined(_MSC_VER)
#define ROCPROFILER_IMPORT_DECORATOR __declspec(dllimport)
#endif /* defined (_MSC_VER) */
#endif /* !defined (ROCPROFILER_IMPORT_DECORATOR) */

#define ROCPROFILER_EXPORT ROCPROFILER_EXPORT_DECORATOR ROCPROFILER_CALL
#define ROCPROFILER_IMPORT ROCPROFILER_IMPORT_DECORATOR ROCPROFILER_CALL

#if !defined(ROCPROFILER)
#if defined(ROCPROFILER_EXPORTS)
#define ROCPROFILER_API ROCPROFILER_EXPORT
#else /* !defined (ROCPROFILER_EXPORTS) */
#define ROCPROFILER_API ROCPROFILER_IMPORT
#endif /* !defined (ROCPROFILER_EXPORTS) */
#endif /* !defined (ROCPROFILER) */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** \defgroup symbol_versions_group Symbol Versions
 *
 * The names used for the shared library versioned symbols.
 *
 * Every function is annotated with one of the version macros defined in this
 * section.  Each macro specifies a corresponding symbol version string.  After
 * dynamically loading the shared library with \p dlopen, the address of each
 * function can be obtained using \p dlsym with the name of the function and
 * its corresponding symbol version string.  An error will be reported by \p
 * dlvsym if the installed library does not support the version for the
 * function specified in this version of the interface.
 *
 * @{
 */

/**
 * The function was introduced in version 9.0 of the interface and has the
 * symbol version string of ``"ROCPROFILER_9.0"``.
 */
#define ROCPROFILER_VERSION_9_0

/** @} */

/** \defgroup versioning_group Library Versioning
 *
 * Version information about the interface and the associated installed
 * library.
 *
 * The semantic version of the interface following semver.org rules. A client
 * that uses this interface is only compatible with the installed library if
 * the major version numbers match and the interface minor version number is
 * less than or equal to the installed library minor version number.
 *
 * @{
 */

/**
 * The major version of the interface as a macro so it can be used by the
 * preprocessor.
 */
#define ROCPROFILER_VERSION_MAJOR 9

/**
 * The minor version of the interface as a macro so it can be used by the
 * preprocessor.
 */
#define ROCPROFILER_VERSION_MINOR 0

/**
 * Query the major version of the installed library.
 *
 * Return the major version of the installed library.  This can be used to
 * check if it is compatible with this interface version.  This function can be
 * used even when the library is not initialized.
 */
ROCPROFILER_API uint32_t rocprofiler_version_major();

/**
 * Query the minor version of the installed library.
 *
 * Return the minor version of the installed library.  This can be used to
 * check if it is compatible with this interface version.  This function can be
 * used even when the library is not initialized.
 */
ROCPROFILER_API uint32_t rocprofiler_version_minor();

/** @} */

#ifndef ROCPROFILER_V1

// TODO(aelwazir): Fix them to use the new Error codes
/** \defgroup status_codes_group Status Codes
 *
 * Most operations return a status code to indicate success or error.
 *
 * @{
 */

/**
 * ROCProfiler API status codes.
 */
typedef enum {
  /**
   * The function has executed successfully.
   */
  ROCPROFILER_STATUS_SUCCESS = 0,
  /**
   * A generic error has occurred.
   */
  ROCPROFILER_STATUS_ERROR = -1,
  /**
   * ROCProfiler is already initialized.
   */
  ROCPROFILER_STATUS_ERROR_ALREADY_INITIALIZED = -2,
  /**
   * ROCProfiler is not initialized.
   */
  ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED = -3,
  /**
   * Missing Buffer for a session.
   */
  ROCPROFILER_STATUS_ERROR_SESSION_MISSING_BUFFER = -4,
  /**
   * Timestamps can't be collected
   */
  ROCPROFILER_STATUS_ERROR_TIMESTAMP_NOT_APPLICABLE = -5,
  /**
   * Agent is not found with given identifier.
   */
  ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND = -6,
  /**
   * Agent information is missing for the given identifier
   */
  ROCPROFILER_STATUS_ERROR_AGENT_INFORMATION_MISSING = -7,
  /**
   * Queue is not found for the given identifier.
   */
  ROCPROFILER_STATUS_ERROR_QUEUE_NOT_FOUND = -8,
  /**
   * The requested information about the queue is not found.
   */
  ROCPROFILER_STATUS_ERROR_QUEUE_INFORMATION_MISSING = -9,
  /**
   * Kernel is not found with given identifier.
   */
  ROCPROFILER_STATUS_ERROR_KERNEL_NOT_FOUND = -10,
  /**
   * The requested information about the kernel is not found.
   */
  ROCPROFILER_STATUS_ERROR_KERNEL_INFORMATION_MISSING = -11,
  /**
   * Counter is not found with the given identifier.
   */
  ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND = -12,
  /**
   * The requested Counter information for the given kernel is missing.
   */
  ROCPROFILER_STATUS_ERROR_COUNTER_INFORMATION_MISSING = -13,
  /**
   * The requested Tracing API Data for the given data identifier is missing.
   */
  ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND = -14,
  /**
   * The requested information for the tracing API Data is missing.
   */
  ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING = -15,
  /**
   * The given Domain is incorrect.
   */
  ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN = -16,
  /**
   * The requested Session given the session identifier is not found.
   */
  ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND = -17,
  /**
   * The requested Session Buffer given the session identifier is corrupted or
   * deleted.
   */
  ROCPROFILER_STATUS_ERROR_CORRUPTED_SESSION_BUFFER = -18,
  /**
   * The requested record given the record identifier is corrupted or deleted.
   */
  ROCPROFILER_STATUS_ERROR_RECORD_CORRUPTED = -19,
  /**
   * Incorrect Replay mode.
   */
  ROCPROFILER_STATUS_ERROR_INCORRECT_REPLAY_MODE = -20,
  /**
   * Missing Filter for a session.
   */
  ROCPROFILER_STATUS_ERROR_SESSION_MISSING_FILTER = -21,
  /**
   * The size given for the buffer is not applicable.
   */
  ROCPROFILER_STATUS_ERROR_INCORRECT_SIZE = -22,
  /**
   * Incorrect Flush interval.
   */
  ROCPROFILER_STATUS_ERROR_INCORRECT_FLUSH_INTERVAL = -23,
  /**
   * The session filter can't accept the given data.
   */
  ROCPROFILER_STATUS_ERROR_SESSION_FILTER_DATA_MISMATCH = -24,
  /**
   * The given filter data is corrupted.
   */
  ROCPROFILER_STATUS_ERROR_FILTER_DATA_CORRUPTED = -25,
  /**
   * The given label is corrupted.
   */
  ROCPROFILER_STATUS_ERROR_CORRUPTED_LABEL_DATA = -26,
  /**
   * There is no label in the labels stack to be popped.
   */
  ROCPROFILER_STATUS_ERROR_RANGE_STACK_IS_EMPTY = -27,
  /**
   * There is no pass that started.
   */
  ROCPROFILER_STATUS_ERROR_PASS_NOT_STARTED = -28,
  /**
   * There is already Active session, Can't activate two session at the same
   * time
   */
  ROCPROFILER_STATUS_ERROR_HAS_ACTIVE_SESSION = -29,
  /**
   * Can't terminate a non active session
   */
  ROCPROFILER_STATUS_ERROR_SESSION_NOT_ACTIVE = -30,
  /**
   * The required filter is not found for the given session
   */
  ROCPROFILER_STATUS_ERROR_FILTER_NOT_FOUND = -31,
  /**
   * The required buffer is not found for the given session
   */
  ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND = -32,
  /**
   * The required Filter is not supported
   */
  ROCPROFILER_STATUS_ERROR_FILTER_NOT_SUPPORTED = -33
} rocprofiler_status_t;

/**
 * Query the textual description of the given error for the current thread.
 *
 * Returns a NULL terminated string describing the error of the given ROCProfiler
 * API call by the calling thread that did not return success.
 *
 * @retval Return the error string.
 */
ROCPROFILER_API const char* rocprofiler_error_str(rocprofiler_status_t status) ROCPROFILER_VERSION_9_0;

/** @} */

/** \defgroup rocprofiler_general_group General ROCProfiler Requirements
 * @{
 */

// TODO(aelwazir): More clear description, (think about nested!!??)

/**
 * Initialize the API Tools
 *
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_API_ALREADY_INITIALIZED If initialize
 * wasn't called or finalized called twice
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_initialize() ROCPROFILER_VERSION_9_0;

/**
 * Finalize the API Tools
 *
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_API_NOT_INITIALIZED If initialize wasn't
 * called or finalized called twice
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_finalize() ROCPROFILER_VERSION_9_0;

/**
 * \addtogroup sessions_handling_group
 * @{
 * ROCProfiler Session Modes.
 */

/**
 * Session Identifier
 */
typedef struct {
  /**
   * Session Identifier to get the session or to be used to call any API that
   * needs to deal with a specific session
   */
  uint64_t handle;
} rocprofiler_session_id_t;

/** @} */

/** @} */

/** \defgroup timestamp_group Timestamp Operations
 *
 * For this group we are focusing on timestamps collection and timestamp
 * definition
 *
 * @{
 */

/**
 * ROCProfiling Timestamp Type.
 */
typedef struct {
  uint64_t value;
} rocprofiler_timestamp_t;

/**
 * Get the system clock timestamp.
 *
 * @param[out] timestamp The system clock timestamp in nano seconds.
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_TIMESTAMP_NOT_APPLICABLE The function
 * failed to get the timestamp using HSA Function.
 *
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_get_timestamp(rocprofiler_timestamp_t* timestamp) ROCPROFILER_VERSION_9_0;

/**
 * Timestamps (start & end), it will be used for kernel dispatch tracing as
 * well as API Tracing
 */
typedef struct {
  rocprofiler_timestamp_t begin;
  rocprofiler_timestamp_t end;
} rocprofiler_record_header_timestamp_t;

/** @} */

/** \defgroup generic_record_group General Records Type
 * @{
 */

/**
 * A unique identifier for every record
 */
typedef struct {
  /**
   * Record ID handle
   */
  uint64_t handle;
} rocprofiler_record_id_t;

/**
 * Record kind
 */
typedef enum {
  /**
   * Represents records that have profiling data (ex. counter collection
   * records)
   */
  ROCPROFILER_PROFILER_RECORD = 0,
  /**
   * Represents records that have tracing data (ex. hip api tracing records)
   */
  ROCPROFILER_TRACER_RECORD = 1,
  /**
   * Represents a ATT tracing record (Not available yet)
   */
  ROCPROFILER_ATT_TRACER_RECORD = 2,
  /**
   * Represents a PC sampling record
   */
  ROCPROFILER_PC_SAMPLING_RECORD = 3,
  /**
   * Represents SPM records
   */
  ROCPROFILER_SPM_RECORD = 4,
  /**
   * Represents Counters sampler records
   */
  ROCPROFILER_COUNTERS_SAMPLER_RECORD = 5
} rocprofiler_record_kind_t;

/**
 * Generic ROCProfiler record header.
 */
typedef struct {
  /**
   * Represents the kind of the record using ::rocprofiler_record_kind_t
   */
  rocprofiler_record_kind_t kind;
  /**
   * Represents the id of the record
   */
  rocprofiler_record_id_t id;
} rocprofiler_record_header_t;

/** \defgroup record_agents_group Agents(AMD CPU/GPU) Handling
 * \ingroup generic_record_group
 * @{
 */

/**
 * Agent ID handle, which represents a unique id to the agent reported as it
 * can be used to retrieve Agent information using
 * ::rocprofiler_query_agent_info, Agents can be CPUs or GPUs
 */
typedef struct {
  /**
   * a unique id to represent every agent on the system, this handle should be
   * unique across all nodes in multi-node system
   */
  uint64_t handle;  // Topology folder serial number
} rocprofiler_agent_id_t;

/**
 * Using ::rocprofiler_query_agent_info, user can determine the type of the agent
 * the following struct will be the output in case of retrieving
 * ::ROCPROFILER_AGENT_TYPE agent info
 */
typedef enum {
  /**
   * CPU Agent
   */
  ROCPROFILER_CPU_AGENT = 0,
  /**
   * GPU Agent
   */
  ROCPROFILER_GPU_AGENT = 1
} rocprofiler_agent_type_t;

// TODO(aelwazir): check if we need to report the family name as well!!?? OR
// return the agent itself so that they can use HSA API
/**
 * Types of information that can be requested about the Agents
 */
typedef enum {
  /**
   * GPU Agent Name
   */
  ROCPROFILER_AGENT_NAME = 0,
  /**
   * GPU Agent Type
   */
  ROCPROFILER_AGENT_TYPE = 1
} rocprofiler_agent_info_kind_t;

/**
 * Query Agent Information size to allow the user to allocate the right size
 * for the information data requested, the information will be collected using
 * ::rocprofiler_agent_id_t to identify one type of information available in
 * ::rocprofiler_agent_info_t
 *
 * @param[in] kind Information kind requested by the user
 * @param[in] agent_id Agent ID
 * @param[out] data_size Size of the information data output
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND, if the agent was not found
 * in the saved agents
 * @retval ::ROCPROFILER_STATUS_ERROR_AGENT_INFORMATION_MISSING, if the agent
 * was found in the saved agents but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_agent_info_size(rocprofiler_agent_info_kind_t kind,
                                                                 rocprofiler_agent_id_t agent_id,
                                                                 size_t* data_size) ROCPROFILER_VERSION_9_0;

/**
 * Query Agent Information Data using an allocated data pointer by the user,
 * user can get the size of the data using ::rocprofiler_query_agent_info_size,
 * the user can get the data using ::rocprofiler_agent_id_t and the user need to
 * identify one type of information available in ::rocprofiler_agent_info_t
 *
 * @param[in] kind Information kind requested by the user
 * @param[in] agent_id Agent ID
 * @param[out] data_size Size of the information data output
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND, if the agent was not found
 * in the saved agents
 * @retval ::ROCPROFILER_STATUS_ERROR_AGENT_INFORMATION_MISSING, if the agent
 * was found in the saved agents but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_agent_info(rocprofiler_agent_info_kind_t kind,
                                                            rocprofiler_agent_id_t descriptor,
                                                            const char** name) ROCPROFILER_VERSION_9_0;

/** @} */

/** \defgroup record_queues_group Queues(AMD HSA QUEUES) Handling
 * \ingroup generic_record_group
 * @{
 */

/**
 * Unique ID handle to represent an HSA Queue of type \p hsa_queue_t, this id
 * can be used by the user to get queue information using
 * ::rocprofiler_query_queue_info
 */
typedef struct {
  /**
   * Unique Id for every queue for one agent for one system
   */
  uint64_t handle;
} rocprofiler_queue_id_t;

// TODO(aelwazir): Check if there is anymore Queue Information needed
/**
 * Types of information that can be requested about the Queues
 */
typedef enum {
  /**
   * AMD HSA Queue Size.
   */
  ROCPROFILER_QUEUE_SIZE = 0
} rocprofiler_queue_info_kind_t;

/**
 * Query Queue Information size to allow the user to allocate the right size
 * for the information data requested, the information will be collected using
 * ::rocprofiler_queue_id_t by using ::rocprofiler_query_queue_info and the user
 * need to identify one type of information available in
 * ::rocprofiler_queue_info_t
 *
 * @param[in] kind Information kind requested by the user
 * @param[in] agent_id Queue ID
 * @param[out] data_size Size of the information data output
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_QUEUE_NOT_FOUND, if the queue was not found
 * in the saved agents
 * @retval ::ROCPROFILER_STATUS_ERROR_QUEUE_INFORMATION_MISSING, if the queue
 * was found in the saved queues but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_queue_info_size(rocprofiler_queue_info_kind_t kind,
                                                                 rocprofiler_queue_id_t agent_id,
                                                                 size_t* data_size) ROCPROFILER_VERSION_9_0;

/**
 * Query Queue Information Data using an allocated data pointer by the user,
 * user can get the size of the data using ::rocprofiler_query_queue_info_size,
 * the user can get the data using ::rocprofiler_queue_id_t and the user need to
 * identify one type of information available in ::rocprofiler_queue_info_t
 *
 * @param[in] kind Information kind requested by the user
 * @param[in] agent_id Queue ID
 * @param[out] data_size Size of the information data output
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_QUEUE_NOT_FOUND, if the queue was not found
 * in the saved agents
 * @retval ::ROCPROFILER_STATUS_ERROR_QUEUE_INFORMATION_MISSING, if the queue
 * was found in the saved agents but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_queue_info(rocprofiler_queue_info_kind_t kind,
                                                            rocprofiler_queue_id_t descriptor,
                                                            const char** name) ROCPROFILER_VERSION_9_0;

/** @} */

/** \defgroup record_kernels_group Kernels Handling
 * \ingroup generic_record_group
 * @{
 */

/**
 * Kernel identifier that represent a unique id for every kernel
 */
typedef struct {
  /**
   * Kernel object identifier
   */
  uint64_t handle;
} rocprofiler_kernel_id_t;

/**
 * Kernel Information Types, can be used by ::rocprofiler_query_kernel_info
 */
typedef enum {
  /**
   * Kernel Name Information Type
   */
  ROCPROFILER_KERNEL_NAME = 0
} rocprofiler_kernel_info_kind_t;

/**
 * Query Kernel Information Data size to allow the user to allocate the right
 * size for the information data requested, the information will be collected
 * using
 * ::rocprofiler_kernel_id_t by using ::rocprofiler_query_kernel_info and the
 * user need to identify one type of information available in
 * ::rocprofiler_kernel_info_t
 *
 * @param[in] kernel_info_type The tyoe of information needed
 * @param[in] kernel_id Kernel ID
 * @param[out] data_size Kernel Information Data size
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_KERNEL_NOT_FOUND, if the kernel was not
 * found in the saved kernels
 * @retval ::ROCPROFILER_STATUS_ERROR_KERNEL_INFORMATION_MISSING, if the kernel
 * was found in the saved counters but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_kernel_info_size(rocprofiler_kernel_info_kind_t kind,
                                                                  rocprofiler_kernel_id_t kernel_id,
                                                                  size_t* data_size) ROCPROFILER_VERSION_9_0;

/**
 * Query Kernel Information Data using an allocated data pointer by the user,
 * user can get the size of the data using ::rocprofiler_query_kernel_info_size,
 * the user can get the data using ::rocprofiler_kernel_id_t and the user need
 * to identify one type of information available in ::rocprofiler_kernel_info_t
 *
 * @param[in] kind Information kind requested by the user
 * @param[in] kernel_id Kernel ID
 * @param[out] data Information Data
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_KERNEL_NOT_FOUND, if the kernel was not
 * found in the saved kernels
 * @retval ::ROCPROFILER_STATUS_ERROR_KERNEL_INFORMATION_MISSING, if the kernel
 * was found in the saved kernels but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_kernel_info(rocprofiler_kernel_info_kind_t kind,
                                                             rocprofiler_kernel_id_t kernel_id,
                                                             const char** data) ROCPROFILER_VERSION_9_0;

/** @} */

/**
 * Holds the thread id
 */
typedef struct {
  /**
   * Thread ID
   */
  uint32_t value;
} rocprofiler_thread_id_t;

/** @} */

/** \defgroup profiling_api_group Profiling Part Handling
 *
 * The profiling records are asynchronously logged to the pool and can be
 * associated with the respective GPU kernels.
 * Profiling API can be used to enable collecting of the records with or
 * without timestamping data for the GPU Application in continuous mode or
 * kernel mode.
 *
 * @{
 */

/** \defgroup profiling_api_counters_group Counter Collection Handling
 * records
 * \ingroup profiling_api_group
 * @{
 */

typedef struct {
  const char* name;
  const char* description;
  const char* expression;
  uint32_t instances_count;
  const char* block_name;
  uint32_t block_counters;
} rocprofiler_counter_info_t;

typedef int (*rocprofiler_counters_info_callback_t)(rocprofiler_counter_info_t counter,
                                                  const char* gpu_name, uint32_t gpu_index) ROCPROFILER_VERSION_9_0;

ROCPROFILER_API rocprofiler_status_t
rocprofiler_iterate_counters(rocprofiler_counters_info_callback_t counters_info_callback) ROCPROFILER_VERSION_9_0;

/**
 * Counter ID to be used to query counter information using
 * ::rocprofiler_query_counter_info
 */
typedef struct {
  /**
   * A unique id generated for every counter requested by the user
   */
  uint64_t handle;
} rocprofiler_counter_id_t;

/**
 * Counter Information Types, can be used by ::rocprofiler_query_counter_info
 */
typedef enum {
  /**
   * Can be used to get the counter name
   */
  ROCPROFILER_COUNTER_NAME = 0,
  /**
   * Can be used to get the block id of a counter
   */
  ROCPROFILER_COUNTER_BLOCK_ID = 2,
  /**
   * This is the level of hierarchy from the GFX_IP where the counter value
   * should be collected
   */
  ROCPROFILER_COUNTER_HIERARCHY_LEVEL = 3
} rocprofiler_counter_info_kind_t;

/**
 * Query Counter Information Data size to allow the user to allocate the right
 * size for the information data requested, the information will be collected
 * using
 * ::rocprofiler_counter_id_t by using ::rocprofiler_query_counter_info and the
 * user need to identify one type of information available in
 * ::rocprofiler_counter_info_t
 *
 * @param[in] session_id Session id where this data was collected
 * @param[in] counter_info_type The tyoe of information needed
 * @param[in] counter_id Counter ID
 * @param[out] data_size Counter Information Data size
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND, if the counter was not
 * found in the saved counters
 * @retval ::ROCPROFILER_STATUS_ERROR_COUNTER_INFORMATION_MISSING, if the counter
 * was found in the saved counters but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_counter_info_size(
    rocprofiler_session_id_t session_id, rocprofiler_counter_info_kind_t counter_info_type,
    rocprofiler_counter_id_t counter_id, size_t* data_size) ROCPROFILER_VERSION_9_0;

/**
 * Query Counter Information Data using an allocated data pointer by the user,
 * user can get the size of the data using ::rocprofiler_query_counter_info_size,
 * the user can get the data using ::rocprofiler_counter_id_t and the user need
 * to identify one type of information available in ::rocprofiler_counter_info_t
 *
 * @param[in] session_id Session id where this data was collected
 * @param[in] kind Information kind requested by the user
 * @param[in] counter_id Counter ID
 * @param[out] data Information Data
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND, if the counter was not
 * found in the saved counters
 * @retval ::ROCPROFILER_STATUS_ERROR_COUNTER_INFORMATION_MISSING, if the counter
 * was found in the saved counters but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_counter_info(rocprofiler_session_id_t session_id,
                                                              rocprofiler_counter_info_kind_t kind,
                                                              rocprofiler_counter_id_t counter_id,
                                                              const char** data) ROCPROFILER_VERSION_9_0;

typedef struct {
  /**
   * queue index value
   */
  uint64_t value;
} rocprofiler_queue_index_t;

// TODO(aelwazir): add more types to the values should we use unions??!!
/**
 * Counter Value Structure
 */
typedef struct {
  /**
   * Counter value
   */
  double value;
} rocprofiler_record_counter_value_t;

/**
 * Counter Instance Structure, it will represent every counter reported in the
 * array of counters reported by every profiler record if counters were needed
 * to be collected
 */
typedef struct {
  /**
   * Counter Instance Identifier
   */
  rocprofiler_counter_id_t counter_handler;  // Counter Handler
  /**
   * Counter Instance Value
   */
  rocprofiler_record_counter_value_t value;  // Counter Value
} rocprofiler_record_counter_instance_t;

/**
 * Counters Instances Count Structure, every profiling record has this
 * structure included to report the number of counters collected for this
 * kernel dispatch
 */
typedef struct {
  /**
   * Counters Instances Count for every record
   */
  uint64_t value;
} rocprofiler_record_counters_instances_count_t;

/**
 * Kernel properties, this will represent the kernel properties
 * such as its grid size, workgroup size, wave_size
 */

typedef struct {
  /**
   * Grid Size
   */
  uint64_t grid_size;
  /**
   * workgroup size
   */
  uint64_t workgroup_size;
  /**
   * lds_size
   */
  uint64_t lds_size;
  /**
   * scratch_size
   */
  uint64_t scratch_size;
  /**
   * arch vgpr count
   */
  uint64_t arch_vgpr_count;
  /**
   * accum vgpr count
   */
  uint64_t accum_vgpr_count;
  /**
   * sgpr_count
   */
  uint64_t sgpr_count;
  /**
   * wave size
   */
  uint64_t wave_size;
  /**
   * Dispatch completion signal handle
   */
  uint64_t signal_handle;

} rocprofiler_kernel_properties_t;
/**
 * Profiling record, this will represent all the information reported by the
 * profiler regarding kernel dispatches and their counters that were collected
 * by the profiler and requested by the user, this can be used as the type of
 * the flushed records that is reported to the user using
 * ::rocprofiler_buffer_callback_t
 */
typedef struct {
  /**
   * ROCProfiler General Record base header to identify the id and kind of every
   * record
   */
  rocprofiler_record_header_t header;
  /**
   * Kernel Identifier to be used by the user to get the kernel info using
   * ::rocprofiler_query_kernel_info
   */
  rocprofiler_kernel_id_t kernel_id;
  /**
   * Agent Identifier to be used by the user to get the Agent Information using
   * ::rocprofiler_query_agent_info
   */
  rocprofiler_agent_id_t gpu_id;
  /**
   * Queue Identifier to be used by the user to get the Queue Information using
   * ::rocprofiler_query_agent_info
   */
  rocprofiler_queue_id_t queue_id;
  /**
   * Timestamps, start and end timestamps of the record data (ex. Kernel
   * Dispatches)
   */
  rocprofiler_record_header_timestamp_t timestamps;
  /**
   * Counters, including identifiers to get counter information and Counters
   * values
   */
  rocprofiler_record_counter_instance_t* counters;
  /**
   * kernel properties, including the grid size, work group size,
   * registers count, wave size and completion signal
   */
  rocprofiler_kernel_properties_t kernel_properties;
  /**
   * Thread id
   */
  rocprofiler_thread_id_t thread_id;
  /**
   * Queue Index - packet index in the queue
   */
  rocprofiler_queue_index_t queue_idx;
  /**
   * The count of the counters that were collected by the profiler
   */
  rocprofiler_record_counters_instances_count_t counters_count; /* Counters Count */
} rocprofiler_record_profiler_t;

typedef struct {
  uint32_t value;

} rocprofiler_event_id_t;

typedef struct {
  uint16_t value;  // Counter Value

} rocprofiler_record_spm_counters_instances_count_t;

/**
 * Counters, including identifiers to get counter information and Counters
 * values
 */
typedef struct {
  rocprofiler_record_spm_counters_instances_count_t counters_data[32];

} rocprofiler_record_se_spm_data_t;


/**
 * SPM record, this will represent all the information reported by the
 * SPM regarding counters and their timestamps this can be used as the type of
 * the flushed records that is reported to the user using
 * ::rocprofiler_buffer_callback_t
 */
typedef struct {
  /**
   * ROCProfiler General Record base header to identify the id and kind of every
   * record
   */
  rocprofiler_record_header_t header;

  /**
   * Timestamps at which the counters were sampled.
   */
  rocprofiler_record_header_timestamp_t timestamps;
  /**
   * Counter values per shader engine
   */
  rocprofiler_record_se_spm_data_t shader_engine_data[4];

} rocprofiler_record_spm_t;

/**
 * struct to store the trace data from a shader engine.
 */
typedef struct {
  void* buffer_ptr;
  uint32_t buffer_size;
} rocprofiler_record_se_att_data_t;

 /**
 * ATT tracing record structure.
 * This will represent all the information reported by the
 * ATT tracer such as the kernel and its thread trace data.
 * This record can be flushed to the user using
 * ::rocprofiler_buffer_callback_t
 */
typedef struct {
  /**
   * ROCProfiler General Record base header to identify the id and kind of every
   * record
   */
  rocprofiler_record_header_t header;
  /**
   * Kernel Identifier to be used by the user to get the kernel info using
   * ::rocprofiler_query_kernel_info
   */
  rocprofiler_kernel_id_t kernel_id;
  /**
   * Agent Identifier to be used by the user to get the Agent Information using
   * ::rocprofiler_query_agent_info
   */
  rocprofiler_agent_id_t gpu_id;
  /**
   * Queue Identifier to be used by the user to get the Queue Information using
   * ::rocprofiler_query_agent_info
   */
  rocprofiler_queue_id_t queue_id;
  /**
   * kernel properties, including the grid size, work group size,
   * registers count, wave size and completion signal
   */
  rocprofiler_kernel_properties_t kernel_properties;
  /**
   * Thread id
   */
  rocprofiler_thread_id_t thread_id;
  /**
   * Queue Index - packet index in the queue
   */
  rocprofiler_queue_index_t queue_idx;
  /**
   * ATT data output from each shader engine.
   */
  rocprofiler_record_se_att_data_t* shader_engine_data;
  /**
   * The count of the shader engine ATT data
   */
  uint64_t shader_engine_data_count;
} rocprofiler_record_att_tracer_t;



/** @} */

/** \defgroup tracing_api_group Tracer Part Handling
 * @{
 */

/**
 * Traced API domains
 */
typedef enum {
  /**
   * HSA API domain
   */
  ACTIVITY_DOMAIN_HSA_API = 0,
  /**
   * HSA async activity domain
   */
  ACTIVITY_DOMAIN_HSA_OPS = 1,
  /**
   * HIP async activity domain
   */
  ACTIVITY_DOMAIN_HIP_OPS = 2,
  /**
   * HIP API domain
   */
  ACTIVITY_DOMAIN_HIP_API = 3,
  /**
   * KFD API domain
   */
  ACTIVITY_DOMAIN_KFD_API = 4,
  /**
   * External ID domain
   */
  ACTIVITY_DOMAIN_EXT_API = 5,
  /**
   * ROCTX domain
   */
  ACTIVITY_DOMAIN_ROCTX = 6,
  // TODO(aelwazir): Used in kernel Info, memcpy, ..etc, refer to hsa_support
  // TODO(aelwazir): Move HSA Events to hsa_support
  /**
   * HSA events (Device Activity)
   */
  ACTIVITY_DOMAIN_HSA_EVT = 7,
  ACTIVITY_DOMAIN_NUMBER
} rocprofiler_tracer_activity_domain_t;

/**
 * Tracing Operation ID for HIP/HSA
 */
typedef struct {
  uint32_t id;
} rocprofiler_tracer_operation_id_t;

/**
 * Correlation identifier
 */
typedef struct {
  /**
   * Correlation ID Value
   */
  uint64_t value;
} rocprofiler_tracer_activity_correlation_id_t;

/**
 * Tracer API Calls Data Handler
 */
typedef struct {
  /**
   * Data Handler Identifier
   */
  const void* handle;
  /**
   * API Data Size
   */
  size_t size;
} rocprofiler_tracer_api_data_handle_t;

/** \defgroup roctx_tracer_api_data_group Tracer ROCTX API Data
 * \ingroup tracing_api_group
 * @{
 */

/**
 * ROCTX Tracer Data Information Kinds
 */
typedef enum {
  /**
   * ROCTX Tracer Data kind that can be used to return ROCTX message
   */
  ROCPROFILER_ROCTX_MESSAGE = 0,
  /**
   * ROCTX Tracer Data kind that can be used to return ROCTX id
   */
  ROCPROFILER_ROCTX_ID = 1
} rocprofiler_tracer_roctx_api_data_info_t;

/**
 * Query Tracer API Call Data Information size to allow the user to allocate
 * the right size for the information data requested, the information will be
 * collected using
 * ::rocprofiler_tracer_api_data_id_t by using
 * ::rocprofiler_query_tracer_api_data_info and the user need to identify one
 * type of information available in
 * ::rocprofiler_query_tracer_api_data_info
 *
 * @param[in] session_id Session id where this data was collected
 * @param[in] kind The tyoe of information needed
 * @param[in] api_data_id API Data ID
 * @param[in] operation_id API Operation ID
 * @param[out] data_size API Data Information size
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND, if the api data
 * was not found in the saved api data
 * @retval ::ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING, if the
 * api data was found in the saved data but the required information is
 * missing
 * @retval ::ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN if the user sent a handle
 * that is not related to the requested domain
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_roctx_tracer_api_data_info_size(
    rocprofiler_session_id_t session_id, rocprofiler_tracer_roctx_api_data_info_t kind,
    rocprofiler_tracer_api_data_handle_t api_data_id, rocprofiler_tracer_operation_id_t operation_id,
    size_t* data_size) ROCPROFILER_VERSION_9_0;

/**
 * Query API Data Information  using an allocated data pointer by the user,
 * user can get the size of the data using
 * ::rocprofiler_query_tracer_api_data_info_length, the user can get the data
 * using ::rocprofiler_tracer_api_data_id_t and the user need to identify one
 * type of information available in ::rocprofiler_tracer_api_data_info_t
 *
 * @param[in] session_id Session id where this data was collected
 * @param[in] kind Information kind requested by the user
 * @param[in] api_data_id API Data ID
 * @param[in] operation_id API Operation ID
 * @param[out] data API Data Data
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND, if the api data
 * was not found in the saved api data
 * @retval ::ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING, if the
 * api data was found in the saved data but the required information is
 * missing
 * @retval ::ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN if the user sent a handle
 * that is not related to the requested domain
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_roctx_tracer_api_data_info(
    rocprofiler_session_id_t session_id, rocprofiler_tracer_roctx_api_data_info_t kind,
    rocprofiler_tracer_api_data_handle_t api_data_id, rocprofiler_tracer_operation_id_t operation_id,
    char** data) ROCPROFILER_VERSION_9_0;

/** @} */

/** \defgroup hsa_tracer_api_data_group Tracer HSA API Data
 * \ingroup tracing_api_group
 * @{
 */

/**
 * hsa Tracer Data Information Kinds
 */
typedef enum {
  /**
   * HSA Tracer Data kind that can be used to return to a pointer to all the
   * API Call Data
   */
  ROCPROFILER_HSA_FUNCTION_NAME = 0,
  /**
   * HSA API Data in string format.
   */
  ROCPROFILER_HSA_API_DATA_STR = 1,
  /**
   * HSA Activity Name
   */
  ROCPROFILER_HSA_ACTIVITY_NAME = 2,
  /**
   * HSA Data
   * User has to reinterpret_cast to hsa_api_data_t*
   */
  ROCPROFILER_HSA_API_DATA = 3
} rocprofiler_tracer_hsa_api_data_info_t;

/**
 * Query Tracer API Call Data Information size to allow the user to allocate
 * the right size for the information data requested, the information will be
 * collected using
 * ::rocprofiler_tracer_api_data_id_t by using
 * ::rocprofiler_query_tracer_api_data_info and the user need to identify one
 * type of information available in
 * ::rocprofiler_query_tracer_api_data_info
 *
 * @param[in] session_id Session id where this data was collected
 * @param[in] kind The tyoe of information needed
 * @param[in] api_data_id API Data ID
 * @param[in] operation_id API Operation ID
 * @param[out] data_size API Data Information size
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND, if the api data
 * was not found in the saved api data
 * @retval ::ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING, if the
 * api data was found in the saved data but the required information is
 * missing
 * @retval ::ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN if the user sent a handle
 * that is not related to the requested domain
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_hsa_tracer_api_data_info_size(
    rocprofiler_session_id_t session_id, rocprofiler_tracer_hsa_api_data_info_t kind,
    rocprofiler_tracer_api_data_handle_t api_data_id, rocprofiler_tracer_operation_id_t operation_id,
    size_t* data_size) ROCPROFILER_VERSION_9_0;

/**
 * Query API Data Information  using an allocated data pointer by the user,
 * user can get the size of the data using
 * ::rocprofiler_query_tracer_api_data_info_length, the user can get the data
 * using ::rocprofiler_tracer_api_data_id_t and the user need to identify one
 * type of information available in ::rocprofiler_tracer_api_data_info_t
 *
 * @param[in] session_id Session id where this data was collected
 * @param[in] kind Information kind requested by the user
 * @param[in] api_data_id API Data ID
 * @param[in] operation_id API Operation ID
 * @param[out] data API Data Data
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND, if the api data
 * was not found in the saved api data
 * @retval ::ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING, if the
 * api data was found in the saved data but the required information is
 * missing
 * @retval ::ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN if the user sent a handle
 * that is not related to the requested domain
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_hsa_tracer_api_data_info(
    rocprofiler_session_id_t session_id, rocprofiler_tracer_hsa_api_data_info_t kind,
    rocprofiler_tracer_api_data_handle_t api_data_id, rocprofiler_tracer_operation_id_t operation_id,
    char** data) ROCPROFILER_VERSION_9_0;

/** @} */

/** \defgroup hip_tracer_api_data_group Tracer HIP API Data
 * \ingroup tracing_api_group
 * @{
 */

/**
 * hip Tracer Data Information Kinds
 */
typedef enum {
  // TODO(aelwazir): Get the data from hip_api_data_t
  /**
   * hip Tracer Data kind that can be used to return to a pointer to all the
   * API Call Data
   */
  ROCPROFILER_HIP_FUNCTION_NAME = 0,
  /**
   * Only available for HIP Functions that lead to kernel launch to get the
   * kernel name
   */
  ROCPROFILER_HIP_KERNEL_NAME = 1,
  /**
   * Only available to hip calls that has memory copy operation with source
   * available
   */
  ROCPROFILER_HIP_MEM_COPY_SRC = 2,
  /**
   * Only available to hip calls that has memory copy operation with
   * destination available
   */
  ROCPROFILER_HIP_MEM_COPY_DST = 3,
  /**
   * Only available to hip calls that has memory copy operation with data size
   * available
   */
  ROCPROFILER_HIP_MEM_COPY_SIZE = 4,
  /**
   * Reporting the whole API data as one string
   */
  ROCPROFILER_HIP_API_DATA_STR = 5,
  /**
   * HIP Activity Name
   */
  ROCPROFILER_HIP_ACTIVITY_NAME = 6,
  /**
   * Stream ID
   */
  ROCPROFILER_HIP_STREAM_ID = 7,
  /**
   * HIP API Data
   * User has to reinterpret_cast to hip_api_data_t*
   */
  ROCPROFILER_HIP_API_DATA = 8
} rocprofiler_tracer_hip_api_data_info_t;

/**
 * Query Tracer API Call Data Information size to allow the user to allocate
 * the right size for the information data requested, the information will be
 * collected using
 * ::rocprofiler_tracer_api_data_id_t by using
 * ::rocprofiler_query_tracer_api_data_info and the user need to identify one
 * type of information available in
 * ::rocprofiler_query_tracer_api_data_info
 *
 * @param[in] session_id Session id where this data was collected
 * @param[in] kind The tyoe of information needed
 * @param[in] api_data_id API Data ID
 * @param[out] data_size API Data Information size
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND, if the api data
 * was not found in the saved api data
 * @retval ::ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING, if the
 * api data was found in the saved data but the required information is
 * missing
 * @retval ::ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN if the user sent a handle
 * that is not related to the requested domain
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_hip_tracer_api_data_info_size(
    rocprofiler_session_id_t session_id, rocprofiler_tracer_hip_api_data_info_t kind,
    rocprofiler_tracer_api_data_handle_t api_data_id, rocprofiler_tracer_operation_id_t operation_id,
    size_t* data_size) ROCPROFILER_VERSION_9_0;

/**
 * Query API Data Information  using an allocated data pointer by the user,
 * user can get the size of the data using
 * ::rocprofiler_query_tracer_api_data_info_length, the user can get the data
 * using ::rocprofiler_tracer_api_data_id_t and the user need to identify one
 * type of information available in ::rocprofiler_tracer_api_data_info_t
 *
 * @param[in] session_id Session id where this data was collected
 * @param[in] kind Information kind requested by the user
 * @param[in] api_data_id API Data ID
 * @param[out] data API Data Data
 * @retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND, if the api data
 * was not found in the saved api data
 * @retval ::ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING, if the
 * api data was found in the saved data but the required information is
 * missing
 * @retval ::ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN if the user sent a handle
 * that is not related to the requested domain
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_hip_tracer_api_data_info(
    rocprofiler_session_id_t session_id, rocprofiler_tracer_hip_api_data_info_t kind,
    rocprofiler_tracer_api_data_handle_t api_data_id, rocprofiler_tracer_operation_id_t operation_id,
    char** data) ROCPROFILER_VERSION_9_0;

/** @} */

/**
 * Tracing external ID
 */
typedef struct {
  uint64_t id;
} rocprofiler_tracer_external_id_t;

/**
 * Tracing record, this will represent all the information reported by the
 * tracer regarding APIs and their data that were traced and collected
 * by the tracer and requested by the user, this can be used as the type of
 * the flushed records that is reported to the user using
 * ::rocprofiler_buffer_async_callback_t
 */
typedef struct {
  /**
   * ROCProfiler General Record base header to identify the id and kind of every
   * record
   */
  rocprofiler_record_header_t header;
  /**
   * Tracing external ID
   */
  rocprofiler_tracer_external_id_t external_id;
  /**
   * Activity domain id, represents the type of the APIs that are being traced
   */
  rocprofiler_tracer_activity_domain_t domain;
  /**
   * Tracing Operation ID for HIP/HSA
   */
  rocprofiler_tracer_operation_id_t operation_id;
  /**
   * API Data Handler to be used by
   * ::rocprofiler_query_roctx_tracer_api_data_info or
   * ::rocprofiler_query_hsa_tracer_api_data_info or
   * ::rocprofiler_query_hip_tracer_api_data_info depending on the domain type
   */
  rocprofiler_tracer_api_data_handle_t api_data_handle;
  /**
   * Activity correlation ID
   */
  rocprofiler_tracer_activity_correlation_id_t correlation_id;
  /**
   * Timestamps
   */
  rocprofiler_record_header_timestamp_t timestamps;
  /**
   * Agent identifier that can be used as a handler in
   * ::rocprofiler_query_agent_info
   */
  rocprofiler_agent_id_t agent_id;
  /**
   * Queue identifier that can be used as a handler in
   * ::rocprofiler_query_queue_info
   */
  rocprofiler_queue_id_t queue_id;
  /**
   * Thread id
   */
  rocprofiler_thread_id_t thread_id;
} rocprofiler_record_tracer_t;

/**
 * Kernel dispatch correlation ID, unique across all dispatches
 */
typedef struct {
  uint64_t value;
} rocprofiler_kernel_dispatch_id_t;

/**
 * An individual PC sample
 */
typedef struct {
  /**
   * Kernel dispatch ID.  This is used by PC sampling to associate samples with
   * individual dispatches and is unrelated to any user-supplied correlation ID
   */
  rocprofiler_kernel_dispatch_id_t dispatch_id;
  union {
    /**
     * Host timestamp
     */
    rocprofiler_timestamp_t timestamp;
    /**
     * GPU clock counter (not currently used)
     */
    uint64_t cycle;
  };
  /**
   * Sampled program counter
   */
  uint64_t pc;
  /**
   * Sampled shader element
   */
  uint32_t se;
  /**
   * Sampled GPU agent
   */
  rocprofiler_agent_id_t gpu_id;
} rocprofiler_pc_sample_t;

/**
 * PC sample record: contains the program counter/instruction pointer observed
 * during periodic sampling of a kernel
 */
typedef struct {
  /**
   * ROCProfiler General Record base header to identify the id and kind of every
   * record
   */
  rocprofiler_record_header_t header;
  /**
   * PC sample data
   */
  rocprofiler_pc_sample_t pc_sample;
} rocprofiler_record_pc_sample_t;

/** @} */

/** \defgroup memory_storage_buffer_group Memory Storage Buffer
 * Sessions
 *
 * In this group, Memory Pools and their types will be discussed.
 * @{
 */

/**
 * Buffer Property Options
 */
typedef enum {
  /**
   * Flush interval
   */
  ROCPROFILER_BUFFER_PROPERTY_KIND_INTERVAL_FLUSH = 0,
  // Periodic Flush
  // Size
  // Think of using the kind as an end of the array!!??
} rocprofiler_buffer_property_kind_t;

typedef struct {
  rocprofiler_buffer_property_kind_t kind;
  uint64_t value;
} rocprofiler_buffer_property_t;

typedef struct {
  uint64_t value;
} rocprofiler_buffer_id_t;

typedef struct {
  uint64_t value;
} rocprofiler_filter_id_t;

/**
 * Memory pool buffer callback.
 * The callback that will be invoked when a memory pool buffer becomes full or
 * is flushed by the user or using flush thread that was initiated using the
 * flush interval set by the user ::rocprofiler_create_session.
 * The user needs to read the record header to identify the record kind and
 * depending on the kind they can reinterpret_cast to either
 * ::rocprofiler_record_profiler_t if the kind was ::ROCPROFILER_PROFILER_RECORD or
 * ::rocprofiler_record_tracer_t if the kind is ::ROCPROFILER_TRACER_RECORD
 *
 * @param[in] begin pointer to first entry in the buffer.
 * @param[in] end pointer to one past the end entry in the buffer.
 * @param[in] session_id The session id associated with that record
 * @param[in] buffer_id The buffer id associated with that record
 */
typedef void (*rocprofiler_buffer_callback_t)(const rocprofiler_record_header_t* begin,
                                            const rocprofiler_record_header_t* end,
                                            rocprofiler_session_id_t session_id,
                                            rocprofiler_buffer_id_t buffer_id);

/**
 * Flush specific Buffer
 *
 * @param[in] session_id The created session id
 * @param[in] buffer_id The buffer ID of the created filter group
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND may return if
 * the session is not found
 * @retval ::ROCPROFILER_STATUS_ERROR_CORRUPTED_SESSION_BUFFER may return if
 * the session buffer is corrupted
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_flush_data(rocprofiler_session_id_t session_id,
                                                      rocprofiler_buffer_id_t buffer_id) ROCPROFILER_VERSION_9_0;

/**
 * Get a pointer to the next profiling record.
 * A memory pool generates buffers that contain multiple profiling records.
 * This function steps to the next profiling record.
 *
 * @param[in] record Pointer to the current profiling record in a memory pool
 * buffer.
 * @param[out] next Pointer to the following profiling record in the memory
 * pool buffer.
 * @param[in] session_id Session ID
 * @param[in] buffer_id Buffer ID
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_RECORD_CORRUPTED if the function couldn't
 * get the next record because of corrupted data reported by the previous
 * record
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_next_record(const rocprofiler_record_header_t* record,
                                                       const rocprofiler_record_header_t** next,
                                                       rocprofiler_session_id_t session_id,
                                                       rocprofiler_buffer_id_t buffer_id) ROCPROFILER_VERSION_9_0;

/** @} */

/** \defgroup sessions_handling_group ROCProfiler Sessions
 * @{
 */

// TODO(aelwazir): Replay mode naming !!?? (If changed, reflect on start&stop)
/**
 * Replay Profiling Modes.
 */
typedef enum {
  /**
   * No Replay to be done, Mostly for tracing tool or if the user wants to make
   * sure that no replays will be done
   */
  ROCPROFILER_NONE_REPLAY_MODE = -1,
  /**
   * Replaying the whole application to get multi passes  (Not Yet Supported)
   */
  ROCPROFILER_APPLICATION_REPLAY_MODE = 0,
  /**
   * Replaying every kernel dispatch to get multi passes
   */
  ROCPROFILER_KERNEL_REPLAY_MODE = 1,
  /**
   * Replaying an user-specified range to get multi passes  (Not Yet Supported)
   */
  ROCPROFILER_USER_REPLAY_MODE = 2
} rocprofiler_replay_mode_t;

/**
 * Create Session
 * A ROCProfiler Session is having enough information about what needs to be
 * collected or traced and it allows the user to start/stop profiling/tracing
 * whenever required.
 * Session will hold multiple mode, that can be added using
 * ::rocprofiler_add_session_mode, it is required to add at least one session
 * mode, if it is tracing or profiling and using ::rocprofiler_session_set_filter
 * can set specific data that is required for the profiler or the tracer such
 * as the counters for profiling or the APIs for tracing before calling
 * ::rocprofiler_start_session, also
 * ::rocprofiler_session_set_filter can be used to set optional filters like
 * specific GPUs/Kernel Names/API Names and more. Session can be started using
 * ::rocprofiler_start_session and can be stopped using
 * ::rocprofiler_terminate_session
 *
 * @param[in] replay_mode The Replay strategy that should be used if replay is
 * needed
 * @param[out] session_id Pointer to the created session id, the session is
 * alive up till ::rocprofiler_destroy_session being called, however, the session
 * id can be
 * used while the session is active which can be activated using
 * ::rocprofiler_start_session and deactivated using
 * ::rocprofiler_terminate_session but ::rocprofiler_flush_data can use session_id
 * even if it is deactivated for flushing the saved records
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_create_session(rocprofiler_replay_mode_t replay_mode,
                                                          rocprofiler_session_id_t* session_id) ROCPROFILER_VERSION_9_0;

/**
 * Destroy Session
 * Destroy session created by ::rocprofiler_create_session, please refer to
 * the samples for how to use.
 * This marks the end of session and its own id life and none of the session
 * related functions will be available after this call.
 *
 * @param[in] session_id The created session id
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND may return if
 * the session is not found
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_destroy_session(rocprofiler_session_id_t session_id) ROCPROFILER_VERSION_9_0;

/** \defgroup session_filter_group Session Filters Handling
 * \ingroup sessions_handling_group
 * @{
 */

typedef enum {
  /**
   * Kernel Dispatch Timestamp collection.
   */
  ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION = 1,
  /**
   * GPU Application counter collection.
   */
  ROCPROFILER_COUNTERS_COLLECTION = 2,
  /**
   * PC Sampling collection. (Not Yet Supported)
   */
  ROCPROFILER_PC_SAMPLING_COLLECTION = 3,
  /**
   * ATT Tracing. (Not Yet Supported)
   */
  ROCPROFILER_ATT_TRACE_COLLECTION = 4,
  /**
   * SPM collection. (Not Yet Supported)
   */
  ROCPROFILER_SPM_COLLECTION = 5,
  /**
   * HIP/HSA/ROCTX/SYS Trace.
   */
  ROCPROFILER_API_TRACE = 6,
  /**
   * Sampled Counters
   */
  ROCPROFILER_COUNTERS_SAMPLER = 7
} rocprofiler_filter_kind_t;

/**
 * Data Filter Types to be used by ::rocprofiler_session_set_filter to add
 * filters to a specific session
 */
typedef enum {
  /**
   * Add HSA API calls that will be only traced (ex. hsa_amd_memory_async_copy)
   */
  ROCPROFILER_FILTER_HSA_TRACER_API_FUNCTIONS = 1,
  /**
   * Add HIP API calls that will be only traced (ex. hipLaunchKernel)
   */
  ROCPROFILER_FILTER_HIP_TRACER_API_FUNCTIONS = 2,
  /**
   * Add GPU names that will be only profiled or traced
   */
  ROCPROFILER_FILTER_GPU_NAME = 3,
  // TODO(aelwazir): Add more clear description on how to use?
  /**
   * Add Range of calls to be traced or kernels to be profiled
   */
  ROCPROFILER_FILTER_RANGE = 4,
  /**
   * Add Kernel names that will be only profiled or traced
   */
  ROCPROFILER_FILTER_KERNEL_NAMES = 5
} rocprofiler_filter_property_kind_t;

// TODO(aelwazir): Another way to define this as needed
typedef const char* rocprofiler_hip_function_name_t;
typedef const char* rocprofiler_hsa_function_name_t;

// ATT tracing parameter names
typedef enum {
  ROCPROFILER_ATT_COMPUTE_UNIT_TARGET = 0,
  ROCPROFILER_ATT_VM_ID_MASK = 1,
  ROCPROFILER_ATT_MASK = 2,
  ROCPROFILER_ATT_TOKEN_MASK = 3,
  ROCPROFILER_ATT_TOKEN_MASK2 = 4,
  ROCPROFILER_ATT_SE_MASK = 5,
  ROCPROFILER_ATT_SAMPLE_RATE = 6,
  ROCPROFILER_ATT_PERF_MASK = 240,
  ROCPROFILER_ATT_PERF_CTRL = 241,
  ROCPROFILER_ATT_PERFCOUNTER = 242,
  ROCPROFILER_ATT_PERFCOUNTER_NAME = 243,
  ROCPROFILER_ATT_MAXVALUE
} rocprofiler_att_parameter_name_t;

// att tracing parameters object
typedef struct {
  rocprofiler_att_parameter_name_t parameter_name;
  union {
    uint32_t value;
    const char* counter_name;
  };
} rocprofiler_att_parameter_t;

/**
 * Filter Data Type
 * filter data will be used to report required and optional filters for the
 * sessions using ::rocprofiler_session_add_filters
 */
typedef struct {
  /**
   * Filter Property kind
   */
  rocprofiler_filter_property_kind_t kind;
  // TODO(aelwazir): get HIP or HSA or counters as enums
  /**
   * Array of data required for the filter type chosen
   */
  union {
    const char** name_regex;
    rocprofiler_hip_function_name_t* hip_functions_names;
    rocprofiler_hsa_function_name_t* hsa_functions_names;
    uint32_t range[2];
  };
  /**
   * Data array count
   */
  uint64_t data_count;
} rocprofiler_filter_property_t;

typedef struct {
  /**
   * Counters to profile
   */
  const char** counters_names;
  /**
   * Counters count
   */
  int counters_count;
  /**
   * Sampling rate
   */
  uint32_t sampling_rate;
  /**
   * Preferred agents to collect SPM on
   */
  rocprofiler_agent_id_t* gpu_agent_id;

} rocprofiler_spm_parameter_t;

typedef enum{
  ROCPROFILER_COUNTERS_SAMPLER_PCIE_COUNTERS = 0,
  ROCPROFILER_COUNTERS_SAMPLER_XGMI_COUNTERS = 1
} rocprofiler_counters_sampler_counter_type_t;

typedef struct{
  char* name;
  rocprofiler_counters_sampler_counter_type_t type;
} rocprofiler_counters_sampler_counter_input_t;

typedef struct{
  rocprofiler_counters_sampler_counter_type_t type;
  rocprofiler_record_counter_value_t value;
} rocprofiler_counters_sampler_counter_output_t;

typedef struct{
  /**
   * Counters to profile
   */
  rocprofiler_counters_sampler_counter_input_t* counters;
  /**
   * Counters count
   */
  int counters_num;
  /**
   * Sampling rate (ms)
   */
  uint32_t sampling_rate;
  /**
   * Total sampling duration (ms); time between sampling start/stop
   */
  uint32_t sampling_duration;
  /**
   * Initial delay (ms)
   */
  uint32_t initial_delay;
  /**
   * Preferred agents to collect counters from
   */
  int gpu_agent_index;
}rocprofiler_counters_sampler_parameters_t;

typedef struct{
  /**
   * ROCMtool General Record base header to identify the id and kind of every
   * record
   */
  rocprofiler_record_header_t header;
  /**
   * Agent Identifier to be used by the user to get the Agent Information using
   * ::rocprofiler_query_agent_info
   */
  rocprofiler_agent_id_t gpu_id;
  /**
   * Counters, including identifiers to get counter information and Counters
   * values
   */
  rocprofiler_counters_sampler_counter_output_t* counters;
  /**
   * Number of counter values
   */
  uint32_t num_counters;
}rocprofiler_record_counters_sampler_t;

/**
 * Filter Kind Data
 */
typedef union {
  /**
   * APIs to trace
   */
  rocprofiler_tracer_activity_domain_t* trace_apis;
  /**
   * Counters to profile
   */
  const char** counters_names;
  /**
   * att parameters
   */
  rocprofiler_att_parameter_t* att_parameters;
  /**
   * spm counters parameters
   */
  rocprofiler_spm_parameter_t* spm_parameters;
  /**
   * sampled counters parameters
   */
  rocprofiler_counters_sampler_parameters_t counters_sampler_parameters;
} rocprofiler_filter_data_t;

/**
 * Create Session Filter
 * This function will create filter and associate it with a specific session
 * For every kind, one filter only is allowed per session
 *
 * @param[in] session_id Session id where these filters will applied to
 * @param[in] filter_kind  Filter kind associated with these filters
 * @param[in] data Pointer to the filter data
 * @param[in] data_count Count of data in the data array given in ::data
 * @param[out] filter_id The id of the filter created
 * @param[in] property property needed for more filteration requests by the
 * user (Only one property is allowed per filter) (Optional)
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND Couldn't find session
 * associated with the given session identifier
 * @retval ::ROCPROFILER_STATUS_ERROR_SESSION_FILTER_DATA_MISMATCH The session
 * filter can't accept the given data
 * @retval ::ROCPROFILER_STATUS_ERROR_FILTER_DATA_CORRUPTED Data can't be read or
 * corrupted
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_create_filter(rocprofiler_session_id_t session_id,
                                                         rocprofiler_filter_kind_t filter_kind,
                                                         rocprofiler_filter_data_t data,
                                                         uint64_t data_count,
                                                         rocprofiler_filter_id_t* filter_id,
                                                         rocprofiler_filter_property_t property) ROCPROFILER_VERSION_9_0;

/**
 * Set Session Filter Buffer
 * This function will associate buffer to a specific filter
 *
 * if the user wants to get the API traces for the api calls synchronously then
 * the user is required to call ::rocprofiler_set_api_trace_sync_callback
 *
 * @param[in] session_id Session id where these filters will applied to
 * @param[in] filter_id The id of the filter
 * @param[in] buffer_id The id of the buffer
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND Couldn't find session
 * associated with the given session identifier
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_set_filter_buffer(rocprofiler_session_id_t session_id,
                                                             rocprofiler_filter_id_t filter_id,
                                                             rocprofiler_buffer_id_t buffer_id) ROCPROFILER_VERSION_9_0;

/**
 * Synchronous Callback
 * To be only used by ::rocprofiler_set_api_trace_sync_callback, please refer to
 * ::rocprofiler_set_api_trace_sync_callback for more details
 *
 * @param[in] record pointer to the record.
 * @param[in] session_id The session id associated with that record
 */
typedef void (*rocprofiler_sync_callback_t)(rocprofiler_record_tracer_t record,
                                          rocprofiler_session_id_t session_id);

/**
 * Set Session API Tracing Filter Synchronous Callback
 * This function will associate buffer to a specific filter
 *
 * Currently Synchronous callbacks are only available to API Tracing filters
 * for the api calls tracing and not available for the api activities or any
 * other filter type, the user is responsible to create and set buffer for the
 * other types
 *
 * @param[in] session_id Session id where these filters will applied to
 * @param[in] filter_id The id of the filter
 * @param[in] callback Synchronous callback
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND, Couldn't find session
 * associated with the given session identifier
 * @retval ::ROCPROFILER_STATUS_ERROR_FILTER_NOT_SUPPORTED, if the filter is not
 * related to API tracing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_set_api_trace_sync_callback(
    rocprofiler_session_id_t session_id, rocprofiler_filter_id_t filter_id,
    rocprofiler_sync_callback_t callback) ROCPROFILER_VERSION_9_0;

/**
 * Destroy Session Filter
 * This function will destroy a specific filter
 *
 * @param[in] session_id Session id where these filters will applied to
 * @param[in] filter_id The id of the filter
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND Couldn't find session
 * associated with the given session identifier
 * @retval ::ROCPROFILER_STATUS_FILTER_NOT_FOUND Couldn't find session filter
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_destroy_filter(rocprofiler_session_id_t session_id,
                                                          rocprofiler_filter_id_t filter_id) ROCPROFILER_VERSION_9_0;

/**
 * Create Buffer
 * This function will create a buffer that can be associated with a filter
 *
 * @param[in] session_id Session id where these filters will applied to
 * @param[in] buffer_callback Providing a callback for the buffer specialized
 * for that filters
 * @param[in] buffer_size Providing size for the buffer that will be created
 * @param[in] buffer_properties Array of Flush Properties provided by the user
 * @param[in] buffer_properties_count The count of the flush properties in the
 * array
 * @param[out] buffer_id Buffer id that was created
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND Couldn't find session
 * associated with the given session identifier
 * @retval ::ROCPROFILER_STATUS_ERROR_SESSION_PROPERTIES_MISMATCH The given
 * properties data are mismatching the properties kind
 * @retval ::ROCPROFILER_STATUS_ERROR_PROPERTY_DATA_CORRUPTED Data can't be read
 * or corrupted
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_create_buffer(
    rocprofiler_session_id_t session_id, rocprofiler_buffer_callback_t buffer_callback,
    size_t buffer_size, rocprofiler_buffer_id_t* buffer_id) ROCPROFILER_VERSION_9_0;

/**
 * Setting Buffer Properties
 * This function will set buffer properties
 *
 * @param[in] session_id Session id where the buffer is associated with
 * @param[in] buffer_id Buffer id of the buffer that the properties are going
 * to be associated with for that filters
 * @param[in] buffer_properties Array of Flush Properties provided by the user
 * @param[in] buffer_properties_count The count of the flush properties in the
 * array
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND Couldn't find session
 * associated with the given session identifier
 * @retval ::ROCPROFILER_STATUS_BUFFER_NOT_FOUND Couldn't find buffer
 * associated with the given buffer identifier
 * @retval ::ROCPROFILER_STATUS_ERROR_SESSION_PROPERTIES_MISMATCH The given
 * properties data are mismatching the properties kind
 * @retval ::ROCPROFILER_STATUS_ERROR_PROPERTY_DATA_CORRUPTED Data can't be read
 * or corrupted
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_set_buffer_properties(
    rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id,
    rocprofiler_buffer_property_t* buffer_properties, uint32_t buffer_properties_count) ROCPROFILER_VERSION_9_0;

/**
 * Destroy Buffer
 * This function will destroy a buffer given its id and session id
 *
 * @param[in] session_id Session id where these filters will applied to
 * @param[in] buffer_id Buffer id that will b e destroyed
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND Couldn't find session
 * associated with the given session identifier
 * @retval ::ROCPROFILER_STATUS_BUFFER_NOT_FOUND Couldn't find buffer
 * associated with the given buffer identifier
 * @retval ::ROCPROFILER_STATUS_ERROR_SESSION_PROPERTIES_MISMATCH The given
 * properties data are mismatching the properties kind
 * @retval ::ROCPROFILER_STATUS_ERROR_PROPERTY_DATA_CORRUPTED Data can't be read
 * or corrupted
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_destroy_buffer(rocprofiler_session_id_t session_id,
                                                          rocprofiler_buffer_id_t buffer_id) ROCPROFILER_VERSION_9_0;

/** @} */

/**
 * Create Ready Session
 * A one call to create a ready profiling or tracing session, so that the
 * session will be ready to collect counters with a one call to
 * ::rocprofiler_start_session.
 * ::rocprofiler_session_set_filter can be used to set optional filters like
 * specific GPUs/Kernel Names/Counter Names and more. The Creation of the
 * session is responsible for the creation of the buffer saving the records
 * generated while the session is active. Session can be started using
 * ::rocprofiler_start_session and can be stopped using
 * ::rocprofiler_terminate_session
 *
 * @param[in] counters counter filter data, it is required from the user to
 * create the filter with ::ROCPROFILER_FILTER_PROFILER_COUNTER_NAMES and to
 * provide an array of counter names needed and their count
 * @param[in] replay_mode The Replay strategy that should be used if replay is
 * needed
 * @param[in] filter_kind  Filter kind associated with these filters
 * @param[in] data Pointer to the filter data
 * @param[in] data_count Filter data array count
 * @param[in] buffer_size Size of the memory pool that will be used to save the
 * data from profiling or/and tracing, if the buffer was allocated before it
 * will be reallocated with the new size in addition to the old size
 * @param[in] buffer_callback Asynchronous callback using Memory buffers saving
 * the data and then it will be flushed if the user called
 * ::rocprofiler_flush_data or if the buffer is full or if the application
 * finished execution
 * @param[out] session_id Pointer to the created session id, the session is
 * alive up till ::rocprofiler_destroy_session being called, however, the session
 * id can be used while the session is active which can be activated using
 * ::rocprofiler_start_session and deactivated using
 * ::rocprofiler_terminate_session but ::rocprofiler_flush_data can use session_id
 * even if it is deactivated for flushing the saved records
 * @param[in] property Filter Property (Optional)
 * @param[in] callback Synchronous callback for API traces (Optional)
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_SESSION_MODE_FILTER_MISMATCH The session
 * doesn't have the required mode for that filter type
 * @retval ::ROCPROFILER_STATUS_ERROR_FILTER_DATA_CORRUPTED Data can't be read or
 * corrupted
 * @retval ::ROCPROFILER_STATUS_ERROR_INCORRECT_SIZE If the size is less than one
 * potential record size
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_create_ready_session(
    rocprofiler_replay_mode_t replay_mode, rocprofiler_filter_kind_t filter_kind,
    rocprofiler_filter_data_t data, uint64_t data_count, size_t buffer_size,
    rocprofiler_buffer_callback_t buffer_callback, rocprofiler_session_id_t* session_id,
    rocprofiler_filter_property_t property, rocprofiler_sync_callback_t callback) ROCPROFILER_VERSION_9_0;

// TODO(aelwazir): Multiple sessions activate for different set of filters
/**
 * Activate Session
 * Activating session created by ::rocprofiler_create_session, please refer to
 * the samples for how to use.
 *
 * @param[in] session_id Session ID representing the created session
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND may return if
 * the session is not found
 * @retval ::ROCPROFILER_STATUS_ERROR_SESSION_MODE_NOT_ADDED if there is no
 * session_mode added
 * @retval ::ROCPROFILER_STATUS_ERROR_MISSING_SESSION_CALLBACK if any
 * session_mode is missing callback set
 * @retval ::ROCPROFILER_STATUS_ERROR_HAS_ACTIVE_SESSION if there is already
 * active session
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_start_session(rocprofiler_session_id_t session_id) ROCPROFILER_VERSION_9_0;

/**
 * Deactivate Session
 * Deactivate session created by ::rocprofiler_create_session, please refer to
 * the samples for how to use.
 *
 * @param[in] session_id Session ID for the session that will be terminated
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully
 * @retval ::ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND may return if
 * the session is not found
 * @retval ::ROCPROFILER_STATUS_ERROR_SESSION_NOT_ACTIVE if the session is not
 * active
 */

ROCPROFILER_API rocprofiler_status_t rocprofiler_terminate_session(rocprofiler_session_id_t session_id) ROCPROFILER_VERSION_9_0;

/** \defgroup session_range_group Session Range Labeling
 * \ingroup sessions_handling_group
 * @{
 */

/**
 * Setting a label to a block range
 * This can be used to label a range of code that is having active profiling
 * session or labeling a pass
 *
 * @param[in] label The label given for a certain block or pass to name/label.
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_CORRUPTED_LABEL_DATA may return if
 * the label pointer can't be read by the API
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_push_range(const char* label) ROCPROFILER_VERSION_9_0;

/**
 * Setting an endpoint for a range
 * This function can be used to set an endpoint to range labeled by
 * ::rocprofiler_push_range
 *
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_RANGE_STACK_IS_EMPTY may return if
 * ::rocprofiler_push_range wasn't called correctly
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_pop_range() ROCPROFILER_VERSION_9_0;

/** @} */

/** \defgroup session_user_replay_pass_group Session User Replay Pass Mode
 * \ingroup sessions_handling_group
 * @{
 */

/**
 * Create and Start a pass
 * A Pass is a block of code that can be replayed if required by the
 * profiling/tracing and it mainly depends on the profiling data given in the
 * ::rocprofiler_create_session
 *
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND If the no active session
 * found
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_start_replay_pass() ROCPROFILER_VERSION_9_0;

/**
 * End a pass
 * End a pass created and started by ::rocprofiler_start_pass
 *
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * @retval ::ROCPROFILER_STATUS_ERROR_PASS_NOT_STARTED if there is no pass
 * started before this call
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_end_replay_pass() ROCPROFILER_VERSION_9_0;

/** @} */
/** @} */

/** \defgroup device_profiling Device Profiling API
 * @{
 */

typedef struct {
  double value;
} rocprofiler_counter_value_t;

typedef struct {
  char metric_name[64];
  rocprofiler_counter_value_t value;
} rocprofiler_device_profile_metric_t;

/**
 * Create a device profiling session
 *
 * A device profiling session allows the user to profile the GPU device
 * for counters irrespective of the running applications on the GPU.
 * This is different from application profiling. device profiling session
 * doesn't care about the host running processes and threads. It directly
 * provides low level profiling information.
 *
 * @param[in] counter_names The names of the counters to be collected.
 * @param[in] num_counters The number of counters specifief to be collected
 * @param[out] session_id Pointer to the created session id.
 * @param[in] cpu_index index of the cpu to be used
 * @param[in] gpu_index index of the gpu to be used
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_device_profiling_session_create(
    const char** counter_names, uint64_t num_counters, rocprofiler_session_id_t* session_id,
    int cpu_index, int gpu_index) ROCPROFILER_VERSION_9_0;

/**
 * Start the device profiling session that was created previously.
 * This will enable the GPU device to start incrementing counters
 *
 * @param[in] session_id session id of the session to start
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
ROCPROFILER_API rocprofiler_status_t
rocprofiler_device_profiling_session_start(rocprofiler_session_id_t session_id) ROCPROFILER_VERSION_9_0;

/**
 * Poll the device profiling session to read counters from the GPU device.
 * This will read out the values of the counters from the GPU device at the
 * specific instant when this API is called. This is a thread-blocking call.
 * Any thread that calls this API will have to wait until
 * the counter values are being read out.
 *
 * @param[in] session_id session id of the session to start
 * @param[out] data records of counter data read out from device
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_device_profiling_session_poll(
    rocprofiler_session_id_t session_id, rocprofiler_device_profile_metric_t* data) ROCPROFILER_VERSION_9_0;

/**
 * Stop the device profiling session that was created previously.
 * This will inform the GPU device to stop counters collection.
 *
 * @param[in] session_id session id of the session to start
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
ROCPROFILER_API rocprofiler_status_t
rocprofiler_device_profiling_session_stop(rocprofiler_session_id_t session_id) ROCPROFILER_VERSION_9_0;

/**
 * Destroy the device profiling session that was created previously.
 *
 * @param[in] session_id session id of the session to start
 * @retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
ROCPROFILER_API rocprofiler_status_t
rocprofiler_device_profiling_session_destroy(rocprofiler_session_id_t session_id) ROCPROFILER_VERSION_9_0;

/** @} */

#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Old ROCProfiler
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <stdint.h>

////////////////////////////////////////////////////////////////////////////////
// Global properties structure

typedef struct {
  uint32_t intercept_mode;
  uint32_t code_obj_tracking;
  uint32_t memcopy_tracking;
  uint32_t trace_size;
  uint32_t trace_local;
  uint64_t timeout;
  uint32_t timestamp_on;
  uint32_t hsa_intercepting;
  uint32_t k_concurrent;
  uint32_t opt_mode;
  uint32_t obj_dumping;
} rocprofiler_settings_t;

////////////////////////////////////////////////////////////////////////////////
// Returning the error string method

hsa_status_t rocprofiler_error_string(
    const char** str);  // [out] the API error string pointer returning

////////////////////////////////////////////////////////////////////////////////
// Profiling features and data
//
// Profiling features objects have profiling feature info, type, parameters and data
// Also profiling data samplaes can be iterated using a callback

// Profiling feature kind
typedef enum {
  ROCPROFILER_FEATURE_KIND_METRIC = 0,
  ROCPROFILER_FEATURE_KIND_TRACE = 1,
  ROCPROFILER_FEATURE_KIND_SPM_MOD = 2,
  ROCPROFILER_FEATURE_KIND_PCSMP_MOD = 4
} rocprofiler_feature_kind_t;

// Profiling feture parameter
typedef hsa_ven_amd_aqlprofile_parameter_t rocprofiler_parameter_t;

// Profiling data kind
typedef enum {
  ROCPROFILER_DATA_KIND_UNINIT = 0,
  ROCPROFILER_DATA_KIND_INT32 = 1,
  ROCPROFILER_DATA_KIND_INT64 = 2,
  ROCPROFILER_DATA_KIND_FLOAT = 3,
  ROCPROFILER_DATA_KIND_DOUBLE = 4,
  ROCPROFILER_DATA_KIND_BYTES = 5
} rocprofiler_data_kind_t;

// Profiling data type
typedef struct {
  rocprofiler_data_kind_t kind;  // result kind
  union {
    uint32_t result_int32;  // 32bit integer result
    uint64_t result_int64;  // 64bit integer result
    float result_float;     // float single-precision result
    double result_double;   // float double-precision result
    struct {
      void* ptr;
      uint32_t size;
      uint32_t instance_count;
      bool copy;
    } result_bytes;  // data by ptr and byte size
  };
} rocprofiler_data_t;

// Profiling feature type
typedef struct {
  rocprofiler_feature_kind_t kind;            // feature kind
  union {
    const char* name;                         // feature name
    struct {
      const char* block;                      // counter block name
      uint32_t event;                         // counter event id
    } counter;
  };
  const rocprofiler_parameter_t* parameters;  // feature parameters array
  uint32_t parameter_count;                   // feature parameters count
  rocprofiler_data_t data;                    // profiling data
} rocprofiler_feature_t;

// Profiling features set type
typedef void rocprofiler_feature_set_t;

////////////////////////////////////////////////////////////////////////////////
// Profiling context
//
// Profiling context object accumuate all profiling information

// Profiling context object
typedef void rocprofiler_t;

// Profiling group object
typedef struct {
  unsigned index;                    // group index
  rocprofiler_feature_t** features;  // profiling info array
  uint32_t feature_count;            // profiling info count
  rocprofiler_t* context;            // context object
} rocprofiler_group_t;

// Profiling mode mask
typedef enum {
  ROCPROFILER_MODE_STANDALONE = 1,   // standalone mode when ROC profiler supports a queue
  ROCPROFILER_MODE_CREATEQUEUE = 2,  // ROC profiler creates queue in standalone mode
  ROCPROFILER_MODE_SINGLEGROUP = 4   // only one group is allowed, failed otherwise
} rocprofiler_mode_t;

// Profiling handler, calling on profiling completion
typedef bool (*rocprofiler_handler_t)(rocprofiler_group_t group, void* arg);

// Profiling preperties
typedef struct {
  hsa_queue_t* queue;             // queue for STANDALONE mode
                                  // the queue is created and returned in CREATEQUEUE mode
  uint32_t queue_depth;           // created queue depth
  rocprofiler_handler_t handler;  // handler on completion
  void* handler_arg;              // the handler arg
} rocprofiler_properties_t;

// Create new profiling context
hsa_status_t rocprofiler_open(hsa_agent_t agent,                      // GPU handle
                              rocprofiler_feature_t* features,        // [in] profiling features array
                              uint32_t feature_count,                 // profiling info count
                              rocprofiler_t** context,                // [out] context object
                              uint32_t mode,                          // profiling mode mask
                              rocprofiler_properties_t* properties);  // profiling properties

// Add feature to a features set
hsa_status_t rocprofiler_add_feature(const rocprofiler_feature_t* feature,     // [in]
                                     rocprofiler_feature_set_t* features_set); // [in/out] profiling features set

// Create new profiling context
hsa_status_t rocprofiler_features_set_open(hsa_agent_t agent,                       // GPU handle
                                           rocprofiler_feature_set_t* features_set, // [in] profiling features set
                                           rocprofiler_t** context,                 // [out] context object
                                           uint32_t mode,                           // profiling mode mask
                                           rocprofiler_properties_t* properties);   // profiling properties

// Delete profiling info
hsa_status_t rocprofiler_close(rocprofiler_t* context);  // [in] profiling context

// Context reset before reusing
hsa_status_t rocprofiler_reset(rocprofiler_t* context,  // [in] profiling context
                               uint32_t group_index);   // group index

// Return context agent
hsa_status_t rocprofiler_get_agent(rocprofiler_t* context,        // [in] profiling context
                                   hsa_agent_t* agent);           // [out] GPU handle

// Supported time value ID
typedef enum {
  ROCPROFILER_TIME_ID_CLOCK_REALTIME = 0, // Linux realtime clock time
  ROCPROFILER_TIME_ID_CLOCK_REALTIME_COARSE = 1, // Linux realtime-coarse clock time
  ROCPROFILER_TIME_ID_CLOCK_MONOTONIC = 2, // Linux monotonic clock time
  ROCPROFILER_TIME_ID_CLOCK_MONOTONIC_COARSE = 3, // Linux monotonic-coarse clock time
  ROCPROFILER_TIME_ID_CLOCK_MONOTONIC_RAW = 4, // Linux monotonic-raw clock time
} rocprofiler_time_id_t;

// Return time value for a given time ID and profiling timestamp
hsa_status_t rocprofiler_get_time(
  rocprofiler_time_id_t time_id, // identifier of the particular time to convert the timesatmp
  uint64_t timestamp, // profiling timestamp
  uint64_t* value_ns, // [out] returned time 'ns' value, ignored if NULL
  uint64_t* error_ns); // [out] returned time error 'ns' value, ignored if NULL

////////////////////////////////////////////////////////////////////////////////
// Queue callbacks
//
// Queue callbacks for initiating profiling per kernel dispatch and to wait
// the profiling data on the queue destroy.

// Dispatch record
typedef struct {
  uint64_t dispatch;                                   // dispatch timestamp, ns
  uint64_t begin;                                      // kernel begin timestamp, ns
  uint64_t end;                                        // kernel end timestamp, ns
  uint64_t complete;                                   // completion signal timestamp, ns
} rocprofiler_dispatch_record_t;

// Profiling callback data
typedef struct {
  hsa_agent_t agent;                                   // GPU agent handle
  uint32_t agent_index;                                // GPU index (GPU Driver Node ID as reported in the sysfs topology)
  const hsa_queue_t* queue;                            // HSA queue
  uint64_t queue_index;                                // Index in the queue
  uint32_t queue_id;                                   // Queue id
  hsa_signal_t completion_signal;                      // Completion signal
  const hsa_kernel_dispatch_packet_t* packet;          // HSA dispatch packet
  const char* kernel_name;                             // Kernel name
  uint64_t kernel_object;                              // Kernel object address
  const amd_kernel_code_t* kernel_code;                // Kernel code pointer
  uint32_t thread_id;                                   // Thread id
  const rocprofiler_dispatch_record_t* record;         // Dispatch record
} rocprofiler_callback_data_t;

// Profiling callback type
typedef hsa_status_t (*rocprofiler_callback_t)(
    const rocprofiler_callback_data_t* callback_data,  // [in] callback data
    void* user_data,                                   // [in/out] user data passed to the callback
    rocprofiler_group_t* group);                       // [out] returned profiling group

// Queue callbacks
typedef struct {
    rocprofiler_callback_t dispatch;                          // dispatch callback
    hsa_status_t (*create)(hsa_queue_t* queue, void* data);   // create callback
    hsa_status_t (*destroy)(hsa_queue_t* queue, void* data);  // destroy callback
} rocprofiler_queue_callbacks_t;

// Set queue callbacks
hsa_status_t rocprofiler_set_queue_callbacks(
    rocprofiler_queue_callbacks_t callbacks,           // callbacks
    void* data);                                       // [in/out] passed callbacks data

// Remove queue callbacks
hsa_status_t rocprofiler_remove_queue_callbacks();

// Start/stop queue callbacks
hsa_status_t rocprofiler_start_queue_callbacks();
hsa_status_t rocprofiler_stop_queue_callbacks();

////////////////////////////////////////////////////////////////////////////////
// Start/stop profiling
//
// Start/stop the context profiling invocation, have to be as many as
// contect.invocations' to collect all profiling data

// Start profiling
hsa_status_t rocprofiler_start(rocprofiler_t* context,     // [in/out] profiling context
                               uint32_t group_index);      // group index

// Stop profiling
hsa_status_t rocprofiler_stop(rocprofiler_t* context,     // [in/out] profiling context
                              uint32_t group_index);      // group index

// Read profiling
hsa_status_t rocprofiler_read(rocprofiler_t* context,     // [in/out] profiling context
                              uint32_t group_index);      // group index

// Read profiling data
hsa_status_t rocprofiler_get_data(rocprofiler_t* context, // [in/out] profiling context
                                  uint32_t group_index);  // group index

// Get profiling groups count
hsa_status_t rocprofiler_group_count(const rocprofiler_t* context,  // [in] profiling context
                                     uint32_t* group_count);        // [out] profiling groups count

// Get profiling group for a given index
hsa_status_t rocprofiler_get_group(rocprofiler_t* context,       // [in] profiling context
                                   uint32_t group_index,         // profiling group index
                                   rocprofiler_group_t* group);  // [out] profiling group

// Start profiling
hsa_status_t rocprofiler_group_start(rocprofiler_group_t* group);  // [in/out] profiling group

// Stop profiling
hsa_status_t rocprofiler_group_stop(rocprofiler_group_t* group);  // [in/out] profiling group

// Read profiling
hsa_status_t rocprofiler_group_read(rocprofiler_group_t* group);  // [in/out] profiling group

// Get profiling data
hsa_status_t rocprofiler_group_get_data(rocprofiler_group_t* group);  // [in/out] profiling group

// Get metrics data
hsa_status_t rocprofiler_get_metrics(const rocprofiler_t* context);  // [in/out] profiling context

// Definition of output data iterator callback
typedef hsa_ven_amd_aqlprofile_data_callback_t rocprofiler_trace_data_callback_t;

// Method for iterating the events output data
hsa_status_t rocprofiler_iterate_trace_data(
    rocprofiler_t* context,                      // [in] profiling context
    rocprofiler_trace_data_callback_t callback,  // callback to iterate the output data
    void* data);                                 // [in/out] callback data

////////////////////////////////////////////////////////////////////////////////
// Profiling features and data
//
// Profiling features objects have profiling feature info, type, parameters and data
// Also profiling data samplaes can be iterated using a callback

// Profiling info kind
typedef enum {
  ROCPROFILER_INFO_KIND_METRIC = 0, // metric info
  ROCPROFILER_INFO_KIND_METRIC_COUNT = 1, // metric features count, int32
  ROCPROFILER_INFO_KIND_TRACE = 2, // trace info
  ROCPROFILER_INFO_KIND_TRACE_COUNT = 3, // trace features count, int32
  ROCPROFILER_INFO_KIND_TRACE_PARAMETER = 4, // trace parameter info
  ROCPROFILER_INFO_KIND_TRACE_PARAMETER_COUNT = 5 // trace parameter count, int32
} rocprofiler_info_kind_t;

// Profiling info query
typedef union {
  rocprofiler_info_kind_t info_kind; // queried profiling info kind
  struct {
    const char* trace_name; // queried info trace name
  } trace_parameter;
} rocprofiler_info_query_t;

// Profiling info data
typedef struct {
  uint32_t agent_index; // GPU HSA agent index (GPU Driver Node ID as reported in the sysfs topology)
  rocprofiler_info_kind_t kind; // info data kind
  union {
    struct {
      const char* name; // metric name
      uint32_t instances; // instances number
      const char* expr; // metric expression, NULL for basic counters
      const char* description; // metric description
      const char* block_name; // block name
      uint32_t block_counters; // number of block counters
    } metric;
    struct {
      const char* name; // trace name
      const char* description; // trace description
      uint32_t parameter_count; // supported by the trace number parameters
    } trace;
    struct {
      uint32_t code; // parameter code
      const char* trace_name; // trace name
      const char* parameter_name; // parameter name
      const char* description; // trace parameter description
    } trace_parameter;
  };
} rocprofiler_info_data_t;

// Return the info for a given info kind
hsa_status_t rocprofiler_get_info(
  const hsa_agent_t* agent, // [in] GFXIP handle
  rocprofiler_info_kind_t kind, // kind of iterated info
  void *data); // [in/out] returned data

// Iterate over the info for a given info kind, and invoke an application-defined callback on every iteration
hsa_status_t rocprofiler_iterate_info(
  const hsa_agent_t* agent, // [in] GFXIP handle
  rocprofiler_info_kind_t kind, // kind of iterated info
  hsa_status_t (*callback)(const rocprofiler_info_data_t info, void *data), // callback
  void *data); // [in/out] data passed to callback

// Iterate over the info for a given info query, and invoke an application-defined callback on every iteration
hsa_status_t rocprofiler_query_info(
  const hsa_agent_t *agent, // [in] GFXIP handle
  rocprofiler_info_query_t query, // iterated info query
  hsa_status_t (*callback)(const rocprofiler_info_data_t info, void *data), // callback
  void *data); // [in/out] data passed to callback

// Create a profiled queue. All dispatches on this queue will be profiled
hsa_status_t rocprofiler_queue_create_profiled(
  hsa_agent_t agent_handle,uint32_t size, hsa_queue_type32_t type,
  void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
  void* data, uint32_t private_segment_size, uint32_t group_segment_size,
  hsa_queue_t** queue);

////////////////////////////////////////////////////////////////////////////////
// Profiling pool
//
// Support for profiling contexts pool
// The API provide capability to create a contexts pool for a given agent and a set of features,
// to fetch/relase a context entry, to register a callback for the contexts completion.

// Profiling pool handle
typedef void rocprofiler_pool_t;

// Profiling pool entry
typedef struct {
  rocprofiler_t* context;             // context object
  void* payload;                      // payload data object
} rocprofiler_pool_entry_t;

// Profiling handler, calling on profiling completion
typedef bool (*rocprofiler_pool_handler_t)(const rocprofiler_pool_entry_t* entry, void* arg);

// Profiling preperties
typedef struct {
  uint32_t num_entries;                // pool size entries
  uint32_t payload_bytes;              // payload size bytes
  rocprofiler_pool_handler_t handler;  // handler on context completion
  void* handler_arg;                   // the handler arg
} rocprofiler_pool_properties_t;

// Open profiling pool
hsa_status_t rocprofiler_pool_open(
  hsa_agent_t agent,                   // GPU handle
  rocprofiler_feature_t* features,     // [in] profiling features array
  uint32_t feature_count,              // profiling info count
  rocprofiler_pool_t** pool,           // [out] context object
  uint32_t mode,                       // profiling mode mask
  rocprofiler_pool_properties_t*);     // pool properties

// Close profiling pool
hsa_status_t rocprofiler_pool_close(
  rocprofiler_pool_t* pool);          // profiling pool handle

// Fetch profiling pool entry
hsa_status_t rocprofiler_pool_fetch(
  rocprofiler_pool_t* pool,           // profiling pool handle
  rocprofiler_pool_entry_t* entry);   // [out] empty profiling pool entry

// Release profiling pool entry
hsa_status_t rocprofiler_pool_release(
  rocprofiler_pool_entry_t* entry);   // released profiling pool entry

// Iterate fetched profiling pool entries
hsa_status_t rocprofiler_pool_iterate(
  rocprofiler_pool_t* pool,           // profiling pool handle
  hsa_status_t (*callback)(rocprofiler_pool_entry_t* entry, void* data), // callback
  void *data); // [in/out] data passed to callback

// Flush completed entries in profiling pool
hsa_status_t rocprofiler_pool_flush(
  rocprofiler_pool_t* pool);          // profiling pool handle

////////////////////////////////////////////////////////////////////////////////
// HSA intercepting API

// HSA callbacks ID enumeration
typedef enum {
  ROCPROFILER_HSA_CB_ID_ALLOCATE = 0, // Memory allocate callback
  ROCPROFILER_HSA_CB_ID_DEVICE = 1,   // Device assign callback
  ROCPROFILER_HSA_CB_ID_MEMCOPY = 2,  // Memcopy callback
  ROCPROFILER_HSA_CB_ID_SUBMIT = 3,   // Packet submit callback
  ROCPROFILER_HSA_CB_ID_KSYMBOL = 4,  // Loading/unloading of kernel symbol
  ROCPROFILER_HSA_CB_ID_CODEOBJ = 5   // Loading/unloading of kernel symbol
} rocprofiler_hsa_cb_id_t;

// HSA callback data type
typedef struct {
  union {
    struct {
      const void* ptr;                                // allocated area ptr
      size_t size;                                    // allocated area size, zero size means 'free' callback
      hsa_amd_segment_t segment;                      // allocated area's memory segment type
      hsa_amd_memory_pool_global_flag_t global_flag;  // allocated area's memory global flag
      int is_code;                                    // equal to 1 if code is allocated
    } allocate;
    struct {
      hsa_device_type_t type;                         // type of assigned device
      uint32_t id;                                    // id of assigned device
      hsa_agent_t agent;                              // device HSA agent handle
      const void* ptr;                                // ptr the device is assigned to
    } device;
    struct {
      const void* dst;                                // memcopy dst ptr
      const void* src;                                // memcopy src ptr
      size_t size;                                    // memcopy size bytes
    } memcopy;
    struct {
      const void* packet;                             // submitted to GPU packet
      const char* kernel_name;                        // kernel name, not NULL if dispatch
      hsa_queue_t* queue;                             // HSA queue the kernel was submitted to
      uint32_t device_type;                           // type of device the packed is submitted to
      uint32_t device_id;                             // id of device the packed is submitted to
    } submit;
    struct {
      uint64_t object;                                // kernel symbol object
      const char* name;                               // kernel symbol name
      uint32_t name_length;                           // kernel symbol name length
      int unload;                                     // symbol executable destroy
    } ksymbol;
    struct {
      uint32_t storage_type;                          // code object storage type
      int storage_file;                               // origin file descriptor
      uint64_t memory_base;                           // origin memory base
      uint64_t memory_size;                           // origin memory size
      uint64_t load_base;                             // codeobj load base
      uint64_t load_size;                             // codeobj load size
      uint64_t load_delta;                            // codeobj load size
      uint32_t uri_length;                            // URI string length
      char* uri;                                      // URI string
      int unload;                                     // unload flag
    } codeobj;
  };
} rocprofiler_hsa_callback_data_t;

// HSA callback function type
typedef hsa_status_t (*rocprofiler_hsa_callback_fun_t)(
  rocprofiler_hsa_cb_id_t id, // callback id
  const rocprofiler_hsa_callback_data_t* data, // [in] callback data
  void* arg); // [in/out] user passed data

// HSA callbacks structure
typedef struct {
  rocprofiler_hsa_callback_fun_t allocate; // memory allocate callback
  rocprofiler_hsa_callback_fun_t device; // agent assign callback
  rocprofiler_hsa_callback_fun_t memcopy; // memory copy callback
  rocprofiler_hsa_callback_fun_t submit; // packet submit callback
  rocprofiler_hsa_callback_fun_t ksymbol; // kernel symbol callback
  rocprofiler_hsa_callback_fun_t codeobj; // codeobject load/unload callback
} rocprofiler_hsa_callbacks_t;

// Set callbacks. If the callback is NULL then it is disabled.
// If callback returns a value that is not HSA_STATUS_SUCCESS the  callback
// will be unregistered.
hsa_status_t rocprofiler_set_hsa_callbacks(
  const rocprofiler_hsa_callbacks_t callbacks, // HSA callback function
  void* arg); // callback user data

#ifdef __cplusplus
}  // extern "C" block
#endif  // __cplusplus

#endif  // INC_ROCPROFILER_H_
