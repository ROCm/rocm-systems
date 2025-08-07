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
// For more information about the AMD ROCm ecosystem, refer to: https://rocm.docs.amd.com/
/**
 * \mainpage rocprofiler_api ROCProfiler API specification
 *
 * The ROCProfiler library provides profiling and tracing APIs for GPU compute applications.
 * The APIs offer functionalities for profiling GPU applications in kernel,
 * application, and user mode. The APIs also support no replay mode and provides
 * the records pool support through a simple sequence of calls. This enables
 * users to profile and trace in small steps. The sample codes demonstrate
 * how to use the API calls for both profiling and tracing.
 *
 * \section supported_amd_gpu_architectures Supported AMD GPU architectures
 *
 * Here is the list of supported AMD GPU architectures:
 *
 * - gfx900 (AMD Vega 10)
 * - gfx906 (AMD Vega 7nm also referred to as AMD Vega 20)
 * - gfx908 (AMD Instinct™ MI100 accelerator)
 * - gfx90a (Aldebaran)
 * - gfx94x (AMD Instinct™ MI300)
 * - gfx1010 (Navi10)
 * - gfx1011 (Navi12)
 * - gfx1012 (Navi14)
 * - gfx1030 (Sienna Cichlid)
 * - gfx1031 (Navy Flounder)
 * - gfx1032 (Dimgrey Cavefish)
 * - gfx1100 (Navi31)
 *
 * \section known_limitations Known limitations and restrictions
 *
 * The AMD Profiler API library implementation has the following
 * restrictions. Future releases aim to address these restrictions.
 *
 * The following profiling modes are not yet implemented:
 *
 * - ::ROCPROFILER_APPLICATION_REPLAY_MODE
 * - ::ROCPROFILER_USER_REPLAY_MODE
 *
 * While setting filters, properties can mix up and might produce undesirable results.
 *
 * This document discusses the following:
 * - @ref symbol_versions_group
 * - @ref versioning_group
 * - @ref status_codes_group
 * - @ref rocprofiler_general_group
 * - @ref timestamp_group
 * - @ref generic_record_group
 *      - @ref record_agents_group
 *      - @ref record_queues_group
 *      - @ref record_kernels_group
 * - @ref profiling_api_group
 *      - @ref profiling_api_counters_group
 * - @ref tracing_api_group
 *      - @ref roctx_tracer_api_data_group
 *      - @ref hsa_tracer_api_data_group
 *      - @ref hip_tracer_api_data_group
 * - @ref memory_storage_buffer_group
 * - @ref sessions_handling_group
 *      - @ref session_filter_group
 *      - @ref session_range_group
 *      - @ref session_user_replay_pass_group
 * - @ref device_profiling
 * - @ref rocprofiler_plugins
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
 * The semantic version of the interface following <https://semver.org> rules. A client
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
  ROCPROFILER_STATUS_ERROR_FILTER_NOT_SUPPORTED = -33,
  /**
   * Invalid Arguments were given to the function
   */
  ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENTS = -34,
  /**
   * The given operation id is not valid.
   */
  ROCPROFILER_STATUS_ERROR_INVALID_OPERATION_ID = -35,
  /**
   * The given domain id is not valid.
   */
  ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID = -36,
  /**
   * The feature requested is not implemented.
   */
  ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED = -37,
  /**
   * External Correlation id pop called without matching push.
   */
  ROCPROFILER_STATUS_ERROR_MISMATCHED_EXTERNAL_CORRELATION_ID = -38,
} rocprofiler_status_t;

/**
 * Query the textual description of the given error for the current thread.
 *
 * Returns a NULL terminated string describing the error of the given ROCProfiler
 * API call by the calling thread that did not return success.
 *
 * \retval Return the error string.
 */
ROCPROFILER_API const char* rocprofiler_error_str(rocprofiler_status_t status)
    ROCPROFILER_VERSION_9_0;

/** @} */

/** \defgroup rocprofiler_general_group General ROCProfiler Requirements
 * @{
 */

// TODO(aelwazir): More clear description, (think about nested!!??)

/**
 * Initialize the API Tools
 *
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_API_ALREADY_INITIALIZED If initialize
 * wasn't called or finalized called twice
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_initialize() ROCPROFILER_VERSION_9_0;

/**
 * Finalize the API Tools
 *
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_API_NOT_INITIALIZED If initialize wasn't
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
 * \param[out] timestamp The system clock timestamp in nano seconds.
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_TIMESTAMP_NOT_APPLICABLE <br />
 * The function failed to get the timestamp using HSA Function.
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_get_timestamp(rocprofiler_timestamp_t* timestamp)
    ROCPROFILER_VERSION_9_0;

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
 * \param[in] kind Information kind requested by the user
 * \param[in] agent_id Agent ID
 * \param[out] data_size Size of the information data output
 * \retval ::ROCPROFILER_STATUS_SUCCESS  if the information was found
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED  if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND <br>, if the agent was not found
 * in the saved agents
 * \retval ::ROCPROFILER_STATUS_ERROR_AGENT_INFORMATION_MISSING \n if the agent
 * was found in the saved agents but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_agent_info_size(
    rocprofiler_agent_info_kind_t kind, rocprofiler_agent_id_t agent_id,
    size_t* data_size) ROCPROFILER_VERSION_9_0;

/**
 * Query Agent Information Data using an allocated data pointer by the user,
 * user can get the size of the data using ::rocprofiler_query_agent_info_size,
 * the user can get the data using ::rocprofiler_agent_id_t and the user need to
 * identify one type of information available in ::rocprofiler_agent_info_t
 *
 * \param[in] kind Information kind requested by the user
 * \param[in] agent_id Agent ID
 * \param[out] data_size Size of the information data output
 * \retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED <br> if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND <br> if the agent was not found
 * in the saved agents
 * \retval ::ROCPROFILER_STATUS_ERROR_AGENT_INFORMATION_MISSING \n if the agent
 * was found in the saved agents but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t
rocprofiler_query_agent_info(rocprofiler_agent_info_kind_t kind, rocprofiler_agent_id_t descriptor,
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
 * \param[in] kind Information kind requested by the user
 * \param[in] agent_id Queue ID
 * \param[out] data_size Size of the information data output
 * \retval ::ROCPROFILER_STATUS_SUCCESS if the information was found
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_ERROR_QUEUE_NOT_FOUND \n if the queue was not found
 * in the saved agents
 * \retval ::ROCPROFILER_STATUS_ERROR_QUEUE_INFORMATION_MISSING \n
 * if the queue was found in the saved queues but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_queue_info_size(
    rocprofiler_queue_info_kind_t kind, rocprofiler_queue_id_t agent_id,
    size_t* data_size) ROCPROFILER_VERSION_9_0;

/**
 * Query Queue Information Data using an allocated data pointer by the user,
 * user can get the size of the data using ::rocprofiler_query_queue_info_size,
 * the user can get the data using ::rocprofiler_queue_id_t and the user need to
 * identify one type of information available in ::rocprofiler_queue_info_t
 *
 * \param[in] kind Information kind requested by the user
 * \param[in] agent_id Queue ID
 * \param[out] data_size Size of the information data output
 * \retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_ERROR_QUEUE_NOT_FOUND \n if the queue was not found
 * in the saved agents
 * \retval ::ROCPROFILER_STATUS_ERROR_QUEUE_INFORMATION_MISSING \n if the queue
 * was found in the saved agents but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t
rocprofiler_query_queue_info(rocprofiler_queue_info_kind_t kind, rocprofiler_queue_id_t descriptor,
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
 * \param[in] kernel_info_type The tyoe of information needed
 * \param[in] kernel_id Kernel ID
 * \param[out] data_size Kernel Information Data size
 * \retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_ERROR_KERNEL_NOT_FOUND \n if the kernel was not
 * found in the saved kernels
 * \retval ::ROCPROFILER_STATUS_ERROR_KERNEL_INFORMATION_MISSING \n if the kernel
 * was found in the saved counters but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_kernel_info_size(
    rocprofiler_kernel_info_kind_t kind, rocprofiler_kernel_id_t kernel_id,
    size_t* data_size) ROCPROFILER_VERSION_9_0;

/**
 * Query Kernel Information Data using an allocated data pointer by the user,
 * user can get the size of the data using ::rocprofiler_query_kernel_info_size,
 * the user can get the data using ::rocprofiler_kernel_id_t and the user need
 * to identify one type of information available in ::rocprofiler_kernel_info_t
 *
 * \param[in] kind Information kind requested by the user
 * \param[in] kernel_id Kernel ID
 * \param[out] data Information Data
 * \retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_ERROR_KERNEL_NOT_FOUND \n if the kernel was not
 * found in the saved kernels
 * \retval ::ROCPROFILER_STATUS_ERROR_KERNEL_INFORMATION_MISSING \n if the kernel
 * was found in the saved kernels but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_kernel_info(
    rocprofiler_kernel_info_kind_t kind, rocprofiler_kernel_id_t kernel_id,
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
                                                    const char* gpu_name,
                                                    uint32_t gpu_index) ROCPROFILER_VERSION_9_0;

ROCPROFILER_API rocprofiler_status_t rocprofiler_iterate_counters(
    rocprofiler_counters_info_callback_t counters_info_callback) ROCPROFILER_VERSION_9_0;

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
 * \param[in] session_id Session id where this data was collected
 * \param[in] counter_info_type The tyoe of information needed
 * \param[in] counter_id Counter ID
 * \param[out] data_size Counter Information Data size
 * \retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED \n  if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND \n if the counter was not
 * found in the saved counters
 * \retval ::ROCPROFILER_STATUS_ERROR_COUNTER_INFORMATION_MISSING \n if the counter
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
 * \param[in] session_id Session id where this data was collected
 * \param[in] kind Information kind requested by the user
 * \param[in] counter_id Counter ID
 * \param[out] data Information Data
 * \retval ::ROCPROFILER_STATUS_SUCCESS, if the information was found
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED \n if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND \n if the counter was not
 * found in the saved counters
 * \retval ::ROCPROFILER_STATUS_ERROR_COUNTER_INFORMATION_MISSING \n if the counter
 * was found in the saved counters but the required information is missing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_counter_info(
    rocprofiler_session_id_t session_id, rocprofiler_counter_info_kind_t kind,
    rocprofiler_counter_id_t counter_id, const char** data) ROCPROFILER_VERSION_9_0;

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
 * Correlation ID
 */
typedef struct {
  uint64_t value;
} rocprofiler_correlation_id_t;

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
  const rocprofiler_record_counter_instance_t* counters;
  /**
   * The count of the counters that were collected by the profiler
   */
  rocprofiler_record_counters_instances_count_t counters_count; /* Counters Count */
  /**
   * The index of the xcc from which these counters were collected
   */
  uint32_t xcc_index;
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
   * Correlation id
   */
  rocprofiler_correlation_id_t correlation_id;
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
 * struct to store the filepaths and their addresses for intercepted code objects
*/
typedef struct {
  /**
   * File path (file://, memory://) of the code object
  */
  const char* filepath;
  /**
   * Addr where codeobj is loaded
  */
  uint64_t base_address;
  /**
   * Maximum offset from base address
  */
  uint64_t mem_size;
  /**
   * If a copy of the codeobj is made, contains the data. Nullptr otherwise.
  */
  const char* data;
  /**
   * If a copy of the codeobj is made, contains the size of the data. 0 otherwise.
  */
  uint64_t data_size;
  /**
   * Timestamp for the time point this codeobj was loaded.
  */
  rocprofiler_timestamp_t clock_start;
  /**
   * Timestamp for the time point this codeobj was unloaded.
   * If the obj is still loaded by the time the record was generated, this value is 0
  */
  rocprofiler_timestamp_t clock_end;
  /**
   * Identifier for code object loading. This is the marker ID sent to the ATT buffer.
  */
  uint32_t att_marker_id;
} rocprofiler_intercepted_codeobj_t;

/**
 * Enum defines how code object is captured for ATT and PC Sampling
*/
typedef enum {
  /**
   * Capture file and memory paths for the loaded code object
  */
  ROCPROFILER_CAPTURE_SYMBOLS_ONLY = 0,
  /**
   * Capture symbols for file:// and memory:// type objects,
   * and generate a copy of all kernel code for objects under memory://
  */
  ROCPROFILER_CAPTURE_COPY_MEMORY = 1,
  /**
   * Capture symbols and all kernel code for file:// and memory:// type objects
  */
  ROCPROFILER_CAPTURE_COPY_FILE_AND_MEMORY = 2
} rocprofiler_codeobj_capture_mode_t;

/**
 * struct to store the filepaths and their addresses for intercepted code objects
*/
typedef struct {
  /**
   * List of symbols
  */
  const rocprofiler_intercepted_codeobj_t* symbols;
  /**
   * Number of symbols
  */
  uint64_t count;
  /**
   * Userdata space for custom capture.
   * For ATT records, it is the address of the kernel being launched.
  */
  uint64_t userdata;
} rocprofiler_codeobj_symbols_t;

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
   * Writer ID for counting how many kernels
   */
  uint64_t writer_id;
  /**
   * ATT data output from each shader engine.
   */
  rocprofiler_record_se_att_data_t* shader_engine_data;
  /**
   * The count of the shader engine ATT data
   */
  uint64_t shader_engine_data_count;
  /**
   * Filepaths for the intercepted code objects at the time of kernel dispatch
  */
  rocprofiler_codeobj_symbols_t intercept_list;
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
  union {
    const struct hip_api_data_s* hip;
    const struct hsa_api_data_s* hsa;
    const struct roctx_api_data_s* roctx;
  };
} rocprofiler_tracer_api_data_t;

/**
 * @brief Get Tracer API Function Name
 *
 * Return NULL if the name is not found for given domain and operation_id.
 *
 * Note: The returned string is NULL terminated.
 *
 * @param[in] domain
 * @param[in] operation_id
 * @param[out] name
 * @return ::rocprofiler_status_t
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_query_tracer_operation_name(
    rocprofiler_tracer_activity_domain_t domain, rocprofiler_tracer_operation_id_t operation_id,
    const char** name);

/**
 * @brief Get Tracer API Operation ID
 *
 * @param [in] domain
 * @param [in] name
 * @param [out] operation_id
 * @return ::rocprofiler_status_t
 */
ROCPROFILER_API rocprofiler_status_t
rocprofiler_tracer_operation_id(rocprofiler_tracer_activity_domain_t domain, const char* name,
                                rocprofiler_tracer_operation_id_t* operation_id);

/**
 * Tracing external ID
 */
typedef struct {
  uint64_t id;
} rocprofiler_tracer_external_id_t;

typedef enum {
  /**
   * No phase, it is an activity record or asynchronous output data
   */
  ROCPROFILER_PHASE_NONE = 0,
  /**
   * Enter phase for API calls
   */
  ROCPROFILER_PHASE_ENTER = 1,
  /**
   * Exit phase for API calls
   */
  ROCPROFILER_PHASE_EXIT = 2
} rocprofiler_api_tracing_phase_t;

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
   * Tracing external ID, and ROCTX ID if domain is ::ACTIVITY_DOMAIN_ROCTX
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
   * API Data
   */
  rocprofiler_tracer_api_data_t api_data;
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
  /**
   * API Tracing phase (Enter/Exit/None(Activity Records/Asynchronous Output Records))
   */
  rocprofiler_api_tracing_phase_t phase;
  /**
   * Kernel Name for HIP API calls that launches kernels or ROCTx message for ROCTx api calls
   */
  const char* name;
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
 * \param[in] begin pointer to first entry in the buffer.
 * \param[in] end pointer to one past the end entry in the buffer.
 * \param[in] session_id The session id associated with that record
 * \param[in] buffer_id The buffer id associated with that record
 */
typedef void (*rocprofiler_buffer_callback_t)(const rocprofiler_record_header_t* begin,
                                              const rocprofiler_record_header_t* end,
                                              rocprofiler_session_id_t session_id,
                                              rocprofiler_buffer_id_t buffer_id);

/**
 * Flush specific Buffer
 *
 * \param[in] session_id The created session id
 * \param[in] buffer_id The buffer ID of the created filter group
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND \n may return if
 * the session is not found
 * \retval ::ROCPROFILER_STATUS_ERROR_CORRUPTED_SESSION_BUFFER \n may return if
 * the session buffer is corrupted
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_flush_data(
    rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) ROCPROFILER_VERSION_9_0;

/**
 * Get a pointer to the next profiling record.
 * A memory pool generates buffers that contain multiple profiling records.
 * This function steps to the next profiling record.
 *
 * \param[in] record Pointer to the current profiling record in a memory pool
 * buffer.
 * \param[out] next Pointer to the following profiling record in the memory
 * pool buffer.
 * \param[in] session_id Session ID
 * \param[in] buffer_id Buffer ID
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_ERROR_RECORD_CORRUPTED \n if the function couldn't
 * get the next record because of corrupted data reported by the previous
 * record
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_next_record(
    const rocprofiler_record_header_t* record, const rocprofiler_record_header_t** next,
    rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) ROCPROFILER_VERSION_9_0;

/** @} */

/** \defgroup sessions_handling_group ROCProfiler Sessions
 * @{
 */

/**
 * Replay Profiling Modes.
 */
typedef enum {
  /**
   * No Replay to be done, Mostly for tracing tool or if the user wants to make
   * sure that no replays will be done
   */
  ROCPROFILER_NONE_REPLAY_MODE = -1,
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
 * \param[in] replay_mode The Replay strategy that should be used if replay is
 * needed
 * \param[out] session_id Pointer to the created session id, the session is
 * alive up till ::rocprofiler_destroy_session being called, however, the session
 * id can be
 * used while the session is active which can be activated using
 * ::rocprofiler_start_session and deactivated using
 * ::rocprofiler_terminate_session but ::rocprofiler_flush_data can use session_id
 * even if it is deactivated for flushing the saved records
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 */
ROCPROFILER_API rocprofiler_status_t
rocprofiler_create_session(rocprofiler_replay_mode_t replay_mode,
                           rocprofiler_session_id_t* session_id) ROCPROFILER_VERSION_9_0;

/**
 * Destroy Session
 * Destroy session created by ::rocprofiler_create_session, please refer to
 * the samples for how to use.
 * This marks the end of session and its own id life and none of the session
 * related functions will be available after this call.
 *
 * \param[in] session_id The created session id
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND \n may return if
 * the session is not found
 */
ROCPROFILER_API rocprofiler_status_t
rocprofiler_destroy_session(rocprofiler_session_id_t session_id) ROCPROFILER_VERSION_9_0;

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
   * ATT Tracing.
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
   * Add Kernel names that will be profiled or traced
   */
  ROCPROFILER_FILTER_KERNEL_NAMES = 5,
  /**
   * Add Kernel correlation ids that will be profiled or traced for ATT
   */
  ROCPROFILER_FILTER_DISPATCH_IDS = 6
} rocprofiler_filter_property_kind_t;

// TODO(aelwazir): Another way to define this as needed
typedef const char* rocprofiler_hip_function_name_t;
typedef const char* rocprofiler_hsa_function_name_t;

/**
 * ATT parameters to be used by for collection
 */
typedef enum {
  /**
   * Select the target compute unit (wgp) for profiling.
  */
  ROCPROFILER_ATT_COMPUTE_UNIT = 0,
  /**
   * VMID Mask
  */
  ROCPROFILER_ATT_VMID_MASK = 1,
  /**
   * Shader engine mask for selection.
  */
  ROCPROFILER_ATT_SE_MASK = 5,
  /**
   * Set SIMD Mask (GFX9) or SIMD ID for collection (Navi)
  */
  ROCPROFILER_ATT_SIMD_SELECT = 8,
  /**
   * Set true for occupancy collection only.
  */
  ROCPROFILER_ATT_OCCUPANCY = 9,
  /**
   * ATT collection max data size, in MB. Shared among shader engines.
  */
  ROCPROFILER_ATT_BUFFER_SIZE = 10,
  /**
   * Set ISA capture during ATT collection (rocprofiler_codeobj_capture_mode_t)
  */
  ROCPROFILER_ATT_CAPTURE_MODE = 11,
  /**
   * Mask of which compute units to generate perfcounters. GFX9 only.
  */
  ROCPROFILER_ATT_PERF_MASK = 240,
  /**
   * Select collection period for perfcounters. GFX9 only.
  */
  ROCPROFILER_ATT_PERF_CTRL = 241,
  /**
   * Select perfcounter ID (SQ block) for collection. GFX9 only.
  */
  ROCPROFILER_ATT_PERFCOUNTER = 242,
  /**
   * Select perfcounter name (SQ block) for collection. GFX9 only.
  */
  ROCPROFILER_ATT_PERFCOUNTER_NAME = 243,
  ROCPROFILER_ATT_MAXVALUE,

  ROCPROFILER_ATT_MASK = 2,           //! Deprecated
  ROCPROFILER_ATT_TOKEN_MASK = 3,     //! Deprecated
  ROCPROFILER_ATT_TOKEN_MASK2 = 4,    //! Deprecated
  ROCPROFILER_ATT_SAMPLE_RATE = 6     //! Deprecated
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
    struct {
      uint64_t start;
      uint64_t end;
    }* dispatch_ids;
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

typedef enum {
  ROCPROFILER_COUNTERS_SAMPLER_PCIE_COUNTERS = 0,
  ROCPROFILER_COUNTERS_SAMPLER_XGMI_COUNTERS = 1
} rocprofiler_counters_sampler_counter_type_t;

typedef struct {
  char* name;
  rocprofiler_counters_sampler_counter_type_t type;
} rocprofiler_counters_sampler_counter_input_t;

typedef struct {
  rocprofiler_counters_sampler_counter_type_t type;
  rocprofiler_record_counter_value_t value;
} rocprofiler_counters_sampler_counter_output_t;

typedef struct {
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
} rocprofiler_counters_sampler_parameters_t;

typedef struct {
  /**
   * ROCProfiler General Record base header to identify the id and kind of every
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
} rocprofiler_record_counters_sampler_t;

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
 * \param[in] session_id Session id where these filters will applied to
 * \param[in] filter_kind  Filter kind associated with these filters
 * \param[in] data Pointer to the filter data
 * \param[in] data_count Count of data in the data array given in ::data
 * \param[out] filter_id The id of the filter created
 * \param[in] property property needed for more filteration requests by the
 * user (Only one property is allowed per filter) (Optional)
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND Couldn't find session
 * associated with the given session identifier
 * \retval ::ROCPROFILER_STATUS_ERROR_SESSION_FILTER_DATA_MISMATCH \n The session
 * filter can't accept the given data
 * \retval ::ROCPROFILER_STATUS_ERROR_FILTER_DATA_CORRUPTED \n Data can't be read or
 * corrupted
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_create_filter(
    rocprofiler_session_id_t session_id, rocprofiler_filter_kind_t filter_kind,
    rocprofiler_filter_data_t data, uint64_t data_count, rocprofiler_filter_id_t* filter_id,
    rocprofiler_filter_property_t property) ROCPROFILER_VERSION_9_0;

/**
 * Set Session Filter Buffer
 * This function will associate buffer to a specific filter
 *
 * if the user wants to get the API traces for the api calls synchronously then
 * the user is required to call ::rocprofiler_set_api_trace_sync_callback
 *
 * \param[in] session_id Session id where these filters will applied to
 * \param[in] filter_id The id of the filter
 * \param[in] buffer_id The id of the buffer
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND Couldn't find session
 * associated with the given session identifier
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_set_filter_buffer(
    rocprofiler_session_id_t session_id, rocprofiler_filter_id_t filter_id,
    rocprofiler_buffer_id_t buffer_id) ROCPROFILER_VERSION_9_0;

/**
 * Synchronous Callback
 * To be only used by ::rocprofiler_set_api_trace_sync_callback, please refer to
 * ::rocprofiler_set_api_trace_sync_callback for more details
 *
 * \param[in] record pointer to the record.
 * \param[in] session_id The session id associated with that record
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
 * \param[in] session_id Session id where these filters will applied to
 * \param[in] filter_id The id of the filter
 * \param[in] callback Synchronous callback
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND, Couldn't find session
 * associated with the given session identifier
 * \retval ::ROCPROFILER_STATUS_ERROR_FILTER_NOT_SUPPORTED \n if the filter is not
 * related to API tracing
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_set_api_trace_sync_callback(
    rocprofiler_session_id_t session_id, rocprofiler_filter_id_t filter_id,
    rocprofiler_sync_callback_t callback) ROCPROFILER_VERSION_9_0;

/**
 * Destroy Session Filter
 * This function will destroy a specific filter
 *
 * \param[in] session_id Session id where these filters will applied to
 * \param[in] filter_id The id of the filter
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND Couldn't find session
 * associated with the given session identifier
 * \retval ::ROCPROFILER_STATUS_FILTER_NOT_FOUND Couldn't find session filter
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_destroy_filter(
    rocprofiler_session_id_t session_id, rocprofiler_filter_id_t filter_id) ROCPROFILER_VERSION_9_0;

/**
 * Create Buffer
 * This function will create a buffer that can be associated with a filter
 *
 * \param[in] session_id Session id where these filters will applied to
 * \param[in] buffer_callback Providing a callback for the buffer specialized
 * for that filters
 * \param[in] buffer_size Providing size for the buffer that will be created
 * \param[in] buffer_properties Array of Flush Properties provided by the user
 * \param[in] buffer_properties_count The count of the flush properties in the
 * array
 * \param[out] buffer_id Buffer id that was created
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND Couldn't find session
 * associated with the given session identifier
 * \retval ::ROCPROFILER_STATUS_ERROR_SESSION_PROPERTIES_MISMATCH The given
 * properties data are mismatching the properties kind
 * \retval ::ROCPROFILER_STATUS_ERROR_PROPERTY_DATA_CORRUPTED Data can't be read
 * or corrupted
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_create_buffer(
    rocprofiler_session_id_t session_id, rocprofiler_buffer_callback_t buffer_callback,
    size_t buffer_size, rocprofiler_buffer_id_t* buffer_id) ROCPROFILER_VERSION_9_0;

/**
 * Setting Buffer Properties
 * This function will set buffer properties
 *
 * \param[in] session_id Session id where the buffer is associated with
 * \param[in] buffer_id Buffer id of the buffer that the properties are going
 * to be associated with for that filters
 * \param[in] buffer_properties Array of Flush Properties provided by the user
 * \param[in] buffer_properties_count The count of the flush properties in the
 * array
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND Couldn't find session
 * associated with the given session identifier
 * \retval ::ROCPROFILER_STATUS_BUFFER_NOT_FOUND Couldn't find buffer
 * associated with the given buffer identifier
 * \retval ::ROCPROFILER_STATUS_ERROR_SESSION_PROPERTIES_MISMATCH The given
 * properties data are mismatching the properties kind
 * \retval ::ROCPROFILER_STATUS_ERROR_PROPERTY_DATA_CORRUPTED Data can't be read
 * or corrupted
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_set_buffer_properties(
    rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id,
    rocprofiler_buffer_property_t* buffer_properties,
    uint32_t buffer_properties_count) ROCPROFILER_VERSION_9_0;

/**
 * Destroy Buffer
 * This function will destroy a buffer given its id and session id
 *
 * \param[in] session_id Session id where these filters will applied to
 * \param[in] buffer_id Buffer id that will b e destroyed
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_SESSION_NOT_FOUND Couldn't find session
 * associated with the given session identifier
 * \retval ::ROCPROFILER_STATUS_BUFFER_NOT_FOUND Couldn't find buffer
 * associated with the given buffer identifier
 * \retval ::ROCPROFILER_STATUS_ERROR_SESSION_PROPERTIES_MISMATCH The given
 * properties data are mismatching the properties kind
 * \retval ::ROCPROFILER_STATUS_ERROR_PROPERTY_DATA_CORRUPTED Data can't be read
 * or corrupted
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_destroy_buffer(
    rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) ROCPROFILER_VERSION_9_0;

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
 * \param[in] counters counter filter data, it is required from the user to
 * create the filter with ::ROCPROFILER_FILTER_PROFILER_COUNTER_NAMES and to
 * provide an array of counter names needed and their count
 * \param[in] replay_mode The Replay strategy that should be used if replay is
 * needed
 * \param[in] filter_kind  Filter kind associated with these filters
 * \param[in] data Pointer to the filter data
 * \param[in] data_count Filter data array count
 * \param[in] buffer_size Size of the memory pool that will be used to save the
 * data from profiling or/and tracing, if the buffer was allocated before it
 * will be reallocated with the new size in addition to the old size
 * \param[in] buffer_callback Asynchronous callback using Memory buffers saving
 * the data and then it will be flushed if the user called
 * ::rocprofiler_flush_data or if the buffer is full or if the application
 * finished execution
 * \param[out] session_id Pointer to the created session id, the session is
 * alive up till ::rocprofiler_destroy_session being called, however, the session
 * id can be used while the session is active which can be activated using
 * ::rocprofiler_start_session and deactivated using
 * ::rocprofiler_terminate_session but ::rocprofiler_flush_data can use session_id
 * even if it is deactivated for flushing the saved records
 * \param[in] property Filter Property (Optional)
 * \param[in] callback Synchronous callback for API traces (Optional)
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_ERROR_SESSION_MODE_FILTER_MISMATCH \n The session
 * doesn't have the required mode for that filter type
 * \retval ::ROCPROFILER_STATUS_ERROR_FILTER_DATA_CORRUPTED \n Data can't be read or
 * corrupted
 * \retval ::ROCPROFILER_STATUS_ERROR_INCORRECT_SIZE If the size is less than one
 * potential record size
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_create_ready_session(
    rocprofiler_replay_mode_t replay_mode, rocprofiler_filter_kind_t filter_kind,
    rocprofiler_filter_data_t data, uint64_t data_count, size_t buffer_size,
    rocprofiler_buffer_callback_t buffer_callback, rocprofiler_session_id_t* session_id,
    rocprofiler_filter_property_t property,
    rocprofiler_sync_callback_t callback) ROCPROFILER_VERSION_9_0;

// TODO(aelwazir): Multiple sessions activate for different set of filters
/**
 * Activate Session
 * Activating session created by ::rocprofiler_create_session, please refer to
 * the samples for how to use.
 *
 * \param[in] session_id Session ID representing the created session
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully
 * \retval ::ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED, if rocprofiler_initialize
 * wasn't called before or if rocprofiler_finalize is called
 * \retval ::ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND \n may return if
 * the session is not found
 * \retval ::ROCPROFILER_STATUS_ERROR_SESSION_MODE_NOT_ADDED if there is no
 * session_mode added
 * \retval ::ROCPROFILER_STATUS_ERROR_MISSING_SESSION_CALLBACK if any
 * session_mode is missing callback set
 * \retval ::ROCPROFILER_STATUS_ERROR_HAS_ACTIVE_SESSION \n if there is already
 * active session
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_start_session(rocprofiler_session_id_t session_id)
    ROCPROFILER_VERSION_9_0;

/**
 * Deactivate Session
 * Deactivate session created by ::rocprofiler_create_session, please refer to
 * the samples for how to use.
 *
 * \param[in] session_id Session ID for the session that will be terminated
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully
 * \retval ::ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND \n may return if
 * the session is not found
 * \retval ::ROCPROFILER_STATUS_ERROR_SESSION_NOT_ACTIVE if the session is not
 * active
 */

ROCPROFILER_API rocprofiler_status_t
rocprofiler_terminate_session(rocprofiler_session_id_t session_id) ROCPROFILER_VERSION_9_0;

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
 * \param[in] counter_names The names of the counters to be collected.
 * \param[in] num_counters The number of counters specifief to be collected
 * \param[out] session_id Pointer to the created session id.
 * \param[in] cpu_index index of the cpu to be used
 * \param[in] gpu_index index of the gpu to be used
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_device_profiling_session_create(
    const char** counter_names, uint64_t num_counters, rocprofiler_session_id_t* session_id,
    int cpu_index, int gpu_index) ROCPROFILER_VERSION_9_0;

/**
 * Start the device profiling session that was created previously.
 * This will enable the GPU device to start incrementing counters
 *
 * \param[in] session_id session id of the session to start
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_device_profiling_session_start(
    rocprofiler_session_id_t session_id) ROCPROFILER_VERSION_9_0;

/**
 * Poll the device profiling session to read counters from the GPU device.
 * This will read out the values of the counters from the GPU device at the
 * specific instant when this API is called. This is a thread-blocking call.
 * Any thread that calls this API will have to wait until
 * the counter values are being read out.
 *
 * \param[in] session_id session id of the session to start
 * \param[out] data records of counter data read out from device
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_device_profiling_session_poll(
    rocprofiler_session_id_t session_id,
    rocprofiler_device_profile_metric_t* data) ROCPROFILER_VERSION_9_0;

/**
 * Stop the device profiling session that was created previously.
 * This will inform the GPU device to stop counters collection.
 *
 * \param[in] session_id session id of the session to start
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_device_profiling_session_stop(
    rocprofiler_session_id_t session_id) ROCPROFILER_VERSION_9_0;

/**
 * Destroy the device profiling session that was created previously.
 *
 * \param[in] session_id session id of the session to start
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
ROCPROFILER_API rocprofiler_status_t rocprofiler_device_profiling_session_destroy(
    rocprofiler_session_id_t session_id) ROCPROFILER_VERSION_9_0;

/**
 * Creates a codeobj capture record, returned in ID.
 * \param[out] id contains a handle for the created record.
 * \param[in] mode Set to capture symbols only,
 *                  make a copy of codeobj under memory://
 *                  or copy all codeobj.
 * \param[in] userdata userdata to be returned in the record. For ATT records, is the kernel addr.
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed successfully.
 */
ROCPROFILER_API rocprofiler_status_t
rocprofiler_codeobj_capture_create(
  rocprofiler_record_id_t* id,
  rocprofiler_codeobj_capture_mode_t mode,
  uint64_t userdata
);

/**
 * API to get the captured codeobj.
 * Each call invalidates the previous pointer for the same ID.
 * \param[in] id record handle.
 * \param[out] capture captured code objects.
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENTS invalid ID.
 */
ROCPROFILER_API rocprofiler_status_t
rocprofiler_codeobj_capture_get(rocprofiler_record_id_t id,
                                rocprofiler_codeobj_symbols_t* capture);

/**
 * API to delete a record.
 * Invalidates the pointer returned from rocprofiler_codeobj_capture_get.
 * \param[in] id record handle.
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed successfully.
 */
ROCPROFILER_API rocprofiler_status_t
rocprofiler_codeobj_capture_free(rocprofiler_record_id_t id);

/**
 * Records the current loaded codeobjs and any following loads until stop() is called.
 * \param[in] id record handle.
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENTS invalid ID.
 */ 
ROCPROFILER_API rocprofiler_status_t
rocprofiler_codeobj_capture_start(rocprofiler_record_id_t id);

/**
 * Stops recording of future codeobjs, until start() is called again.
 * Calling stop() immediately after a start() snapshots the current state of loaded codeobjs.
 * \param[in] id record handle.
 * \retval ::ROCPROFILER_STATUS_SUCCESS The function has been executed successfully.
 * \retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENTS invalid ID.
 */
ROCPROFILER_API rocprofiler_status_t
rocprofiler_codeobj_capture_stop(rocprofiler_record_id_t id);

/** @} */

#ifdef __cplusplus
}  // extern "C" block
#endif  // __cplusplus

#endif  // INC_ROCPROFILER_H_