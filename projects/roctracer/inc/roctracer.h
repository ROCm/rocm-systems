/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

/** \mainpage ROC Tracer API Specification
 *
 * \section introduction Introduction
 *
 * ROC-tracer library, Runtimes Generic Callback/Activity APIs.
 *
 * The goal of the implementation is to provide a generic independent from
 * specific runtime profiler to trace API and asynchronous activity.
 *
 * The API provides functionality for registering the runtimes API callbacks and
 * asynchronous activity records pool support.
*/

/**
 * \file
 * ROC tracer API interface.
 */

#ifndef INC_ROCTRACER_H_
#define INC_ROCTRACER_H_

#include <stdint.h>
#include <stddef.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#include <ext/prof_protocol.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/** \defgroup versioning_group Versioning
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
#define ROCTRACER_VERSION_MAJOR 4

/**
 * The minor version of the interface as a macro so it can be used by the
 * preprocessor.
 */
#define ROCTRACER_VERSION_MINOR 0

/**
 * Query the major version of the installed library.
 *
 * Return the major version of the installed library.  This can be used to
 * check if it is compatible with this interface version.  This function can be
 * used even when the library is not initialized.
 */
uint32_t roctracer_version_major();

/**
 * Query the minor version of the installed library.
 *
 * Return the minor version of the installed library.  This can be used to
 * check if it is compatible with this interface version.  This function can be
 * used even when the library is not initialized.
 */
uint32_t roctracer_version_minor();

/** @} */

/** \defgroup status_codes_group Status Codes
 *
 * Most operations return a status code to indicate success or error.
 *
 * @{
 */

/**
 * ROC Tracer API status codes.
 */
typedef enum {
  /**
   * The function has executed successfully.
   */
  ROCTRACER_STATUS_SUCCESS = 0,
  /**
   * A generic error has occurred.
   */
  ROCTRACER_STATUS_ERROR = -1,
  /**
   * The domain ID is invalid.
   */
  ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID = -2,
  /**
   * An invalid argument was given to the function.
   */
  ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT = -3,
  /**
   * No default pool is defined.
   */
  ROCTRACER_STATUS_ERROR_DEFAULT_POOL_UNDEFINED = -4,
  /**
   * The default pool is already defined.
   */
  ROCTRACER_STATUS_ERROR_DEFAULT_POOL_ALREADY_DEFINED = -5,
  /**
   * Memory allocation error.
   */
  ROCTRACER_STATUS_ERROR_MEMORY_ALLOCATION = -6,
  /**
   * External correlation ID pop mismatch.
   */
  ROCTRACER_STATUS_ERROR_MISMATCHED_EXTERNAL_CORRELATION_ID = -7,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_UNINIT = 2,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_BREAK = 3,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_BAD_DOMAIN = ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_BAD_PARAMETER = ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_HIP_API_ERR = 6,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_HIP_OPS_ERR = 7,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_HCC_OPS_ERR = ROCTRACER_STATUS_HIP_OPS_ERR,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_HSA_ERR = 7,
  /**
   * Deprecated error code.
   */
  ROCTRACER_STATUS_ROCTX_ERR = 8,
} roctracer_status_t;

/**
 * Query the textual description of the last error for the current thread.
 *
 * Returns a NUL terminated string describing the error of the last ROC Tracer
 * API call by the calling thread that did not return success.  The empty
 * string is returned if there is no previous error.  The last error is not
 * cleared.
 *
 * \return Return the error string.  The caller owns the returned string and
 * should use \p free() to deallocate it.
 */
const char* roctracer_error_string();

/** @} */

/** \defgroup domain_group Traced Runtime Domains
 *
 * The ROC Tracer API can trace multiple runtime libraries.  Each library can
 * have API operations and asynchronous operations that can be traced.
 *
 * @{
 */

/**
 * Enumeration of domains that can be traced.
 */
typedef activity_domain_t roctracer_domain_t;

/**
 * Query textual name of an operation of a domain.
 *
 * @param[in] domain Domain being queried.
 *
 * @param[in] op Operation within \p domain.
 *
 * @param[in] kind \todo Define kind.
 *
 * @return Returns the NUL terminated string for the operation name, or NULL if
 * the domain or operation are invalid.  The string is owned by the ROC Tracer
 * library.
 */
const char* roctracer_op_string(uint32_t domain,
                                uint32_t op,
                                uint32_t kind);

/**
 * Query the operation code given a domain and the name of an operation.
 *
 * @param[in] domain The domain being queried.
 *
 * @param[in] str The NUL terminated name of the operation name being queried.
 *
 * @param[out] op The operation code.
 *
 * @param[out] kind If not NULL then the operation kind code.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.  \p op and \p kind have been updated.
 *
 * @retval ::ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT The \p op is invalid for
 * \p domain.
 *
 * @retval ::ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID The domain is invalid or
 * not supported.
 */
roctracer_status_t roctracer_op_code(uint32_t domain,
                                     const char* str,
                                     uint32_t* op,
                                     uint32_t* kind);

/**
 * Set the properties of a domain.
 *
 * @param[in] domain The domain.
 *
 * @param[in] properties The properties. Each domain defines its own type for
 * the properties. Some domains require the properties to be set before they
 * can be enabled.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
roctracer_status_t roctracer_set_properties(roctracer_domain_t domain,
                                            void* properties);

/** @} */

/** \defgroup callback_api_group Callback API
 *
 * ROC tracer provides support for runtime API callbacks and activity
 * records logging. The API callbacks provide the API calls arguments and are
 * called on different phases, on enter, on exit, on kernel completion.
 *
 * @{
 */

/**
 * Runtime API callback type.
 *
 * The callback that will be invoked when an enabled runtime API is called. The
 * callback is invoked on entry and on exit.
 */
typedef activity_rtapi_callback_t roctracer_rtapi_callback_t;

/**
 * Enable runtime API callback for a specific operation of a domain.
 *
 * @param domain The domain.
 *
 * @param op The operation ID in \p domain.
 *
 * @param callback The callback to invoke each time the operation is performed
 * on entry and exit.
 *
 * @param arg Value to pass as last argument of \p callback.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID \p domain is invalid.
 *
 * @retval ::ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT \p op is invalid for \p
 * domain.
 */
roctracer_status_t roctracer_enable_op_callback(
    activity_domain_t domain,
    uint32_t op,
    activity_rtapi_callback_t callback,
    void* arg);

/**
 * Enable runtime API callback for all operations of a domain.
 *
 * @param domain The domain
 *
 * @param callback The callback to invoke each time the operation is performed
 * on entry and exit.
 *
 * @param arg Value to pass as last argument of \p callback.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID \p domain is invalid.
 */
roctracer_status_t roctracer_enable_domain_callback(
    activity_domain_t domain,
    activity_rtapi_callback_t callback,
    void* arg);

/**
 * Enable runtime API callback for all operations of all domains.
 *
 * @param callback The callback to invoke each time the operation is performed
 * on entry and exit.
 *
 * @param arg Value to pass as last argument of \p callback.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
roctracer_status_t roctracer_enable_callback(
    activity_rtapi_callback_t callback,
    void* arg);

/**
 * Disable runtime API callback for a specific operation of a domain.
 *
 * @param domain The domain
 *
 * @param op The operation in \p domain.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID \p domain is invalid.
 *
 * @retval ::ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT \p op is invalid for \p
 * domain.
 */
roctracer_status_t roctracer_disable_op_callback(activity_domain_t domain,
                                                 uint32_t op);

/**
 * Disable runtime API callback for all operations of a domain.
 *
 * @param domain The domain
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ::ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID \p domain is invalid.
 */
roctracer_status_t roctracer_disable_domain_callback(activity_domain_t domain);

/**
 * Disable runtime API callback for all operations of all domains.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
roctracer_status_t roctracer_disable_callback();

/** @} */

/** \defgroup activity_api_group Activity API
 *
 * The activity records are asynchronously logged to the pool and can be associated
 * with the respective API callbacks using the correlation ID. Activity API can
 * be used to enable collecting of the records with timestamping data for API
 * calls and the kernel submits.
 *
 * @{
 */

/**
 * Activity record.
 *
 * Asynchronous activity events generate activity records.
 */
typedef activity_record_t roctracer_record_t;

/**
 * Get a pointer to the next activity record.
 *
 * A memory pool generates buffers that contain multiple activity records.
 * This function steps to the next activity record.
 *
 * @param[in] record Pointer to ac activity record in a memory pool buffer.
 *
 * @param[out] next Pointer to the following activity record in the memory pool
 * buffer.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
roctracer_status_t roctracer_next_record(
    const activity_record_t* record,
    const activity_record_t** next);

/**
 * Memory pool allocator callback.
 *
 * If \p *ptr is NULL, then allocate memory of \p size bytes and save address
 * in \p *ptr.
 * 
 * If \p *ptr is non-NULL and size is non-0, then reallocate the memory at \p
 * *ptr with size \p size and save the address in \p *ptr. The memory will have
 * been allocated by the same callback.
 * 
 * If \p *ptr is non-NULL and size is 0, then deallocate the memory at \p *ptr.
 * The memory will have been allocated by the same callback.
 *
 * \p size is the size of the memory allocation or reallocation, or 0 if
 * deallocating.
 *
 * \p arg Argument provided in the ::roctracer_properties_t passed to the
 * ::roctracer_open_pool function.
 */
typedef void (*roctracer_allocator_t)(char** ptr,
                                      size_t size,
                                      void* arg);

/**
 * Memory pool buffer callback.
 *
 * The callback that will be invoked when a memory pool buffer becomes full or
 * is flushed.
 *
 * \p begin pointer to first entry entry in the buffer.
 *
 * \p end pointer to one past the end entry in the buffer.
 *
 * \p arg the argument specified when the callback was defined.
 */
typedef void (*roctracer_buffer_callback_t)(
    const char* begin,  // [in] available buffered trace records
    const char* end,    // [in] end of buffered trace records
    void* arg);         // [in/out] callback arg

/**
 * Memory pool properties.
 *
 * Defines the propertis when a tracer memory pool is created.
 */
typedef struct {

  /**
   * ROC Tracer mode.
   */
  uint32_t mode;

  /**
   * Size of buffer in bytes.
   */
  size_t buffer_size;

  /**
   * The allocator function to use to allocate and deallocate the buffer. If
   * NULL then \p malloc, \p realloc, and \p free are used.
   */
  roctracer_allocator_t alloc_fun;

  /**
   * The argument to pass when invoking the \p alloc_fun allocator.
   */
  void* alloc_arg;

  /**
   * The function to call when a buffer becomes full or is flushed.
   */
  roctracer_buffer_callback_t buffer_callback_fun;

  /**
   * The argument to pass when invoking the \p buffer_callback_fun callback.
   */
  void* buffer_callback_arg;
} roctracer_properties_t;

/**
 * Tracer memory pool type.
 */
typedef void roctracer_pool_t;

/**
 * Create tracer memory pool.
 *
 * If \p pool is not NULL, returns the created memory pool. Does not change the
 * default memory pool.
 *
 * If \p pool is NULL, sets the default memory pool to the created pool if not
 * already defined. Otherwise, return an error.
 *
 * @param[in] properties Tracer memory pool properties.
 *
 * @param[out] pool Tracer memory pool created if not NULL.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ROCTRACER_STATUS_ERROR_DEFAULT_POOL_ALREADY_DEFINED \p pool is NULL
 * and the default pool is already defined. Unable to create the pool.
 *
 * @retval ROCTRACER_STATUS_ERROR_MEMORY_ALLOCATION Unable to allocate memory
 * for the \p pool. Unable to create the pool.
 */
roctracer_status_t roctracer_open_pool_expl(
    const roctracer_properties_t* properties,
    roctracer_pool_t** pool);

/**
 * Create tracer memory pool.
 *
 * Sets the default memory pool to the created pool if not already defined.
 * Otherwise, return an error.
 *
 * @param[in] properties Tracer memory pool properties.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ROCTRACER_STATUS_ERROR_DEFAULT_POOL_ALREADY_DEFINED The default pool
 * is already defined. Unable to create the pool.
 *
 * @retval ROCTRACER_STATUS_ERROR_MEMORY_ALLOCATION Unable to allocate memory
 * for the \p pool. Unable to create the pool.
 */
roctracer_status_t roctracer_open_pool(const roctracer_properties_t* properties);

/**
 * Close tracer memory pool.
 *
 * All enabled activities that use the pool must have completed writing to the
 * pool, before deleting the pool. Deleting a pool automatically disables any
 * activities that specify the pool, and flushes it.
 *
 * @param[in] pool Memory pool to close. If NULL, the default memory pool is
 * closed if defined. The default memory pool is set to undefined if closed.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully or pool was NULL and there is no default pool.
 */
roctracer_status_t roctracer_close_pool_expl(
    roctracer_pool_t* pool);

/**
 * Close default tracer memory pool, if defined, and set to undefined.
 *
 * All enabled activities that use the pool must have completed writing to the
 * pool, before deleting the pool. Deleting a pool automatically disables any
 * activities that specify the pool, and flushes it.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully or there is no default pool.
 */
roctracer_status_t roctracer_close_pool();

/**
 * Query and set the default memory pool.
 *
 * @param[in] pool If not NULL, change the current default pool to \p pool. If
 * NULL, the default pool is not changed.
 *
 * @return Return the current default memory pool before any change, or NULL if
 * none is defined.
 */
// Return current default pool
// Set new default pool if the argument is not NULL
roctracer_pool_t* roctracer_default_pool_expl(
    roctracer_pool_t* pool);

/**
 * Query the current default memory pool.
 *
 * @return Return the current default memory pool, or NULL is none is defined.
 */
roctracer_pool_t* roctracer_default_pool();

/**
 * Enable activity record logging for a specified operation of a domain
 * providing a memory pool.
 *
 * @param[in] domain The domain.
 *
 * @param[in] op The activity operation ID in \p domain.
 *
 * @param[in] pool The memory pool to write the activity record. If NULL, use the
 * default memory pool.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ROCTRACER_STATUS_ERROR \p pool is NULL and no default pool is defined.
 */
roctracer_status_t roctracer_enable_op_activity_expl(
    activity_domain_t domain,
    uint32_t op,
    roctracer_pool_t* pool);

/**
 * Enable activity record logging for a specified operation of a domain using
 * the default memory pool.
 *
 * @param[in] domain The domain.
 *
 * @param[in] op The activity operation ID in \p domain.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ROCTRACER_STATUS_ERROR No default pool is defined.
 */
roctracer_status_t roctracer_enable_op_activity(
    activity_domain_t domain,
    uint32_t op);

/**
 * Enable activity record logging for all operations of a domain providing a
 * memory pool.
 *
 * @param[in] domain The domain.
 *
 * @param[in] pool The memory pool to write the activity record. If NULL, use the
 * default memory pool.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ROCTRACER_STATUS_ERROR \p pool is NULL and no default pool is defined.
 */
roctracer_status_t roctracer_enable_domain_activity_expl(
    activity_domain_t domain,
    roctracer_pool_t* pool);

/**
 * Enable activity record logging for all operations of a domain using the
 * default memory pool.
 *
 * @param[in] domain The domain.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ROCTRACER_STATUS_ERROR No default pool is defined.
 */
roctracer_status_t roctracer_enable_domain_activity(activity_domain_t domain);

/**
 * Enable activity record logging for all operations of all domains providing a
 * memory pool.
 *
 * @param[in] pool The memory pool to write the activity record. If NULL, use the
 * default memory pool.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ROCTRACER_STATUS_ERROR \p pool is NULL and no default pool is defined.
 */
roctracer_status_t roctracer_enable_activity_expl(
    roctracer_pool_t* pool);

/**
 * Enable activity record logging for all operations of all domains using the
 * default memory pool.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 *
 * @retval ROCTRACER_STATUS_ERROR No default pool is defined.
 */
roctracer_status_t roctracer_enable_activity();

/**
 * Disable activity record logging for a specified operation of a domain.
 *
 * @param[in] domain The domain.
 *
 * @param[in] op The activity operation ID in \p domain.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
//
roctracer_status_t roctracer_disable_op_activity(activity_domain_t domain,
                                                 uint32_t op);

/**
 * Disable activity record logging for all operations of a domain.
 *
 * @param[in] domain The domain.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
roctracer_status_t roctracer_disable_domain_activity(activity_domain_t domain);

/**
 * Disable activity record logging for all operations of all domains.
 *
 * @param[in] op The activity operation ID in \p domain.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
roctracer_status_t roctracer_disable_activity();

/**
 * Flush available activity records for a memory pool.
 *
 * If flushing encounters an activity record still being written, flushing
 * stops. Use a subsequent flush when the record has completed being written to
 * resume the flush.
 *
 * @param[in] pool The memory pool to flush. If NULL, flushes the default memory
 * pool.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
roctracer_status_t roctracer_flush_activity_expl(
    roctracer_pool_t* pool);

/**
 * Flush available activity records for the default memory pool.
 *
 * If flushing encounters an activity record still being written, flushing
 * stops. Use a subsequent flush when the record has completed being written to
 * resume the flush.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
roctracer_status_t roctracer_flush_activity();

/** @} */

/** \defgroup timestamp_group Timestamp Operations
 *
 *
 *
 * @{
 */

/**
 * Get the system clock timestamp.
 *
 * @param[out] timestamp The system clock timestamp in nano seconds.
 *
 * @retval ::ROCTRACER_STATUS_SUCCESS The function has been executed
 * successfully.
 */
roctracer_status_t roctracer_get_timestamp(uint64_t* timestamp);

/** @} */

#ifdef __cplusplus
}  // extern "C" block
#endif  // __cplusplus

#endif  // INC_ROCTRACER_H_
