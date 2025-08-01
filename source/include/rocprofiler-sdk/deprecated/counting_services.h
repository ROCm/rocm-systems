// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

/**
 * @file counting_services.h
 * @brief Backward compatibility support for counting service API changes
 *
 * This header provides backward compatibility for applications that use the old
 * counting service function and type names from ROCm 6.4. Include this header
 * to use the old names with the new implementation.
 */

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>

ROCPROFILER_EXTERN_C_INIT

//==============================================================================
// DISPATCH COUNTING SERVICE BACKWARD COMPATIBILITY
//==============================================================================

/**
 * @brief (deprecated) Replaced by ::rocprofiler_dispatch_counting_service_cb_t
 */
ROCPROFILER_SDK_DEPRECATED("rocprofiler_dispatch_counting_service_callback_t renamed to "
                           "rocprofiler_dispatch_counting_service_cb_t")
typedef rocprofiler_dispatch_counting_service_cb_t rocprofiler_dispatch_counting_service_callback_t;

/**
 * @brief (deprecated) Replaced by ::rocprofiler_dispatch_counting_record_cb_t
 */
ROCPROFILER_SDK_DEPRECATED("rocprofiler_profile_counting_record_callback_t renamed to "
                           "rocprofiler_dispatch_counting_record_cb_t")
typedef rocprofiler_dispatch_counting_record_cb_t rocprofiler_profile_counting_record_callback_t;

/**
 * @brief (deprecated) Replaced by ::rocprofiler_configure_buffer_dispatch_counting_service
 */
ROCPROFILER_SDK_DEPRECATED("rocprofiler_configure_buffered_dispatch_counting_service renamed to "
                           "rocprofiler_configure_buffer_dispatch_counting_service")
static inline rocprofiler_status_t
rocprofiler_configure_buffered_dispatch_counting_service(
    rocprofiler_context_id_t                         context_id,
    rocprofiler_buffer_id_t                          buffer_id,
    rocprofiler_dispatch_counting_service_callback_t callback,
    void*                                            callback_data_args)
{
    return rocprofiler_configure_buffer_dispatch_counting_service(
        context_id, buffer_id, callback, callback_data_args);
}

//==============================================================================
// DEVICE COUNTING SERVICE BACKWARD COMPATIBILITY
//==============================================================================

/**
 * @brief (deprecated) Replaced by ::rocprofiler_device_counting_service_cb_t
 */
ROCPROFILER_SDK_DEPRECATED("rocprofiler_device_counting_service_callback_t renamed to "
                           "rocprofiler_device_counting_service_cb_t")
typedef rocprofiler_device_counting_service_cb_t rocprofiler_device_counting_service_callback_t;

//==============================================================================
// CORRELATION ID BACKWARD COMPATIBILITY
//==============================================================================

/**
 * @brief (deprecated) Replaced by ::rocprofiler_async_correlation_id_t
 *
 * Note: This provides basic compatibility but ::rocprofiler_async_correlation_id_t
 * may have additional fields not present in the original ::rocprofiler_correlation_id_t
 */
ROCPROFILER_SDK_DEPRECATED(
    "rocprofiler_correlation_id_t enhanced to rocprofiler_async_correlation_id_t")
typedef rocprofiler_async_correlation_id_t rocprofiler_correlation_id_t;

//==============================================================================
// ADDITIONAL TYPE ALIASES FOR COMMON PATTERNS
//==============================================================================

/**
 * @brief (deprecated) Old dimension info type name
 */
ROCPROFILER_SDK_DEPRECATED(
    "rocprofiler_record_dimension_info_t renamed to rocprofiler_counter_record_dimension_info_t")
typedef rocprofiler_counter_record_dimension_info_t rocprofiler_record_dimension_info_t;

/**
 * @brief (deprecated) Counter info version 0 - for backward compatibility with pre-v1 API
 */
ROCPROFILER_SDK_DEPRECATED("Use rocprofiler_counter_info_v1_t for enhanced dimension support")
typedef rocprofiler_counter_info_v0_t rocprofiler_counter_info_t;

//==============================================================================
// FUNCTION WRAPPER COMPATIBILITY NOTES
//==============================================================================

/*
 * The following functions have been renamed but cannot be easily aliased with
 * static inline functions due to experimental markings or complex signatures.
 * Users should update their code to use the new names:
 *
 * OLD NAME                                          → NEW NAME
 * ===============================================      =====================================
 * rocprofiler_configure_buffered_dispatch_counting_service  →
 * rocprofiler_configure_buffer_dispatch_counting_service
 * rocprofiler_configure_callback_dispatch_counting_service  → (same name, but parameter types
 * changed) rocprofiler_configure_device_counting_service            → (same name, but parameter
 * types changed) rocprofiler_sample_device_counting_service               → (same name, but
 * parameter types changed)
 *
 * Parameter type changes:
 * - rocprofiler_profile_config_id_t          → rocprofiler_counter_config_id_t
 * - rocprofiler_record_counter_t             → rocprofiler_counter_record_t
 * - rocprofiler_correlation_id_t             → rocprofiler_async_correlation_id_t
 * - All callback typedefs renamed with _cb_t suffix
 */

//==============================================================================
// USAGE EXAMPLE
//==============================================================================

/*
 * To use this backward compatibility header, include it after the main headers:
 *
 * #include <rocprofiler-sdk/rocprofiler.h>
 * #include <rocprofiler-sdk/deprecated/counting_services.h>
 *
 * // Now you can use old names:
 * rocprofiler_dispatch_counting_service_callback_t old_callback;
 * rocprofiler_record_counter_t* old_records;
 * rocprofiler_configure_buffered_dispatch_counting_service(...);
 */

ROCPROFILER_EXTERN_C_FINI