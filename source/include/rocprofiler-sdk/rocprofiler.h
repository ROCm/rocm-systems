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
 *
 * @file rocprofiler.h
 * @brief ROCProfiler-SDK API interface.
 *
 * @mainpage ROCProfiler-SDK API Specification
 *
 * @section introduction Introduction
 * ROCprofiler-SDK is a library that implements the profiling and performance monitoring
 * capabilities for AMD's ROCm platform. It provides a comprehensive set of APIs for:
 *
 * - Hardware performance counters monitoring
 * - PC sampling for kernel execution analysis
 * - Buffer and callback-based tracing mechanisms
 * - Device and dispatch counting services
 * - External correlation tracking
 * - HIP and HSA runtime profiling support
 *
 * The library is designed to help developers analyze and optimize the performance
 * of applications running on AMD GPUs. It offers both low-level hardware access
 * and high-level profiling abstractions to accommodate different profiling needs.
 *
 *
 * \section compatibility Compatibility
 * ROCprofiler-SDK is compatible with AMD ROCm platform and supports
 * profiling of applications using HIP and HSA runtimes.
 */

#include <stddef.h>
#include <stdint.h>

#include "rocprofiler-sdk/defines.h"
#include "rocprofiler-sdk/fwd.h"

/**
 * @defgroup VERSIONING_GROUP Library Versioning
 * @brief Version information about the interface and the associated installed library.
 *
 * The semantic version of the interface following semver.org rules. A context
 * that uses this interface is only compatible with the installed library if
 * the major version numbers match and the interface minor version number is
 * less than or equal to the installed library minor version number.
 *
 * @{
 */

#include "rocprofiler-sdk/version.h"

ROCPROFILER_EXTERN_C_INIT

/**
 * @fn rocprofiler_status_t rocprofiler_get_version(uint32_t* major, uint32_t* minor, uint32_t*
 * patch)
 * @brief Query the version of the installed library.
 *
 * Returns the version of the rocprofiler-sdk library loaded at runtime.  This can be used to check
 * if the runtime version is equal to or compatible with the version of rocprofiler-sdk used during
 * compilation time. This function can be invoked before tool initialization.
 *
 * @param [out] major The major version number is stored if non-NULL.
 * @param [out] minor The minor version number is stored if non-NULL.
 * @param [out] patch The patch version number is stored if non-NULL.
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS Always returned
 */
rocprofiler_status_t
rocprofiler_get_version(uint32_t* major, uint32_t* minor, uint32_t* patch) ROCPROFILER_API;

/**
 * @brief Simplified alternative to ::rocprofiler_get_version
 *
 * Returns the version of the rocprofiler-sdk library loaded at runtime.  This can be used to check
 * if the runtime version is equal to or compatible with the version of rocprofiler-sdk used during
 * compilation time. This function can be invoked before tool initialization.
 *
 * @param [out] info Pointer to version triplet struct which will be populated by the function call.
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS Always returned
 */
rocprofiler_status_t
rocprofiler_get_version_triplet(rocprofiler_version_triplet_t* info) ROCPROFILER_API
    ROCPROFILER_NONNULL(1);

ROCPROFILER_EXTERN_C_FINI

/** @} */

#include "rocprofiler-sdk/agent.h"
#include "rocprofiler-sdk/buffer.h"
#include "rocprofiler-sdk/buffer_tracing.h"
#include "rocprofiler-sdk/callback_tracing.h"
#include "rocprofiler-sdk/context.h"
#include "rocprofiler-sdk/counter_config.h"
#include "rocprofiler-sdk/counters.h"
#include "rocprofiler-sdk/device_counting_service.h"
#include "rocprofiler-sdk/dispatch_counting_service.h"
#include "rocprofiler-sdk/external_correlation.h"
#include "rocprofiler-sdk/hip.h"
#include "rocprofiler-sdk/hsa.h"
#include "rocprofiler-sdk/intercept_table.h"
#include "rocprofiler-sdk/internal_threading.h"
#include "rocprofiler-sdk/marker.h"
#include "rocprofiler-sdk/ompt.h"
#include "rocprofiler-sdk/pc_sampling.h"
#include "rocprofiler-sdk/rccl.h"
#include "rocprofiler-sdk/rocdecode.h"
#include "rocprofiler-sdk/rocjpeg.h"
// #include "rocprofiler-sdk/spm.h"

// subject to removal
#include "rocprofiler-sdk/deprecated/profile_config.h"

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup MISCELLANEOUS_GROUP Miscellaneous Utility Functions
 * @brief utility functions for library
 * @{
 */

/**
 * @fn rocprofiler_status_t rocprofiler_get_timestamp(rocprofiler_timestamp_t* ts)
 * @brief Get the timestamp value that rocprofiler uses
 * @param [out] ts Output address of the rocprofiler timestamp value
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS Always returned
 */
rocprofiler_status_t
rocprofiler_get_timestamp(rocprofiler_timestamp_t* ts) ROCPROFILER_API ROCPROFILER_NONNULL(1);

/**
 * @fn rocprofiler_status_t rocprofiler_get_thread_id(rocprofiler_thread_id_t* tid)
 * @brief Get the identifier value of the current thread that is used by rocprofiler
 * @param [out] tid Output address of the rocprofiler thread id value
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS Always returned
 */
rocprofiler_status_t
rocprofiler_get_thread_id(rocprofiler_thread_id_t* tid) ROCPROFILER_API ROCPROFILER_NONNULL(1);

/**
 * @fn const char* rocprofiler_get_status_name(rocprofiler_status_t status)
 * @brief Return the string encoding of ::rocprofiler_status_t value
 * @param [in] status error code value
 * @return Will return a nullptr if invalid/unsupported ::rocprofiler_status_t value is provided.
 */
const char*
rocprofiler_get_status_name(rocprofiler_status_t status) ROCPROFILER_API;

/**
 * @fn const char* rocprofiler_get_status_string(rocprofiler_status_t status)
 * @brief Return the message associated with ::rocprofiler_status_t value
 * @param [in] status error code value
 * @return Will return a nullptr if invalid/unsupported ::rocprofiler_status_t value is provided.
 */
const char*
rocprofiler_get_status_string(rocprofiler_status_t status) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
