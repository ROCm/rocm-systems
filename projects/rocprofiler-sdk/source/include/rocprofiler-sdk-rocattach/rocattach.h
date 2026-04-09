// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * @file rocattach.h
 * @brief rocAttach API interface for AMD profiling data analysis
 *
 * @mainpage rocAttach API Specification
 *
 */

#include "rocprofiler-sdk-rocattach/defines.h"
#include "rocprofiler-sdk-rocattach/types.h"

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

#include "rocprofiler-sdk-rocattach/version.h"

ROCATTACH_EXTERN_C_INIT

/**
 * @fn rocattach_status_t rocattach_get_version(uint32_t* major, uint32_t* minor, uint32_t*
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
 * @return ::rocattach_status_t
 * @retval ::ROCATTACH_STATUS_SUCCESS Always returned
 */
rocattach_status_t
rocattach_get_version(uint32_t* major, uint32_t* minor, uint32_t* patch) ROCATTACH_API;

/**
 * @brief Simplified alternative to ::rocattach_get_version
 *
 * Returns the version of the rocprofiler-sdk library loaded at runtime.  This can be used to check
 * if the runtime version is equal to or compatible with the version of rocprofiler-sdk used during
 * compilation time. This function can be invoked before tool initialization.
 *
 * @param [out] info Pointer to version triplet struct which will be populated by the function call.
 * @return ::rocattach_status_t
 * @retval ::ROCATTACH_STATUS_SUCCESS Always returned
 */
rocattach_status_t
rocattach_get_version_triplet(rocattach_version_triplet_t* info) ROCATTACH_API ROCATTACH_NONNULL(1);

/**
 * @brief Attach to a process ID
 *
 * Attempts to attach to a rocm process at the given process identifier (PID). If successful, the
 * target process will then load rocprofiler-sdk, which will subsequently load any tool libraries
 * given in the environment variable ROCPROF_ATTACH_TOOL_LIBRARY. This environment variable should
 * be set for the attacher process before calling rocattach_attach(). It does not need to be set
 * for the attachee.
 *
 * @param [in] pid Process ID to attach to
 * @return ::rocattach_status_t
 * @retval ::ROCATTACH_STATUS_SUCCESS Attachment successful
 */
rocattach_status_t
rocattach_attach(int pid) ROCATTACH_API;

/**
 * @brief Detach from a process ID
 *
 * Detaches from a previous attachment to the given process identifier (PID). If successful, the
 * target process pauses rocprofiler-sdk, but the library will remain loaded. The PID can be
 * attached to again after detach is completed. A PID of 0 can be specified to detach from all
 * current sessions.
 *
 * @param [in] pid Process ID to detach from
 * @return ::rocattach_status_t
 * @retval ::ROCATTACH_STATUS_SUCCESS Detachment successful
 */
rocattach_status_t
rocattach_detach(int pid) ROCATTACH_API;

/**
 * @defgroup MISCELLANEOUS_GROUP Miscellaneous Utility Functions
 * @brief utility functions for library
 * @{
 */

/**
 * @fn const char* rocattach_get_status_name(rocattach_status_t status)
 * @brief Return the string encoding of ::rocattach_status_t value
 * @param [in] status error code value
 * @return Will return a nullptr if invalid/unsupported ::rocattach_status_t value is provided.
 */
const char*
rocattach_get_status_name(rocattach_status_t status) ROCATTACH_API;

/**
 * @fn const char* rocattach_get_status_string(rocattach_status_t status)
 * @brief Return the message associated with ::rocattach_status_t value
 * @param [in] status error code value
 * @return Will return a nullptr if invalid/unsupported ::rocattach_status_t value is provided.
 */
const char*
rocattach_get_status_string(rocattach_status_t status) ROCATTACH_API;

/** @} */

ROCATTACH_EXTERN_C_FINI
