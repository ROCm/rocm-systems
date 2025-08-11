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
 * @file rocpd.h
 * @brief rocPD API interface for AMD profiling data analysis
 *
 * @mainpage rocPD API Specification
 *
 */

#include "rocprofiler-sdk-rocpd/defines.h"
#include "rocprofiler-sdk-rocpd/types.h"

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

#include "rocprofiler-sdk-rocpd/version.h"

ROCPD_EXTERN_C_INIT

/**
 * @fn rocpd_status_t rocpd_get_version(uint32_t* major, uint32_t* minor, uint32_t*
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
 * @return ::rocpd_status_t
 * @retval ::ROCPD_STATUS_SUCCESS Always returned
 */
rocpd_status_t
rocpd_get_version(uint32_t* major, uint32_t* minor, uint32_t* patch) ROCPD_API;

/**
 * @brief Simplified alternative to ::rocpd_get_version
 *
 * Returns the version of the rocprofiler-sdk library loaded at runtime.  This can be used to check
 * if the runtime version is equal to or compatible with the version of rocprofiler-sdk used during
 * compilation time. This function can be invoked before tool initialization.
 *
 * @param [out] info Pointer to version triplet struct which will be populated by the function call.
 * @return ::rocpd_status_t
 * @retval ::ROCPD_STATUS_SUCCESS Always returned
 */
rocpd_status_t
rocpd_get_version_triplet(rocpd_version_triplet_t* info) ROCPD_API ROCPD_NONNULL(1);

ROCPD_EXTERN_C_FINI

/** @} */

#include "rocprofiler-sdk-rocpd/sql.h"

ROCPD_EXTERN_C_INIT

/**
 * @defgroup MISCELLANEOUS_GROUP Miscellaneous Utility Functions
 * @brief utility functions for library
 * @{
 */

/**
 * @fn const char* rocpd_get_status_name(rocpd_status_t status)
 * @brief Return the string encoding of ::rocpd_status_t value
 * @param [in] status error code value
 * @return Will return a nullptr if invalid/unsupported ::rocpd_status_t value is provided.
 */
const char*
rocpd_get_status_name(rocpd_status_t status) ROCPD_API;

/**
 * @fn const char* rocpd_get_status_string(rocpd_status_t status)
 * @brief Return the message associated with ::rocpd_status_t value
 * @param [in] status error code value
 * @return Will return a nullptr if invalid/unsupported ::rocpd_status_t value is provided.
 */
const char*
rocpd_get_status_string(rocpd_status_t status) ROCPD_API;

/** @} */

ROCPD_EXTERN_C_FINI
