// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocprofiler-sdk-rocattach/defines.h"

#include <stdint.h>

/** @defgroup DATA_TYPE rocAttach Data types
 *
 * Data types defined or aliased by rocAttach
 *
 * @{
 */

//--------------------------------------------------------------------------------------//
//
//                                      ENUMERATIONS
//
//--------------------------------------------------------------------------------------//

/**
 * @defgroup BASIC_DATA_TYPES Basic data types
 * @brief Basic data types and typedefs
 *
 * @{
 */

/**
 * @brief Status codes.
 */
typedef enum rocattach_status_t  // NOLINT(performance-enum-size)
{
    ROCATTACH_STATUS_SUCCESS = 0,             ///< No error occurred
    ROCATTACH_STATUS_ERROR,                   ///< Generalized error
    ROCATTACH_STATUS_ERROR_INVALID_ARGUMENT,  ///< Invalid function argument
    ROCATTACH_STATUS_ERROR_NOT_SUPPORTED,     ///< Attachment is not supported on this platform
    ROCATTACH_STATUS_ERROR_PTRACE_ERROR,      ///< General ptrace error
    ROCATTACH_STATUS_ERROR_PTRACE_OPERATION_NOT_PERMITTED,  ///< ptrace returned EPERM, operation
                                                            ///< not permitted
    ROCATTACH_STATUS_ERROR_PTRACE_PROCESS_NOT_FOUND,  ///< ptrace returned ESRCH, no such process
    ROCATTACH_STATUS_LAST,
} rocattach_status_t;

//--------------------------------------------------------------------------------------//
//
//                                      STRUCTS
//
//--------------------------------------------------------------------------------------//

/**
 * @brief Versioning info.
 */
typedef struct rocattach_version_triplet_t
{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
} rocattach_version_triplet_t;

/** @} */
