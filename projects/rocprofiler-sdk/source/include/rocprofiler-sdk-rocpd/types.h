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

#include "rocprofiler-sdk-rocpd/defines.h"

#include <stdint.h>

/** @defgroup DATA_TYPE rocPD Data types
 *
 * Data types defined or aliased by rocPD
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
typedef enum rocpd_status_t  // NOLINT(performance-enum-size)
{
    ROCPD_STATUS_SUCCESS = 0,                         ///< No error occurred
    ROCPD_STATUS_ERROR,                               ///< Generalized error
    ROCPD_STATUS_ERROR_INVALID_ARGUMENT,              ///< Argument is invalid
    ROCPD_STATUS_ERROR_SQL_ERROR,                     ///< General SQL error
    ROCPD_STATUS_ERROR_SQL_INVALID_ENGINE,            ///< SQL engine is invalid
    ROCPD_STATUS_ERROR_SQL_INVALID_SCHEMA_KIND,       ///< SQL schema kind not found
    ROCPD_STATUS_ERROR_SQL_SCHEMA_NOT_FOUND,          ///< SQL schema does not exist
    ROCPD_STATUS_ERROR_SQL_SCHEMA_PERMISSION_DENIED,  ///< SQL schema not accessible
    ROCPD_STATUS_LAST,
} rocpd_status_t;

//--------------------------------------------------------------------------------------//
//
//                                      STRUCTS
//
//--------------------------------------------------------------------------------------//

/**
 * @brief Versioning info.
 */
typedef struct rocpd_version_triplet_t
{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
} rocpd_version_triplet_t;

/** @} */
