// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include <stddef.h>
#include <stdint.h>

#include "rocprofiler/defines.h"

/** @defgroup VERSIONING_GROUP Library Versioning
 *
 * Version information about the interface and the associated installed
 * library.
 *
 * The semantic version of the interface following semver.org rules. A context
 * that uses this interface is only compatible with the installed library if
 * the major version numbers match and the interface minor version number is
 * less than or equal to the installed library minor version number.
 */

#include "rocprofiler/agent.h"
#include "rocprofiler/agent_profile.h"
#include "rocprofiler/buffer.h"
#include "rocprofiler/buffer_tracing.h"
#include "rocprofiler/callback_tracing.h"
#include "rocprofiler/context.h"
#include "rocprofiler/counters.h"
#include "rocprofiler/dispatch_profile.h"
#include "rocprofiler/external_correlation.h"
#include "rocprofiler/fwd.h"
#include "rocprofiler/hip.h"
#include "rocprofiler/hsa.h"
#include "rocprofiler/marker.h"
#include "rocprofiler/pc_sampling.h"
#include "rocprofiler/profile_config.h"
#include "rocprofiler/spm.h"
#include "rocprofiler/version.h"

ROCPROFILER_EXTERN_C_INIT

/**
 * @brief Query the version of the installed library.
 *
 * Return the version of the installed library.  This can be used to check if
 * it is compatible with this interface version.  This function can be used
 * even when the library is not initialized.
 *
 * @param [out] major The major version number is stored if non-NULL.
 * @param [out] minor The minor version number is stored if non-NULL.
 * @param [out] patch The patch version number is stored if non-NULL.
 * @addtogroup VERSIONING_GROUP
 */
rocprofiler_status_t
rocprofiler_get_version(uint32_t* major, uint32_t* minor, uint32_t* patch) ROCPROFILER_API;

/**
 * @defgroup MISCELLANEOUS_GROUP Miscellaneous utility functions
 */

/**
 * @brief Get the timestamp value that rocprofiler uses
 * @param [out] ts Output address of the rocprofiler timestamp value
 * @addtogroup MISCELLANEOUS_GROUP
 */
rocprofiler_status_t
rocprofiler_get_timestamp(rocprofiler_timestamp_t* ts) ROCPROFILER_API ROCPROFILER_NONNULL(1);

ROCPROFILER_EXTERN_C_FINI
