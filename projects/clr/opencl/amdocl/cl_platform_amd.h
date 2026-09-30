/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// AMD-specific platform management extensions

#ifndef __CL_PLATFORM_AMD_H
#define __CL_PLATFORM_AMD_H

#include "CL/cl_platform.h"

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/

/*! \brief Unloads the specified platform, handling all required cleanup.
 *
 *  @todo This is still somewhat of a stub. It only works for the AMD
 *        platform and just forces shutdown of all devices (to get PM4
 *        capture working). It should handle ICD unregistration as well.
 */
extern CL_API_ENTRY cl_int CL_API_CALL clUnloadPlatformAMD(cl_platform_id platform)
    CL_API_SUFFIX__VERSION_1_0;

#ifdef __cplusplus
} /*extern "C"*/
#endif /*__cplusplus*/

#endif /*__CL_AMD_PROFILE_H*/
