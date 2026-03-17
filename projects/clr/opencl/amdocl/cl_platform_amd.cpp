/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "cl_common.hpp"
#include "vdi_common.hpp"
#include "cl_platform_amd.h"
#include <cstring>

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup AMD_Extensions
 *  @{
 *
 */

RUNTIME_ENTRY(cl_int, clUnloadPlatformAMD, (cl_platform_id platform)) {
  if (AMD_PLATFORM == platform) {
    amd::Runtime::tearDown();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT


/*! @}
 *  @}
 */
