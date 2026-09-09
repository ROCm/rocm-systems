// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef GO_AMD_SMI_H_
#define GO_AMD_SMI_H_

#include <stdbool.h>
#include <stdio.h>

#define GOAMDSMI_VALUE_0 0
#define GOAMDSMI_UINT16_MAX 0xFFFF
#define GOAMDSMI_UINT32_MAX 0xFFFFFFFF
#define GOAMDSMI_UINT64_MAX 0xFFFFFFFFFFFFFFFF
#define GOAMDSMI_STRING_NA "NA"

/**
 *  @brief Go language stub to initialize the Debug Level prints
 *         -DENABLE_DEBUG_LEVEL=1 (or) -DENABLE_DEBUG_LEVEL=<Enable_Debug_level_number> must be
 * passed at cmake time
 *
 *  @retval ::bool value of true upon enabling logs
 *  @retval false is returned upon if user does not want to enable logs.
 *
 */
#define enable_debug_level(debug_level) ((ENABLE_DEBUG_LEVEL >= debug_level) ? true : false)

typedef enum {
  GOAMDSMI_STATUS_SUCCESS = 0x0,  //!< Operation successful
  GOAMDSMI_STATUS_FAILURE = 0x1,  //!< Operation failed
} goamdsmi_status_t;

typedef enum {
  GOAMDSMI_CPU_INIT = 0x0,  //!< CPU Init
  GOAMDSMI_GPU_INIT = 0x1,  //!< GPU Init
} goamdsmi_Init_t;

typedef enum {
  GOAMDSMI_DEBUG_LEVEL_0 = 0x0,  //!< Debug Level as 0
  GOAMDSMI_DEBUG_LEVEL_1 = 0x1,  //!< Debug Level as 1
  GOAMDSMI_DEBUG_LEVEL_2 = 0x2,  //!< Debug Level as 2
  GOAMDSMI_DEBUG_LEVEL_3 = 0x3,  //!< Debug Level as 3
} goamdsmi_Enable_Debug_Level_t;

#endif
