// SPDX-License-Identifier: MIT
/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sellcopies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *  - The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * Except as contained in this notice, the name of the Advanced Micro Devices,
 * Inc. shall not be used in advertising or otherwise to promote the sale, use
 * or other dealings in this Software without prior written authorization from
 * the Advanced Micro Devices, Inc.
 *
 */

#ifndef GO_AMD_SMI_H_
#define GO_AMD_SMI_H_

#include <stdbool.h>
#include <stdio.h>

#define GOAMDSMI_VALUE_0           0
#define GOAMDSMI_UINT16_MAX        0xFFFF
#define GOAMDSMI_UINT32_MAX        0xFFFFFFFF
#define GOAMDSMI_UINT64_MAX        0xFFFFFFFFFFFFFFFF
#define GOAMDSMI_STRING_NA         "NA"

/**
 *  @brief Go language stub to initialize the Debug Level prints
 *         -DENABLE_DEBUG_LEVEL=1 (or) -DENABLE_DEBUG_LEVEL=<Enable_Debug_level_number> must be passed at cmake time
 *
 *  @retval ::bool value of true upon enabling logs
 *  @retval false is returned upon if user does not want to enable logs.
 *
 */
#define enable_debug_level(debug_level) ((ENABLE_DEBUG_LEVEL >= debug_level)?true:false)

typedef enum {
  GOAMDSMI_STATUS_SUCCESS = 0x0,               //!< Operation successful
  GOAMDSMI_STATUS_FAILURE = 0x1,               //!< Operation failed
} goamdsmi_status_t;

typedef enum {
  GOAMDSMI_CPU_INIT = 0x0,               //!< CPU Init
  GOAMDSMI_GPU_INIT = 0x1,               //!< GPU Init
} goamdsmi_Init_t;

typedef enum {
  GOAMDSMI_DEBUG_LEVEL_0 = 0x0,               //!< Debug Level as 0
  GOAMDSMI_DEBUG_LEVEL_1 = 0x1,               //!< Debug Level as 1
  GOAMDSMI_DEBUG_LEVEL_2 = 0x2,               //!< Debug Level as 2
  GOAMDSMI_DEBUG_LEVEL_3 = 0x3,               //!< Debug Level as 3
} goamdsmi_Enable_Debug_Level_t;

#endif
