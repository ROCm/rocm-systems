/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __CL_P2P_AMD_H
#define __CL_P2P_AMD_H

#include "CL/cl_ext.h"

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/

extern CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBufferP2PAMD(
    cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset,
    size_t dst_offset, size_t cb, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
    cl_event* event) CL_EXT_SUFFIX__VERSION_1_2;

#ifdef __cplusplus
} /*extern "C"*/
#endif /*__cplusplus*/

#endif
