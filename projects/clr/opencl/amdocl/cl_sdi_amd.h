/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __CL_SDI_AMD_H
#define __CL_SDI_AMD_H

#include "CL/cl_ext.h"

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/


extern CL_API_ENTRY cl_int CL_API_CALL clEnqueueWaitSignalAMD(
    cl_command_queue command_queue, cl_mem mem_object, cl_uint value, cl_uint num_events,
    const cl_event* event_wait_list, cl_event* event) CL_EXT_SUFFIX__VERSION_1_2;


extern CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteSignalAMD(
    cl_command_queue command_queue, cl_mem mem_object, cl_uint value, cl_ulong offset,
    cl_uint num_events, const cl_event* event_list, cl_event* event) CL_EXT_SUFFIX__VERSION_1_2;


extern CL_API_ENTRY cl_int CL_API_CALL clEnqueueMakeBuffersResidentAMD(
    cl_command_queue command_queue, cl_uint num_mem_objs, cl_mem* mem_objects,
    cl_bool blocking_make_resident, cl_bus_address_amd* bus_addresses, cl_uint num_events,
    const cl_event* event_list, cl_event* event) CL_EXT_SUFFIX__VERSION_1_2;


#ifdef __cplusplus
} /*extern "C"*/
#endif /*__cplusplus*/

#endif
