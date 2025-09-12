/* Copyright (c) 2012 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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
