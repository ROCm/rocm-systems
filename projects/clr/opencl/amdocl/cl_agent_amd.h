/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

#ifndef __OPENCL_CL_AGENT_AMD_H
#define __OPENCL_CL_AGENT_AMD_H

#include <CL/cl.h>
#include "cl_icd_amd.h"

#define cl_amd_agent 1

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef const struct _cl_agent cl_agent;

#define CL_AGENT_VERSION_1_0 100

/* Context Callbacks */

typedef void(CL_CALLBACK* acContextCreate_fn)(cl_agent* /* agent */, cl_context /* context */);

typedef void(CL_CALLBACK* acContextFree_fn)(cl_agent* /* agent */, cl_context /* context */);

/* Command Queue Callbacks */

typedef void(CL_CALLBACK* acCommandQueueCreate_fn)(cl_agent* /* agent */,
                                                   cl_command_queue /* queue */);

typedef void(CL_CALLBACK* acCommandQueueFree_fn)(cl_agent* /* agent */,
                                                 cl_command_queue /* queue */);

/* Event Callbacks */

typedef void(CL_CALLBACK* acEventCreate_fn)(cl_agent* /* agent */, cl_event /* event */,
                                            cl_command_type /* type */);

typedef void(CL_CALLBACK* acEventFree_fn)(cl_agent* /* agent */, cl_event /* event */);

typedef void(CL_CALLBACK* acEventStatusChanged_fn)(cl_agent* /* agent */, cl_event /* event */,
                                                   cl_int /* execution_status */,
                                                   cl_long /* epoch_time_stamp */);

/* Memory Object Callbacks */

typedef void(CL_CALLBACK* acMemObjectCreate_fn)(cl_agent* /* agent */, cl_mem /* memobj */);

typedef void(CL_CALLBACK* acMemObjectFree_fn)(cl_agent* /* agent */, cl_mem /* memobj */);

typedef void(CL_CALLBACK* acMemObjectAcquired_fn)(cl_agent* /* agent */, cl_mem /* memobj */,
                                                  cl_device_id /* device */,
                                                  cl_long /* elapsed_time */);

/* Sampler Callbacks */

typedef void(CL_CALLBACK* acSamplerCreate_fn)(cl_agent* /* agent */, cl_sampler /* sampler */);

typedef void(CL_CALLBACK* acSamplerFree_fn)(cl_agent* /* agent */, cl_sampler /* sampler */);

/* Program Callbacks */

typedef void(CL_CALLBACK* acProgramCreate_fn)(cl_agent* /* agent */, cl_program /* program */);

typedef void(CL_CALLBACK* acProgramFree_fn)(cl_agent* /* agent */, cl_program /* program */);

typedef void(CL_CALLBACK* acProgramBuild_fn)(cl_agent* /* agent */, cl_program /* program */);

/* Kernel Callbacks */

typedef void(CL_CALLBACK* acKernelCreate_fn)(cl_agent* /* agent */, cl_kernel /* kernel */);

typedef void(CL_CALLBACK* acKernelFree_fn)(cl_agent* /* agent */, cl_kernel /* kernel */);

typedef void(CL_CALLBACK* acKernelSetArg_fn)(cl_agent* /* agent */, cl_kernel /* kernel */,
                                             cl_int /* arg_index */, size_t /* size */,
                                             const void* /* value_ptr */);

typedef struct _cl_agent_callbacks {
  /* Context Callbacks */
  acContextCreate_fn ContextCreate;
  acContextFree_fn ContextFree;

  /* Command Queue Callbacks */
  acCommandQueueCreate_fn CommandQueueCreate;
  acCommandQueueFree_fn CommandQueueFree;

  /* Event Callbacks */
  acEventCreate_fn EventCreate;
  acEventFree_fn EventFree;
  acEventStatusChanged_fn EventStatusChanged;

  /* Memory Object Callbacks */
  acMemObjectCreate_fn MemObjectCreate;
  acMemObjectFree_fn MemObjectFree;
  acMemObjectAcquired_fn MemObjectAcquired;

  /* Sampler Callbacks */
  acSamplerCreate_fn SamplerCreate;
  acSamplerFree_fn SamplerFree;

  /* Program Callbacks */
  acProgramCreate_fn ProgramCreate;
  acProgramFree_fn ProgramFree;
  acProgramBuild_fn ProgramBuild;

  /* Kernel Callbacks */
  acKernelCreate_fn KernelCreate;
  acKernelFree_fn KernelFree;
  acKernelSetArg_fn KernelSetArg;

} cl_agent_callbacks;

typedef cl_uint cl_agent_capability_action;

#define CL_AGENT_ADD_CAPABILITIES 0x0
#define CL_AGENT_RELINQUISH_CAPABILITIES 0x1

typedef struct _cl_agent_capabilities {
  cl_bitfield canGenerateContextEvents : 1;
  cl_bitfield canGenerateCommandQueueEvents : 1;
  cl_bitfield canGenerateEventEvents : 1;
  cl_bitfield canGenerateMemObjectEvents : 1;
  cl_bitfield canGenerateSamplerEvents : 1;
  cl_bitfield canGenerateProgramEvents : 1;
  cl_bitfield canGenerateKernelEvents : 1;

} cl_agent_capabilities;

struct _cl_agent {
  cl_int(CL_API_CALL* GetVersionNumber)(cl_agent* /* agent */, cl_int* /* version_ret */);

  cl_int(CL_API_CALL* GetPlatform)(cl_agent* /* agent */, cl_platform_id* /* platform_id_ret */);

  cl_int(CL_API_CALL* GetTime)(cl_agent* /* agent */, cl_long* /* time_nanos */);

  cl_int(CL_API_CALL* SetCallbacks)(cl_agent* /* agent */,
                                    const cl_agent_callbacks* /* callbacks */, size_t /* size */);


  cl_int(CL_API_CALL* GetPotentialCapabilities)(cl_agent* /* agent */,
                                                cl_agent_capabilities* /* capabilities */);

  cl_int(CL_API_CALL* GetCapabilities)(cl_agent* /* agent */,
                                       cl_agent_capabilities* /* capabilities */);

  cl_int(CL_API_CALL* SetCapabilities)(cl_agent* /* agent */,
                                       const cl_agent_capabilities* /* capabilities */,
                                       cl_agent_capability_action /* action */);


  cl_int(CL_API_CALL* GetICDDispatchTable)(cl_agent* /* agent */,
                                           cl_icd_dispatch_table* /* table */, size_t /* size */);

  cl_int(CL_API_CALL* SetICDDispatchTable)(cl_agent* /* agent */,
                                           const cl_icd_dispatch_table* /* table */,
                                           size_t /* size */);

  /* add Kernel/Program helper functions, etc... */
};

extern cl_int CL_CALLBACK clAgent_OnLoad(cl_agent* /* agent */);

extern void CL_CALLBACK clAgent_OnUnload(cl_agent* /* agent */);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __OPENCL_CL_AGENT_AMD_H */
