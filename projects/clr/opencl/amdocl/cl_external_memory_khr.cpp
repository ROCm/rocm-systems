/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

//
// cl_khr_external_memory
//
// The acquire/release pair reuses the generic interop acquire/release machinery
// already used by cl_khr_gl_sharing and the D3D sharing extensions; only the
// reported command type differs.
//
// The import itself happens in clCreateBufferWithProperties /
// clCreateImageWithProperties (cl_memobj.cpp).
//

#include "cl_common.hpp"

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup cl_khr_external_memory
 *  @{
 */

RUNTIME_ENTRY(cl_int, clEnqueueAcquireExternalMemObjectsKHR,
              (cl_command_queue command_queue, cl_uint num_mem_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clEnqueueAcquireExtObjectsAMD(command_queue, num_mem_objects, mem_objects,
                                            num_events_in_wait_list, event_wait_list, event,
                                            CL_COMMAND_ACQUIRE_EXTERNAL_MEM_OBJECTS_KHR);
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clEnqueueReleaseExternalMemObjectsKHR,
              (cl_command_queue command_queue, cl_uint num_mem_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  return amd::clEnqueueReleaseExtObjectsAMD(command_queue, num_mem_objects, mem_objects,
                                            num_events_in_wait_list, event_wait_list, event,
                                            CL_COMMAND_RELEASE_EXTERNAL_MEM_OBJECTS_KHR);
}
RUNTIME_EXIT

/*! @}
 *  @}
 */
