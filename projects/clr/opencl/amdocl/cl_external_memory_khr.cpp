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

#include "platform/interop.hpp"

namespace {

//! Validates the object list for both entry points.
//!
//! The shared clSetInteropObjects() helper only rejects a null interop object, so on its own it
//! would accept a GL or D3D object here, accept an imported object belonging to a different
//! context, and report CL_INVALID_GL_OBJECT for an ordinary buffer. Check the two things that
//! are specific to this extension before handing the list on.
cl_int ValidateExternalMemObjects(cl_command_queue command_queue, cl_uint num_mem_objects,
                                  const cl_mem* mem_objects) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (queue == nullptr) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  if ((num_mem_objects == 0) != (mem_objects == nullptr)) {
    return CL_INVALID_VALUE;
  }

  for (cl_uint i = 0; i < num_mem_objects; ++i) {
    if (!is_valid(mem_objects[i])) {
      return CL_INVALID_MEM_OBJECT;
    }
    amd::Memory* memory = as_amd(mem_objects[i]);
    amd::InteropObject* interop = memory->getInteropObj();
    if (interop == nullptr || interop->asExternalMemory() == nullptr) {
      // An ordinary buffer/image, or one shared through GL/D3D rather than imported.
      return CL_INVALID_MEM_OBJECT;
    }
    if (&memory->getContext() != &queue->context()) {
      return CL_INVALID_CONTEXT;
    }
  }
  return CL_SUCCESS;
}

}  // namespace

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup cl_khr_external_memory
 *  @{
 */

RUNTIME_ENTRY(cl_int, clEnqueueAcquireExternalMemObjectsKHR,
              (cl_command_queue command_queue, cl_uint num_mem_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  cl_int err = ValidateExternalMemObjects(command_queue, num_mem_objects, mem_objects);
  if (err != CL_SUCCESS) {
    return err;
  }
  return amd::clEnqueueAcquireExtObjectsAMD(command_queue, num_mem_objects, mem_objects,
                                            num_events_in_wait_list, event_wait_list, event,
                                            CL_COMMAND_ACQUIRE_EXTERNAL_MEM_OBJECTS_KHR);
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clEnqueueReleaseExternalMemObjectsKHR,
              (cl_command_queue command_queue, cl_uint num_mem_objects, const cl_mem* mem_objects,
               cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event)) {
  cl_int err = ValidateExternalMemObjects(command_queue, num_mem_objects, mem_objects);
  if (err != CL_SUCCESS) {
    return err;
  }
  return amd::clEnqueueReleaseExtObjectsAMD(command_queue, num_mem_objects, mem_objects,
                                            num_events_in_wait_list, event_wait_list, event,
                                            CL_COMMAND_RELEASE_EXTERNAL_MEM_OBJECTS_KHR);
}
RUNTIME_EXIT

/*! @}
 *  @}
 */
