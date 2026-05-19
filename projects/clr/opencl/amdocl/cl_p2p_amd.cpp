/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "cl_common.hpp"
#include <CL/cl_ext.h>

#include "cl_p2p_amd.h"
#include "platform/object.hpp"

RUNTIME_ENTRY(cl_int, clEnqueueCopyBufferP2PAMD,
              (cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer,
               size_t src_offset, size_t dst_offset, size_t cb, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(src_buffer) || !is_valid(dst_buffer)) {
    return CL_INVALID_MEM_OBJECT;
  }
  amd::Buffer* srcBuffer = as_amd(src_buffer)->asBuffer();
  amd::Buffer* dstBuffer = as_amd(dst_buffer)->asBuffer();
  if (srcBuffer == NULL || dstBuffer == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if ((hostQueue.context() != srcBuffer->getContext()) &&
      (hostQueue.context() != dstBuffer->getContext())) {
    return CL_INVALID_CONTEXT;
  }

  amd::Coord3D srcOffset(src_offset, 0, 0);
  amd::Coord3D dstOffset(dst_offset, 0, 0);
  amd::Coord3D size(cb, 1, 1);

  if (!srcBuffer->validateRegion(srcOffset, size) || !dstBuffer->validateRegion(dstOffset, size)) {
    return CL_INVALID_VALUE;
  }

  if (srcBuffer == dstBuffer && ((src_offset <= dst_offset && dst_offset < src_offset + cb) ||
                                 (dst_offset <= src_offset && src_offset < dst_offset + cb))) {
    return CL_MEM_COPY_OVERLAP;
  }

  amd::Command::EventWaitList eventWaitList;
  if ((num_events_in_wait_list == 0 && event_wait_list != NULL) ||
      (num_events_in_wait_list != 0 && event_wait_list == NULL)) {
    return CL_INVALID_EVENT_WAIT_LIST;
  }

  while (num_events_in_wait_list-- > 0) {
    cl_event event = *event_wait_list++;
    amd::Event* amdEvent = as_amd(event);
    if (!is_valid(event)) {
      return CL_INVALID_EVENT_WAIT_LIST;
    }
    eventWaitList.push_back(amdEvent);
  }

  amd::CopyMemoryP2PCommand* command =
      new amd::CopyMemoryP2PCommand(hostQueue, CL_COMMAND_COPY_BUFFER, eventWaitList, *srcBuffer,
                                    *dstBuffer, srcOffset, dstOffset, size);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT
