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

#include "cl_common.hpp"
#include "cl_sdi_amd.h"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/memory.hpp"
#include <cstring>


RUNTIME_ENTRY(cl_int, clEnqueueWaitSignalAMD,
              (cl_command_queue command_queue, cl_mem mem_object, cl_uint value, cl_uint num_events,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(mem_object)) {
    return CL_INVALID_MEM_OBJECT;
  }

  amd::Buffer* buffer = as_amd(mem_object)->asBuffer();
  if (buffer == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  if (!(buffer->getMemFlags() & CL_MEM_BUS_ADDRESSABLE_AMD)) {
    return CL_INVALID_MEM_OBJECT;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != buffer->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err = amd::clSetEventWaitList(eventWaitList, hostQueue, num_events, event_wait_list);

  if (err != CL_SUCCESS) {
    return err;
  }

  amd::SignalCommand* command =
      new amd::SignalCommand(hostQueue, CL_COMMAND_WAIT_SIGNAL_AMD, eventWaitList, *buffer, value);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_OUT_OF_RESOURCES;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT


RUNTIME_ENTRY(cl_int, clEnqueueWriteSignalAMD,
              (cl_command_queue command_queue, cl_mem mem_object, cl_uint value, cl_ulong offset,
               cl_uint num_events, const cl_event* event_wait_list, cl_event* event))

{
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (!is_valid(mem_object)) {
    return CL_INVALID_MEM_OBJECT;
  }

  amd::Buffer* buffer = as_amd(mem_object)->asBuffer();
  if (buffer == NULL) {
    return CL_INVALID_MEM_OBJECT;
  }

  if (!(buffer->getMemFlags() & CL_MEM_EXTERNAL_PHYSICAL_AMD)) {
    return CL_INVALID_MEM_OBJECT;
  }

  if ((offset + sizeof(value)) > (buffer->getSize() + amd::Os::pageSize())) {
    return CL_INVALID_BUFFER_SIZE;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (hostQueue.context() != buffer->getContext()) {
    return CL_INVALID_CONTEXT;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err = amd::clSetEventWaitList(eventWaitList, hostQueue, num_events, event_wait_list);

  if (err != CL_SUCCESS) {
    return err;
  }

  amd::SignalCommand* command = new amd::SignalCommand(hostQueue, CL_COMMAND_WRITE_SIGNAL_AMD,
                                                       eventWaitList, *buffer, value, offset);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_OUT_OF_RESOURCES;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT


RUNTIME_ENTRY(cl_int, clEnqueueMakeBuffersResidentAMD,
              (cl_command_queue command_queue, cl_uint num_mem_objs, cl_mem* mem_objects,
               cl_bool blocking_make_resident, cl_bus_address_amd* bus_addresses,
               cl_uint num_events, const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (mem_objects == 0) {
    return CL_INVALID_MEM_OBJECT;
  }

  if (bus_addresses == 0 || num_mem_objs == 0) {
    return CL_INVALID_VALUE;
  }

  memset(bus_addresses, 0, sizeof(cl_bus_address_amd) * num_mem_objs);

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  std::vector<amd::Memory*> memObjects;
  for (unsigned int i = 0; i < num_mem_objs; ++i) {
    if (!is_valid(mem_objects[i])) {
      return CL_INVALID_MEM_OBJECT;
    }

    amd::Buffer* buffer = as_amd(mem_objects[i])->asBuffer();
    if (buffer == NULL) {
      return CL_INVALID_MEM_OBJECT;
    }

    if (!(buffer->getMemFlags() & CL_MEM_BUS_ADDRESSABLE_AMD)) {
      return CL_INVALID_MEM_OBJECT;
    }

    if (hostQueue.context() != buffer->getContext()) {
      return CL_INVALID_CONTEXT;
    }

    memObjects.push_back(buffer);
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err = amd::clSetEventWaitList(eventWaitList, hostQueue, num_events, event_wait_list);

  if (err != CL_SUCCESS) {
    return err;
  }

  amd::MakeBuffersResidentCommand* command = new amd::MakeBuffersResidentCommand(
      hostQueue, CL_COMMAND_MAKE_BUFFERS_RESIDENT_AMD, eventWaitList, memObjects, bus_addresses);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_OUT_OF_RESOURCES;
  }

  command->enqueue();

  if (blocking_make_resident) {
    command->awaitCompletion();
  }

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT
