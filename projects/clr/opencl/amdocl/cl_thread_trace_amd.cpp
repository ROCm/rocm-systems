/* Copyright (c) 2009 - 2021 Advanced Micro Devices, Inc.

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
#include "cl_thread_trace_amd.h"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/threadtrace.hpp"
#include <cstring>
#include <vector>
#include <memory>

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup AMD_Extensions
 *  @{
 *
 */

/*! \brief Creates a new HW threadTrace
 *
 *  \param device must be a valid OpenCL device.
 *
 *  \param threadTrace the created cl_threadtrace_amd object
 *
 *  \param errcode_ret A non zero value if OpenCL failed to create cl_threadtrace_amd
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_INVALID_DEVICE if the specified context is invalid.
 *  - CL_INVALID_OPERATION if we couldn't create the object
 *
 *  \return Created cl_threadtrace_amd object
 */
RUNTIME_ENTRY_RET(cl_threadtrace_amd, clCreateThreadTraceAMD,
                  (cl_device_id device, cl_int* errcode_ret)) {
  // Make sure we have a valid device object
  if (!is_valid(device)) {
    *not_null(errcode_ret) = CL_INVALID_DEVICE;
    return NULL;
  }

  // Create the device thread trace object
  amd::ThreadTrace* threadTrace = new amd::ThreadTrace(*as_amd(device));

  if (threadTrace == NULL) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    return NULL;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(threadTrace);
}
RUNTIME_EXIT

///*! \brief Destroy a threadTrace object.
// *
// *  \param threadTrace the cl_threadtrace_amd object for release
// *
// *  \return A non zero value if OpenCL failed to release cl_threadtrace_amd
// *  - CL_SUCCESS if the function is executed successfully.
// *  - CL_INVALID_OPERATION if we failed to release the object
// */
RUNTIME_ENTRY(cl_int, clReleaseThreadTraceAMD, (cl_threadtrace_amd threadTrace)) {
  if (!is_valid(threadTrace)) {
    return CL_INVALID_OPERATION;
  }
  as_amd(threadTrace)->release();
  return CL_SUCCESS;
}
RUNTIME_EXIT
//
// *! \brief Increments the cl_threadtrace_amd object reference count.
// *
// *  \param threadTrace the cl_threadtrace_amd object for retain
// *
// *  \return A non zero value if OpenCL failed to retain cl_threadtrace_amd
// *  - CL_SUCCESS if the function is executed successfully.
// *  - CL_INVALID_OPERATION if we failed to release the object
// */
RUNTIME_ENTRY(cl_int, clRetainThreadTraceAMD, (cl_threadtrace_amd threadTrace)) {
  if (!is_valid(threadTrace)) {
    return CL_INVALID_OPERATION;
  }
  as_amd(threadTrace)->retain();
  return CL_SUCCESS;
}
RUNTIME_EXIT

//
// *! \brief Sets the cl_threadtrace_amd object configuration parameter.
// *
// *  \param thread_trace the cl_threadtrace_amd object to set configuration parameter
// *
// *  \param config_param the cl_thread_trace_param
// *
// *  \param param_value corresponding to configParam
// *
// *  \return A non zero value if OpenCL failed to set threadTrace buffer parameter
// *  - CL_INVALID_VALUE if the thread_trace  is invalid thread trace object.
// *  - CL_INVALID_VALUE if the invalid config_param or param_value enum values , are used.
// *  - CL_INVALID_EVENT_WAIT_LIST if event_wait_list is NULL and num_events_in_wait_list > 0, or
// event_wait_list is not NULL and num_events_in_wait_list is 0,
// *  -                            or if event objects in event_wait_list are not valid events.
// *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required by the OpenCL
// implementation on the device.
// *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the
//                           OpenCL implementation on the host.
// */
RUNTIME_ENTRY(cl_int, clSetThreadTraceParamAMD,
              (cl_threadtrace_amd thread_trace, cl_thread_trace_param config_param,
               cl_uint param_value)) {
  if (!is_valid(thread_trace)) {
    return CL_INVALID_OPERATION;
  }
  switch (config_param) {
    case CL_THREAD_TRACE_PARAM_TOKEN_MASK:
      if (param_value > CL_THREAD_TRACE_TOKEN_MASK_ALL_SI) {
        return CL_INVALID_VALUE;
      }
      as_amd(thread_trace)->setTokenMask(param_value);
      break;
    case CL_THREAD_TRACE_PARAM_REG_MASK:
      if (param_value > CL_THREAD_TRACE_REG_MASK_ALL_SI) {
        return CL_INVALID_VALUE;
      }
      as_amd(thread_trace)->setRegMask(param_value);
      break;
    case CL_THREAD_TRACE_PARAM_VM_ID_MASK:
      if (param_value > CL_THREAD_TRACE_VM_ID_MASK_SINGLE_DETAIL) {
        return CL_INVALID_VALUE;
      }
      as_amd(thread_trace)->setVmIdMask(param_value);
      break;
    case CL_THREAD_TRACE_PARAM_INSTRUCTION_MASK:
      if (param_value > CL_THREAD_TRACE_INST_MASK_IMMEDIATE_CI) {
        return CL_INVALID_VALUE;
      }
      as_amd(thread_trace)->setInstMask(param_value);
      break;
    case CL_THREAD_TRACE_PARAM_COMPUTE_UNIT_TARGET:
      as_amd(thread_trace)->setCU(param_value);
      break;
    case CL_THREAD_TRACE_PARAM_SHADER_ARRAY_TARGET:
      as_amd(thread_trace)->setSH(param_value);
      break;
    case CL_THREAD_TRACE_PARAM_SIMD_MASK:
      as_amd(thread_trace)->setSIMD(param_value);
      break;
    case CL_THREAD_TRACE_PARAM_USER_DATA:
      as_amd(thread_trace)->setUserData(param_value);
      break;
    case CL_THREAD_TRACE_PARAM_CAPTURE_MODE:
      if (param_value > CL_THREAD_TRACE_CAPTURE_SELECT_DETAIL) {
        return CL_INVALID_VALUE;
      }
      as_amd(thread_trace)->setCaptureMode(param_value);
      break;
    case CL_THREAD_TRACE_PARAM_IS_WRAPPED:
      as_amd(thread_trace)->setIsWrapped(true);
      break;
    case CL_THREAD_TRACE_PARAM_RANDOM_SEED:
      as_amd(thread_trace)->setRandomSeed(param_value);
      break;
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Get specific information about the OpenCL Thread Trace.
 *
 *  \param threadTrace_info_param is an enum that identifies the Thread Trace information being
 *  queried.
 *
 *  \param param_value is a pointer to memory location where appropriate values
 *  for a given \a threadTrace_info_param will be returned. If \a param_value is NULL,
 *  it is ignored.
 *
 *  \param param_value_size specifies the size in bytes of memory pointed to by
 *  \a param_value. This size in bytes must be >= size of return type.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data being
 *  queried by param_value. If \a param_value_size_ret is NULL, it is ignored.
 *
 *  \return One of the following values:
 *      CL_INVALID_OPERATION if cl_threadtrace_amd object is not valid
 *    - CL_INVALID_VALUE if \a param_name is not one of the supported
 *      values or if size in bytes specified by \a param_value_size is < size of
 *      return type and \a param_value is not a NULL value.
 *    - CL_SUCCESS if the function is executed successfully.
 *
 */
RUNTIME_ENTRY(cl_int, clGetThreadTraceInfoAMD,
              (cl_threadtrace_amd thread_trace /* threadTrace */,
               cl_threadtrace_info thread_trace_info_param, size_t param_value_size,
               void* param_value, size_t* param_value_size_ret)) {
  if (!is_valid(thread_trace)) {
    return CL_INVALID_OPERATION;
  }

  // Find the thread trace object, associated with the specified device
  const device::ThreadTrace* devThreadTrace = as_amd(thread_trace)->getDeviceThreadTrace();

  const size_t seNum = as_amd(thread_trace)->deviceSeNumThreadTrace();
  switch (thread_trace_info_param) {
    case CL_THREAD_TRACE_SE: {
      return amd::clGetInfo(seNum, param_value_size, param_value, param_value_size_ret);
    }
    case CL_THREAD_TRACE_BUFFERS_SIZE: {
      // Make sure we found a valid thread trace object
      if (devThreadTrace == NULL) {
        return CL_INVALID_OPERATION;
      }

      std::unique_ptr<uint> bufSize2Se(new uint[seNum]);

      if (bufSize2Se.get() == NULL) {
        return CL_OUT_OF_HOST_MEMORY;
      }

      if (!devThreadTrace->info(thread_trace_info_param, bufSize2Se.get(), seNum)) {
        return CL_INVALID_VALUE;
      }

      const size_t valueSize = seNum * sizeof(unsigned int);

      if (param_value != NULL && param_value_size < valueSize) {
        return CL_INVALID_VALUE;
      }

      *not_null(param_value_size_ret) = valueSize;

      if (param_value != NULL) {
        ::memcpy(param_value, bufSize2Se.get(), valueSize);
        if (param_value_size > valueSize) {
          ::memset(static_cast<address>(param_value) + valueSize, '\0',
                   param_value_size - valueSize);
        }
      }

      return CL_SUCCESS;
    }
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/* \brief Enqueues the  command for the specified thread trace object.
 *
 *  \param command_queue must be a valid OpenCL command queue.
 *
 *  \param thread_trace specifies the cl_threadtrace_amd object.
 *
 *  \param event_wait_list specify [is a pointer to] events that need to
 *  complete before this particular command can be executed.
 *  If \a event_wait_list is NULL, then this particular command does not wait
 *  on any event to complete. If \a event_wait_list is NULL,
 *  \a num_events_in_wait_list must be 0. If \a event_wait_list is not NULL,
 *  the list of events pointed to by \a event_wait_list must be valid and
 *  \a num_events_in_wait_list must be greater than 0. The events specified in
 *  \a event_wait_list act as synchronization points.
 *
 *  \param num_events_in_wait_list specify the number of events in
 *  \a event_wait_list. It must be 0 if \a event_wait_list is NULL. It  must be
 *  greater than 0 if \a event_wait_list is not NULL.
 *
 *  \param event returns an event object that identifies this particular
 *  command and can be used to query or queue a wait for this particular
 *  command to complete. \a event can be NULL in which case it will not be
 *  possible for the application to query the status of this command or queue a
 *  wait for this command to complete.
 * \return A non zero value if OpenCL failed to release threadTrace
 *  - CL_INVALID_COMMAND_QUEUE if command_queue is not a valid command-queue.
 *  - CL_INVALID_CONTEXT if the context associated with command_queue and  events in event_wait_list
 * are not the same.
 *  - CL_INVALID_VALUE if the thread_trace is invalid thread trace object .
 *  - CL_INVALID_VALUE if the invalid command name enum value , not  described in the
 * cl_threadtrace_command_name_amd, is used.
 *  - CL_INVALID_OPERATION if the command enqueue failed. It can happen in the following cases:
 *          o BEGIN_COMMAND is queued for thread trace object for which memory object/s was/were not
 * bound..
 *          o END_COMMAND is queued for thread trace object, for which BEGIN_COMMAND was not queued.
 *          o PAUSE_COMMAND is queued for thread trace object, for which BEGIN_COMMAND was not
 * queued.
 *          o RESUME_COMMAND is queued for thread trace object, for which  PAUSE_COMMAND was not
 * queued.
 *  - CL_INVALID_EVENT_WAIT_LIST if event_wait_list is NULL and num_events_in_wait_list > 0, or
 * event_wait_list is not NULL and num_events_in_wait_list is 0, or if event objects in
 * event_wait_list are not valid events.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required by the OpenCL
 * implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the OpenCL
 * implementation on the host.
 */

RUNTIME_ENTRY(cl_int, clEnqueueThreadTraceCommandAMD,
              (cl_command_queue command_queue, cl_threadtrace_amd thread_trace,
               cl_threadtrace_command_name_amd command_name, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  // Check if command queue is valid
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  // Check if thread trace is valid
  if (!is_valid(thread_trace)) {
    return CL_INVALID_OPERATION;
  }

  amd::ThreadTrace* amdThreadTrace = as_amd(thread_trace);
  amd::HostQueue* hostQueue = as_amd(command_queue)->asHostQueue();
  if (NULL == hostQueue) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  // Check that device associated with the command queue is the same as with thread trace
  if (&hostQueue->device() != &amdThreadTrace->device()) {
    return CL_INVALID_DEVICE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, *hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  // Create a new command for the threadTraces
  amd::ThreadTraceCommand* command = NULL;
  switch (command_name) {
    case CL_THREAD_TRACE_BEGIN_COMMAND:
      if ((amdThreadTrace->getState() != amd::ThreadTrace::MemoryBound) &&
          (amdThreadTrace->getState() != amd::ThreadTrace::End)) {
        return CL_INVALID_OPERATION;
      }
      amdThreadTrace->setState(amd::ThreadTrace::Begin);
      command = new amd::ThreadTraceCommand(
          *hostQueue, eventWaitList, static_cast<const void*>(&amdThreadTrace->threadTraceConfig()),
          *amdThreadTrace, amd::ThreadTraceCommand::Begin, CL_COMMAND_THREAD_TRACE);
      break;
    case CL_THREAD_TRACE_END_COMMAND:
      if ((amdThreadTrace->getState() != amd::ThreadTrace::Begin) &&
          (amdThreadTrace->getState() != amd::ThreadTrace::Pause)) {
        return CL_INVALID_OPERATION;
      }
      amdThreadTrace->setState(amd::ThreadTrace::End);
      command = new amd::ThreadTraceCommand(*hostQueue, eventWaitList,
                                            &amdThreadTrace->threadTraceConfig(), *amdThreadTrace,
                                            amd::ThreadTraceCommand::End, CL_COMMAND_THREAD_TRACE);
      break;
    case CL_THREAD_TRACE_PAUSE_COMMAND:
      if (amdThreadTrace->getState() != amd::ThreadTrace::Begin) {
        return CL_INVALID_OPERATION;
      }
      amdThreadTrace->setState(amd::ThreadTrace::Pause);
      command = new amd::ThreadTraceCommand(
          *hostQueue, eventWaitList, &amdThreadTrace->threadTraceConfig(), *amdThreadTrace,
          amd::ThreadTraceCommand::Pause, CL_COMMAND_THREAD_TRACE);
      break;
    case CL_THREAD_TRACE_RESUME_COMMAND:
      if (amdThreadTrace->getState() != amd::ThreadTrace::Pause) {
        return CL_INVALID_OPERATION;
      }
      amdThreadTrace->setState(amd::ThreadTrace::Begin);
      command = new amd::ThreadTraceCommand(
          *hostQueue, eventWaitList, &amdThreadTrace->threadTraceConfig(), *amdThreadTrace,
          amd::ThreadTraceCommand::Resume, CL_COMMAND_THREAD_TRACE);
      break;
  }

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Submit the command to the device
  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

//
///*! \brief Enqueues the binding command to bind cl_threadtrace_amd to cl_mem object for trace
/// recording..
// *
// *  \param command_queue must be a valid OpenCL command queue.
// *
// *  \param thread_trace specifies the cl_threadtrace_amd object.
// *
// *  \param mem_objects the cl_mem objects for trace recording
// *
// *  \param mem_objects_num the number of cl_mem objects in the mem_objects
// *
// *  \param buffer_size the size of each cl_mem object from mem_objects
// *
// *  \param event_wait_list specify [is a pointer to] events that need to
// *  complete before this particular command can be executed.
// *  If \a event_wait_list is NULL, then this particular command does not wait
// *  on any event to complete. If \a event_wait_list is NULL,
// *  \a num_events_in_wait_list must be 0. If \a event_wait_list is not NULL,
// *  the list of events pointed to by \a event_wait_list must be valid and
// *  \a num_events_in_wait_list must be greater than 0. The events specified in
// *  \a event_wait_list act as synchronization points.
// *
// *  \param num_events_in_wait_list specify the number of events in
// *  \a event_wait_list. It must be 0 if \a event_wait_list is NULL. It  must be
// *  greater than 0 if \a event_wait_list is not NULL.
// *
// *  \param event returns an event object that identifies this particular
// *  command and can be used to query or queue a wait for this particular
// *  command to complete. \a event can be NULL in which case it will not be
// *  possible for the application to query the status of this command or queue a
// *  wait for this command to complete.
// *  \return A non zero value if OpenCL failed to set threadTrace buffer parameter
// *  - CL_INVALID_COMMAND_QUEUE if command_queue is not a valid command-queue.
// *  - CL_INVALID_CONTEXT if the context associated with command_queue and  events in
// event_wait_list are not the same.
// *  - CL_INVALID_VALUE if the thread_trace  is invalid thread trace object.
// *  - CL_INVALID_VALUE if the buffer_size is negative or zero.
// *  - CL_INVALID_VALUE if the  sub_buffers_num I less than 1.
// *  - CL_INVALID_OPERATION if the mem_objects_num is not equal to the number of Shader Engines of
// the [GPU] device.
// *  - CL_INVALID_MEM_OBJECT if one on memory objects in the mem_objects array is not a valid
// memory object or memory_objects is NULL.
// *  - CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory for the data store
// associated from the memory objects of the mem_objects array.
// *  - CL_INVALID_EVENT_WAIT_LIST if event_wait_list is NULL and num_events_in_wait_list > 0, or
// event_wait_list is not NULL and num_events_in_wait_list is 0, or if event objects in
// event_wait_list are not valid events.
// *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required by the OpenCL
// implementation on the device.
// *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the
// *     OpenCL implementation on the host.
// */
RUNTIME_ENTRY(cl_int, clEnqueueBindThreadTraceBufferAMD,
              (cl_command_queue command_queue, cl_threadtrace_amd thread_trace, cl_mem* mem_objects,
               cl_uint mem_objects_num, cl_uint buffer_size, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  // Check if command queue is valid
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  // Check if thread trace is valid
  if (!is_valid(thread_trace)) {
    return CL_INVALID_OPERATION;
  }

  // Check if input values are valid
  if ((mem_objects == NULL) || (buffer_size <= 0)) {
    return CL_INVALID_VALUE;
  }

  amd::ThreadTrace* amdThreadTrace = as_amd(thread_trace);

  // Check if the number of bound memory objects is the same as the number of SEs
  if (amdThreadTrace->deviceSeNumThreadTrace() != mem_objects_num) {
    return CL_INVALID_OPERATION;
  }
  // Check if memory objects ,bound the thread trace,are valid
  for (size_t i = 0; i < mem_objects_num; ++i) {
    cl_mem obj = mem_objects[i];
    if (!is_valid(obj)) {
      return CL_INVALID_MEM_OBJECT;
    }
  }

  amd::HostQueue* hostQueue = as_amd(command_queue)->asHostQueue();
  if (NULL == hostQueue) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  // Check that device associated with the command queue is the same as with thread trace
  if (&hostQueue->device() != &amdThreadTrace->device()) {
    return CL_INVALID_DEVICE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, *hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amdThreadTrace->setState(amd::ThreadTrace::MemoryBound);
  // Create a new ThreadTraceMemObjectsCommand command
  amd::ThreadTraceMemObjectsCommand* command = new amd::ThreadTraceMemObjectsCommand(
      *hostQueue, eventWaitList, mem_objects_num, mem_objects, buffer_size, *amdThreadTrace,
      CL_COMMAND_THREAD_TRACE_MEM);
  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Make sure we have memory for the command execution
  if (!command->validateMemory()) {
    delete command;
    return CL_OUT_OF_RESOURCES;
  }
  // Submit the command to the device
  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  @}
 */
