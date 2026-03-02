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
#include "cl_profile_amd.h"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/perfctr.hpp"
#include "device/device.hpp"
#include <cstring>

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup AMD_Extensions
 *  @{
 *
 */

/*! \brief Creates a new HW performance counter
 *   for the specified OpenCL context.
 *
 *  \param device must be a valid OpenCL device.
 *
 *  \param block_index index of the HW block to configure.
 *
 *  \param counter_index index of the hardware counter
 *  within the block to configure.
 *
 *  \param event_index Event you wish to count with
 *  the counter specified by block_index + counter_index
 *
 *  \param perf_counter the created perfcounter object
 *
 *  \param errcode_ret A non zero value if OpenCL failed to create PerfCounter
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_INVALID_DEVICE if the specified context is invalid.
 *  - CL_INVALID_OPERATION if we couldn't create the object
 *
 *  \return Created perfcounter object
 */
RUNTIME_ENTRY_RET(cl_perfcounter_amd, clCreatePerfCounterAMD,
                  (cl_device_id device, cl_perfcounter_property* properties, cl_int* errcode_ret)) {
  // Make sure we have a valid device object
  if (!is_valid(device)) {
    *not_null(errcode_ret) = CL_INVALID_DEVICE;
    return NULL;
  }

  // Make sure we have a valid pointer to the performance counter properties
  if (NULL == properties) {
    return NULL;
  }

  amd::PerfCounter::Properties perfProperties;
  size_t size = 0;
  while (properties[size] != CL_PERFCOUNTER_NONE) {
    if (properties[size] < CL_PERFCOUNTER_LAST) {
      perfProperties[properties[size]] = static_cast<ulong>(properties[size + 1]);
      size += 2;
    } else {
      return NULL;
    }
  }

  // Create the device perf counter
  amd::PerfCounter* perfCounter = new amd::PerfCounter(*as_amd(device), perfProperties);

  if (perfCounter == NULL) {
    *not_null(errcode_ret) = CL_INVALID_OPERATION;
    return NULL;
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(perfCounter);
}
RUNTIME_EXIT

/*! \brief Destroy a performance counter object.
 *
 *  \param perf_counter the perfcounter object for release
 *
 *  \return A non zero value if OpenCL failed to release PerfCounter
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_INVALID_OPERATION if we failed to release the object
 */
RUNTIME_ENTRY(cl_int, clReleasePerfCounterAMD, (cl_perfcounter_amd perf_counter)) {
  if (!is_valid(perf_counter)) {
    return CL_INVALID_OPERATION;
  }
  as_amd(perf_counter)->release();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Increments the perfcounter object reference count.
 *
 *  \param perf_counter the perfcounter object for retain
 *
 *  \return A non zero value if OpenCL failed to retain PerfCounter
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_INVALID_OPERATION if we failed to release the object
 */
RUNTIME_ENTRY(cl_int, clRetainPerfCounterAMD, (cl_perfcounter_amd perf_counter)) {
  if (!is_valid(perf_counter)) {
    return CL_INVALID_OPERATION;
  }
  as_amd(perf_counter)->retain();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Enqueues the begin command for the specified counters.
 *
 *  \param command_queue must be a valid OpenCL command queue.
 *
 *  \param num_perf_counters the number of perfcounter objects in the array.
 *
 *  \param perf_counters specifies an array of perfcounter objects.
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
 *
 *  \return A non zero value if OpenCL failed to release PerfCounter
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_INVALID_OPERATION if we failed to enqueue the begin operation
 *  - CL_INVALID_COMMAND_QUEUE if the queue is
 */
RUNTIME_ENTRY(cl_int, clEnqueueBeginPerfCounterAMD,
              (cl_command_queue command_queue, cl_uint num_perf_counters,
               cl_perfcounter_amd* perf_counters, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if ((num_perf_counters == 0) || (perf_counters == NULL)) {
    return CL_INVALID_OPERATION;
  }

  amd::HostQueue* hostQueue = as_amd(command_queue)->asHostQueue();
  if (NULL == hostQueue) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::PerfCounterCommand::PerfCounterList counters;

  // Place all counters into the list
  for (cl_uint i = 0; i < num_perf_counters; ++i) {
    amd::PerfCounter* amdPerf = as_amd(perf_counters[i]);
    if (&hostQueue->device() == &amdPerf->device()) {
      counters.push_back(amdPerf);
    } else {
      return CL_INVALID_DEVICE;
    }
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, *hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  // Create a new command for the performance counters
  amd::PerfCounterCommand* command = new amd::PerfCounterCommand(
      *hostQueue, eventWaitList, counters, amd::PerfCounterCommand::Begin);
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

/*! \brief Enqueues the end command for the specified counters.
 *
 *  \param command_queue must be a valid OpenCL command queue.
 *
 *  \param num_perf_counters the number of perfcounter objects in the array.
 *
 *  \param perf_counters specifies an array of perfcounter objects.
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
 *
 *  \return A non zero value if OpenCL failed to release PerfCounter
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_INVALID_OPERATION if we failed to enqueue the end operation
 */
RUNTIME_ENTRY(cl_int, clEnqueueEndPerfCounterAMD,
              (cl_command_queue command_queue, cl_uint num_perf_counters,
               cl_perfcounter_amd* perf_counters, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if ((num_perf_counters == 0) || (perf_counters == NULL)) {
    return CL_INVALID_OPERATION;
  }

  amd::HostQueue* hostQueue = as_amd(command_queue)->asHostQueue();
  if (NULL == hostQueue) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::PerfCounterCommand::PerfCounterList counters;

  // Place all counters into the list
  for (cl_uint i = 0; i < num_perf_counters; ++i) {
    amd::PerfCounter* amdPerf = as_amd(perf_counters[i]);
    if (&hostQueue->device() == &amdPerf->device()) {
      counters.push_back(amdPerf);
    } else {
      return CL_INVALID_DEVICE;
    }
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, *hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  // Create a new command for the performance counters
  amd::PerfCounterCommand* command = new amd::PerfCounterCommand(
      *hostQueue, eventWaitList, counters, amd::PerfCounterCommand::End);
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

/*! \brief Retrieves the results from the counter objects.
 *
 *  \param num_perf_counter the perfcounter object for the information query.
 *
 *  \param perf_counters specifies an array of perfcounter objects.
 *
 *  \param wait_event specifies the wait event, returned in
 *  the clEnqueueEndPerfCounterAMD.
 *
 *  \param wait true if OpenCL should wait for the perfcounter data.
 *
 *  \param values must be a valid pointer to an array of 64-bit values
 *  and the array size must be equal to num_perf_counters.
 *
 *  \return
 *  - CL_SUCCESS if the function is executed successfully.
 *  - CL_PROFILING_INFO_NOT_AVAILABLE if event isn't finished.
 *  - CL_INVALID_OPERATION if we failed to get the data
 */
RUNTIME_ENTRY(cl_int, clGetPerfCounterInfoAMD,
              (cl_perfcounter_amd perf_counter, cl_perfcounter_info param_name,
               size_t param_value_size, void* param_value, size_t* param_value_size_ret)) {
  // Check if we have a valid performance counter
  if (!is_valid(perf_counter)) {
    return CL_INVALID_OPERATION;
  }

  // Find the kernel, associated with the specified device
  const device::PerfCounter* devCounter = as_amd(perf_counter)->getDeviceCounter();

  // Make sure we found a valid performance counter
  if (devCounter == NULL) {
    return CL_INVALID_OPERATION;
  }

  // Get the corresponded parameters
  switch (param_name) {
    case CL_PERFCOUNTER_REFERENCE_COUNT: {
      cl_uint count = as_amd(perf_counter)->referenceCount();
      // Return the reference counter
      return amd::clGetInfo(count, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PERFCOUNTER_GPU_BLOCK_INDEX:
    case CL_PERFCOUNTER_GPU_COUNTER_INDEX:
    case CL_PERFCOUNTER_GPU_EVENT_INDEX: {
      cl_ulong data = devCounter->getInfo(param_name);
      // Return the device performance counter information
      return amd::clGetInfo(data, param_value_size, param_value, param_value_size_ret);
    }
    case CL_PERFCOUNTER_DATA: {
      cl_ulong data = devCounter->getInfo(param_name);
      if (static_cast<cl_ulong>(0xffffffffffffffffULL) == data) {
        return CL_PROFILING_INFO_NOT_AVAILABLE;
      }
      // Return the device performance counter result
      return amd::clGetInfo(data, param_value_size, param_value, param_value_size_ret);
    }
    default:
      return CL_INVALID_VALUE;
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

RUNTIME_ENTRY(cl_int, clSetDeviceClockModeAMD,
              (cl_device_id device, cl_set_device_clock_mode_input_amd set_clock_mode_input,
               cl_set_device_clock_mode_output_amd* set_clock_mode_output)) {
  // Make sure we have a valid device object
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }
  if (set_clock_mode_input.clock_mode >= CL_DEVICE_CLOCK_MODE_COUNT_AMD) {
    return CL_INVALID_VALUE;
  }
  amd::Device* amdDevice = as_amd(device);
  bool ret = amdDevice->SetClockMode(set_clock_mode_input, set_clock_mode_output);
  return (ret == true) ? CL_SUCCESS : CL_INVALID_OPERATION;
}
RUNTIME_EXIT

/*! @}
 *  @}
 */
