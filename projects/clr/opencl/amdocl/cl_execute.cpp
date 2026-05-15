/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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
#include "vdi_common.hpp"

#include "platform/kernel.hpp"
#include "platform/ndrange.hpp"
#include "platform/command.hpp"
#include "platform/program.hpp"
#include "os/os.hpp"

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup CL_Exec Executing Kernel Objects
 *
 *  @{
 */

/*! \brief Enqueue a command to execute a kernel on a device.
 *
 *  \param command_queue is a valid command-queue. The kernel  will  be  queued
 *  for execution on the device associated with \a command_queue.
 *
 *  \param kernel is a valid kernel object. The OpenCL context associated  with
 *  \a kernel and \a command-queue must be the same.
 *
 *  \param work_dim is the number of dimensions  used  to  specify  the  global
 *  work-items and work-items in the work-group.  \a work_dim  must  be greater
 *  than zero and less than or equal to three.
 *
 *  \param global_work_offset must currently be  a  NULL  value.  In  a  future
 *  revision of OpenCL, \a global_work_offset can be used to specify  an  array
 *  of \a work_dim unsigned values that describe the offset used  to  calculate
 *  the global ID of a work-item instead of having the global IDs always  start
 *  at offset (0, 0, 0).
 *
 *  \param global_work_size points to an array of \a work_dim  unsigned  values
 *  that describe the number of global work-items  in  \a  work_dim  dimensions
 *  that will  execute  the  kernel  function.   The  total  number  of  global
 *  work-items is computed as global_work_size[0] * ...
 *  * global_work_size[work_dim - 1].
 *
 *  \param local_work_size points to an array of \a  work_dim  unsigned  values
 *  that describe the number of work-items that  make  up  a  work-group  (also
 *  referred to as the size of the work-group)  that  will  execue  the  kernel
 *  specified by kernel.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list
 *
 *  \param event_wait_list specifies events that need to complete  before  this
 *  particular command can be executed. If  \a event_wait_list  is  NULL,  then
 *  this particular command does not wait on any event to complete.
 *  If \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points.
 *
 *  \param event returns an event object that identifies this particular kernel
 *  execution instance. Event objects are unique and can be used to identify  a
 *  particular kernel execution instance later on.  If \a  event  is  NULL,  no
 *  event will be created for this kernel execution instance and  therefore  it
 *  will not be possible for the application to query or queue a wait for  this
 *  particular kernel execution instance.
 *
 *  The total number of work-items in a work-group is computed as
 *  local_work_size[0] * ... * local_work_size[work_dim - 1].
 *  The total number of work-items in the work-group must be less than or equal
 *  to the CL_DEVICE_MAX_WORK_GROUP_SIZE. The explicitly specified
 *  \a local_work_size will be used to determine how to break the global  work-
 *  items specified by global_work_size into appropriate work-group  instances.
 *  If \a local_work_size is specified, the values specified in
 *  \a global_work_size[0], ...,  global_work_size[work_dim - 1] must be evenly
 *  divisable by the corresponding values specified in  \a  local_work_size[0],
 *  ..., local_work_size[work_dim - 1]. \a local_work_size can also be  a  NULL
 *  value in which case the OpenCL implementation  will  determine  how  to  be
 *  break the global work-items into appropriate work-groups.
 *
 *  If \a local_work_size is NULL and no work-group size is specified when  the
 *  kernel is compiled, the OpenCL implementation will determine how  to  break
 *  the global work-items specified  by  \a  global_work_size into  appropriate
 *  work-group instances. The work-group size to be used for kernel can also be
 *  specified in the program source using the
 *  __attribute__((reqd_work_group_size(X, Y, Z))) qualifier. In this case  the
 *  size of work group specified by \a  local_work_size must  match  the  value
 *  specified by the \a reqd_work_group_size attribute qualifier.
 *
 *  These  work-group  instances  are  executed  in  parallel  across  multiple
 *  compute units or concurrently on the  same  compute  unit.  Each  work-item
 *  is  uniquely identified by a global identifier. The global ID, which can be
 *  read inside the kernel is computed using the value given by
 *  \a global_work_size and \a global_work_offset.
 *
 *  \return One of the following values:
 *
 *  - CL_SUCCESS if the kernel execution was successfully queued
 *
 *  - CL_INVALID_PROGRAM_EXECUTABLE if there is no successfully  built  program
 *    executable available for device associated with \a command_queue.
 *
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue
 *
 *  - CL_INVALID_KERNEL if \a kernel is not a valid kernel object.
 *
 *  - CL_INVALID_CONTEXT if context associated with command_queue and kernel are
 *    not the same or if the context associated with command_queue and events in
 *    event_wait_list are not the same.
 *
 *  - CL_INVALID_KERNEL_ARGS if the kernel argument values have not been
 *    specified or are not valid for the device on which kernel will be
 *    executed.
 *
 *  - CL_INVALID_WORK_DIMENSION if \a work_dim is not a valid value
 *    (i.e. a value between 1 and 3).
 *
 *  - CL_INVALID_WORK_GROUP_SIZE if \a local_work_size is specified and  number
 *    of workitems specified by \a global_work_size is not evenly divisable  by
 *    size of work-given by \a local_work_size or does not match the work-group
 *    size specified for kernel using the
 *    __attribute__((reqd_work_group_size(X, Y, Z))) qualifier in program
 *    source.
 *
 *  - CL_INVALID_GLOBAL_OFFSET if \a global_work_offset is not NULL.
 *
 *  - CL_OUT_OF_RESOURCES if there is a failure to queue the execution instance
 *    of \a kernel on  the  command-queue  because  of  insufficient  resources
 *    needed to execute the  kernel.  For  example,  the  explicitly  specified
 *    \a local_work_dim in range causes a failure to execute the kernel because
 *    of insufficient resources such as  registers  or  local  memory.  Another
 *    example would be the number of read-only image args used in kernel exceed
 *    the CL_DEVICE_MAX_READ_IMAGE_ARGS value  for  device  or  the  number  of
 *    write-only image args used in kernel exceed the
 *    CL_DEVICE_MAX_WRITE_IMAGE_ARGS value for device or the number of samplers
 *    used in kernel exceed CL_DEVICE_MAX_SAMPLERS for device.
 *
 *  - CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory
 *    for image or buffer objects specified as arguments to kernel.
 *
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in
 *    \a event_wait_list are not valid events.
 *
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *    required by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueNDRangeKernel,
              (cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
               const size_t* global_work_offset, const size_t* global_work_size,
               const size_t* local_work_size, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  *not_null(event) = NULL;

  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  if (!is_valid(kernel)) {
    return CL_INVALID_KERNEL;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  const amd::Kernel* amdKernel = as_amd(kernel);
  if (&hostQueue.context() != &amdKernel->program().context()) {
    return CL_INVALID_CONTEXT;
  }

  const amd::Device& device = hostQueue.device();
  const device::Kernel* devKernel = amdKernel->getDeviceKernel(device);
  if (devKernel == NULL) {
    return CL_INVALID_PROGRAM_EXECUTABLE;
  }

  if (amdKernel->parameters().getSvmSystemPointersSupport() == FGS_YES &&
      !(device.info().svmCapabilities_ & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) {
    // The user indicated that this kernel will access SVM system pointers,
    // but the device does not support them.
    return CL_INVALID_OPERATION;
  }

  if (work_dim < 1 || work_dim > 3) {
    return CL_INVALID_WORK_DIMENSION;
  }
#if !defined(CL_VERSION_1_1)
  if (global_work_offset != NULL) {
    return CL_INVALID_GLOBAL_OFFSET;
  }
#endif  // CL_VERSION
  if (global_work_size == NULL) {
    if ((OPENCL_MAJOR == 2 && OPENCL_MINOR >= 1) || (OPENCL_MAJOR > 2)) {
      static size_t inputZeroes[3] = {0, 0, 0};
      global_work_size = inputZeroes;
    } else {
      return CL_INVALID_VALUE;
    }
  }

  if (local_work_size == NULL) {
    static size_t zeroes[3] = {0, 0, 0};
    local_work_size = zeroes;
  } else {
    size_t numWorkItems = 1;
    for (cl_uint dim = 0; dim < work_dim; ++dim) {
      if ((devKernel->workGroupInfo()->compileSize_[0] != 0) &&
          (local_work_size[dim] != devKernel->workGroupInfo()->compileSize_[dim])) {
        return CL_INVALID_WORK_GROUP_SIZE;
      }
      // >32bits global work size is not supported.
      if ((global_work_size[dim] == 0) ||
          (global_work_size[dim] > static_cast<size_t>(0xffffffff))) {
        return CL_INVALID_GLOBAL_WORK_SIZE;
      }
      numWorkItems *= local_work_size[dim];
    }
    // Make sure local work size is valid
    if ((numWorkItems == 0) || (numWorkItems > devKernel->workGroupInfo()->size_)) {
      return CL_INVALID_WORK_GROUP_SIZE;
    }
    // Check if uniform was requested and validate dimensions
    if (devKernel->workGroupInfo()->uniformWorkGroupSize_) {
      for (cl_uint dim = 0; dim < work_dim; ++dim) {
        if ((global_work_size[dim] % local_work_size[dim]) != 0) {
          return CL_INVALID_WORK_GROUP_SIZE;
        }
      }
    }
  }

  // Check that all parameters have been defined.
  if (!amdKernel->parameters().check()) {
    return CL_INVALID_KERNEL_ARGS;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::NDRangeContainer ndrange((size_t)work_dim, global_work_offset, global_work_size,
                                local_work_size);
  amd::NDRangeKernelCommand* command =
      new amd::NDRangeKernelCommand(hostQueue, eventWaitList, *as_amd(kernel), ndrange);
  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }
  // ndrange is now owned by command. Do not delete it!

  // Make sure we have memory for the command execution
  cl_int result = command->captureAndValidate();
  if (result != CL_SUCCESS) {
    delete command;
    return result;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Enqueue a command to execute a kernel on a device.
 *  The kernel is executed using a single work-item.
 *
 *  \param command_queue is a valid command-queue. The kernel will be queued
 *  for execution on the device associated with \a command_queue.
 *
 *  \param kernel is a valid kernel object. The OpenCL context associated with
 *  \a kernel and \a command-queue must be the same.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list
 *
 *  \param event_wait_list specifies events that need to complete before this
 *  particular command can be executed. If \a event_wait_list is NULL, then
 *  this particular command does not wait on any event to complete.
 *  If \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points.
 *
 *  \param event returns an event objects that identifies this particular kernel
 *  execution instance. Event objects are unique and can be used to identify a
 *  particular kernel execution instance later on. If \a event is NULL, no event
 *  will be created for this kernel execution instance and therefore it will not
 *  be possible for the application to query or queue a wait for this particular
 *  kernel execution instance.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the kernel execution was successfully queued.
 *  - CL_INVALID_PROGRAM_EXECUTABLE if there is no successfully built program
 *    executable available for device associated with \a command_queue.
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_KERNEL if \a kernel is not a valid kernel object.
 *  - CL_INVALID_KERNEL_ARGS if the kernel argument values have not been
 *    specified or are not valid for the device on which kernel will be
 *    executed.
 *  - CL_INVALID_WORK_GROUP_SIZE if a work-group size is specified for
 *    kernel using the __attribute__((reqd_work_group_size(X, Y, Z)))
 *    qualifier in program source and is not (1, 1, 1).
 *  - CL_OUT_OF_RESOURCES if there is a failure to queue the execution instance
 *    of kernel on the command-queue because of insufficient resources needed
 *    to execute the kernel. For example, the explicitly specified
 *    \a local_work_dim in range causes a failure to execute the kernel because
 *    of insufficient resources such as registers or local memory. Another
 *    example would be the number of read-only image args used in kernel exceed
 *    the CL_DEVICE_MAX_READ_IMAGE_ARGS value for device or the number of
 *    write-only image args used in kernel exceed the
 *    CL_DEVICE_MAX_WRITE_IMAGE_ARGS value for device or the number of samplers
 *    used in kernel exceed CL_DEVICE_MAX_SAMPLERS for device.
 *  - CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory
 *    for image or buffer objects specified as arguments to kernel.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueTask,
              (cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  static size_t const globalWorkSize[3] = {1, 0, 0};
  static size_t const localWorkSize[3] = {1, 0, 0};

  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::HostQueue* hostQueue = as_amd(command_queue)->asHostQueue();
  if (NULL == hostQueue) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  return hostQueue->dispatch_->clEnqueueNDRangeKernel(
      command_queue, kernel, 1, NULL, globalWorkSize, localWorkSize, num_events_in_wait_list,
      event_wait_list, event);
}
RUNTIME_EXIT

/*! \brief Enqueue a command to execute a native C/C++ function not compiled
 *  using the OpenCL compiler.
 *
 *  \param command_queue is a valid command-queue. A native user function can
 *  only be executed on a command-queue created on a device that has
 *  CL_EXEC_NATIVE_KERNEL capability set in CL_DEVICE_EXECUTION_CAPABILITIES.
 *
 *  \param user_func is a pointer to a host-callable user function.
 *
 *  \param args is a pointer to the args list that \a user_func should be called
 *  with.
 *
 *  \param cb_args is the size in bytes of the args list that args points to.
 *  The data pointed to by \a args and \a cb_args bytes in size will be copied
 *  and a pointer to this copied region will be passed to \a user_func. The copy
 *  needs to be done because the memory objects (cl_mem values) that args may
 *  contain need to be modified and replaced by appropriate pointers to global
 *  memory. When clEnqueueNativeKernel returns, the memory region pointed to by
 *  args can be reused by the application.
 *
 *  \param num_mem_objects is the number of buffer objects that are passed in
 *  args.
 *
 *  \param mem_list is a list of valid buffer objects, if \a num_mem_objects > 0
 *
 *  \param args_mem_loc is a pointer to appropriate locations that args points
 *  to where memory object handles (cl_mem values) are stored. Before the user
 *  function is executed, the memory object handles are replaced by pointers to
 *  global memory.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list
 *
 *  \param event_wait_list as described in clEnqueueNDRangeKernel.
 *
 *  \param event returns an event objects that identifies this particular kernel
 *  execution instance. Event objects are unique and can be used to identify a
 *  particular kernel execution instance later on. If \a event is NULL, no event
 *  will be created for this kernel execution instance and therefore it will not
 *  be possible for the application to query or queue a wait for this particular
 *  kernel execution instance.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the user function execution instance was successfully queued
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue.
 *  - CL_INVALID_VALUE if \a user_func is NULL, or if \a args is a NULL value
 *    and \a num_mem_objects > 0 or if \a num_mem_objects > 0 and \a mem_list
 *    is NULL.
 *  - CL_INVALID_OPERATION if device cannot execute the native kernel.
 *  - CL_INVALID_MEM_OBJECT if one or more memory objects specified in
 *    \a mem_list are not valid or are not buffer objects.
 *  - CL_OUT_OF_RESOURCES if there is a failure to queue the execution instance
 *    of kernel on the command-queue because of insufficient resources needed
 *    to execute the kernel.
 *  - CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory
 *    for buffer objects specified as arguments to \a kernel.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueNativeKernel,
              (cl_command_queue command_queue, void(CL_CALLBACK* user_func)(void*), void* args,
               size_t cb_args, cl_uint num_mem_objects, const cl_mem* mem_list,
               const void** args_mem_loc, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  *not_null(event) = NULL;

  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  const amd::Device& device = hostQueue.device();

  if (!(device.info().executionCapabilities_ & CL_EXEC_NATIVE_KERNEL)) {
    return CL_INVALID_OPERATION;
  }

  if (user_func == NULL || (num_mem_objects > 0 && (mem_list == NULL || args_mem_loc == NULL)) ||
      (num_mem_objects == 0 && (mem_list != NULL || args_mem_loc != NULL)) ||
      (args == NULL && (cb_args > 0 || num_mem_objects > 0)) || (args != NULL && cb_args == 0)) {
    return CL_INVALID_VALUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  for (size_t i = 0; i < num_mem_objects; ++i) {
    cl_mem obj = mem_list[i];
    if (!is_valid(obj)) {
      return CL_INVALID_MEM_OBJECT;
    }
  }

  amd::NativeFnCommand* command = new amd::NativeFnCommand(
      hostQueue, eventWaitList, user_func, args, cb_args, num_mem_objects, mem_list, args_mem_loc);
  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *
 *  \addtogroup CL_Order Out of order Execution of Kernels and Memory Commands
 *
 *  The OpenCL functions that are submitted to a command-queue are queued in
 *  the order the calls are made but can be configured to execute in-order or
 *  out-of-order. The properties argument in clCreateCommandQueue can be used
 *  to specify the execution order.
 *
 *  If the CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE property of a command-queue
 *  is not set, the commands queued to a command-queue execute in order.
 *  For example, if an application calls clEnqueueNDRangeKernel to execute
 *  kernel A followed by a clEnqueueNDRangeKernel to execute kernel B,
 *  the application can assume that kernel A finishes first and then kernel B
 *  is executed. If the memory objects output by kernel A are inputs to kernel B
 *  then kernel B will see the correct data in memory objects produced
 *  by execution of kernel A. If the CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
 *  property of a commandqueue is set, then there is no guarantee that kernel A
 *  will finish before kernel B starts execution.
 *
 *  Applications can configure the commands queued to a command-queue to
 *  execute out-of-order by setting the CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
 *  property of the commandqueue.  This can be specified when the command-queue
 *  is created or can be changed dynamically using clSetCommandQueueProperty.
 *  In out-of-order execution mode there is no guarantee that the queued
 *  commands will finish execution in the order they were queued. As there is
 *  no guarantee that kernels will be executed in order i.e. based on when
 *  the clEnqueueNDRangeKernel calls are made within a command-queue, it is
 *  therefore possible that an earlier clEnqueueNDRangeKernel call to execute
 *  kernel A identified by event A may execute and/or finish later than a
 *  clEnqueueNDRangeKernel call to execute kernel B which was called by the
 *  application at a later point in time. To guarantee a specific order of
 *  execution of kernels, a wait on a particular event (in this case event A)
 *  can be used. The wait for event A can be specified in the event_wait_list
 *  argument to clEnqueueNDRangeKernel for kernel B.
 *
 *  In addition, a wait for events or a barrier function can be queued to the
 *  command-queue. The wait for events command ensures that previously queued
 *  commands identified by the list of events to wait for have finished before
 *  the next batch of commands is executed. The barrier ensures that all
 *  previously queued commands in a command-queue have finished execution
 *  before the next batch of commands is executed.
 *
 *  Similarly, commands to read, write, copy or map memory objects that are
 *  queued after clEnqueueNDRangeKernel, clEnqueueTask or clEnqueueNativeKernel
 *  commands are not guaranteed to wait for kernels scheduled for execution
 *  to have completed (if the CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE property
 *  is set). To ensure correct ordering of commands, the event object returned
 *  by clEnqueueNDRangeKernel, clEnqueueTask or clEnqueueNativeKernel can be
 *  used to queue a wait for event or a barrier command can be queued that must
 *  complete before reads or writes to the memory object(s) occur.
 *
 *  @{
 */

/*! \brief Enqueue a marker command to \a command_queue.
 *
 *  The marker command returns an event which can be used by to queue a wait on
 *  this marker event i.e. wait for all commands queued before the marker
 *  command to complete.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is successfully executed
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue
 *  - CL_INVALID_VALUE if \a event is a NULL value
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueMarker, (cl_command_queue command_queue, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::HostQueue* hostQueue = as_amd(command_queue)->asHostQueue();
  if (NULL == hostQueue) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::Command* command = new amd::Marker(*hostQueue, true);
  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief enqueues a marker command which waits for either a list of events
 *  to complete, or if the list is empty it waits for all commands previously
 *  enqueued in \a command_queue to complete before it completes. This command
 *  returns an event which can be waited on, i.e. this event can be waited on
 *  to insure that all events either in the \a event_wait_list or all
 *  previously enqueued commands, queued before this command to
 *  \a command_queue, have completed.
 *
 *  \param command_queue is a valid command-queue.
 *
 *  \param num_events_in_wait_list specifies the number of events given
 *  by \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete before this
 *  particular command can be executed.
 *  If \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must
 *  be greater than 0. The events specified in event_wait_list act as
 *  synchronization points. The context associated with events in
 *  \a event_wait_list and \a command_queue must be the same. The
 *  memory associated with \a event_wait_list can be reused or freed after
 *  the function returns.
 *  If \a event_wait_list is NULL, then this particular command waits until
 *  all previous enqueued commands to \a command_queue have completed.
 *
 *  \param event returns an event object that identifies this particular
 *  kernel execution instance. Event objects are unique and can be used to
 *  identify this marker command later on.
 *
 *  \return CL_SUCCESS if the function is successfully executed.
 *  Otherwise, it returns one of the following errors:
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid \a command-queue.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required by
 *    the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY(cl_int, clEnqueueMarkerWithWaitList,
              (cl_command_queue command_queue, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::HostQueue* hostQueue = as_amd(command_queue)->asHostQueue();
  if (NULL == hostQueue) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, *hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::Command* command = new amd::Marker(*hostQueue, true, eventWaitList);
  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }
  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Enqueue a wait for a specific event or a list of events to complete
 *  before any future commands queued in the command-queue are executed.
 *
 *  \param command_queue is a valid command-queue.
 *
 *  \param num_events specifies the number of events given by \a event_list.
 *
 *  \param event_list is the list of events. Each event in \a event_list must
 *  be a valid event object returned by a previous call to:
 *  - clEnqueueNDRangeKernel
 *  - clEnqueueTask
 *  - clEnqueueNativeKernel
 *  - clEnqueue{Read|Write|Map}{Buffer|Image}
 *  - clEnqueueCopy{Buffer|Image}
 *  - clEnqueueCopyBufferToImage
 *  - clEnqueueCopyImageToBuffer
 *  - clEnqueueMarker.
 *  The events specified in \a event_list act as synchronization points.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function was successfully executed.
 *  - CL_INVALID_COMMAND_QUEUE if c\a ommand_queue is not a valid command-queue
 *  - CL_INVALID_VALUE if \a num_events is zero or \a event_list is NULL
 *  - CL_INVALID_EVENT if event objects specified in \a event_list are not valid
 *    events
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueWaitForEvents,
              (cl_command_queue command_queue, cl_uint num_events, const cl_event* event_list)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  amd::Command::EventWaitList eventWaitList;
  cl_int err = amd::clSetEventWaitList(eventWaitList, hostQueue, num_events, event_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::Command* command = new amd::Marker(hostQueue, false, eventWaitList);
  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  command->enqueue();
  command->release();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Enqueue a barrier operation.
 *
 *  The clEnqueueBarrier command ensures that all queued commands in
 *  \a command_queue have finished execution before the next batch of commands
 *  can begin execution. clEnqueueBarrier is a synchronization point.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function was executed successfully
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clEnqueueBarrier, (cl_command_queue command_queue)) {
  //! @todo: Unimplemented();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief enqueues a barrier command which waits for either a list of events
 *  to complete, or if the list is empty it waits for all commands previously
 *  enqueued in \a command_queue to complete before it completes. This command
 *  blocks command execution, that is, any following commands enqueued after it
 *  do not execute until it completes. This command returns an event which can
 *  be waited on, i.e. this event can be waited on to insure that all events
 *  either in the \a event_wait_list or all previously enqueued commands,
 *  queued before this command to command_queue, have completed
 *
 *  \param command_queue is a valid command-queue.
 *
 *  \param num_events_in_wait_list specifies the number of events given
 *  by \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete before this
 *  particular command can be executed.
 *  If \a event_wait_list is NULL, \a num_events_in_wait_list must be 0.
 *  If \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must
 *  be greater than 0. The events specified in event_wait_list act as
 *  synchronization points. The context associated with events in
 *  \a event_wait_list and \a command_queue must be the same. The
 *  memory associated with \a event_wait_list can be reused or freed after
 *  the function returns.
 *  If \a event_wait_list is NULL, then this particular command waits until
 *  all previous enqueued commands to \a command_queue have completed.
 *
 *  \param event returns an event object that identifies this particular
 *  kernel execution instance. Event objects are unique and can be used to
 *  identify this marker command later on.
 *
 *  \return CL_SUCCESS if the function is successfully executed.
 *  Otherwise, it returns one of the following errors:
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid \a command-queue.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required by
 *    the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 1.2r07
 */
RUNTIME_ENTRY(cl_int, clEnqueueBarrierWithWaitList,
              (cl_command_queue command_queue, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::HostQueue* hostQueue = as_amd(command_queue)->asHostQueue();
  if (NULL == hostQueue) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, *hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  //!@note: with the current runtime architecture and in-order execution
  //! barrier and marker should be the same operation
  amd::Command* command = new amd::Marker(*hostQueue, true, eventWaitList);
  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }
  command->enqueue();

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *
 *  \addtogroup CL_Profiling Profiling Operations on Memory Objects and Kernels
 *
 *  Profiling of OpenCL functions that are enqueued as commands to a
 *  command-queue. The specific functions being referred to are:
 *    - clEnqueue{Read|Write|Map}Buffer,
 *    - clEnqueue{Read|Write|Map}Image,
 *    - clEnqueueCopy{Buffer|Image},
 *    - clEnqueueCopyImageToBuffer,
 *    - clEnqueueCopyBufferToImage,
 *    - clEnqueueNDRangeKernel ,
 *    - clEnqueueTask and
 *    - clEnqueueNativeKernel.
 *  These enqueued commands are identified by unique event objects.
 *
 *  Event objects can be used to capture profiling information that measure
 *  execution time of a command. Profiling of OpenCL commands can be enabled
 *  either by using a command-queue created with CL_QUEUE_PROFILING_ENABLE
 *  flag set in properties arguments to clCreateCommandQueue or by setting the
 *  CL_QUEUE_PROFILING_ENABLE flag in properties arguments to
 *  clSetCommandQueueProperty.
 *
 *  @{
 */

/*! \brief Return profiling information for the command associated with event.
 *
 *  \param event specifies the event object.
 *
 *  \param param_name specifies the profiling data to query.
 *
 *  \param param_value is a pointer to memory where the appropriate result being
 *  queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type
 *
 *  \param param_value_size_ret returns the actual size in bytes of data copied
 *  to \a param_value. If \a param_value_size_ret is NULL, it is ignored.
 *
 *  The unsigned 64-bit values returned can be used to measure the time in
 *  nano-seconds consumed by OpenCL commands. OpenCL devices are required to
 *  correctly track time across changes in frequency and p-states. The
 *  CL_DEVICE_PROFILING_TIMER_RESOLUTION specifies the resolution of the timer
 *  i.e. the number of nanoseconds elapsed before the timer is incremented.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully and the profiling
 *    information has been recorded
 *  - CL_PROFILING_INFO_NOT_AVAILABLE if the profiling information is currently
 *    not available (because the command identified by event has not completed)
 *  - CL_INVALID_VALUE if \a param_name is not valid, or if size in bytes
 *    specified by param_value_size is < size of return type and \a param_value
 *    is not NULL
 *  - CL_INVALID_EVENT if \a event is a not a valid event object.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clGetEventProfilingInfo,
              (cl_event event, cl_profiling_info param_name, size_t param_value_size,
               void* param_value, size_t* param_value_size_ret)) {
  if (!is_valid(event)) {
    return CL_INVALID_EVENT;
  }

  if (!as_amd(event)->profilingInfo().enabled_) {
    return CL_PROFILING_INFO_NOT_AVAILABLE;
  }

  if (param_value != NULL && param_value_size < sizeof(cl_ulong)) {
    return CL_INVALID_VALUE;
  }

  *not_null(param_value_size_ret) = sizeof(cl_ulong);
  if (param_value != NULL) {
    cl_ulong value = 0;
    switch (param_name) {
      case CL_PROFILING_COMMAND_END:
      case CL_PROFILING_COMMAND_COMPLETE:
        value = as_amd(event)->profilingInfo().end_;
        break;

      case CL_PROFILING_COMMAND_START:
        value = as_amd(event)->profilingInfo().start_;
        break;

      case CL_PROFILING_COMMAND_SUBMIT:
        value = as_amd(event)->profilingInfo().submitted_;
        break;

      case CL_PROFILING_COMMAND_QUEUED:
        value = as_amd(event)->profilingInfo().queued_;
        break;

      default:
        return CL_INVALID_VALUE;
    }
    if (value == 0) {
      return CL_PROFILING_INFO_NOT_AVAILABLE;
    }
    *(cl_ulong*)param_value = value;
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Returns a reasonably synchronized pair of timestamps from the device
 *  timer and the host timer as seen by device.
 *
 *  \param device a device returned by clGetDeviceIDs.
 *
 *  \param device_timestamp will be updated with the value of the current timer
 *  in nanoseconds. The resolution of the timer is the same as the device
 *  profiling timer returned by clGetDeviceInfo and the
 *  CL_DEVICE_PROFILING_TIMER_RESOLUTION query.
 *
 *  \param host_timestamp will be updated with the value of the current timer
 *  in nanoseconds at the closest possible point in time to that at which
 *  device_timer was returned. The resolution of the timer may be queried
 *  via clGetPlatformInfo and the flag CL_PLATFORM_HOST_TIMER_RESOLUTION.
 *
 *  Returns a reasonably synchronized pair of timestamps from the device
 *  timer and the host timer as seen by device. Implementations may need
 *  to execute this query with a high latency in order to provide reasonable
 *  synchronization of the timestamps. The host timestamp and device timestamp
 *  returned by this function and clGetHostTimer each have an implementation
 *  defined timebase. The timestamps will always be in their respective timebases
 *  regardless of which query function is used. The timestamp returned from
 *  clGetEventProfilingInfo for an event on a device and a device timestamp
 *  queried from the same device will always be in the same timebase.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if a time value in host_timestamp is provided
 *  - CL_INVALID_DEVICE if device is not a valid OpenCL device.
 *  - CL_INVALID_VALUE if host_timestamp is NULL.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 */
RUNTIME_ENTRY(cl_int, clGetDeviceAndHostTimer,
              (cl_device_id device, cl_ulong* device_timestamp, cl_ulong* host_timestamp)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (!device_timestamp || !host_timestamp) {
    return CL_INVALID_VALUE;
  }

  // The device timestamp and host timestamp use the same timebase.
  *device_timestamp = *host_timestamp = amd::Os::timeNanos();

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Return the current value of the host clock as seen by device.
 *
 *  \param device a device returned by clGetDeviceIDs.
 *
 *  \param host_timestamp will be updated with the value of the current timer
 *  in nanoseconds. The resolution of the timer may be queried via
 *  clGetPlatformInfo and the flag CL_PLATFORM_HOST_TIMER_RESOLUTION.
 *
 *  Return the current value of the host clock as seen by device. This value
 *  is in the same timebase as the host_timestamp returned from
 *  clGetDeviceAndHostTimer. The implementation will return with as low a
 *  latency as possible to allow a correlation with a subsequent application
 *  sampled time. The host timestamp and device timestamp returned by this
 *  function and clGetDeviceAndHostTimer each have an implementation defined
 *  timebase. The timestamps will always be in their respective timebases
 *  regardless of which query function is used. The timestamp returned from
 *  clGetEventProfilingInfo for an event on a device and a device timestamp
 *  queried from the same device will always be in the same timebase.
 *
 *  \return One of the following values:
 *
 *  - CL_SUCCESS if a time value in host_timestamp is provided
 *  - CL_INVALID_DEVICE if device is not a valid OpenCL device.
 *  - CL_INVALID_VALUE if host_timestamp is NULL.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 */
RUNTIME_ENTRY(cl_int, clGetHostTimer, (cl_device_id device, cl_ulong* host_timestamp)) {
  if (!is_valid(device)) {
    return CL_INVALID_DEVICE;
  }

  if (!host_timestamp) {
    return CL_INVALID_VALUE;
  }

  *host_timestamp = amd::Os::timeNanos();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  \addtogroup CL_FlushFinish Flush and Finish
 *  @{
 */

/*! \brief Issue all previously queued OpenCL commands in \a command_queue to
 *  the device associated with command_queue.
 *
 *  clFlush only guarantees that all queued commands to \a command_queue get
 *  issued to the appropriate device. There is no guarantee that they will be
 *  complete after clFlush returns.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function call was executed successfully
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  Any blocking commands queued in a command-queue such as
 *  clEnqueueRead{Image|Buffer} with \a blocking_read set to CL_TRUE,
 *  clEnqueueWrite{Image|Buffer} with \a blocking_write set to CL_TRUE,
 *  clEnqueueMap{Buffer|Image} with \a blocking_map set to CL_TRUE or
 *  clWaitForEvents perform an implicit flush of the command-queue.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clFlush, (cl_command_queue command_queue)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::HostQueue* hostQueue = as_amd(command_queue)->asHostQueue();
  if (NULL == hostQueue) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::Command* command = new amd::Marker(*hostQueue, false);
  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  command->enqueue();
  command->release();

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Block until all previously queued OpenCL runtime commands in
 *  \a command_queue are issued to the associated device and have completed.
 *
 *  clFinish does not return until all queued commands in \a command_queue have
 *  been processed and completed. clFinish is also a synchronization point.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function call was executed successfully.
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clFinish, (cl_command_queue command_queue)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::HostQueue* hostQueue = as_amd(command_queue)->asHostQueue();
  if (NULL == hostQueue) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  hostQueue->finish();

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  @}
 */
