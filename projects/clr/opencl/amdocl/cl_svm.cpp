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
#include "platform/command.hpp"
#include "platform/kernel.hpp"
#include "platform/program.hpp"

/*! \brief Helper function to validate SVM allocation flags
 *
 *  \return true if flags are valid, otherwise - false
 */
static bool validateSvmFlags(cl_svm_mem_flags flags) {
  if (!flags) {
    // coarse-grained allocation
    return true;
  }
  const cl_svm_mem_flags rwFlags = CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY;
  const cl_svm_mem_flags setFlags =
      flags & (rwFlags | CL_MEM_SVM_ATOMICS | CL_MEM_SVM_FINE_GRAIN_BUFFER);
  if (flags != setFlags) {
    // invalid flags value
    return false;
  }

  if (amd::countBitsSet(flags & rwFlags) > 1) {
    // contradictory R/W flags
    return false;
  }

  if ((flags & CL_MEM_SVM_ATOMICS) && !(flags & CL_MEM_SVM_FINE_GRAIN_BUFFER)) {
    return false;
  }

  return true;
}

/*! \brief Helper function to validate cl_map_flags
 *
 *  \return true if flags are valid, otherwise - false
 */
static bool validateMapFlags(cl_map_flags flags) {
  const cl_map_flags maxFlag = CL_MAP_WRITE_INVALIDATE_REGION;
  if (flags >= (maxFlag << 1)) {
    // at least one flag is out-of-range
    return false;
  } else if ((flags & CL_MAP_WRITE_INVALIDATE_REGION) && (flags & (CL_MAP_READ | CL_MAP_WRITE))) {
    // CL_MAP_READ or CL_MAP_WRITE and CL_MAP_WRITE_INVALIDATE_REGION are
    // mutually exclusive.
    return false;
  }
  return true;
}

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup SVM
 *  @{
 *
 */

/*! \brief Allocate a shared virtual memory buffer that can be shared by the
 * host and all devices in an OpenCL context.
 *
 *  \param context is a valid OpenCL context used to create the SVM buffer.
 *
 *  \param flags is a bit-field that is used to specify allocation and usage
 *  information. If CL_MEM_SVM_FINE_GRAIN_BUFFER is not specified, the
 *  buffer is created as a coarse grained SVM allocation. Similarly, if
 *  CL_MEM_SVM_ATOMICS is not specified, the buffer is created without
 *  support for SVM atomic operations.
 *
 *  \param size is the size in bytes of the SVM buffer to be allocated.
 *
 *  \param alignment is the minimum alignment in bytes that is required for the
 *  newly created buffer?s memory region. It must be a power of two up to the
 *  largest data type supported by the OpenCL device. For the full profile, the
 *  largest data type is long16. For the embedded profile, it is long16 if the
 *  device supports 64-bit integers; otherwise it is int16. If alignment is 0, a
 *  default alignment will be used that is equal to the size of largest data
 *  type supported by the OpenCL implementation.
 *
 *  \return A valid non-NULL shared virtual memory address if the SVM buffer
 *  is successfully allocated. Otherwise, like malloc, it returns a NULL pointer
 *  value. clSVMAlloc will fail if
 *  - \a context is not a valid context.
 *  - \a flags does not contain CL_MEM_SVM_FINE_GRAIN_BUFFER but does
 *  contain CL_MEM_SVM_ATOMICS.
 *  - Values specified in \a flags do not follow rules for that particular type.
 *  - CL_MEM_SVM_FINE_GRAIN_BUFFER or CL_MEM_SVM_ATOMICS is specified
 *  in \a flags and these are not supported by at least one device in
 *  \a context.
 *  - The values specified in \a flags are not valid.
 *  - \a size is 0 or > CL_DEVICE_MAX_MEM_ALLOC_SIZE value for any device in
 *  \a context.
 *  - \a alignment is not a power of two or the OpenCL implementation cannot
 *  support the specified alignment for at least one device in \a context.
 *  - There was a failure to allocate resources.
 *
 *  \version 2.0r15
 */
RUNTIME_ENTRY_RET_NOERRCODE(void*, clSVMAlloc,
                            (cl_context context, cl_svm_mem_flags flags, size_t size,
                             unsigned int alignment)) {
  if (!is_valid(context)) {
    LogWarning("invalid parameter \"context\"");
    return NULL;
  }

  if (size == 0) {
    LogWarning("invalid parameter \"size = 0\"");
    return NULL;
  }

  if (!validateSvmFlags(flags)) {
    LogWarning("invalid parameter \"flags\"");
    return NULL;
  }

  if (!amd::isPowerOfTwo(alignment)) {
    LogWarning("invalid parameter \"alignment\"");
    return NULL;
  }

  const std::vector<amd::Device*>& devices = as_amd(context)->svmDevices();
  bool sizePass = false;
  cl_device_svm_capabilities combinedSvmCapabilities = 0;
  const cl_uint hostAddressBits = LP64_SWITCH(32, 64);
  cl_uint minContextAlignment = std::numeric_limits<uint>::max();
  for (const auto& it : devices) {
    cl_device_svm_capabilities svmCapabilities = it->info().svmCapabilities_;
    if (svmCapabilities == 0) {
      continue;
    }
    combinedSvmCapabilities |= svmCapabilities;

    if (it->info().maxMemAllocSize_ >= size) {
      sizePass = true;
    }

    if (it->info().addressBits_ < hostAddressBits) {
      LogWarning("address mode mismatch between host and device");
      return NULL;
    }

    // maximum alignment for a device is given in bits.
    cl_uint baseAlignment = it->info().memBaseAddrAlign_ >> 3;
    if (alignment > baseAlignment) {
      LogWarning("invalid parameter \"alignment\"");
      return NULL;
    }

    minContextAlignment = std::min(minContextAlignment, baseAlignment);
  }
  if ((flags & CL_MEM_SVM_FINE_GRAIN_BUFFER) &&
      !(combinedSvmCapabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
    LogWarning("No device in context supports SVM fine grained buffers");
    return NULL;
  }
  if ((flags & CL_MEM_SVM_ATOMICS) && !(combinedSvmCapabilities & CL_DEVICE_SVM_ATOMICS)) {
    LogWarning("No device in context supports SVM atomics");
    return NULL;
  }
  if (!sizePass) {
    LogWarning("invalid parameter \"size\"");
    return NULL;
  }

  // if alignment not specified, use largest data type alignment supported
  if (alignment == 0) {
    alignment = minContextAlignment;
    ClPrint(amd::LOG_INFO, amd::LOG_API, "Assumed alignment %d\n", alignment);
  }

  amd::Context& amdContext = *as_amd(context);
  return amd::SvmBuffer::malloc(amdContext, flags, size, alignment);
}
RUNTIME_EXIT

/*! \brief Free a shared virtual memory buffer allocated using clSVMAlloc.
 *
 *  \param context is a valid OpenCL context used to create the SVM buffer.
 *
 *  \param svm_pointer must be the value returned by a call to clSVMAlloc. If a
 *  NULL pointer is passed in \a svm_pointer, no action occurs.
 *
 *  \version 2.0r15
 */
RUNTIME_ENTRY_VOID(void, clSVMFree, (cl_context context, void* svm_pointer)) {
  if (!is_valid(context)) {
    LogWarning("invalid parameter \"context\"");
    return;
  }

  if (svm_pointer == NULL) {
    return;
  }

  amd::Context& amdContext = *as_amd(context);
  amd::SvmBuffer::free(amdContext, svm_pointer);
}
RUNTIME_EXIT

/*! \brief enqueues a command to free shared virtual memory allocated using
 *  clSVMAlloc or a shared system memory pointer.
 *
 *  \param command_queue is a valid host command-queue.
 *
 *  \param num_svm_pointers specifies the number of elements in \a svm_pointers.
 *
 *  \param svm_pointers is a list of shared virtual memory pointers to
 *  be freed. Each pointer in \a svm_pointers that was allocated using SVMAlloc
 *  must have been allocated from the same context from which \a command_queue
 *  was created. The memory associated with \a svm_pointers can be reused or
 *  freed after the function returns.
 *
 *  \param pfn_free_func specifies the callback function to be called to free
 *  the SVM pointers. \a pfn_free_func takes four arguments: \a queue which is
 *  the command queue in which clEnqueueSVMFree was enqueued, the count and list
 *  of SVM pointers to free and \a user_data which is a pointer to user
 *  specified data. If \a pfn_free_func is NULL, all the pointers specified in
 *  \a svm_pointers array must be allocated using clSVMAlloc. \a pfn_free_func
 *  must however be a valid callback function if any SVM pointer to be freed is
 *  a shared system memory pointer i.e. not allocated using clSVMAlloc.
 *
 *  \param user_data will be passed as the user_data argument when
 *  \a pfn_free_func is called. \a user_data can be NULL.
 *
 *  \param even_wait_list specifies the  events that need to complete before
 *  this particular command can be executed. If \a event_wait_list is NULL, then
 *  this particular command does not wait on any event to complete. If
 *  \a event_wait_list is NULL, \a num_events_in_wait_list must be 0. If
 *  \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points. The context associated with events in
 *  \a event_wait_list and \a command_queue must be the same. The memory
 *  associated with \a event_wait_list can be reused or freed after the function
 *  returns.
 *
 *  \param num_events_in_wait_list specifies the number of elements in
 *  \a even_wait_list
 *
 *  \param event returns an event object that identifies this particular command
 *  and can be used to query or queue a wait for this particular command to
 *  complete. \a event can be NULL in which case it will not be possible for the
 *  application to query the status of this command or queue a wait for this
 *  command to complete. If the \a event_wait_list and the \a event arguments
 *  are not NULL, the \a event argument should not refer to an element of the
 *  \a event_wait_list array.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function was executed successfully
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid host
 *    command-queue
 *  - CL_INVALID_VALUE if \a num_svm_pointers is 0 or if \a svm_pointers is
 *    NULL or if any of the pointers specified in \a svm_pointers array is NULL
 *  - CL_INVALID_CONTEXT if context associated with \a command_queue and
 *    events in \a event_wait_list are not the same
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 2.0r15
 */
RUNTIME_ENTRY(cl_int, clEnqueueSVMFree,
              (cl_command_queue command_queue, cl_uint num_svm_pointers, void* svm_pointers[],
               void(CL_CALLBACK* pfn_free_func)(cl_command_queue queue, cl_uint num_svm_pointers,
                                                void* svm_pointers[], void* user_data),
               void* user_data, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
               cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (num_svm_pointers == 0) {
    LogWarning("invalid parameter \"num_svm_pointers = 0\"");
    return CL_INVALID_VALUE;
  }

  if (svm_pointers == NULL) {
    LogWarning("invalid parameter \"svm_pointers = NULL\"");
    return CL_INVALID_VALUE;
  }

  //!@todo why are NULL pointers disallowed here but not in clSVMFree?
  for (cl_uint i = 0; i < num_svm_pointers; i++) {
    if (svm_pointers[i] == NULL) {
      LogWarning("Null pointers are not allowed");
      return CL_INVALID_VALUE;
    }
  }

  //!@todo what if the callback is NULL but \a user_data is not?

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::Command* command = new amd::SvmFreeMemoryCommand(hostQueue, eventWaitList, num_svm_pointers,
                                                        svm_pointers, pfn_free_func, user_data);

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

/*! \brief enqueues a command to do a memcpy operation.
 *
 *  \param command_queue refers to the host command-queue in which the read/
 *  write commands will be queued.
 *
 *  \param blocking_copy indicates if the copy operation is blocking or
 *  non-blocking. If \a blocking_copy is CL_TRUE i.e. the copy command is
 *  blocking, clEnqueueSVMMemcpy does not return until the buffer data has been
 *  copied into memory pointed to by \a dst_ptr. If \a blocking_copy is CL_FALSE
 *  i.e. the copy command is non-blocking, clEnqueueSVMMemcpy queues a
 *  non-blocking copy command and returns. The contents of the buffer that
 *  \a dst_ptr point to cannot be used until the copy command has completed.
 *  The \a event argument returns an event object which can be used to query the
 *  execution status of the read command. When the copy command has completed,
 *  the contents of the buffer that \a dst_ptr points to can be used by the
 *  application.
 *
 *  \param dst_ptr is the pointer to a memory region where data is copied to.
 *
 *  \param src_ptr is the pointer to a memory region where data is copied from.
 *  If \a dst_ptr and/or \a src_ptr are allocated using clSVMAlloc then they
 *  must be allocated from the same context from which \a command_queue was
 *  created. Otherwise the behavior is undefined.
 *
 *  \param size is the size in bytes of data being copied.
 *
 *  \param even_wait_list specifies the  events that need to complete before
 *  this particular command can be executed. If \a event_wait_list is NULL, then
 *  this particular command does not wait on any event to complete. If
 *  \a event_wait_list is NULL, \a num_events_in_wait_list must be 0. If
 *  \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points. The context associated with events in
 *  \a event_wait_list and \a command_queue must be the same. The memory
 *  associated with \a event_wait_list can be reused or freed after the function
 *  returns.
 *
 *  \param num_events_in_wait_list specifies the number of elements in
 *  \a even_wait_list
 *
 *  \param event returns an event object that identifies this particular command
 *  and can be used to query or queue a wait for this particular command to
 *  complete. \a event can be NULL in which case it will not be possible for the
 *  application to query the status of this command or queue a wait for this
 *  command to complete. If the \a event_wait_list and the \a event arguments
 *  are not NULL, the \a event argument should not refer to an element of the
 *  \a event_wait_list array.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function was executed successfully
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid host
 *    command-queue
 *  - CL_INVALID_CONTEXT if the context associated with \a command_queue and
 *    events in \a event_wait_list are not the same
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST if the operation is
 *    blocking and the execution status of any of the events in
 *    \a event_wait_list is a negative integer value.
 *  - CL_INVALID_VALUE if \a dst_ptr or \a src_ptr are NULL.
 *  - CL_INVALID_VALUE if \a size is 0.
 *  - CL_MEM_COPY_OVERLAP if the values specified for \a dst_ptr, \a src_ptr
 *    and \a size result in an overlapping copy.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 2.0r15
 */
RUNTIME_ENTRY(cl_int, clEnqueueSVMMemcpy,
              (cl_command_queue command_queue, cl_bool blocking_copy, void* dst_ptr,
               const void* src_ptr, size_t size, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (dst_ptr == NULL || src_ptr == NULL) {
    return CL_INVALID_VALUE;
  }

  if (size == 0) {
    return CL_INVALID_VALUE;
  }

  char* dst = reinterpret_cast<char*>(dst_ptr);
  const char* src = reinterpret_cast<const char*>(src_ptr);
  if ((dst > src - size) && (dst < src + size)) {
    return CL_MEM_COPY_OVERLAP;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::Command* command =
      new amd::SvmCopyMemoryCommand(hostQueue, eventWaitList, dst_ptr, src_ptr, size);

  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  command->enqueue();

  if (blocking_copy) {
    command->awaitCompletion();
  }

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief enqueues a command to fill a region in memory with a pattern of a
 * given pattern size.
 *
 *  \param command_queue refers to the host command-queue in which the fill
 *  command will be queued. The OpenCL context associated with \a command_queue
 *  and SVM pointer referred to by \a svm_ptr must be the same..
 *
 *  \param svm_ptr is a pointer to a memory region that will be filled with
 *  \a pattern. It must be aligned to \a pattern_size bytes. If \a svm_ptr is
 *  allocated using clSVMAlloc then it must be allocated from the same context
 *  from which \a command_queue was created. Otherwise the behavior is
 *  undefined.
 *
 *  \a pattern is a pointer to the data pattern of size \a pattern_size in
 *  bytes. \a pattern will be used to fill a region in buffer starting at
 *  \a svm_ptr and is \a size bytes in size. The data pattern must be a scalar
 *  or vector integer or floating-point data type supported by OpenCL. For
 *  example, if region pointed to by \a svm_ptr is to be filled with a pattern
 *  of float4 values, then \a pattern will be a pointer to a cl_float4 value
 *  and \a pattern_size will be sizeof(cl_float4). The maximum value of
 *  \a pattern_size is the size of the largest integer or floating-point vector
 *  data type supported by the OpenCL device. The memory associated with
 *  \a pattern can be reused or freed after the function returns.
 *
 *  \param size is the size in bytes of region being filled starting with
 *  \a svm_ptr and must be a multiple of \a pattern_size.
 *
 *  \param even_wait_list specifies the  events that need to complete before
 *  this particular command can be executed. If \a event_wait_list is NULL, then
 *  this particular command does not wait on any event to complete. If
 *  \a event_wait_list is NULL, \a num_events_in_wait_list must be 0. If
 *  \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points. The context associated with events in
 *  \a event_wait_list and \a command_queue must be the same. The memory
 *  associated with \a event_wait_list can be reused or freed after the function
 *  returns.
 *
 *  \param num_events_in_wait_list specifies the number of elements in
 *  \a even_wait_list
 *
 *  \param event returns an event object that identifies this particular command
 *  and can be used to query or queue a wait for this particular command to
 *  complete. \a event can be NULL in which case it will not be possible for the
 *  application to query the status of this command or queue a wait for this
 *  command to complete. clEnqueueBarrierWithWaitList can be used instead. If
 *  the \a event_wait_list and the \a event arguments are not NULL, the \a event
 *  argument should not refer to an element of the \a event_wait_list array.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function was executed successfully
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid host
 *    command-queue
 *  - CL_INVALID_CONTEXT if context associated with \a command_queue and
 *    events in \a event_wait_list are not the same
 *  - CL_INVALID_VALUE if \a svm_ptr is NULL.
 *  - CL_INVALID_VALUE if \a svm_ptr is not aligned to \a pattern_size bytes.
 *  - CL_INVALID_VALUE if \a pattern is NULL or if \a pattern_size is 0 or if
 *    \a pattern_size is not one of {1, 2, 4, 8, 16, 32, 64, 128}.
 *  - CL_INVALID_VALUE if \a size is 0 or is not a multiple of \a pattern_size.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 2.0r15
 */
RUNTIME_ENTRY(cl_int, clEnqueueSVMMemFill,
              (cl_command_queue command_queue, void* svm_ptr, const void* pattern,
               size_t pattern_size, size_t size, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (svm_ptr == NULL) {
    return CL_INVALID_VALUE;
  }

  char* dst = reinterpret_cast<char*>(svm_ptr);
  if (!amd::isMultipleOf(dst, pattern_size)) {
    return CL_INVALID_VALUE;
  }

  if (pattern == NULL) {
    return CL_INVALID_VALUE;
  }

  if (!amd::isPowerOfTwo(pattern_size) || pattern_size == 0 ||
      pattern_size > amd::FillMemoryCommand::MaxFillPatterSize) {
    return CL_INVALID_VALUE;
  }

  if (size == 0 || !amd::isMultipleOf(size, pattern_size)) {
    return CL_INVALID_VALUE;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::Command* command =
      new amd::SvmFillMemoryCommand(hostQueue, eventWaitList, svm_ptr, pattern, pattern_size, size);

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

/*! \brief enqueues a command that will allow the host to update a region of a
 *  SVM buffer
 *
 *  \param command_queue is a valid host command-queue.
 *
 *  \param blocking_map indicates if the map operation is blocking or
 *  non-blocking. If \a blocking_map is CL_TRUE, clEnqueueSVMMap does not return
 *  until the application can access the contents of the SVM region specified by
 *  \a svm_ptr and \a size on the host. If blocking_map is CL_FALSE i.e. map
 *  operation is non-blocking, the region specified by \a svm_ptr and \a size
 *  cannot be used until the map command has completed. The \a event argument
 *  returns an event object which can be used to query the execution status of
 *  the map command. When the map command is completed, the application can
 *  access the contents of the region specified by \a svm_ptr and \a size.
 *
 *  \param maps_flag is a valid cl_map_flags flag.
 *
 *  \param svm_ptr is a pointer to a memory region that will be updated by the
 *   host. If \a svm_ptr is allocated using clSVMAlloc then it must be allocated
 *   from the same context from which \a command_queue was created. Otherwise
 *   the behavior is undefined.
 *
 *  \param size is the size in bytes of the memory region that will be updated
 *  by the host.
 *
 *  \param even_wait_list specifies the  events that need to complete before
 *  this particular command can be executed. If \a event_wait_list is NULL, then
 *  this particular command does not wait on any event to complete. If
 *  \a event_wait_list is NULL, \a num_events_in_wait_list must be 0. If
 *  \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points. The context associated with events in
 *  \a event_wait_list and \a command_queue must be the same. The memory
 *  associated with \a event_wait_list can be reused or freed after the function
 *  returns.
 *
 *  \param num_events_in_wait_list specifies the number of elements in
 *  \a even_wait_list
 *
 *  \param event returns an event object that identifies this particular command
 *  and can be used to query or queue a wait for this particular command to
 *  complete. \a event can be NULL in which case it will not be possible for the
 *  application to query the status of this command or queue a wait for this
 *  command to complete. clEnqueueBarrierWithWaitList can be used instead. If
 *  the \a event_wait_list and the \a event arguments are not NULL, the \a event
 *  argument should not refer to an element of the \a event_wait_list array.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function was executed successfully
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid host
 *    command-queue
 *  - CL_INVALID_CONTEXT if context associated with \a command_queue and
 *    events in \a event_wait_list are not the same
 *  - CL_INVALID_VALUE if \a svm_ptr is NULL.
 *  - CL_INVALID_VALUE if \a size is 0 or if values specified in \a map_flags
 *    are not valid.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST if the operation is
 *    blocking and the execution status of any of the events in
 *    \a event_wait_list is a negative integer value.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 2.0r15
 */
RUNTIME_ENTRY(cl_int, clEnqueueSVMMap,
              (cl_command_queue command_queue, cl_bool blocking_map, cl_map_flags map_flags,
               void* svm_ptr, size_t size, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (svm_ptr == NULL) {
    return CL_INVALID_VALUE;
  }

  if (size == 0) {
    return CL_INVALID_VALUE;
  }

  if (!validateMapFlags(map_flags)) {
    return CL_INVALID_VALUE;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;
  size_t offset = 0;
  amd::Memory* svmMem = NULL;
  if ((queue->device()).isFineGrainedSystem()) {
    // leave blank on purpose for FGS no op
  } else {
    svmMem = amd::MemObjMap::FindMemObj(svm_ptr);
    if (NULL != svmMem) {
      // make sure the context is the same as the context of creation of svm space
      if (hostQueue.context() != svmMem->getContext()) {
        LogWarning("different contexts");
        return CL_INVALID_CONTEXT;
      }

      offset = static_cast<address>(svm_ptr) - static_cast<address>(svmMem->getSvmPtr());
      if (offset < 0 || offset + size > svmMem->getSize()) {
        LogWarning("wrong svm address ");
        return CL_INVALID_VALUE;
      }
      amd::Buffer* srcBuffer = svmMem->asBuffer();

      amd::Coord3D srcSize(size);
      amd::Coord3D srcOffset(offset);
      if (NULL != srcBuffer) {
        if (!srcBuffer->validateRegion(srcOffset, srcSize)) {
          return CL_INVALID_VALUE;
        }
      }

      // Make sure we have memory for the command execution
      device::Memory* mem = svmMem->getDeviceMemory(queue->device());
      if (NULL == mem) {
        LogPrintfError("Can't allocate memory size - 0x%08X bytes!", svmMem->getSize());
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
      }
      // Attempt to allocate the map target now (whether blocking or non-blocking)
      void* mapPtr = mem->allocMapTarget(srcOffset, srcSize, map_flags);
      if (NULL == mapPtr || mapPtr != svm_ptr) {
        return CL_OUT_OF_RESOURCES;
      }
    }
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::Command* command = new amd::SvmMapMemoryCommand(hostQueue, eventWaitList, svmMem, size,
                                                       offset, map_flags, svm_ptr);
  if (command == NULL) {
    return CL_OUT_OF_HOST_MEMORY;
  }
  command->enqueue();

  if (blocking_map) {
    command->awaitCompletion();
  }

  *not_null(event) = as_cl(&command->event());
  if (event == NULL) {
    command->release();
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief enqueues a command to indicate that the host has completed updating
 * a memory region which was specified in a previous call to clEnqueueSVMUnmap.
 *
 *  \param command_queue is a valid host command-queue.
 *
 *  \param svm_ptr is a pointer that was specified in a previous call to
 *  clEnqueueSVMMap. If \a svm_ptr is allocated using clSVMAlloc then it must be
 *  allocated from the same context from which \a command_queue was created.
 *  Otherwise the behavior is undefined.
 *
 *  \param even_wait_list specifies the events that need to complete before
 *  this particular command can be executed. If \a event_wait_list is NULL, then
 *  this particular command does not wait on any event to complete. If
 *  \a event_wait_list is NULL, \a num_events_in_wait_list must be 0. If
 *  \a event_wait_list is not NULL, the list of events pointed to by
 *  \a event_wait_list must be valid and \a num_events_in_wait_list must be
 *  greater than 0. The events specified in \a event_wait_list act as
 *  synchronization points. The context associated with events in
 *  \a event_wait_list and \a command_queue must be the same. The memory
 *  associated with \a event_wait_list can be reused or freed after the function
 *  returns.
 *
 *  \param num_events_in_wait_list specifies the number of elements in
 *  \a even_wait_list
 *
 *  \param event returns an event object that identifies this particular command
 *  and can be used to query or queue a wait for this particular command to
 *  complete. \a event can be NULL in which case it will not be possible for the
 *  application to query the status of this command or queue a wait for this
 *  command to complete. clEnqueueBarrierWithWaitList can be used instead. If
 *  the \a event_wait_list and the \a event arguments are not NULL, the \a event
 *  argument should not refer to an element of the \a event_wait_list array.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function was executed successfully
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid host
 *    command-queue
 *  - CL_INVALID_CONTEXT if context associated with \a command_queue and
 *    events in \a event_wait_list are not the same
 *  - CL_INVALID_VALUE if \a svm_ptr is NULL.
 *  - CL_INVALID_EVENT_WAIT_LIST if \a event_wait_list is NULL and
 *    \a num_events_in_wait_list > 0, or \a event_wait_list is not NULL and
 *    \a num_events_in_wait_list is 0, or if event objects in \a event_wait_list
 *    are not valid events.
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 2.0r15
 */
RUNTIME_ENTRY(cl_int, clEnqueueSVMUnmap,
              (cl_command_queue command_queue, void* svm_ptr, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  if (svm_ptr == NULL) {
    return CL_INVALID_VALUE;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;
  amd::Memory* svmMem = NULL;
  if (!(queue->device()).isFineGrainedSystem()) {
    // check if the ptr is in the svm space
    svmMem = amd::MemObjMap::FindMemObj(svm_ptr);
    // Make sure we have memory for the command execution
    if (NULL != svmMem) {
      // Make sure we have memory for the command execution
      device::Memory* mem = svmMem->getDeviceMemory(queue->device());
      if (NULL == mem) {
        LogPrintfError("Can't allocate memory size - 0x%08X bytes!", svmMem->getSize());
        return CL_INVALID_VALUE;
      }
    }
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::Command* command = new amd::SvmUnmapMemoryCommand(hostQueue, eventWaitList, svmMem, svm_ptr);
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

/*! \brief Set the argument value for a specific argument of a kernel to be
 *  a SVM pointer.
 *
 *  \param kernel is a valid kernel object.
 *
 *  \param arg_index is the argument index. Arguments to the kernel are referred
 *  by indices that go from 0 for the leftmost argument to n - 1, where n is the
 *  total number of arguments declared by a kernel.
 *
 *  \param arg_value is the SVM pointer that should be used as the argument
 *  value for argument specified by \a arg_index. The SVM pointer pointed to by
 *  \a arg_value is copied and the \a arg_value pointer can therefore be reused
 *  by the application after clSetKernelArgSVMPointer returns. The SVM pointer
 *  specified is the value used by all API calls that enqueue kernel
 *  (clEnqueueNDRangeKernel) until the argument value is changed by a call to
 *  clSetKernelArgSVMPointer for \a kernel. The SVM pointer can only be used for
 *  arguments that are declared to be a pointer to global or constant memory.
 *  The SVM pointer value must be aligned according to the argument?s type. For
 *  example, if the argument is declared to be global float4 *p, the SVM pointer
 *  value passed for p must be at a minimum aligned to a float4. The SVM pointer
 *  value specified as the argument value can be the pointer returned by
 *  clSVMAlloc or can be a pointer + offset into the SVM region.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function was executed successfully
 *  - CL_INVALID_KERNEL if \a kernel is not a valid kernel object
 *  - CL_INVALID_ARG_INDEX if \a arg_index is not a valid argument index
 *  - CL_INVALID_ARG_VALUE if \a arg_value is not a valid value
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 2.0r15
 */
RUNTIME_ENTRY(cl_int, clSetKernelArgSVMPointer,
              (cl_kernel kernel, cl_uint arg_index, const void* arg_value)) {
  if (!is_valid(kernel)) {
    return CL_INVALID_KERNEL;
  }

  const amd::KernelSignature& signature = as_amd(kernel)->signature();
  if (arg_index >= signature.numParameters()) {
    return CL_INVALID_ARG_INDEX;
  }

  const amd::KernelParameterDescriptor& desc = signature.at(arg_index);
  if (desc.type_ != T_POINTER ||
      !(desc.addressQualifier_ & (CL_KERNEL_ARG_ADDRESS_GLOBAL | CL_KERNEL_ARG_ADDRESS_CONSTANT))) {
    as_amd(kernel)->parameters().reset(static_cast<size_t>(arg_index));
    return CL_INVALID_ARG_VALUE;
  }

  //! @todo We need to check that the alignment of \a arg_value. For instance,
  // if the argument is of type 'global float4*', then \a arg_value must be
  // aligned to sizeof(float4*). Note that desc.size_ contains the size of the
  // pointer type itself and the size of the pointed type.


  // We do not perform additional pointer validations:
  // -verifying pointers returned by SVMAlloc would imply keeping track
  //  of every allocation range and then matching the pointer against that
  //  range. Note that even if the pointer would look correct, nothing
  //  prevents the user from using an offset within the kernel that would
  //  result on an invalid access.
  // -verifying system pointers (if supported) requires matching the pointer
  //  against the address space of the current process.

  as_amd(kernel)->parameters().set(static_cast<size_t>(arg_index), sizeof(arg_value), &arg_value,
                                   true);
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Pass additional information other than argument values to a kernel.
 *
 *  \param kernel is a valid kernel object.
 *
 *  \param param_name specifies the information to be passed to \a kernel. It
 *  must be a cl_kernel_exec_info value.
 *
 *  \param param_value_size specifies the size in bytes of the memory pointed to
 *  by \a param_value.
 *
 *  \param param_value is a pointer to memory where the appropiate values
 *  determined by \a param_name are specified.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function was executed successfully
 *  - CL_INVALID_KERNEL if \a kernel is not a valid kernel object.
 *  - CL_INVALID_VALUE if \a param_name is not valid, if  \a param_value is
 *    NULL or if the size specified by \a param_value_size is not valid
 *  - CL_INVALID_OPERATION if \a param_name is
 *    CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM and \a param_value = CL_TRUE
 *    but no devices in context associated with \a kernel support fine-grained
 *    system SVM allocations
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 2.0r15
 */
RUNTIME_ENTRY(cl_int, clSetKernelExecInfo,
              (cl_kernel kernel, cl_kernel_exec_info param_name, size_t param_value_size,
               const void* param_value)) {
  if (!is_valid(kernel)) {
    return CL_INVALID_KERNEL;
  }

  if (param_value == NULL) {
    return CL_INVALID_VALUE;
  }

  const amd::Kernel* amdKernel = as_amd(kernel);

  switch (param_name) {
    case CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM:
      if (param_value_size != sizeof(cl_bool)) {
        return CL_INVALID_VALUE;
      } else {
        const bool flag = *(static_cast<const bool*>(param_value));
        const amd::Context* amdContext = &amdKernel->program().context();
        bool foundFineGrainedSystemDevice = false;
        const std::vector<amd::Device*>& devices = amdContext->devices();
        for (const auto it : devices) {
          if (it->info().svmCapabilities_ & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) {
            foundFineGrainedSystemDevice = true;
            break;
          }
        }
        if (flag && !foundFineGrainedSystemDevice) {
          return CL_INVALID_OPERATION;
        }
        amdKernel->parameters().setSvmSystemPointersSupport(flag ? FGS_YES : FGS_NO);
      }
      break;
    case CL_KERNEL_EXEC_INFO_SVM_PTRS:
      if (param_value_size == 0 || !amd::isMultipleOf(param_value_size, sizeof(void*))) {
        return CL_INVALID_VALUE;
      } else {
        size_t count = param_value_size / sizeof(void*);
        void* const* execInfoArray = reinterpret_cast<void* const*>(param_value);
        for (size_t i = 0; i < count; i++) {
          if (NULL == execInfoArray[i]) {
            return CL_INVALID_VALUE;
          }
        }
        amdKernel->parameters().addSvmPtr(execInfoArray, count);
      }
      break;
    case CL_KERNEL_EXEC_INFO_NEW_VCOP_AMD:
      if (param_value_size != sizeof(cl_bool)) {
        return CL_INVALID_VALUE;
      } else {
        const bool newVcopFlag = (*(reinterpret_cast<const cl_bool*>(param_value))) ? true : false;
        amdKernel->parameters().setExecNewVcop(newVcopFlag);
      }
      break;
    case CL_KERNEL_EXEC_INFO_PFPA_VCOP_AMD:
      if (param_value_size != sizeof(cl_bool)) {
        return CL_INVALID_VALUE;
      } else {
        const bool pfpaVcopFlag = (*(reinterpret_cast<const cl_bool*>(param_value))) ? true : false;
        amdKernel->parameters().setExecPfpaVcop(pfpaVcopFlag);
      }
      break;
    default:
      return CL_INVALID_VALUE;
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Enqueues a command to indicate which device a set of ranges of SVM
 *  allocations should be associated with. Once the event returned by
 *  \a clEnqueueSVMMigrateMem has become CL_COMPLETE, the ranges specified by
 *  svm pointers and sizes have been successfully migrated to the device
 *  associated with command queue.
 *  The user is responsible for managing the event dependencies associated with
 *  this command in order to avoid overlapping access to SVM allocations.
 *  Improperly specified event dependencies passed to clEnqueueSVMMigrateMem
 *  could result in undefined results
 *
 *  \param command_queue is a valid host command queue. The specified set of
 *  allocation ranges will be migrated to the OpenCL device associated with
 *  command_queue.
 *
 *  \param num_svm_pointers is the number of pointers in the specified
 *  svm_pointers array, and the number of sizes in the sizes array, if sizes
 *  is not NULL.
 *
 *  \param svm_pointers is a pointer to an array of pointers. Each pointer in
 *  this array must be within an allocation produced by a call to clSVMAlloc.
 *
 *  \param sizes is an array of sizes. The pair svm_pointers[i] and sizes[i]
 *  together define the starting address and number of bytes in a range to be
 *  migrated. sizes may be NULL indicating that every allocation containing
 *  any svm_pointer[i] is to be migrated. Also, if sizes[i] is zero, then the
 *  entire allocation containing svm_pointer[i] is migrated.
 *
 *  \param flags is a bit-field that is used to specify migration options.
 *  Table 5.12 describes the possible values for flags.
 *
 *  \param num_events_in_wait_list specifies the number of event objects in
 *  \a event_wait_list.
 *
 *  \param event_wait_list specifies events that need to complete before this
 *  particular command can be executed. If event_wait_list is NULL, then this
 *  particular command does not wait on any event to complete. If
 *  event_wait_list is NULL, num_events_in_wait_list must be 0. If
 *  event_wait_list is not NULL, the list of events pointed to by
 *  event_wait_list must be valid and num_events_in_wait_list must be greater
 *  than 0. The events specified in event_wait_list act as synchronization
 *  points. The context associated with events in event_wait_list and
 *  command_queue must be the same. The memory associated with
 *  event_wait_list can be reused or freed after the function returns.
 *
 *  \param event an returned event object that identifies this particular write
 *  command and can be used to query or queue a wait for this particular
 *  command to complete. event can be NULL in which case it will not be
 *  possible for the application to query the status of this command or queue
 *  another command that waits for this command to complete. If the
 *  event_wait_list and the event arguments are not NULL, the event argument
 *  should not refer to an element of the event_wait_list array.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-queue
 *  - CL_INVALID_VALUE if num_svm_pointers is zero or svm_pointers is NULL
 *  - CL_INVALID_VALUE if sizes[i] is non-zero range [svm_pointers[i],
 *    svm_pointers[i]+sizes[i]) is not contained within an existing clSVMAlloc
 *    allocation
 *  - CL_INVALID_EVENT_WAIT_LIST if event_wait_list is NULL and
 *    num_events_in_wait_list > 0, or event_wait_list is not NULL and
 *    num_events_in_wait_list is 0, or if event objects in event_wait_list are
 *    not valid events
 *  - CL_OUT_OF_RESOURCES if there is a failure to allocate resources required
 *    by the OpenCL implementation on the device.
 *  - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required
 *    by the OpenCL implementation on the host.
 *
 *  \version 2.1r00
 */
RUNTIME_ENTRY(cl_int, clEnqueueSVMMigrateMem,
              (cl_command_queue command_queue, cl_uint num_svm_pointers, const void** svm_pointers,
               const size_t* size, cl_mem_migration_flags flags, cl_uint num_events_in_wait_list,
               const cl_event* event_wait_list, cl_event* event)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::HostQueue* queue = as_amd(command_queue)->asHostQueue();
  if (NULL == queue) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  amd::HostQueue& hostQueue = *queue;

  if (num_svm_pointers == 0) {
    LogWarning("invalid parameter \"num_svm_pointers = 0\"");
    return CL_INVALID_VALUE;
  }

  if (svm_pointers == NULL) {
    LogWarning("invalid parameter \"svm_pointers = NULL\"");
    return CL_INVALID_VALUE;
  }

  for (cl_uint i = 0; i < num_svm_pointers; i++) {
    if (svm_pointers[i] == NULL) {
      LogWarning("Null pointers are not allowed");
      return CL_INVALID_VALUE;
    }
  }

  if (flags & ~(CL_MIGRATE_MEM_OBJECT_HOST | CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED)) {
    LogWarning("Invalid flag is specified");
    return CL_INVALID_VALUE;
  }

  std::vector<amd::Memory*> memObjects;
  for (cl_uint i = 0; i < num_svm_pointers; i++) {
    const void* svm_ptr = svm_pointers[i];

    amd::Memory* svmMem = amd::MemObjMap::FindMemObj(svm_ptr);
    if (NULL != svmMem) {
      // make sure the context is the same as the context of creation of svm space
      if (hostQueue.context() != svmMem->getContext()) {
        LogWarning("different contexts");
        return CL_INVALID_CONTEXT;
      }

      // Make sure the specified size[i] is within a valid range
      // TODO: handle the size parameter properly
      size_t svm_size = (size == NULL) ? 0 : size[i];
      size_t offset =
          reinterpret_cast<const_address>(svm_ptr) - reinterpret_cast<address>(svmMem->getSvmPtr());
      if ((offset + svm_size) > svmMem->getSize()) {
        LogWarning("wrong svm address ");
        return CL_INVALID_VALUE;
      }

      memObjects.push_back(svmMem);
    }
  }

  amd::Command::EventWaitList eventWaitList;
  cl_int err =
      amd::clSetEventWaitList(eventWaitList, hostQueue, num_events_in_wait_list, event_wait_list);
  if (err != CL_SUCCESS) {
    return err;
  }

  amd::MigrateMemObjectsCommand* command = new amd::MigrateMemObjectsCommand(
      hostQueue, CL_COMMAND_MIGRATE_MEM_OBJECTS, eventWaitList, memObjects, flags);

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
/*! @}
 *  @}
 */
