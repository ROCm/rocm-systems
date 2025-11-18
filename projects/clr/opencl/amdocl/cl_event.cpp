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

#include "platform/object.hpp"
#include "platform/context.hpp"
#include "platform/command.hpp"

/*! \addtogroup API
 *  @{
 * \addtogroup CL_Events
 *
 *  Event objects can be used to refer to a kernel execution command:
 *    - clEnqueueNDRangeKernel
 *    - clEnqueueTask
 *    - clEnqueueNativeKernel
 *
 *  or read, write, map and copy commands on memory objects:
 *    - clEnqueue{Read|Write|Map}{Buffer|Image}
 *    - clEnqueueCopy{Buffer|Image}
 *    - clEnqueueCopyBufferToImage
 *    - clEnqueueCopyImageToBuffer
 *
 *  An event object can be used to track the execution status of a command.
 *  The execution status of a command at any given point in time can be
 *  CL_QUEUED (is currently in the command queue),
 *  CL_RUNNING (device is currently executing this command),
 *  CL_COMPLETE (command has successfully completed) or the appropriate error
 *  code if the command was abnormally terminated (this may be caused by a bad
 *  memory access etc.). The error code returned by a terminated command is
 *  a negative integer value. A command is considered to be complete if its
 *  execution status is CL_COMPLETE or is a negative integer value.
 *
 *  If the execution of a command is terminated, the command-queue associated
 *  with this terminated command, and the associated context (and all other
 *  command-queues in this context) may no longer be available. The behavior of
 *  OpenCL API calls that use this context (and command-queues associated with
 *  this context) are now considered to be implementationdefined. The user
 *  registered callback function specified when context is created can be used
 *  to report appropriate error information.
 *
 *  @{
 */


/*! \brief Wait on the host thread for commands identified by event objects in
 *  event_list to complete.
 *
 *  A command is considered complete if its execution status is CL_COMPLETE or
 *  a negative value. The events specified in event_list act as synchronization
 *  points.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function was executed successfully.
 *  - CL_INVALID_VALUE if \a num_events is zero
 *  - CL_INVALID_CONTEXT if events specified in \a event_list do not belong to
 *    the same context
 *  - CL_INVALID_EVENT if event objects specified in \a event_list are not valid
 *    event objects.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clWaitForEvents, (cl_uint num_events, const cl_event* event_list)) {
  if (num_events == 0 || event_list == NULL) {
    return CL_INVALID_VALUE;
  }

  const amd::Context* prevContext = NULL;
  const amd::HostQueue* prevQueue = NULL;

  for (cl_uint i = 0; i < num_events; ++i) {
    cl_event event = event_list[i];

    if (!is_valid(event)) {
      return CL_INVALID_EVENT;
    }

    // Make sure all the events are associated with the same context
    amd::Event* amdEvent = as_amd(event);
    if (amdEvent->status() == CL_COMPLETE) continue;

    const amd::Context* context = &amdEvent->context();
    if (prevContext != NULL && prevContext != context) {
      return CL_INVALID_CONTEXT;
    }
    prevContext = context;

    // Flush the command queues associated with event1...eventN
    amd::HostQueue* queue = amdEvent->command().queue();
    if (queue != NULL && prevQueue != queue) {
      queue->flush();
    }
    prevQueue = queue;
  }

  bool allSucceeded = true;
  while (num_events-- > 0) {
    allSucceeded &= as_amd(*event_list++)->awaitCompletion();
  }
  return allSucceeded ? CL_SUCCESS : CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
}
RUNTIME_EXIT

/*! \brief Return information about the event object.
 *
 *  \param event specifies the event object being queried.
 *
 *  \param param_name specifies the information to query.
 *
 *  \param param_value is a pointer to memory where the appropriate result being
 *  queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data copied
 *  to \a param_value. If \a param_value_size_ret is NULL, it is ignored.
 *
 *  Using clGetEventInfo to determine if a command identified by event has
 *  finished execution (i.e. CL_EVENT_COMMAND_EXECUTION_STATUS returns
 *  CL_COMPLETE) is not a synchronization point i.e. there are no guarantees
 *  that the memory objects being modified by command associated with event will
 *  be visible to other enqueued commands.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the function is executed successfully
 *  - CL_INVALID_VALUE if \a param_name is not valid, or if size in bytes
 *    specified by \a param_value_size is < size of return type and
 *    \a param_value is not NULL
 *  - CL_INVALID_EVENT if \a event is a not a valid event object.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clGetEventInfo,
              (cl_event event, cl_event_info param_name, size_t param_value_size, void* param_value,
               size_t* param_value_size_ret)) {
  if (!is_valid(event)) {
    return CL_INVALID_EVENT;
  }

  switch (param_name) {
    case CL_EVENT_CONTEXT: {
      amd::Context& amdCtx = const_cast<amd::Context&>(as_amd(event)->context());
      cl_context context = as_cl(&amdCtx);
      return amd::clGetInfo(context, param_value_size, param_value, param_value_size_ret);
    }
    case CL_EVENT_COMMAND_QUEUE: {
      amd::Command& command = as_amd(event)->command();
      cl_command_queue queue =
          command.queue() == NULL
              ? NULL
              : const_cast<cl_command_queue>(as_cl(command.queue()->asCommandQueue()));
      return amd::clGetInfo(queue, param_value_size, param_value, param_value_size_ret);
    }
    case CL_EVENT_COMMAND_TYPE: {
      cl_command_type type = as_amd(event)->command().type();
      return amd::clGetInfo(type, param_value_size, param_value, param_value_size_ret);
    }
    case CL_EVENT_COMMAND_EXECUTION_STATUS: {
      as_amd(event)->notifyCmdQueue();
      cl_int status = as_amd(event)->command().status();
      return amd::clGetInfo(status, param_value_size, param_value, param_value_size_ret);
    }
    case CL_EVENT_REFERENCE_COUNT: {
      cl_uint count = as_amd(event)->referenceCount();
      return amd::clGetInfo(count, param_value_size, param_value, param_value_size_ret);
    }
    default:
      break;
  }

  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! \brief Increment the event reference count.
 *
 *  \return CL_SUCCESS if the function is executed successfully. It returns
 *  CL_INVALID_EVENT if \a event is not a valid event object.
 *
 *  The OpenCL commands that return an event perform an implicit retain.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clRetainEvent, (cl_event event)) {
  if (!is_valid(event)) {
    return CL_INVALID_EVENT;
  }
  as_amd(event)->retain();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Decrement the event reference count.
 *
 *  \return CL_SUCCESS if the function is executed successfully. It returns
 *  CL_INVALID_EVENT if \a event is not a valid event object.
 *
 *  The event object is deleted once the reference count becomes zero, the
 *  specific command identified by this event has completed (or terminated) and
 *  there are no commands in the command-queues of a context that require a wait
 *  for this event to complete.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clReleaseEvent, (cl_event event)) {
  if (!is_valid(event)) {
    return CL_INVALID_EVENT;
  }
  as_amd(event)->release();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Creates a user event object.
 *
 * User events allow applications to enqueue commands that wait on a user event
 * to finish before the command is executed by the device.
 *
 * \return a valid non-zero event object and errcode_ret is set to CL_SUCCESS
 * if the user event object is created successfully. Otherwise, it returns
 * a NULL value with one of the following error values returned in errcode_ret:
 *   - CL_INVALID_CONTEXT if context is not a valid context.
 *   - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *     required by the OpenCL implementation on the host.
 *
 * The execution status of the user event object created is set to CL_SUBMITTED.
 *
 * \version 1.1r15
 */
RUNTIME_ENTRY_RET(cl_event, clCreateUserEvent, (cl_context context, cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    return (cl_event)0;
  }

  auto event = new amd::UserEvent(*as_amd(context));
  if (event == nullptr || !event->Create()) {
    delete event;
    *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
    return (cl_event)0;
  }

  event->retain();
  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(reinterpret_cast<amd::Event*>(event));
}
RUNTIME_EXIT

/*! \brief Sets the execution status of a user event object.
 *
 * \a event is a user event object created using clCreateUserEvent.
 * \a execution_status specifies the new execution status to be set and can be
 * CL_COMPLETE or a negative integer value to indicate an error.
 * clSetUserEventStatus can only be called once to change the execution status
 * of event.
 *
 * \return CL_SUCCESS if the function was executed successfully. Otherwise,
 * it returns one of the following errors:
 *   - CL_INVALID_EVENT if event is not a valid user event object.
 *   - CL_INVALID_VALUE if the execution_status is not CL_COMPLETE or
 *     a negative integer value.
 *   - CL_INVALID_OPERATION if the execution_status for event has already been
 *     changed by a previous call to clSetUserEventStatus.
 *
 * \version 1.1r15
 */
RUNTIME_ENTRY(cl_int, clSetUserEventStatus, (cl_event event, cl_int execution_status)) {
  if (!is_valid(event)) {
    return CL_INVALID_EVENT;
  }
  if (execution_status > CL_COMPLETE) {
    return CL_INVALID_VALUE;
  }
  auto user_event = reinterpret_cast<amd::UserEvent*>(as_amd(event));
  if (!user_event->SetExecutionStatus(execution_status)) {
    return CL_INVALID_OPERATION;
  }
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Registers a user callback function for a specific command execution
 *  status.
 *
 * The registered callback function will be called when the execution status
 * of command associated with event changes to the execution status specified
 * by command_exec_status.
 *
 * Each call to clSetEventCallback registers the specified user callback
 * function on a callback stack associated with event. The order in which the
 * registered user callback functions are called is undefined.
 *
 * \a event is a valid event object.
 * \a command_exec_callback_type specifies the command execution status for
 *    which the callback is registered.    The command execution callback mask
 *    values for which a callback can be registered are: CL_COMPLETE.
 *    There is no guarantee that the callback functions registered for various
 *    execution status values for an event will be called in the exact order
 *    that the execution status of a command changes.
 * \a pfn_event_notify is the event callback function that can be registered
 *    by the application. This callback function may be called asynchronously
 *    by the OpenCL implementation. It is the application’s responsibility to
 *    ensure that the callback function is thread-safe. The parameters to this
 *    callback function are:
 *        event is the event object for which the callback function is invoked.
 *        event_command_exec_status represents the execution status of command
 *        for which this callback function is invoked. If the callback is called
 *        as the result of the command associated with event being abnormally
 *        terminated, an appropriate error code for the error that caused the
 *        termination will be passed to event_command_exec_status instead.
 * \a user_data is a pointer to user supplied data. user_data will be passed as
 *    the user_data argument when pfn_notify is called. user_data can be NULL.
 *
 * All callbacks registered for an event object must be called. All enqueued
 * callbacks shall be called before the event object is destroyed. Callbacks
 * must return promptly. The behavior of calling expensive system routines,
 * OpenCL API calls to create contexts or command-queues, or blocking OpenCL
 * operations from the following list below, in a callback is undefined.
 *     clFinish, clWaitForEvents, blocking calls to clEnqueueReadBuffer,
 *     clEnqueueReadBufferRect, clEnqueueWriteBuffer, clEnqueueWriteBufferRect,
 *     blocking calls to clEnqueueReadImage and clEnqueueWriteImage, blocking
 *     calls to clEnqueueMapBuffer and clEnqueueMapImage, blocking calls to
 *     clBuildProgram
 *
 * If an application needs to wait for completion of a routine from the above
 * list in a callback, please use the non-blocking form of the function, and
 * assign a completion callback to it to do the remainder of your work.
 * Note that when a callback (or other code) enqueues commands to a
 * command-queue, the commands are not required to begin execution until the
 * queue is flushed. In standard usage, blocking enqueue calls serve this role
 * by implicitly flushing the queue. Since blocking calls are not permitted in
 * callbacks, those callbacks that enqueue commands on a command queue should
 * either call clFlush on the queue before returning or arrange for clFlush
 * to be called later on another thread.
 *
 * \return CL_SUCCESS if the function is executed successfully. Otherwise,
 * it returns one of the following errors:
 *   - CL_INVALID_EVENT if event is not a valid event object or is a user event
 *     object created using clCreateUserEvent.
 *   - CL_INVALID_VALUE if pfn_event_notify is NULL or if
 *     command_exec_callback_type is not a valid command execution status.
 *   - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *     required by the OpenCL implementation on the host.
 *
 * \version 1.1r15
 */
RUNTIME_ENTRY(cl_int, clSetEventCallback,
              (cl_event event, cl_int command_exec_callback_type,
               void(CL_CALLBACK* pfn_notify)(cl_event event, cl_int command_exec_status,
                                             void* user_data),
               void* user_data)) {
  if (!is_valid(event)) {
    return CL_INVALID_EVENT;
  }

  if (pfn_notify == NULL || command_exec_callback_type < CL_COMPLETE ||
      command_exec_callback_type > CL_QUEUED) {
    return CL_INVALID_VALUE;
  }

  if (!as_amd(event)->setCallback(command_exec_callback_type, pfn_notify, user_data)) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  as_amd(event)->notifyCmdQueue();

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! @}
 *  @}
 */
