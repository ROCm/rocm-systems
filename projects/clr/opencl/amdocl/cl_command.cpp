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
#include "platform/agent.hpp"

/*! \addtogroup API
 *  @{
 *
 *  \addtogroup CL_Queues
 *
 *  OpenCL objects such as memory objects, program and kernel objects are
 *  created using a context. Operations on these objects are performed using
 *  a command-queue. The command-queue can be used to queue a set of operations
 *  (referred to as commands) in order. Having multiple command-queues allows
 *  applications to queue multiple independent commands without requiring
 *  synchronization. Note that this should work as long as these objects are
 *  not being shared. Sharing of objects across multiple command-queues will
 *  require the application to perform appropriate synchronization.
 *
 *  @{
 */

/*! \brief Create a command-queue on a specific device.
 *
 *  \param context must be a valid OpenCL context.
 *
 *  \param device must be a device associated with context. It can either be
 *  in the list of devices specified when context is created using
 *  clCreateContext or have the same device type as device type specified wheni
 *  context is created using clCreateContextFromType.
 *
 *  \param properties specifies a list of properties for the command-queue.
 *
 *  \param errcode_ret will return an appropriate error code. If \a errcode_ret
 *  is NULL, no error code is returned.
 *
 *  \return A valid non-zero command-queue and \a errcode_ret is set to
 *  CL_SUCCESS if the command-queue is created successfully or a NULL value
 *  with one of the following error values returned \a in errcode_ret:
 *    - CL_INVALID_CONTEXT if context is not a valid.
 *    - CL_INVALID_DEVICE if device is not a valid device or is not associated
 *      with context
 *    - CL_INVALID_VALUE if values specified in properties are not valid.
 *    - CL_INVALID_QUEUE_PROPERTIES if values specified in properties are valid
 *      but are not supported by the device.
 *    - CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources
 *      required by the runtime.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY_RET(cl_command_queue, clCreateCommandQueueWithProperties,
                  (cl_context context, cl_device_id device,
                   const cl_queue_properties* queue_properties, cl_int* errcode_ret)) {
  if (!is_valid(context)) {
    *not_null(errcode_ret) = CL_INVALID_CONTEXT;
    return (cl_command_queue)0;
  }

  amd::Context& amdContext = *as_amd(context);
  amd::Device& amdDevice = *as_amd(device);

  if (!is_valid(device) || !amdContext.containsDevice(&amdDevice)) {
    *not_null(errcode_ret) = CL_INVALID_DEVICE;
    return (cl_command_queue)0;
  }

  cl_command_queue_properties properties = 0;
  const struct QueueProperty {
    cl_queue_properties name;
    union {
      cl_queue_properties raw;
      // FIXME_lmoriche: Check with Khronos. cl_queue_properties is an intptr,
      // but cl_command_queue_properties is a bitfield (truncate?).
      // cl_command_queue_properties properties;
      cl_uint size;
    } value;
  }* p = reinterpret_cast<const QueueProperty*>(queue_properties);

  uint queueSize = amdDevice.info().queueOnDevicePreferredSize_;
  uint queueRTCUs = amd::CommandQueue::RealTimeDisabled;
  amd::CommandQueue::Priority priority = amd::CommandQueue::Priority::Normal;
  if (p != NULL)
    while (p->name != 0) {
      switch (p->name) {
        case CL_QUEUE_PROPERTIES:
          // FIXME_lmoriche: See comment above.
          // properties = p->value.properties;
          properties = static_cast<cl_command_queue_properties>(p->value.raw);
          if (properties == CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
            *not_null(errcode_ret) = CL_INVALID_QUEUE_PROPERTIES;
            return (cl_command_queue)0;
          }
          break;
        case CL_QUEUE_SIZE:
          queueSize = p->value.size;
          break;
#define CL_QUEUE_REAL_TIME_COMPUTE_UNITS_AMD 0x404f
        case CL_QUEUE_REAL_TIME_COMPUTE_UNITS_AMD:
          queueRTCUs = p->value.size;
          break;
#define CL_QUEUE_MEDIUM_PRIORITY_AMD 0x4050
        case CL_QUEUE_MEDIUM_PRIORITY_AMD:
          priority = amd::CommandQueue::Priority::Medium;
          if (p->value.size != 0) {
            queueRTCUs = p->value.size;
          }
          break;
        default:
          *not_null(errcode_ret) = CL_INVALID_QUEUE_PROPERTIES;
          LogWarning("invalid property name");
          return (cl_command_queue)0;
      }
      ++p;
    }

  if (queueSize > amdDevice.info().queueOnDeviceMaxSize_) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return (cl_command_queue)0;
  }

  if ((queueRTCUs != amd::CommandQueue::RealTimeDisabled) &&
      ((queueRTCUs > amdDevice.info().numRTCUs_) || (queueRTCUs == 0) ||
       (queueRTCUs < amdDevice.info().granularityRTCUs_))) {
    *not_null(errcode_ret) = CL_INVALID_VALUE;
    return (cl_command_queue)0;
  }

  amd::CommandQueue* queue = NULL;
  {
    amd::ScopedLock lock(amdContext.lock());

    // Check if the app creates a host queue
    if (!(properties & CL_QUEUE_ON_DEVICE)) {
      queue = new amd::HostQueue(amdContext, amdDevice, properties, queueRTCUs, priority);
    } else {
      // Is it a device default queue
      if (properties & CL_QUEUE_ON_DEVICE_DEFAULT) {
        queue = amdContext.defDeviceQueue(amdDevice);
        // If current context has one already then return it
        if (NULL != queue) {
          queue->retain();
          *not_null(errcode_ret) = CL_SUCCESS;
          return as_cl(queue);
        }
      }
      // Check if runtime can allocate a new device queue on this context
      if (amdContext.isDevQueuePossible(amdDevice)) {
        queue = new amd::DeviceQueue(amdContext, amdDevice, properties, queueSize);
      }
    }

    if (queue == NULL || !queue->create()) {
      *not_null(errcode_ret) = CL_OUT_OF_HOST_MEMORY;
      delete queue;
      return (cl_command_queue)0;
    }
  }

  if (amd::Agent::shouldPostCommandQueueEvents()) {
    amd::Agent::postCommandQueueCreate(as_cl(queue->asCommandQueue()));
  }

  *not_null(errcode_ret) = CL_SUCCESS;
  return as_cl(queue);
}
RUNTIME_EXIT

RUNTIME_ENTRY_RET(cl_command_queue, clCreateCommandQueue,
                  (cl_context context, cl_device_id device, cl_command_queue_properties properties,
                   cl_int* errcode_ret)) {
  const cl_queue_properties cprops[] = {CL_QUEUE_PROPERTIES,
                                        static_cast<cl_queue_properties>(properties), 0};
  return clCreateCommandQueueWithProperties(context, device, properties ? cprops : NULL,
                                            errcode_ret);
}
RUNTIME_EXIT

/*! \brief Replaces the default command queue on the device
 *
 *  \param context must be a valid OpenCL context.
 *
 *  \param device must be a device associated with context.
 *
 *  \param command_queue specifies the default command-queue.
 *
 *  \reture One of the following values:
 *    - CL_SUCCESS if the function executed successfully.
 *    - CL_INVALID_CONTEXT if \a context is not a valid context.
 *    - CL_INVALID_DEVICE if \a device is not a valid device or is not
 *      associated with context.
 *    - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid command-
 *      queue for device.
 */
RUNTIME_ENTRY(cl_int, clSetDefaultDeviceCommandQueue,
              (cl_context context, cl_device_id device, cl_command_queue command_queue)) {
  if (!is_valid(context)) {
    return CL_INVALID_CONTEXT;
  }

  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  amd::Context* amdContext = as_amd(context);
  amd::Device* amdDevice = as_amd(device);
  if (!is_valid(device) || !amdContext->containsDevice(amdDevice)) {
    return CL_INVALID_DEVICE;
  }

  amd::DeviceQueue* deviceQueue = as_amd(command_queue)->asDeviceQueue();
  if ((deviceQueue == NULL) || (amdContext != &deviceQueue->context()) ||
      (amdDevice != &deviceQueue->device())) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  {
    amd::ScopedLock lock(amdContext->lock());
    amdContext->setDefDeviceQueue(*amdDevice, deviceQueue);
  }

  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Increment the \a command_queue reference count.
 *
 *  \return One of the following values:
 *    - CL_SUCCESS if the function is executed successfully.
 *    - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid
 *      command-queue.
 *
 *  clCreateCommandQueue performs an implicit retain. This is very helpful for
 *  3rd party libraries, which typically get a command-queue passed to them
 *  by the application. However, it is possible that the application may delete
 *  the command-queue without informing the library.  Allowing functions to
 *  attach to (i.e. retain) and release a command-queue solves the problem of a
 *  command-queue being used by a library no longer being valid.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clRetainCommandQueue, (cl_command_queue command_queue)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  as_amd(command_queue)->retain();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Decrement the \a command_queue reference count.
 *
 *  \return One of the following values:
 *    - CL_SUCCESS if the function is executed successfully.
 *    - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid
 *      command-queue.
 *
 *  After the command_queue reference count becomes zero and all commands queued
 *  to \a command_queue have finished (eg. kernel executions, memory object
 *  updates etc.), the command-queue is deleted.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clReleaseCommandQueue, (cl_command_queue command_queue)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }
  as_amd(command_queue)->release();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Query information about a command-queue.
 *
 *  \param command_queue specifies the command-queue being queried.
 *
 *  \param param_name specifies the information to query.
 *
 *  \param param_value is a pointer to memory where the appropriate result
 *  being queried is returned. If \a param_value is NULL, it is ignored.
 *
 *  \param param_value_size is used to specify the size in bytes of memory
 *  pointed to by \a param_value. This size must be >= size of return type.
 *  If param_value is NULL, it is ignored.
 *
 *  \param param_value_size_ret returns the actual size in bytes of data being
 *  queried by \a param_value. If \a param_value_size_ret is NULL,
 *  it is ignored.
 *
 *  \return One of the following values:
 *    - CL_SUCCESS if the function is executed successfully.
 *    - CL_INVALID_COMMAND_QUEUE if \a command_queue is not a valid
 *      command-queue.
 *    - CL_INVALID_VALUE if \a param_name is not one of the supported
 *      values or if size in bytes specified by \a param_value_size is < size of
 *      return type and \a param_value is not a NULL value.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clGetCommandQueueInfo,
              (cl_command_queue command_queue, cl_command_queue_info param_name,
               size_t param_value_size, void* param_value, size_t* param_value_size_ret)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  switch (param_name) {
    case CL_QUEUE_CONTEXT: {
      cl_context context = const_cast<cl_context>(as_cl(&as_amd(command_queue)->context()));
      return amd::clGetInfo(context, param_value_size, param_value, param_value_size_ret);
    }
    case CL_QUEUE_DEVICE: {
      cl_device_id device = const_cast<cl_device_id>(as_cl(&as_amd(command_queue)->device()));
      return amd::clGetInfo(device, param_value_size, param_value, param_value_size_ret);
    }
    case CL_QUEUE_PROPERTIES: {
      cl_command_queue_properties properties = as_amd(command_queue)->properties().value_;
      return amd::clGetInfo(properties, param_value_size, param_value, param_value_size_ret);
    }
    case CL_QUEUE_REFERENCE_COUNT: {
      cl_uint count = as_amd(command_queue)->referenceCount();
      return amd::clGetInfo(count, param_value_size, param_value, param_value_size_ret);
    }
    case CL_QUEUE_SIZE: {
      const amd::DeviceQueue* deviceQueue = as_amd(command_queue)->asDeviceQueue();
      if (NULL == deviceQueue) {
        return CL_INVALID_COMMAND_QUEUE;
      }
      cl_uint size = deviceQueue->size();
      return amd::clGetInfo(size, param_value_size, param_value, param_value_size_ret);
    }
    case CL_QUEUE_THREAD_HANDLE_AMD: {
      const amd::HostQueue* hostQueue = as_amd(command_queue)->asHostQueue();
      if (NULL == hostQueue) {
        return CL_INVALID_COMMAND_QUEUE;
      }
      const void* handle = hostQueue->thread().handle();
      return amd::clGetInfo(handle, param_value_size, param_value, param_value_size_ret);
    }
    case CL_QUEUE_DEVICE_DEFAULT: {
      const amd::Device& device = as_amd(command_queue)->device();
      amd::CommandQueue* defQueue = as_amd(command_queue)->context().defDeviceQueue(device);
      cl_command_queue queue = defQueue ? as_cl(defQueue) : NULL;
      return amd::clGetInfo(queue, param_value_size, param_value, param_value_size_ret);
    }
    default:
      break;
  }

  return CL_INVALID_VALUE;
}
RUNTIME_EXIT

/*! \brief Enable or disable the properties of a command-queue.
 *
 *  \param command_queue specifies the command-queue being queried.
 *
 *  \param properties specifies the new command-queue properties to be applied
 *  to \a command_queue .
 *
 *  \param enable determines whether the values specified by properties are
 *  enabled (if enable is CL_TRUE) or disabled (if enable is CL_FALSE) for the
 *  command-queue .
 *
 *  \param old_properties returns the command-queue properties before they were
 *  changed by clSetCommandQueueProperty. If \a old_properties is NULL,
 *  it is ignored.
 *
 *  \return One of the following values:
 *  - CL_SUCCESS if the command-queue properties are successfully updated.
 *  - CL_INVALID_COMMAND_QUEUE if command_queue is not a valid command-queue.
 *  - CL_INVALID_VALUE if the values specified in properties are not valid.
 *  - CL_INVALID_QUEUE_PROPERTIES if values specified in properties are
 *    not supported by the device.
 *
 *  \version 1.0r33
 */
RUNTIME_ENTRY(cl_int, clSetCommandQueueProperty,
              (cl_command_queue command_queue, cl_command_queue_properties properties,
               cl_bool enable, cl_command_queue_properties* old_properties)) {
  if (!is_valid(command_queue)) {
    return CL_INVALID_COMMAND_QUEUE;
  }

  *not_null(old_properties) = as_amd(command_queue)->properties().value_;

  if (properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
    clFinish(command_queue);
  }

  bool success;
  if (enable == CL_TRUE) {
    success = as_amd(command_queue)->properties().set(properties);
  } else {
    success = as_amd(command_queue)->properties().clear(properties);
  }

  return success ? CL_SUCCESS : CL_INVALID_QUEUE_PROPERTIES;
}
RUNTIME_EXIT

/*! @}
 *  @}
 */
