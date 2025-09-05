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

#ifndef CL_COMMON_HPP_
#define CL_COMMON_HPP_

#ifdef _WIN32
#include <CL/cl_d3d11.h>
#include <CL/cl_d3d10.h>
#include <CL/cl_dx9_media_sharing.h>
#endif
#include <CL/cl_icd.h>

#include "top.hpp"
#include "vdi_common.hpp"

//! Helper function to check "properties" parameter in various functions
int checkContextProperties(const cl_context_properties* properties, bool* offlineDevices);

namespace amd {

template <typename T> static inline cl_int clGetInfo(T& field, size_t param_value_size,
                                                     void* param_value,
                                                     size_t* param_value_size_ret) {
  const void* valuePtr;
  size_t valueSize;

  std::tie(valuePtr, valueSize) =
      detail::ParamInfo<typename std::remove_const<T>::type>::get(field);

  *not_null(param_value_size_ret) = valueSize;

  cl_int ret = CL_SUCCESS;
  if (param_value != NULL && param_value_size < valueSize) {
    if ((param_value_size == 0) || !std::is_pointer<T>() ||
        !std::is_same<typename std::remove_const<typename std::remove_pointer<T>::type>::type,
                      char>()) {
      return CL_INVALID_VALUE;
    }
    // For char* and char[] params, we will at least fill up to
    // param_value_size, then return an error.
    valueSize = param_value_size;
    static_cast<char*>(param_value)[--valueSize] = '\0';
    ret = CL_INVALID_VALUE;
  }

  if (param_value != NULL) {
    ::memcpy(param_value, valuePtr, valueSize);
    if (param_value_size > valueSize) {
      ::memset(static_cast<address>(param_value) + valueSize, '\0', param_value_size - valueSize);
    }
  }

  return ret;
}

static inline cl_int clSetEventWaitList(Command::EventWaitList& eventWaitList,
                                        const amd::HostQueue& hostQueue,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event* event_wait_list) {
  if ((num_events_in_wait_list == 0 && event_wait_list != NULL) ||
      (num_events_in_wait_list != 0 && event_wait_list == NULL)) {
    return CL_INVALID_EVENT_WAIT_LIST;
  }

  while (num_events_in_wait_list-- > 0) {
    cl_event event = *event_wait_list++;
    Event* amdEvent = as_amd(event);
    if (!is_valid(event)) {
      return CL_INVALID_EVENT_WAIT_LIST;
    }
    if (&hostQueue.context() != &amdEvent->context()) {
      return CL_INVALID_CONTEXT;
    }
    if ((amdEvent->command().queue() != &hostQueue) && !amdEvent->notifyCmdQueue()) {
      return CL_INVALID_EVENT_WAIT_LIST;
    }
    eventWaitList.push_back(amdEvent);
  }
  return CL_SUCCESS;
}

//! Common function declarations for CL-external graphics API interop
cl_int clEnqueueAcquireExtObjectsAMD(cl_command_queue command_queue, cl_uint num_objects,
                                     const cl_mem* mem_objects, cl_uint num_events_in_wait_list,
                                     const cl_event* event_wait_list, cl_event* event,
                                     cl_command_type cmd_type);
cl_int clEnqueueReleaseExtObjectsAMD(cl_command_queue command_queue, cl_uint num_objects,
                                     const cl_mem* mem_objects, cl_uint num_events_in_wait_list,
                                     const cl_event* event_wait_list, cl_event* event,
                                     cl_command_type cmd_type);
static inline cl_int clDXTranslateErrorCode(cl_int err) {
  return err == CL_INVALID_GL_OBJECT ? CL_INVALID_MEM_OBJECT : err;
}

}  // namespace amd

extern "C" {

#if defined(CL_VERSION_1_1)
extern CL_API_ENTRY cl_int CL_API_CALL clSetCommandQueueProperty(
    cl_command_queue command_queue, cl_command_queue_properties properties, cl_bool enable,
    cl_command_queue_properties* old_properties) CL_API_SUFFIX__VERSION_1_0;
#endif  // CL_VERSION_1_1

extern CL_API_ENTRY cl_mem CL_API_CALL clConvertImageAMD(cl_context context, cl_mem image,
                                                         const cl_image_format* image_format,
                                                         cl_int* errcode_ret);

extern CL_API_ENTRY cl_mem CL_API_CALL clCreateBufferFromImageAMD(cl_context context, cl_mem image,
                                                                  cl_int* errcode_ret);

extern CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithAssemblyAMD(cl_context context,
                                                                          cl_uint count,
                                                                          const char** strings,
                                                                          const size_t* lengths,
                                                                          cl_int* errcode_ret);

}  // extern "C"

//! \endcond

#endif /*CL_COMMON_HPP_*/
