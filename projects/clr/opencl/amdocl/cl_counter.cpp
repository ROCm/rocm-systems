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
#include <CL/cl_ext.h>

#include "platform/object.hpp"
#include "platform/context.hpp"
#include "platform/command.hpp"
#include "platform/counter.hpp"

#ifdef cl_amd_atomic_counters

/*! \addtogroup API
 *  @{
 * \addtogroup CL_Counters
 *
 *  Counter objects ...
 *
 *  @{
 */

/*! \brief
 *
 *  \version 1.1r18
 */
RUNTIME_ENTRY_RET(cl_counter_amd, clCreateCounterAMD,
                  (cl_context context, cl_counter_flags_amd flags, cl_uint value,
                   cl_int* errcode_ret)) {
  *not_null(errcode_ret) = CL_INVALID_CONTEXT;
  return (cl_counter_amd)0;
}
RUNTIME_EXIT

/*! \brief
 *
 *  \version 1.1r18
 */
RUNTIME_ENTRY(cl_int, clGetCounterInfoAMD,
              (cl_counter_amd counter, cl_counter_info_amd param_name, size_t param_value_size,
               void* param_value, size_t* param_value_size_ret)) {
  return CL_INVALID_COUNTER_AMD;
}
RUNTIME_EXIT

/*! \brief Increment the counter reference count.
 *
 *  \return CL_SUCCESS if the function is executed successfully. It returns
 *  CL_INVALID_COUNTER if \a counter is not a valid counter object.
 *
 *  The OpenCL commands that return a counter perform an implicit retain.
 *
 *  \version 1.1r18
 */
RUNTIME_ENTRY(cl_int, clRetainCounterAMD, (cl_counter_amd counter)) {
  if (!is_valid(counter)) {
    return CL_INVALID_COUNTER_AMD;
  }
  as_amd(counter)->retain();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief Decrement the counter reference count.
 *
 *  \return CL_SUCCESS if the function is executed successfully. It returns
 *  CL_INVALID_EVENT if \a counter is not a valid counter object.
 *
 *  The counter object is deleted once the reference count becomes zero.
 *
 *  \version 1.1r18
 */
RUNTIME_ENTRY(cl_int, clReleaseCounterAMD, (cl_counter_amd counter)) {
  if (!is_valid(counter)) {
    return CL_INVALID_COUNTER_AMD;
  }
  as_amd(counter)->release();
  return CL_SUCCESS;
}
RUNTIME_EXIT

/*! \brief
 *
 *  \version 1.1r18
 */
RUNTIME_ENTRY(cl_int, clEnqueueReadCounterAMD,
              (cl_command_queue command_queue, cl_counter_amd counter, cl_bool blocking_read,
               cl_uint* value, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
               cl_event* event)) {
  return CL_INVALID_COUNTER_AMD;
}
RUNTIME_EXIT

/*! \brief
 *
 *  \version 1.1r18
 */
RUNTIME_ENTRY(cl_int, clEnqueueWriteCounterAMD,
              (cl_command_queue command_queue, cl_counter_amd counter, cl_bool blocking_write,
               cl_uint value, cl_uint num_events_in_wait_list, const cl_event* event_wait_list,
               cl_event* event)) {
  return CL_INVALID_COUNTER_AMD;
}
RUNTIME_EXIT

/*! @}
 *  @}
 */

#endif  // cl_amd_atomic_counters
