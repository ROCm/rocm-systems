/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
