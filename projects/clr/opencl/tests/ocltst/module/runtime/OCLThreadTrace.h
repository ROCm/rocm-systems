/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_THREAD_TRACE_H_
#define _OCL_THREAD_TRACE_H_

#include "OCLTestImp.h"
#include "cl_thread_trace_amd.h"

// Thread Trace API
typedef CL_API_ENTRY cl_threadtrace_amd(CL_API_CALL* fnp_clCreateThreadTraceAMD)(cl_device_id,
                                                                                 cl_int*);
typedef CL_API_ENTRY cl_int(CL_API_CALL* fnp_clReleaseThreadTraceAMD)(cl_threadtrace_amd);
typedef CL_API_ENTRY cl_int(CL_API_CALL* fnp_clRetainThreadTraceAMD)(cl_threadtrace_amd);
typedef CL_API_ENTRY cl_int(CL_API_CALL* fnp_clGetThreadTraceInfoAMD)(cl_threadtrace_amd,
                                                                      cl_threadtrace_info, size_t,
                                                                      void*, size_t*);
typedef CL_API_ENTRY cl_int(CL_API_CALL* fnp_clSetThreadTraceParamAMD)(cl_threadtrace_amd,
                                                                       cl_thread_trace_param,
                                                                       cl_uint);
typedef CL_API_ENTRY cl_int(CL_API_CALL* fnp_clEnqueueThreadTraceCommandAMD)(
    cl_command_queue, cl_threadtrace_amd, cl_threadtrace_command_name_amd, cl_uint, const cl_event*,
    cl_event*);
typedef CL_API_ENTRY cl_int(CL_API_CALL* fnp_clEnqueueBindThreadTraceBufferAMD)(
    cl_command_queue, cl_threadtrace_amd, cl_mem*, cl_uint, cl_uint, cl_uint, const cl_event*,
    cl_event*);

class OCLThreadTrace : public OCLTestImp {
 public:
  OCLThreadTrace();
  virtual ~OCLThreadTrace();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  cl_uint** ioBuf_;
  cl_uint** ttBuf_;
  cl_threadtrace_amd threadTrace_;

  fnp_clCreateThreadTraceAMD clCreateThreadTraceAMD_;
  fnp_clReleaseThreadTraceAMD clReleaseThreadTraceAMD_;
  fnp_clRetainThreadTraceAMD clRetainThreadTraceAMD_;
  fnp_clGetThreadTraceInfoAMD clGetThreadTraceInfoAMD_;
  fnp_clSetThreadTraceParamAMD clSetThreadTraceParamAMD_;
  fnp_clEnqueueThreadTraceCommandAMD clEnqueueThreadTraceCommandAMD_;
  fnp_clEnqueueBindThreadTraceBufferAMD clEnqueueBindThreadTraceBufferAMD_;
};

#endif  // _OCL_THREAD_TRACE_H_
