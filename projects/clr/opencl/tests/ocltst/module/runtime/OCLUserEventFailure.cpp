/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "OCLUserEventFailure.h"

#include "CL/cl.h"

static const unsigned int NumTasks = 4;

static const char* strKernel =
    "__kernel void simple_task(__global uint* out) {\n"
    "  out[0] += 1;\n"
    "}\n";

// Sub-test 0: tasks wait on one user event, which fails. The application releases the kernel
//             and program before the queue drains.
// Sub-test 1: tasks wait on two user events and only one fails. The wait must report the
//             failure without the other event being resolved, and no kernel reference may leak.
OCLUserEventFailure::OCLUserEventFailure() {
  _numSubTests = 2;
  testID_ = 0;
}

OCLUserEventFailure::~OCLUserEventFailure() {}

void OCLUserEventFailure::open(unsigned int test, char* units, double& conversion,
                               unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");
  testID_ = test;

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource() failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], NULL, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

  kernel_ = _wrapper->clCreateKernel(program_, "simple_task", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_uint init = 0;
  cl_mem buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                           sizeof(init), &init, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");
}

void OCLUserEventFailure::failSingleEvent() {
  cl_command_queue queue = cmdQueues_[_deviceId];
  cl_event userEvent = clCreateUserEvent(context_, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateUserEvent() failed");

  cl_event events[NumTasks];
  for (unsigned int i = 0; i < NumTasks; ++i) {
    error_ = _wrapper->clEnqueueTask(queue, kernel_, 1, &userEvent, &events[i]);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueTask() failed");
  }

  error_ = clSetUserEventStatus(userEvent, -1);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetUserEventStatus() failed");

  error_ = _wrapper->clWaitForEvents(NumTasks, events);
  CHECK_RESULT((error_ != CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST),
               "clWaitForEvents() didn't report the failed wait list");

  // The commands must keep the kernel and program alive until the device is done with them
  _wrapper->clReleaseKernel(kernel_);
  kernel_ = 0;
  _wrapper->clReleaseProgram(program_);
  program_ = 0;

  error_ = _wrapper->clFinish(queue);
  CHECK_RESULT((error_ != CL_SUCCESS), "clFinish() failed");

  for (unsigned int i = 0; i < NumTasks; ++i) {
    cl_int status = CL_COMPLETE;
    error_ = _wrapper->clGetEventInfo(events[i], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status),
                                      &status, NULL);
    CHECK_RESULT((error_ != CL_SUCCESS), "clGetEventInfo() failed");
    CHECK_RESULT((status >= 0), "A task waiting on a failed user event didn't fail");
    _wrapper->clReleaseEvent(events[i]);
  }
  _wrapper->clReleaseEvent(userEvent);

  // The queue must still be usable
  cl_uint value = 0;
  error_ = _wrapper->clEnqueueReadBuffer(queue, buffers_[0], CL_TRUE, 0, sizeof(value), &value, 0,
                                         NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");
}

void OCLUserEventFailure::failOneOfTwoEvents() {
  cl_command_queue queue = cmdQueues_[_deviceId];
  cl_event userEvents[2];
  for (unsigned int i = 0; i < 2; ++i) {
    userEvents[i] = clCreateUserEvent(context_, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateUserEvent() failed");
  }

  cl_event events[NumTasks];
  for (unsigned int i = 0; i < NumTasks; ++i) {
    error_ = _wrapper->clEnqueueTask(queue, kernel_, 2, userEvents, &events[i]);
    CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueTask() failed");
  }

  error_ = clSetUserEventStatus(userEvents[0], -1);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetUserEventStatus() failed");

  // The second user event is still unresolved, so this must not block
  error_ = _wrapper->clWaitForEvents(NumTasks, events);
  CHECK_RESULT((error_ != CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST),
               "clWaitForEvents() didn't report the failed wait list");

  error_ = clSetUserEventStatus(userEvents[1], CL_COMPLETE);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetUserEventStatus() failed");

  error_ = _wrapper->clFinish(queue);
  CHECK_RESULT((error_ != CL_SUCCESS), "clFinish() failed");

  for (unsigned int i = 0; i < NumTasks; ++i) {
    _wrapper->clReleaseEvent(events[i]);
  }
  for (unsigned int i = 0; i < 2; ++i) {
    _wrapper->clReleaseEvent(userEvents[i]);
  }

  // Once the queue drains, the failed tasks must not hold kernel references
  cl_uint refCount = 0;
  error_ = _wrapper->clGetKernelInfo(kernel_, CL_KERNEL_REFERENCE_COUNT, sizeof(refCount),
                                     &refCount, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetKernelInfo() failed");
  CHECK_RESULT((refCount != 1), "Kernel reference count is %u after clFinish (expected 1)",
               refCount);
}

void OCLUserEventFailure::run(void) {
  if (testID_ == 0) {
    failSingleEvent();
  } else {
    failOneOfTwoEvents();
  }
}

unsigned int OCLUserEventFailure::close(void) { return OCLTestImp::close(); }
