/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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

#include "OCLGetQueueThreadID.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "CL/cl_ext.h"

#if !defined(__linux__)
#include "WinBase.h"
typedef DWORD(WINAPI* GetThreadId)(__in HANDLE Thread);
#endif
bool badThread = false;

OCLGetQueueThreadID::OCLGetQueueThreadID() {
  _numSubTests = 1;
  failed_ = false;
}

OCLGetQueueThreadID::~OCLGetQueueThreadID() {}

void OCLGetQueueThreadID::open(unsigned int test, char* units, double& conversion,
                               unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  char name[1024] = {0};
  size_t size = 0;

  if (deviceId >= deviceCount_) {
    failed_ = true;
    return;
  }

  cl_mem buffer;
  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
  buffers_.push_back(buffer);
}

static void CL_CALLBACK notify_callback(cl_event event, cl_int event_command_exec_status,
                                        void* user_data) {
#if defined(__linux__)
  pthread_t id = (pthread_t)user_data;
  pthread_t handle = pthread_self();
#else
  HMODULE module = GetModuleHandle("kernel32.dll");
  GetThreadId getThreadId = reinterpret_cast<GetThreadId>(GetProcAddress(module, "GetThreadId"));
  if (NULL == getThreadId) {
    return;
  }
  DWORD id = getThreadId((HANDLE)user_data);
  DWORD handle = GetCurrentThreadId();
#endif
  if (id != handle) {
    badThread = true;
  }
}

void OCLGetQueueThreadID::run(void) {
  if (failed_) {
    return;
  }
  void* handle;
  cl_event clEvent;
  cl_event userEvent = clCreateUserEvent(context_, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateUserEvent() failed");

  cl_uint initVal[2] = {5, 10};
  error_ = _wrapper->clGetCommandQueueInfo(cmdQueues_[_deviceId], CL_QUEUE_THREAD_HANDLE_AMD,
                                           sizeof(void*), &handle, NULL);
  error_ = _wrapper->clEnqueueWriteBuffer(cmdQueues_[_deviceId], buffers()[0], false, 0,
                                          sizeof(cl_uint), &initVal[0], 1, &userEvent, &clEvent);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueWriteBuffer() failed");

  error_ = _wrapper->clSetEventCallback(clEvent, CL_SUBMITTED, notify_callback, handle);

  clSetUserEventStatus(userEvent, CL_COMPLETE);

  clFinish(cmdQueues_[_deviceId]);

  clReleaseEvent(clEvent);

  clReleaseEvent(userEvent);

  CHECK_RESULT(badThread, "Thread ID is incorrect!");
}

unsigned int OCLGetQueueThreadID::close(void) { return OCLTestImp::close(); }
