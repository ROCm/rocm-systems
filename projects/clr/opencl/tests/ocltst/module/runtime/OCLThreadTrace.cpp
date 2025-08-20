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

#include "OCLThreadTrace.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"

const static unsigned int IOThreadTrace = 3;  // number of input/oputput buffers
static size_t SeNum = 1;                      // number of SEs
// size of thread trace buffer
#if EMU_ENV
const static unsigned int ttBufSize = 5000;
#else
const static unsigned int ttBufSize = 30000;
#endif
const static unsigned int InputElements = 2048;  // elements in each vector

const static char* strKernel =
    "__kernel void thread_trace_test(                                       \n"
    "   __global int *A,__global int *B,__global int *C)                    \n"
    "{                                                                      \n"
    "   int idx = get_global_id(0);                                         \n"
    "   C[idx] = A[idx] + B[idx];                                           \n"
    "}                                                                      \n";

OCLThreadTrace::OCLThreadTrace() {
  _numSubTests = 1;
  failed_ = false;
  clCreateThreadTraceAMD_ = 0;
  clReleaseThreadTraceAMD_ = 0;
  clRetainThreadTraceAMD_ = 0;
  clGetThreadTraceInfoAMD_ = 0;
  clSetThreadTraceParamAMD_ = 0;
  clEnqueueThreadTraceCommandAMD_ = 0;
  clEnqueueBindThreadTraceBufferAMD_ = 0;
  ioBuf_ = 0;
  ttBuf_ = 0;
}

OCLThreadTrace::~OCLThreadTrace() {}

void OCLThreadTrace::open(unsigned int test, char* units, double& conversion,
                          unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening");

  if (deviceId >= deviceCount_) {
    failed_ = true;
    return;
  }

  cl_device_type deviceType;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_TYPE, sizeof(deviceType),
                                     &deviceType, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_TYPE failed");

  if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
    printf("GPU device is required for this test!\n");
    failed_ = true;
    return;
  }

  size_t threadTraceEnabled;
  size_t retsize;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD,
                                     sizeof(threadTraceEnabled), &threadTraceEnabled, &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  if (!threadTraceEnabled) {
    failed_ = true;
    testDescString = "Not supported";
    return;
  }

  unsigned int datasize = sizeof(unsigned int) * InputElements;

  ioBuf_ = (unsigned int**)malloc(IOThreadTrace * sizeof(unsigned int*));
  CHECK_RESULT((ioBuf_ == NULL), "malloc  failed");

  memset(ioBuf_, 0, IOThreadTrace * sizeof(unsigned int*));
  for (unsigned i = 0; i < IOThreadTrace; ++i) {
    ioBuf_[i] = (unsigned int*)malloc(datasize);
    CHECK_RESULT((ioBuf_[i] == NULL), "malloc  failed");
    for (unsigned j = 0; j < InputElements; ++j) {
      ioBuf_[i][j] = j;
    }
  }

  clCreateThreadTraceAMD_ =
      (fnp_clCreateThreadTraceAMD)_wrapper->clGetExtensionFunctionAddress("clCreateThreadTraceAMD");
  CHECK_RESULT((clCreateThreadTraceAMD_ == 0),
               "clGetExtensionFunctionAddress(clCreateThreadTraceAMD) failed");
  clGetThreadTraceInfoAMD_ = (fnp_clGetThreadTraceInfoAMD)_wrapper->clGetExtensionFunctionAddress(
      "clGetThreadTraceInfoAMD");
  CHECK_RESULT((clGetThreadTraceInfoAMD_ == 0),
               "clGetExtensionFunctionAddress(clGetThreadTraceInfoAMD) failed");

  threadTrace_ = clCreateThreadTraceAMD_(devices_[_deviceId], &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateThreadTraceAMD() failed");

  // Get number of shader engines
  clGetThreadTraceInfoAMD_(threadTrace_, CL_THREAD_TRACE_SE, sizeof(SeNum), &SeNum, NULL);

  ttBuf_ = (unsigned int**)malloc(SeNum * sizeof(unsigned int*));
  CHECK_RESULT((ttBuf_ == NULL), "malloc  failed");

  memset(ttBuf_, 0, SeNum * sizeof(unsigned int*));

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], NULL, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed");

  kernel_ = _wrapper->clCreateKernel(program_, "thread_trace_test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed");

  cl_mem buffer;
  for (unsigned int i = 0; i < IOThreadTrace; ++i) {
    buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, datasize,
                                      ioBuf_[i], &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
    buffers_.push_back(buffer);
  }

  for (unsigned int i = 0; i < SeNum; ++i) {
    buffer = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, ttBufSize, NULL, &error_);
    CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer() failed");
    buffers_.push_back(buffer);
  }

  clReleaseThreadTraceAMD_ = (fnp_clReleaseThreadTraceAMD)_wrapper->clGetExtensionFunctionAddress(
      "clReleaseThreadTraceAMD");
  CHECK_RESULT((clReleaseThreadTraceAMD_ == 0),
               "clGetExtensionFunctionAddress(clReleaseThreadTraceAMD) failed");
  clRetainThreadTraceAMD_ =
      (fnp_clRetainThreadTraceAMD)_wrapper->clGetExtensionFunctionAddress("clRetainThreadTraceAMD");
  CHECK_RESULT((clRetainThreadTraceAMD_ == 0),
               "clGetExtensionFunctionAddress(clRetainThreadTraceAMD) failed");
  clSetThreadTraceParamAMD_ = (fnp_clSetThreadTraceParamAMD)_wrapper->clGetExtensionFunctionAddress(
      "clSetThreadTraceParamAMD");
  CHECK_RESULT((clSetThreadTraceParamAMD_ == 0),
               "clGetExtensionFunctionAddress(clSetThreadTraceParamAMD) failed");
  clEnqueueThreadTraceCommandAMD_ =
      (fnp_clEnqueueThreadTraceCommandAMD)_wrapper->clGetExtensionFunctionAddress(
          "clEnqueueThreadTraceCommandAMD");
  CHECK_RESULT((clEnqueueThreadTraceCommandAMD_ == 0),
               "clGetExtensionFunctionAddress(clEnqueueThreadTraceCommandAMD) failed");
  clEnqueueBindThreadTraceBufferAMD_ =
      (fnp_clEnqueueBindThreadTraceBufferAMD)_wrapper->clGetExtensionFunctionAddress(
          "clEnqueueBindThreadTraceBufferAMD");
  CHECK_RESULT((clEnqueueBindThreadTraceBufferAMD_ == 0),
               "clGetExtensionFunctionAddress("
               "clEnqueueBindThreadTraceBufferAMD) failed");
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

static void DumpTraceSI(unsigned int index, cl_ushort* tracePtr, size_t numOfBytes) {
  FILE* outFile;
  char file_name[16] = {0};
  static unsigned int iii = 0;
  sprintf(file_name, "TTrace%d%d.out", index, iii++);

  outFile = fopen(file_name, "w");

  for (size_t i = 0; i < numOfBytes / 2; i++) {
    fprintf(outFile, "%04x\n", (cl_ushort)(*tracePtr));
    tracePtr++;
  }

  fclose(outFile);
}

#define DUMPTRACE 0

void OCLThreadTrace::run(void) {
  cl_mem* ttArrBuf = 0;
  unsigned int* ttBufRecordedSizes = 0;
  unsigned int i = 0, j = 0;

  if (failed_) {
    return;
  }

  for (i = 0; i < IOThreadTrace; ++i) {
    cl_mem buffer = buffers()[i];
    error_ = _wrapper->clSetKernelArg(kernel_, i, sizeof(cl_mem), &buffer);
    CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg() failed");
  }

  size_t globalWorkSize[1];
  size_t localWorkSize[1];
  globalWorkSize[0] = InputElements;
  localWorkSize[0] = 32;

  ttArrBuf = (cl_mem*)malloc(sizeof(cl_mem) * SeNum);
  ;
  for (i = 0; i < SeNum; i++) ttArrBuf[i] = buffers()[IOThreadTrace + i];

  cl_event clEvent;
  error_ = clEnqueueBindThreadTraceBufferAMD_(cmdQueues_[_deviceId], threadTrace_, ttArrBuf,
                                              (cl_uint)SeNum, ttBufSize, 0, NULL, &clEvent);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueBindThreadTraceBufferAMD() failed");

  error_ = clEnqueueThreadTraceCommandAMD_(cmdQueues_[_deviceId], threadTrace_,
                                           CL_THREAD_TRACE_BEGIN_COMMAND, 0, NULL, &clEvent);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueThreadTraceCommandAMD() failed");

  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, globalWorkSize,
                                            localWorkSize, 0, NULL, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel() failed");
  clFinish(cmdQueues_[_deviceId]);

  error_ = clEnqueueThreadTraceCommandAMD_(cmdQueues_[_deviceId], threadTrace_,
                                           CL_THREAD_TRACE_END_COMMAND, 0, NULL, &clEvent);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueThreadTraceCommandAMD() failed");

  ttBufRecordedSizes = (unsigned int*)malloc(sizeof(unsigned int) * SeNum);
  memset(ttBufRecordedSizes, 0, sizeof(unsigned int) * SeNum);
  size_t ttBufRecordedSize;
  error_ = clGetThreadTraceInfoAMD_(threadTrace_, CL_THREAD_TRACE_BUFFERS_SIZE, 1, NULL,
                                    &ttBufRecordedSize);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetThreadTraceInfoAMD() failed");

  if (ttBufRecordedSize > sizeof(unsigned int) * SeNum) {
    free(ttBufRecordedSizes);
    ttBufRecordedSizes = (unsigned int*)malloc(ttBufRecordedSize);
    memset(ttBufRecordedSizes, 0, ttBufRecordedSize);
  }

  error_ = clGetThreadTraceInfoAMD_(threadTrace_, CL_THREAD_TRACE_BUFFERS_SIZE, ttBufRecordedSize,
                                    ttBufRecordedSizes, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetThreadTraceInfoAMD() failed");

  for (i = 0; i < SeNum; ++i) {
    ttBuf_[i] = (cl_uint*)malloc(ttBufRecordedSizes[i] * sizeof(cl_uint));
    CHECK_RESULT((ttBuf_[i] == NULL), "malloc  failed");
  }

  for (i = 0; i < SeNum; ++i) {
    if (ttBufRecordedSizes[i] != 0) {
      error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers()[IOThreadTrace + i],
                                             CL_TRUE, 0, ttBufRecordedSizes[i], ttBuf_[i], 0, NULL,
                                             NULL);
      CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer() failed");
#if DUMPTRACE
      DumpTraceSI(i, (cl_ushort*)ttBuf_[i], ttBufRecordedSizes[i]);
#endif
    }
  }

  bool validRes = true;
  for (i = 0; i < SeNum; ++i) {
    unsigned j;
    for (j = 0; j < ttBufRecordedSizes[i]; ++j) {
      if (ttBuf_[i][j] != 0) {
        break;
      }
    }
    if (j >= ttBufRecordedSizes[i] && ttBufRecordedSizes[i] > 0) {
      validRes = false;
      break;
    }
  }
  if (!validRes) {
    CHECK_RESULT(true, " - Incorrect result for thread trace. no output data was recorded.\n");
  }

  if (ttArrBuf) free(ttArrBuf);
  if (ttBufRecordedSizes) free(ttBufRecordedSizes);
}

unsigned int OCLThreadTrace::close(void) {
  if (clReleaseThreadTraceAMD_ && threadTrace_) clReleaseThreadTraceAMD_(threadTrace_);

  if (ioBuf_) {
    for (unsigned i = 0; i < IOThreadTrace; ++i) {
      if (ioBuf_[i]) {
        free(ioBuf_[i]);
      }
    }
    free(ioBuf_);
  }
  if (ttBuf_) {
    for (unsigned i = 0; i < SeNum; ++i) {
      if (ttBuf_[i]) {
        free(ttBuf_[i]);
      }
    }
    free(ttBuf_);
  }
  return OCLTestImp::close();
}
