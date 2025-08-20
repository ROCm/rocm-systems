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

#include "OCLProgramScopeVariables.h"

#include "CL/cl.h"

OCLProgramScopeVariables::OCLProgramScopeVariables() { _numSubTests = 3; }

OCLProgramScopeVariables::~OCLProgramScopeVariables() {}

void OCLProgramScopeVariables::open(unsigned int test, char* units, double& conversion,
                                    unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "error_ opening test");
  silentFailure = false;
  _openTest = test;
  size_t param_size = 0;
  program_ = 0;
  kernel1_ = kernel2_ = 0;
  char* strVersion = 0;
  error_ =
      _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_OPENCL_C_VERSION, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformInfo failed");
  strVersion = (char*)malloc(param_size);
  error_ = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_OPENCL_C_VERSION, param_size,
                                     strVersion, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformInfo failed");
  if (strVersion[9] < '2') {
    printf("\nOpenCL C 2.0 not supported\n");
    silentFailure = true;
  }
  free(strVersion);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLProgramScopeVariables::run(void) {
  if (silentFailure) return;
  switch (_openTest) {
    case 0:
      test0();
      break;
    case 1:
      test1();
      break;
    case 2:
      test2();
      break;
  }
  return;
}

void OCLProgramScopeVariables::test0(void) {
  const char* kernel_str =
      "global int g[1000] = {0}; \n\
        __kernel void test1 (global unsigned int * A) \n\
        { \n\
            int id = get_global_id(0);  \n\
            g[id] = id; \n\
        } \n\
        __kernel void test2 (global unsigned int * A) \n\
        { \n\
            int id = get_global_id(0);  \n\
            A[id] = g[id]; \n\
        } \n";
  const size_t arrSize = 1000;
  cl_uint* output_arr = (cl_uint*)malloc(arrSize * sizeof(cl_uint));
  cl_mem buffer =
      _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, arrSize * sizeof(cl_uint), 0, &error_);
  buffers_.push_back(buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer failed");
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &kernel_str, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource failed");
  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], "-cl-std=CL2.0", NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char log[400];
    _wrapper->clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG, 400, log,
                                    0);
    printf("\n\n%s\n\n", log);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram failed");
  kernel1_ = _wrapper->clCreateKernel(program_, "test1", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel1 failed");
  kernel2_ = _wrapper->clCreateKernel(program_, "test2", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel2 failed");
  error_ = _wrapper->clSetKernelArg(kernel1_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  error_ = _wrapper->clSetKernelArg(kernel2_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  cl_event evt;
  size_t global_work_size = arrSize;
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel1_, 1, NULL,
                                            &global_work_size, NULL, 0, NULL, &evt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel2_, 1, NULL,
                                            &global_work_size, NULL, 0, NULL, &evt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[0], CL_TRUE, 0,
                                         sizeof(cl_uint) * arrSize, output_arr, 1, &evt, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed");
  bool bResult = true;
  for (unsigned int i = 0; i < arrSize; ++i) {
    if (output_arr[i] != i) {
      bResult = false;
      break;
    }
  }
  free(output_arr);
  CHECK_RESULT((bResult == false), "Program Scope Variables - test0 failed");
}

void OCLProgramScopeVariables::test1(void) {
  const char* kernel_str =
      "global int temp = 0; \n\
        __kernel void test1 (global unsigned int * A) \n\
        { \n\
            int id = get_global_id(0);  \n\
            if (id == 0) temp = 55; \n\
        } \n\
        __kernel void test2 (global unsigned int * A) \n\
        { \n\
            int id = get_global_id(0);  \n\
            if (id == 0) A[0] = temp; \n\
        } \n";
  cl_uint* output_arr = (cl_uint*)malloc(sizeof(cl_uint));
  cl_mem buffer =
      _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(cl_uint), 0, &error_);
  buffers_.push_back(buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer failed");
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &kernel_str, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource failed");
  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], "-cl-std=CL2.0", NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char log[400];
    _wrapper->clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG, 400, log,
                                    0);
    printf("\n\n%s\n\n", log);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram failed");
  kernel1_ = _wrapper->clCreateKernel(program_, "test1", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel1 failed");
  kernel2_ = _wrapper->clCreateKernel(program_, "test2", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel2 failed");
  error_ = _wrapper->clSetKernelArg(kernel1_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  error_ = _wrapper->clSetKernelArg(kernel2_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  cl_event evt;
  size_t global_work_size = 1;
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel1_, 1, NULL,
                                            &global_work_size, NULL, 0, NULL, &evt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel2_, 1, NULL,
                                            &global_work_size, NULL, 0, NULL, &evt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[0], CL_TRUE, 0,
                                         sizeof(cl_uint), output_arr, 1, &evt, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed");
  bool bResult = (output_arr[0] == 55);
  free(output_arr);
  CHECK_RESULT((bResult == false), "Program Scope Variables - test1 failed");
}

void OCLProgramScopeVariables::test2(void) {
  const char* kernel_str =
      "global int temp = 0; \n\
        global int* ptr[] = {&temp}; \n\
        __kernel void test1 (global unsigned int * A) \n\
        { \n\
            int id = get_global_id(0);  \n\
            if (id == 0) temp = 65; \n\
        } \n\
        __kernel void test2 (global unsigned int * A) \n\
        { \n\
            int id = get_global_id(0);  \n\
            if (id == 0) A[0] = *ptr[0]; \n\
        } \n";
  cl_uint* output_arr = (cl_uint*)malloc(sizeof(cl_uint));
  cl_mem buffer =
      _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(cl_uint), 0, &error_);
  buffers_.push_back(buffer);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer failed");
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &kernel_str, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource failed");
  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], "-cl-std=CL2.0", NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char log[400];
    _wrapper->clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG, 400, log,
                                    0);
    printf("\n\n%s\n\n", log);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram failed");
  kernel1_ = _wrapper->clCreateKernel(program_, "test1", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel1 failed");
  kernel2_ = _wrapper->clCreateKernel(program_, "test2", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel2 failed");
  error_ = _wrapper->clSetKernelArg(kernel1_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  error_ = _wrapper->clSetKernelArg(kernel2_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  cl_event evt;
  size_t global_work_size = 1;
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel1_, 1, NULL,
                                            &global_work_size, NULL, 0, NULL, &evt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel2_, 1, NULL,
                                            &global_work_size, NULL, 0, NULL, &evt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[0], CL_TRUE, 0,
                                         sizeof(cl_uint), output_arr, 1, &evt, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed");
  bool bResult = (output_arr[0] == 65);
  free(output_arr);
  CHECK_RESULT((bResult == false), "Program Scope Variables - test2 failed");
}

unsigned int OCLProgramScopeVariables::close(void) {
  if (kernel1_) {
    error_ = _wrapper->clReleaseKernel(kernel1_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel1 failed");
    kernel1_ = 0;
  }
  if (kernel2_) {
    error_ = _wrapper->clReleaseKernel(kernel2_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel2 failed");
    kernel2_ = 0;
  }
  return OCLTestImp::close();
}
