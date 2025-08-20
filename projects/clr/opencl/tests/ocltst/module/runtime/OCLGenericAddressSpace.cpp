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

#include "OCLGenericAddressSpace.h"

#include "CL/cl.h"

#define TO_LOCAL_FAIL 0x000f0
#define TO_GLOBAL_FAIL 0x00e00
#define TO_PRIVATE_FAIL 0x0d000
#define WRONG_VALUE 0xc0000

OCLGenericAddressSpace::OCLGenericAddressSpace() { _numSubTests = 7; }

OCLGenericAddressSpace::~OCLGenericAddressSpace() {}

void OCLGenericAddressSpace::open(unsigned int test, char* units, double& conversion,
                                  unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "error_ opening test");
  silentFailure = false;
  _openTest = test;
  size_t param_size = 0;
  program_ = 0;
  kernel_ = 0;
  char* strVersion = 0;
#if EMU_ENV
  arrSize = 10;
#else
  arrSize = 1000;
#endif  // EMU_ENV
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

void OCLGenericAddressSpace::run(void) {
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
    case 3:
      test3();
      break;
    case 4:
      test4();
      break;
    case 5:
      test5();
      break;
    case 6:
      test6();
      break;
  }
  return;
}

void OCLGenericAddressSpace::test6(void) {
  const char* kernel_str =
      "\n\
        __global unsigned int gint = 1; \n\
        __kernel void test(__global ulong *results) \n\
        { \n\
            uint tid = get_global_id(0); \n\
            unsigned int *ptr; \n\
            __private unsigned int pint = tid + 2; \n\
            if ((tid % 2) == 0) { \n\
                ptr = &pint; \n\
            } \n\
            else { \n\
                ptr = &gint; \n\
            } \n\
            results[0] = *ptr;\n\
            results[1] = pint;\n\
            results[2] = (ulong)ptr;\n\
            results[3] = (ulong)to_private(ptr);\n\
            results[4] = (ulong)&pint;\n\
        } \n";
  const size_t global_work_size = 1;
  const size_t arrSize = global_work_size * 5;
  cl_ulong* output_arr = (cl_ulong*)malloc(arrSize * sizeof(cl_ulong));
  memset(output_arr, 0, arrSize * sizeof(cl_ulong));
  cl_mem buffer =
      _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, arrSize * sizeof(cl_ulong), 0, &error_);
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
  kernel_ = _wrapper->clCreateKernel(program_, "test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel failed");
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  cl_event evt;

  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                            &global_work_size, NULL, 0, NULL, &evt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[0], CL_TRUE, 0,
                                         sizeof(cl_ulong) * arrSize, output_arr, 1, &evt, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed");
  if (output_arr[0] != 2) {
    printf(
        "\n*ptr:0x%llx, pint:0x%llx, ptr:0x%llx, to_private(ptr):0x%llx, "
        "&pint:0x%llx",
        (unsigned long long)output_arr[0], (unsigned long long)output_arr[1],
        (unsigned long long)output_arr[2], (unsigned long long)output_arr[3],
        (unsigned long long)output_arr[4]);
    printf("\n\n");
    error_ = 1;
  }
  free(output_arr);
  CHECK_RESULT((error_ != CL_SUCCESS), "Generic Address Space - test2 failed");
}

void OCLGenericAddressSpace::test5(void) {
  const char* kernel_str =
      "\n\
        __global unsigned int gint = 1; \n\
        __kernel void test(__global ulong *results) \n\
        { \n\
            uint tid = get_global_id(0); \n\
            results[tid] = 0; \n\
            unsigned int *ptr; \n\
            __local unsigned int lint; \n\
            lint = 2; \n\
            if ((tid % 2) == 0) { \n\
                ptr = &lint; \n\
            } \n\
            else { \n\
                ptr = &gint; \n\
            } \n\
            barrier(CLK_GLOBAL_MEM_FENCE); \n\
            if ((tid % 2) == 0) { \n\
                results[tid*5] = *ptr;\n\
                results[tid*5+1] = lint;\n\
                results[tid*5+2] = (ulong)ptr;\n\
                results[tid*5+3] = (ulong)to_local(ptr);\n\
                results[tid*5+4] = (ulong)&lint;\n\
            } \n\
            else { \n\
                results[tid*5] = *ptr;\n\
                results[tid*5+1] = gint;\n\
                results[tid*5+2] = (ulong)ptr;\n\
                results[tid*5+3] = (ulong)to_global(ptr);\n\
                results[tid*5+4] = (ulong)&gint;\n\
            } \n\
        } \n";
  const size_t global_work_size = 2;
  const size_t arrSize = global_work_size * 5;
  cl_ulong* output_arr = (cl_ulong*)malloc(arrSize * sizeof(cl_ulong));
  memset(output_arr, 0, arrSize * sizeof(cl_ulong));
  cl_mem buffer =
      _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, arrSize * sizeof(cl_ulong), 0, &error_);
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
  kernel_ = _wrapper->clCreateKernel(program_, "test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel failed");
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  cl_event evt;

  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                            &global_work_size, NULL, 0, NULL, &evt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[0], CL_TRUE, 0,
                                         sizeof(cl_ulong) * arrSize, output_arr, 1, &evt, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed");
  int error_cnt = 0;
  for (unsigned int i = 0; i < global_work_size; ++i) {
    if (((i % 2 == 0) && (output_arr[i * 5] != 2)) || ((i % 2 == 1) && (output_arr[i * 5] != 1))) {
      ++error_cnt;
    }
  }
  if (error_cnt) {
    printf("\nNumber of wrong results: %d/%d\n\n", error_cnt, (int)global_work_size);
    for (unsigned int i = 0; i < global_work_size; ++i) {
      if (i % 2 == 0) {
        printf(
            "\n*ptr:0x%llx, lint:0x%llx, ptr:0x%llx, to_local(ptr):0x%llx, "
            "&lint:0x%llx",
            (unsigned long long)output_arr[i * 5], (unsigned long long)output_arr[i * 5 + 1],
            (unsigned long long)output_arr[i * 5 + 2], (unsigned long long)output_arr[i * 5 + 3],
            (unsigned long long)output_arr[i * 5 + 4]);
      } else {
        printf(
            "\n*ptr:0x%llx, gint:0x%llx, ptr:0x%llx, to_global(ptr):0x%llx, "
            "&gint:0x%llx",
            (unsigned long long)output_arr[i * 5], (unsigned long long)output_arr[i * 5 + 1],
            (unsigned long long)output_arr[i * 5 + 2], (unsigned long long)output_arr[i * 5 + 3],
            (unsigned long long)output_arr[i * 5 + 4]);
      }
    }
    printf("\n\n");
  }
  free(output_arr);
  CHECK_RESULT((error_cnt != 0), "Generic Address Space - test2 failed");
}

void OCLGenericAddressSpace::test4(void) {
  const char* kernel_str =
      "\n\
        __global unsigned int gint = 1; \n\
        __kernel void test(__global ulong *results) \n\
        { \n\
            uint tid = get_global_id(0); \n\
            results[tid] = 0; \n\
            unsigned int *ptr; \n\
            __private unsigned int pint = 2; \n\
            if ((tid % 2) == 0) { \n\
                ptr = &pint; \n\
            } \n\
            else { \n\
                ptr = &gint; \n\
            } \n\
            barrier(CLK_GLOBAL_MEM_FENCE); \n\
            if ((tid % 2) == 0) { \n\
                results[tid*5] = *ptr;\n\
                results[tid*5+1] = pint;\n\
                results[tid*5+2] = (ulong)ptr;\n\
                results[tid*5+3] = (ulong)to_private(ptr);\n\
                results[tid*5+4] = (ulong)&pint;\n\
            } \n\
            else { \n\
                results[tid*5] = *ptr;\n\
                results[tid*5+1] = gint;\n\
                results[tid*5+2] = (ulong)ptr;\n\
                results[tid*5+3] = (ulong)to_global(ptr);\n\
                results[tid*5+4] = (ulong)&gint;\n\
            } \n\
        } \n";
  const size_t global_work_size = 2;
  const size_t arrSize = global_work_size * 5;
  cl_ulong* output_arr = (cl_ulong*)malloc(arrSize * sizeof(cl_ulong));
  memset(output_arr, 0, arrSize * sizeof(cl_ulong));
  cl_mem buffer =
      _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, arrSize * sizeof(cl_ulong), 0, &error_);
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
  kernel_ = _wrapper->clCreateKernel(program_, "test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel failed");
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  cl_event evt;

  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                            &global_work_size, NULL, 0, NULL, &evt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[0], CL_TRUE, 0,
                                         sizeof(cl_ulong) * arrSize, output_arr, 1, &evt, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed");
  int error_cnt = 0;
  for (unsigned int i = 0; i < global_work_size; ++i) {
    if (((i % 2 == 0) && (output_arr[i * 5] != 2)) || ((i % 2 == 1) && (output_arr[i * 5] != 1))) {
      ++error_cnt;
    }
  }
  if (error_cnt) {
    printf("\nNumber of wrong results: %d/%d\n\n", error_cnt, (int)global_work_size);
    for (unsigned int i = 0; i < global_work_size; ++i) {
      if (i % 2 == 0) {
        printf(
            "\n*ptr:0x%llx, pint:0x%llx, ptr:0x%llx, to_private(ptr):0x%llx, "
            "&pint:0x%llx",
            (unsigned long long)output_arr[i * 5], (unsigned long long)output_arr[i * 5 + 1],
            (unsigned long long)output_arr[i * 5 + 2], (unsigned long long)output_arr[i * 5 + 3],
            (unsigned long long)output_arr[i * 5 + 4]);
      } else {
        printf(
            "\n*ptr:0x%llx, gint:0x%llx, ptr:0x%llx, to_global(ptr):0x%llx, "
            "&gint:0x%llx",
            (unsigned long long)output_arr[i * 5], (unsigned long long)output_arr[i * 5 + 1],
            (unsigned long long)output_arr[i * 5 + 2], (unsigned long long)output_arr[i * 5 + 3],
            (unsigned long long)output_arr[i * 5 + 4]);
      }
    }
    printf("\n\n");
  }
  free(output_arr);
  CHECK_RESULT((error_cnt != 0), "Generic Address Space - test2 failed");
}

void OCLGenericAddressSpace::test3(void) {
  const char* kernel_str =
      "\n\
        #define TO_LOCAL_FAIL   0x000f0\n\
        #define TO_GLOBAL_FAIL  0x00e00\n\
        #define TO_PRIVATE_FAIL 0x0d000\n\
        #define WRONG_VALUE     0xc0000\n\
        __global unsigned int gint = 1; \n\
        __kernel void test(__global uint *results) \n\
        { \n\
            uint tid = get_global_id(0); \n\
            results[tid] = 0; \n\
            unsigned int *ptr; \n\
            __local unsigned int lint; \n\
            lint = 2; \n\
            __private unsigned int pint = 3; \n\
            switch (tid % 3) \n\
            {\n\
                case 0:\n\
                    ptr = &gint; break; \n\
                case 1:\n\
                    ptr = &lint; break; \n\
                case 2:\n\
                    ptr = &pint; break; \n\
            }\n\
            barrier(CLK_GLOBAL_MEM_FENCE); \n\
            switch (tid % 3) \n\
            {\n\
                case 0:\n\
                    if(to_global(ptr) && (*ptr == 1))\n\
                    {\n\
                        results[tid] = *ptr;\n\
                    }\n\
                    else\n\
                    {\n\
                        if (*ptr != 1) results[tid] = WRONG_VALUE;\n\
                        if(!to_global(ptr)) results[tid] |= TO_GLOBAL_FAIL;\n\
                    }\n\
                    break; \n\
                case 1:\n\
                    if(to_local(ptr) && (*ptr == 2))\n\
                    {\n\
                        results[tid] = *ptr;\n\
                    }\n\
                    else\n\
                    {\n\
                        if (*ptr != 2) results[tid] = WRONG_VALUE;\n\
                        if(!to_local(ptr)) results[tid] |= TO_LOCAL_FAIL;\n\
                    }\n\
                    break; \n\
                case 2:\n\
                    if(to_private(ptr) && (*ptr == 3))\n\
                    {\n\
                        results[tid] = *ptr;\n\
                    }\n\
                    else\n\
                    {\n\
                        if (*ptr != 3) results[tid] = WRONG_VALUE;\n\
                        if(!to_private(ptr)) results[tid] |= TO_PRIVATE_FAIL;\n\
                    }\n\
                    break; \n\
            }\n\
        } \n";
  cl_uint* output_arr = (cl_uint*)malloc(arrSize * sizeof(cl_uint));
  memset(output_arr, 0, arrSize * sizeof(cl_uint));
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
  kernel_ = _wrapper->clCreateKernel(program_, "test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel failed");
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  cl_event evt;
  size_t global_work_size = arrSize;
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                            &global_work_size, NULL, 0, NULL, &evt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[0], CL_TRUE, 0,
                                         sizeof(cl_uint) * arrSize, output_arr, 1, &evt, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed");
  int error_cnt = 0;
  int wrong_values = 0;
  int to_local_error = 0;
  int to_global_error = 0;
  int to_private_error = 0;
  for (unsigned int i = 0; i < arrSize; ++i) {
    switch (i % 3) {
      case 0:
        error_cnt += (output_arr[i] != 1);
        break;
      case 1:
        error_cnt += (output_arr[i] != 2);
        break;
      case 2:
        error_cnt += (output_arr[i] != 3);
        break;
    }
    if (output_arr[i] & WRONG_VALUE) ++wrong_values;
    if (output_arr[i] & TO_LOCAL_FAIL) ++to_local_error;
    if (output_arr[i] & TO_GLOBAL_FAIL) ++to_global_error;
    if (output_arr[i] & TO_PRIVATE_FAIL) ++to_private_error;
  }
  if (error_cnt) {
    printf("\nNumber of wrong results: %d/%d ", error_cnt, (int)arrSize);
    printf(
        "wrong values: %d to_local_error: %d, to_global_error: %d, "
        "to_private_error: %d\n",
        wrong_values, to_local_error, to_global_error, to_private_error);
  }
  free(output_arr);
  CHECK_RESULT((error_cnt != 0), "Generic Address Space - test3 failed");
}

void OCLGenericAddressSpace::test2(void) {
  const char* kernel_str =
      "\n\
        #define TO_LOCAL_FAIL   0x000f0\n\
        #define TO_GLOBAL_FAIL  0x00e00\n\
        #define TO_PRIVATE_FAIL 0x0d000\n\
        #define WRONG_VALUE     0xc0000\n\
        __global unsigned int gint = 1; \n\
        __kernel void test(__global uint *results) \n\
        { \n\
            uint tid = get_global_id(0); \n\
            results[tid] = 0; \n\
            unsigned int *ptr; \n\
            __private unsigned int pint = 2; \n\
            if ((tid % 2) == 0) { \n\
                ptr = &pint; \n\
            } \n\
            else { \n\
                ptr = &gint; \n\
            } \n\
            barrier(CLK_GLOBAL_MEM_FENCE); \n\
            if ((tid % 2) == 0) { \n\
                if (to_private(ptr) && *ptr == 2) {\n\
                    results[tid] = *ptr;\n\
                }\n\
                else {\n\
                    if (*ptr != 2) results[tid] = WRONG_VALUE;\n\
                    if(!to_private(ptr)) results[tid] |= TO_PRIVATE_FAIL;\n\
                }\n\
            } \n\
            else { \n\
                if (to_global(ptr) && *ptr == 1) {\n\
                    results[tid] = *ptr;\n\
                }\n\
                else {\n\
                    if (*ptr != 1) results[tid] = WRONG_VALUE;\n\
                    if(!to_global(ptr)) results[tid] |= TO_GLOBAL_FAIL;\n\
                }\n\
            } \n\
        } \n";
  cl_uint* output_arr = (cl_uint*)malloc(arrSize * sizeof(cl_uint));
  memset(output_arr, 0, arrSize * sizeof(cl_uint));
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
  kernel_ = _wrapper->clCreateKernel(program_, "test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel failed");
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  cl_event evt;
  size_t global_work_size = arrSize;
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                            &global_work_size, NULL, 0, NULL, &evt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[0], CL_TRUE, 0,
                                         sizeof(cl_uint) * arrSize, output_arr, 1, &evt, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed");
  int error_cnt = 0;
  int wrong_values = 0;
  int to_local_error = 0;
  int to_global_error = 0;
  int to_private_error = 0;

  for (unsigned int i = 0; i < arrSize; ++i) {
    if (((i % 2 == 0) && (output_arr[i] != 2)) || ((i % 2 == 1) && (output_arr[i] != 1))) {
      if (output_arr[i] & WRONG_VALUE) ++wrong_values;
      if (output_arr[i] & TO_LOCAL_FAIL) ++to_local_error;
      if (output_arr[i] & TO_GLOBAL_FAIL) ++to_global_error;
      if (output_arr[i] & TO_PRIVATE_FAIL) ++to_private_error;
      ++error_cnt;
    }
  }
  free(output_arr);
  if (error_cnt) {
    printf("\nNumber of wrong results: %d/%d", error_cnt, (int)arrSize);
    printf(
        "wrong values: %d to_local_error: %d, to_global_error: %d, "
        "to_private_error: %d\n",
        wrong_values, to_local_error, to_global_error, to_private_error);
  }
  CHECK_RESULT((error_cnt != 0), "Generic Address Space - test2 failed");
}

void OCLGenericAddressSpace::test1(void) {
  const char* kernel_str =
      "\n\
        #define TO_LOCAL_FAIL   0x000f0\n\
        #define TO_GLOBAL_FAIL  0x00e00\n\
        #define TO_PRIVATE_FAIL 0x0d000\n\
        #define WRONG_VALUE     0xc0000\n\
        __global unsigned int gint1 = 1; \n\
        __global unsigned int gint2 = 2; \n\
        __kernel void test(__global uint *results) \n\
        { \n\
            uint tid = get_global_id(0); \n\
            results[tid] = 0; \n\
            unsigned int *ptr; \n\
            if ((tid % 2) == 0) { \n\
                ptr = &gint2; \n\
            } \n\
            else { \n\
                ptr = &gint1; \n\
            } \n\
            barrier(CLK_GLOBAL_MEM_FENCE); \n\
            if ((tid % 2) == 0) { \n\
                if (to_global(ptr) && *ptr == 2) {\n\
                    results[tid] = *ptr;\n\
                }\n\
                else {\n\
                    if (*ptr != 2) results[tid] = WRONG_VALUE;\n\
                    if(!to_global(ptr)) results[tid] |= TO_GLOBAL_FAIL;\n\
                }\n\
            } \n\
            else { \n\
                if (to_global(ptr) && *ptr == 1) {\n\
                    results[tid] = *ptr;\n\
                }\n\
                else {\n\
                    if (*ptr != 1) results[tid] = WRONG_VALUE;\n\
                    if(!to_global(ptr)) results[tid] |= TO_GLOBAL_FAIL;\n\
                }\n\
            } \n\
        } \n";
  cl_uint* output_arr = (cl_uint*)malloc(arrSize * sizeof(cl_uint));
  memset(output_arr, 0, arrSize * sizeof(cl_uint));
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
  kernel_ = _wrapper->clCreateKernel(program_, "test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel failed");
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  cl_event evt;
  size_t global_work_size = arrSize;
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                            &global_work_size, NULL, 0, NULL, &evt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[0], CL_TRUE, 0,
                                         sizeof(cl_uint) * arrSize, output_arr, 1, &evt, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed");
  int error_cnt = 0;
  int wrong_values = 0;
  int to_local_error = 0;
  int to_global_error = 0;
  int to_private_error = 0;

  for (unsigned int i = 0; i < arrSize; ++i) {
    if (((i % 2 == 0) && (output_arr[i] != 2)) || ((i % 2 == 1) && (output_arr[i] != 1))) {
      if (output_arr[i] & WRONG_VALUE) ++wrong_values;
      if (output_arr[i] & TO_LOCAL_FAIL) ++to_local_error;
      if (output_arr[i] & TO_GLOBAL_FAIL) ++to_global_error;
      if (output_arr[i] & TO_PRIVATE_FAIL) ++to_private_error;
      ++error_cnt;
    }
  }
  free(output_arr);
  if (error_cnt) {
    printf("\nNumber of wrong results: %d/%d", error_cnt, (int)arrSize);
    printf(
        "wrong values: %d to_local_error: %d, to_global_error: %d, "
        "to_private_error: %d\n",
        wrong_values, to_local_error, to_global_error, to_private_error);
  }
  CHECK_RESULT((error_cnt != 0), "Generic Address Space - test1 failed");
}

void OCLGenericAddressSpace::test0(void) {
  const char* kernel_str =
      "\n\
        #define TO_LOCAL_FAIL   0x000f0\n\
        #define TO_GLOBAL_FAIL  0x00e00\n\
        #define TO_PRIVATE_FAIL 0x0d000\n\
        #define WRONG_VALUE     0xc0000\n\
        __global unsigned int gint = 1; \n\
        __kernel void test(__global uint *results) \n\
        { \n\
            uint tid = get_global_id(0); \n\
            results[tid] = 0; \n\
            unsigned int *ptr; \n\
            __local unsigned int lint; \n\
            lint = 2; \n\
            if ((tid % 2) == 0) { \n\
                ptr = &lint; \n\
            } \n\
            else { \n\
                ptr = &gint; \n\
            } \n\
            barrier(CLK_GLOBAL_MEM_FENCE); \n\
            if ((tid % 2) == 0) { \n\
                if (to_local(ptr) && *ptr == 2) {\n\
                    results[tid] = *ptr;\n\
                }\n\
                else {\n\
                    if (*ptr != 2) results[tid] = WRONG_VALUE;\n\
                    if(!to_local(ptr)) results[tid] |= TO_LOCAL_FAIL;\n\
                }\n\
            } \n\
            else { \n\
                if (to_global(ptr) && *ptr == 1) {\n\
                    results[tid] = *ptr;\n\
                }\n\
                else {\n\
                    if (*ptr != 1) results[tid] = WRONG_VALUE;\n\
                    if(!to_global(ptr)) results[tid] |= TO_GLOBAL_FAIL;\n\
                }\n\
            } \n\
        } \n";
  cl_uint* output_arr = (cl_uint*)malloc(arrSize * sizeof(cl_uint));
  memset(output_arr, 0, arrSize * sizeof(cl_uint));
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
  kernel_ = _wrapper->clCreateKernel(program_, "test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel failed");
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&buffers_[0]);
  CHECK_RESULT((error_ != CL_SUCCESS), "clSetKernelArg failed");
  cl_event evt;
  size_t global_work_size = arrSize;
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL,
                                            &global_work_size, NULL, 0, NULL, &evt);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueNDRangeKernel");
  _wrapper->clFinish(cmdQueues_[_deviceId]);
  error_ = _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], buffers_[0], CL_TRUE, 0,
                                         sizeof(cl_uint) * arrSize, output_arr, 1, &evt, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueReadBuffer failed");
  int error_cnt = 0;
  int wrong_values = 0;
  int to_local_error = 0;
  int to_global_error = 0;
  int to_private_error = 0;

  for (unsigned int i = 0; i < arrSize; ++i) {
    if (((i % 2 == 0) && (output_arr[i] != 2)) || ((i % 2 == 1) && (output_arr[i] != 1))) {
      if (output_arr[i] & WRONG_VALUE) ++wrong_values;
      if (output_arr[i] & TO_LOCAL_FAIL) ++to_local_error;
      if (output_arr[i] & TO_GLOBAL_FAIL) ++to_global_error;
      if (output_arr[i] & TO_PRIVATE_FAIL) ++to_private_error;
      ++error_cnt;
    }
  }
  free(output_arr);
  if (error_cnt) {
    printf("\nNumber of wrong results: %d/%d", error_cnt, (int)arrSize);
    printf(
        "wrong values: %d to_local_error: %d, to_global_error: %d, "
        "to_private_error: %d\n",
        wrong_values, to_local_error, to_global_error, to_private_error);
  }
  CHECK_RESULT((error_cnt != 0), "Generic Address Space - test0 failed");
}

unsigned int OCLGenericAddressSpace::close(void) {
  if (kernel_) {
    error_ = _wrapper->clReleaseKernel(kernel_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel failed");
    kernel_ = 0;
  }
  return OCLTestImp::close();
}
