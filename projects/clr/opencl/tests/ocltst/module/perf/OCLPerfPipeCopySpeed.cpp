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

#include "OCLPerfPipeCopySpeed.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <complex>

#include "CL/opencl.h"
#include "Timer.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define KERNEL_CODE(...) #__VA_ARGS__

const static char * strKernel =
{
    KERNEL_CODE(
    \n
        kernel void initPipe(global DATA_TYPE* inBuf, write_only pipe DATA_TYPE outPipe)\n
        {\n
            int gid = get_global_id(0);\n
            write_pipe(outPipe, &inBuf[gid]);\n
        }\n
    \n
        kernel void copyPipe(read_only pipe DATA_TYPE inPipe, write_only pipe DATA_TYPE outPipe)\n
        {\n
            DATA_TYPE tmp;\n
            read_pipe(inPipe, &tmp);\n
            write_pipe(outPipe, &tmp);\n
        }\n
    \n
        kernel void readPipe(read_only pipe DATA_TYPE inPipe, global DATA_TYPE* outBuf)\n
        {\n
            int gid = get_global_id(0);\n
            DATA_TYPE tmp;\n
            read_pipe(inPipe, &tmp);\n
            outBuf[gid] = tmp;\n
        }\n
    \n
        kernel void initPipe_reserve(global DATA_TYPE* inBuf, write_only pipe DATA_TYPE outPipe)\n
        {\n
            int gid = get_global_id(0);\n
            local reserve_id_t resId;\n
            resId = reserve_write_pipe(outPipe, 1);\n
            if (is_valid_reserve_id(resId)) {\n
                write_pipe(outPipe, resId, 0, &inBuf[gid]);\n
                commit_write_pipe(outPipe, resId);\n
            }\n
        }\n
    \n
        kernel void copyPipe_reserve(read_only pipe DATA_TYPE inPipe, write_only pipe DATA_TYPE outPipe)\n
        {\n
            local reserve_id_t resId;\n
            resId = reserve_read_pipe(inPipe, 1);\n
            if (is_valid_reserve_id(resId)) {\n
                DATA_TYPE tmp;\n
                read_pipe(inPipe, resId, 0, &tmp);\n
                commit_read_pipe(inPipe, resId);\n
                resId = reserve_write_pipe(outPipe, 1);\n
                if (is_valid_reserve_id(resId)) {\n
                    write_pipe(outPipe, resId, 0, &tmp);\n
                    commit_write_pipe(outPipe, resId);\n
                }\n
            }\n
        }\n
    \n
        kernel void readPipe_reserve(read_only pipe DATA_TYPE inPipe, global DATA_TYPE* outBuf)\n
        {\n
            int gid = get_global_id(0);\n
            local reserve_id_t resId;\n
            resId = reserve_read_pipe(inPipe, 1);\n
            if (is_valid_reserve_id(resId)) {\n
                DATA_TYPE tmp;\n
                read_pipe(inPipe, resId, 0, &tmp);\n
                commit_read_pipe(inPipe, resId);\n
                outBuf[gid] = tmp;\n
            }\n
        }\n
    \n
        kernel void initPipe_wg(global DATA_TYPE* inBuf, write_only pipe DATA_TYPE outPipe)\n
        {\n
            int gid = get_global_id(0);\n
            local reserve_id_t resId;\n
            resId = work_group_reserve_write_pipe(outPipe, get_local_size(0));\n
            if (is_valid_reserve_id(resId)) {\n
                write_pipe(outPipe, resId, get_local_id(0), &inBuf[gid]);\n
                work_group_commit_write_pipe(outPipe, resId);\n
            }\n
        }\n
    \n
        kernel void copyPipe_wg(read_only pipe DATA_TYPE inPipe, write_only pipe DATA_TYPE outPipe)\n
        {\n
            local reserve_id_t resId;\n
            resId = work_group_reserve_read_pipe(inPipe, get_local_size(0));\n
            if (is_valid_reserve_id(resId)) {\n
                DATA_TYPE tmp;\n
                read_pipe(inPipe, resId, get_local_id(0), &tmp);\n
                work_group_commit_read_pipe(inPipe, resId);\n
                resId = work_group_reserve_write_pipe(outPipe, get_local_size(0));\n
                if (is_valid_reserve_id(resId)) {\n
                    write_pipe(outPipe, resId, get_local_id(0), &tmp);\n
                    work_group_commit_write_pipe(outPipe, resId);\n
                }\n
            }\n
        }\n
    \n
        kernel void readPipe_wg(read_only pipe DATA_TYPE inPipe, global DATA_TYPE* outBuf)\n
        {\n
            int gid = get_global_id(0);\n
            local reserve_id_t resId;\n
            resId = work_group_reserve_read_pipe(inPipe, get_local_size(0));\n
            if (is_valid_reserve_id(resId)) {\n
                DATA_TYPE tmp;\n
                read_pipe(inPipe, resId, get_local_id(0), &tmp);\n
                work_group_commit_read_pipe(inPipe, resId);\n
                outBuf[gid] = tmp;\n
            }\n
        }\n
    \n
\x23 ifdef SUBGROUPS\n
        \x23 pragma OPENCL EXTENSION cl_khr_subgroups : enable\n
        kernel __attribute__((reqd_work_group_size(64,1,1))) void initPipe_sg(global DATA_TYPE* inBuf, write_only pipe DATA_TYPE outPipe)\n
        {\n
            int gid = get_global_id(0);\n
            local reserve_id_t resId;\n
            resId = sub_group_reserve_write_pipe(outPipe, get_sub_group_size());\n
            if (is_valid_reserve_id(resId)) {\n
                write_pipe(outPipe, resId, get_sub_group_local_id(), &inBuf[gid]);\n
                sub_group_commit_write_pipe(outPipe, resId);\n
            }\n
        }\n
    \n
        kernel __attribute__((reqd_work_group_size(64,1,1))) void copyPipe_sg(read_only pipe DATA_TYPE inPipe, write_only pipe DATA_TYPE outPipe)\n
        {\n
            local reserve_id_t resId;\n
            resId = sub_group_reserve_read_pipe(inPipe, get_sub_group_size());\n
            if (is_valid_reserve_id(resId)) {\n
                DATA_TYPE tmp;\n
                read_pipe(inPipe, resId, get_sub_group_local_id(), &tmp);\n
                sub_group_commit_read_pipe(inPipe, resId);\n
                resId = sub_group_reserve_write_pipe(outPipe, get_sub_group_size());\n
                if (is_valid_reserve_id(resId)) {\n
                    write_pipe(outPipe, resId, get_sub_group_local_id(), &tmp);\n
                    sub_group_commit_write_pipe(outPipe, resId);\n
                }\n
            }\n
        }\n
    \n
        kernel __attribute__((reqd_work_group_size(64,1,1))) void readPipe_sg(read_only pipe DATA_TYPE inPipe, global DATA_TYPE* outBuf)\n
        {\n
            int gid = get_global_id(0);\n
            local reserve_id_t resId;\n
            resId = sub_group_reserve_read_pipe(inPipe, get_sub_group_size());\n
            if (is_valid_reserve_id(resId)) {\n
                DATA_TYPE tmp;\n
                read_pipe(inPipe, resId, get_sub_group_local_id(), &outBuf[gid]);\n
                sub_group_commit_read_pipe(inPipe, resId);\n
                outBuf[gid] = tmp;\n
            }\n
        }\n
\x23 endif\n
    \n
    )
};

#define NUM_SIZES 6
// 4KB, 8KB, 64KB, 256KB, 1 MB, 4MB
static const unsigned int Sizes[NUM_SIZES] = {4096, 8192, 65536, 262144, 1048576, 4194304};

#define NUM_TYPES 3
static const char* types[NUM_TYPES] = {"int", "int4", "int16"};
static const unsigned int typeSize[NUM_TYPES] = {4, 16, 64};

#define NUM_TESTS 4

OCLPerfPipeCopySpeed::OCLPerfPipeCopySpeed() { _numSubTests = NUM_TESTS * NUM_SIZES * NUM_TYPES; }

OCLPerfPipeCopySpeed::~OCLPerfPipeCopySpeed() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfPipeCopySpeed::setData(cl_mem buffer) {
  int* mem;
  int dwTypeSize = (int)(typeSize[typeIdx_]) >> 2;
  mem = (int*)_wrapper->clEnqueueMapBuffer(cmd_queue_, buffer, CL_TRUE, CL_MAP_WRITE, 0, bufSize_,
                                           0, NULL, NULL, &error_);
  CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
  for (int i = 0; i < (int)numElements; i++) {
    for (int j = 0; j < dwTypeSize; j++) {
      mem[i * dwTypeSize + j] = i;
    }
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, (void*)mem, 0, NULL, NULL);
  CHECK_RESULT(error_, "clEnqueueUnmapBuffer failed");
  clFinish(cmd_queue_);
}

void OCLPerfPipeCopySpeed::checkData(cl_mem buffer) {
  int* mem;
  int dwTypeSize = (int)(typeSize[typeIdx_]) >> 2;
  char* histo;
  histo = (char*)malloc(numElements * sizeof(char));
  memset(histo, 0, sizeof(char) * numElements);
  mem = (int*)_wrapper->clEnqueueMapBuffer(cmd_queue_, buffer, CL_TRUE, CL_MAP_READ, 0, bufSize_, 0,
                                           NULL, NULL, &error_);
  CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
  int errCnt = 0;
  for (int i = 0; (i < (int)numElements) && (errCnt < 5); i++) {
    int tmp = mem[dwTypeSize * i];
    for (int j = 1; (j < dwTypeSize) && (errCnt < 5); j++) {
      if (mem[i * dwTypeSize + j] != tmp) {
        // BAD DATA!
        printf("BAD DATA at element %d, ref %d, got %d\n", i, tmp, mem[i * dwTypeSize + j]);
        errCnt++;
      }
    }
    if (histo[tmp] == 1) {
      printf("BAD DATA at element %d, val %d already found!\n", i, tmp);
      errCnt++;
    }
    histo[tmp] = 1;
  }
  errCnt = 0;
  for (int i = 0; (i < (int)numElements) && (errCnt < 5); i++) {
    if (histo[i] != 1) {
      printf("BAD DATA at element %d, val not found!\n", i);
      errCnt++;
    }
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, (void*)mem, 0, NULL, NULL);
  CHECK_RESULT(error_, "clEnqueueUnmapBuffer failed");
  clFinish(cmd_queue_);
  free(histo);
}

void OCLPerfPipeCopySpeed::open(unsigned int test, char* units, double& conversion,
                                unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  _crcword = 0;
  conversion = 1.0f;

  cl_device_id device = devices_[deviceId];
  cmd_queue_ = cmdQueues_[_deviceId];

  program_ = 0;
  initPipe_ = 0;
  copyPipe_ = 0;
  readPipe_ = 0;
  srcBuffer_ = 0;
  dstBuffer_ = 0;
  pipe_[0] = 0;
  pipe_[1] = 0;
  failed_ = false;
  subgroupSupport_ = false;

  bufSize_ = Sizes[test % NUM_SIZES];
  typeIdx_ = (test / NUM_SIZES) % NUM_TYPES;
  testIdx_ = test / (NUM_SIZES * NUM_TYPES);

  numIter = NUM_ITER;

  char getVersion[128];
  error_ =
      _wrapper->clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(getVersion), getVersion, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if (getVersion[7] < '2') {
    failed_ = true;
    _errorMsg = "OpenCL 2.0 not supported";
    return;
  }

  srcBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY, bufSize_, NULL, &error_);
  CHECK_RESULT(srcBuffer_ == 0, "clCreateBuffer(srcBuffer) failed");

  numElements = bufSize_ / typeSize[typeIdx_];
  char args[100];

#if defined(CL_VERSION_2_0)
  pipe_[0] = _wrapper->clCreatePipe(context_, CL_MEM_HOST_NO_ACCESS, typeSize[typeIdx_],
                                    numElements, NULL, &error_);
  CHECK_RESULT(pipe_[0] == 0, "clCreatePipe(pipe_[0]) failed");

  pipe_[1] = _wrapper->clCreatePipe(context_, CL_MEM_HOST_NO_ACCESS, typeSize[typeIdx_],
                                    numElements, NULL, &error_);
  CHECK_RESULT(pipe_[1] == 0, "clCreatePipe(pipe_[1]) failed");

  char charbuf[1024];
  size_t retsize;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1024, charbuf, &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  char* p = strstr(charbuf, "cl_khr_subgroups");
  if (p) {
    subgroupSupport_ = true;
    SNPRINTF(args, sizeof(args), "-cl-std=CL2.0 -D DATA_TYPE=%s -D SUBGROUPS", types[typeIdx_]);
  } else {
    if (test >= (NUM_SIZES * NUM_TYPES * 3)) {
      // No support for subgroups, so skip these tests
      failed_ = true;
      _errorMsg = "Subgroup extension not supported";
      return;
    }
    SNPRINTF(args, sizeof(args), "-cl-std=CL2.0 -D DATA_TYPE=%s", types[typeIdx_]);
  }
#endif

  dstBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bufSize_, NULL, &error_);
  CHECK_RESULT(dstBuffer_ == 0, "clCreateBuffer(dstBuffer) failed");

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &device, args, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    printf("\nerror: %d\n", error_);
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);

    CHECK_RESULT(0, "clBuildProgram failed");
  }
  if (testIdx_ == 0) {
    initPipe_ = _wrapper->clCreateKernel(program_, "initPipe", &error_);
    CHECK_RESULT(initPipe_ == 0, "clCreateKernel(initPipe) failed");
    copyPipe_ = _wrapper->clCreateKernel(program_, "copyPipe", &error_);
    CHECK_RESULT(copyPipe_ == 0, "clCreateKernel(copyPipe) failed");
    readPipe_ = _wrapper->clCreateKernel(program_, "readPipe", &error_);
    CHECK_RESULT(readPipe_ == 0, "clCreateKernel(readPipe) failed");
    testName_ = "r/w";
  } else if (testIdx_ == 1) {
    initPipe_ = _wrapper->clCreateKernel(program_, "initPipe_reserve", &error_);
    CHECK_RESULT(initPipe_ == 0, "clCreateKernel(initPipe) failed");
    copyPipe_ = _wrapper->clCreateKernel(program_, "copyPipe_reserve", &error_);
    CHECK_RESULT(copyPipe_ == 0, "clCreateKernel(copyPipe) failed");
    readPipe_ = _wrapper->clCreateKernel(program_, "readPipe_reserve", &error_);
    CHECK_RESULT(readPipe_ == 0, "clCreateKernel(readPipe) failed");
    numIter = 10;  // Limit iteration count because this test is very slow
    testName_ = "r/w w/ reserve";
  } else if (testIdx_ == 2) {
    initPipe_ = _wrapper->clCreateKernel(program_, "initPipe_wg", &error_);
    CHECK_RESULT(initPipe_ == 0, "clCreateKernel(initPipe) failed");
    copyPipe_ = _wrapper->clCreateKernel(program_, "copyPipe_wg", &error_);
    CHECK_RESULT(copyPipe_ == 0, "clCreateKernel(copyPipe) failed");
    readPipe_ = _wrapper->clCreateKernel(program_, "readPipe_wg", &error_);
    CHECK_RESULT(readPipe_ == 0, "clCreateKernel(readPipe) failed");
    testName_ = "wg r/w w/ reserve";
  } else if (testIdx_ == 3) {
    initPipe_ = _wrapper->clCreateKernel(program_, "initPipe_sg", &error_);
    CHECK_RESULT(initPipe_ == 0, "clCreateKernel(initPipe) failed");
    copyPipe_ = _wrapper->clCreateKernel(program_, "copyPipe_sg", &error_);
    CHECK_RESULT(copyPipe_ == 0, "clCreateKernel(copyPipe) failed");
    readPipe_ = _wrapper->clCreateKernel(program_, "readPipe_sg", &error_);
    CHECK_RESULT(readPipe_ == 0, "clCreateKernel(readPipe) failed");
    testName_ = "sg r/w w/ reserve";
  } else {
    CHECK_RESULT(1, "Invalid test index!");
  }
  setData(srcBuffer_);
}

void OCLPerfPipeCopySpeed::run(void) {
  if (failed_) return;
  CPerfCounter timer;
  size_t global_work_size[1] = {(size_t)numElements};
  size_t local_work_size[1] = {64};

  error_ = _wrapper->clSetKernelArg(initPipe_, 0, sizeof(cl_mem), (void*)&srcBuffer_);
  error_ = _wrapper->clSetKernelArg(initPipe_, 1, sizeof(cl_mem), (void*)&pipe_[0]);
  // Warm up
  error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, initPipe_, 1, NULL,
                                            (const size_t*)global_work_size,
                                            (const size_t*)local_work_size, 0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");

  error_ = _wrapper->clSetKernelArg(copyPipe_, 0, sizeof(cl_mem), (void*)&pipe_[0]);
  error_ = _wrapper->clSetKernelArg(copyPipe_, 1, sizeof(cl_mem), (void*)&pipe_[1]);
  error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, copyPipe_, 1, NULL,
                                            (const size_t*)global_work_size,
                                            (const size_t*)local_work_size, 0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");

  error_ = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(error_, "clFinish failed");

  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < numIter; i++) {
    error_ = _wrapper->clSetKernelArg(copyPipe_, 0, sizeof(cl_mem), (void*)&pipe_[(i + 1) % 2]);
    error_ = _wrapper->clSetKernelArg(copyPipe_, 1, sizeof(cl_mem), (void*)&pipe_[i % 2]);
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, copyPipe_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);

    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  }
  error_ = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(error_, "clFinish failed");

  timer.Stop();

  // pipe[(numIter-1)%2 has the data
  error_ = _wrapper->clSetKernelArg(readPipe_, 0, sizeof(cl_mem), (void*)&pipe_[(numIter - 1) % 2]);
  error_ = _wrapper->clSetKernelArg(readPipe_, 1, sizeof(cl_mem), (void*)&dstBuffer_);
  error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, readPipe_, 1, NULL,
                                            (const size_t*)global_work_size,
                                            (const size_t*)local_work_size, 0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueNDRangeKernel(readPipe) failed");
  error_ = _wrapper->clFinish(cmd_queue_);
  checkData(dstBuffer_);
  double sec = timer.GetElapsedTime();

  // Pipe copy total bandwidth in GB/s
  double perf = 2. * ((double)bufSize_ * numIter * (double)(1e-09)) / sec;

  _perfInfo = (float)perf;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " %17s (%8d bytes) block size: %2d i:%4d (GB/s) ", testName_.c_str(),
           bufSize_, typeSize[typeIdx_], numIter);
  testDescString = buf;
}

unsigned int OCLPerfPipeCopySpeed::close(void) {
  if (srcBuffer_) {
    error_ = _wrapper->clReleaseMemObject(srcBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(srcBuffer_) failed");
  }
  if (pipe_[0]) {
    error_ = _wrapper->clReleaseMemObject(pipe_[0]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(pipe_[0]) failed");
  }
  if (pipe_[1]) {
    error_ = _wrapper->clReleaseMemObject(pipe_[1]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(pipe_[1]) failed");
  }
  if (dstBuffer_) {
    error_ = _wrapper->clReleaseMemObject(dstBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(dstBuffer_) failed");
  }

  return OCLTestImp::close();
}
