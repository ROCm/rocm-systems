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

/*******************************************************************************
 * Kernel Throughput
 *
 *
 *
 *
 *
 *
 ******************************************************************************/

#ifndef _OCL_KernelThroughput_H_
#define _OCL_KernelThroughput_H_

#ifdef WIN32
#include "xmmintrin.h"
#endif

#include "OCLTestImp.h"
// #include <sstream>
// #define WIN32_LEAN_AND_MEAN //Restricts windows.h to include only the core
// API. #include "windows.h" #undef Yield #include <process.h> #include
//<xmmintrin.h> #include <emmintrin.h> #include <pmmintrin.h>

#define LARGE_INT long long
#define UNSIGNED_LARGE_INT unsigned long long
#define MAX_LOOP_ITER 10
typedef cl_float4 float4;
typedef void (*CPUKernel)(__m128*, __m128*, unsigned int);

class OCLPerfKernelThroughput : public OCLTestImp {
 public:
  OCLPerfKernelThroughput();
  virtual ~OCLPerfKernelThroughput();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void genShaderMadds();
  void genShaderMatrixMultiply();
  void checkData();
  // void allocateBuffers();
  void launchKernel();

  // test parameters
  int kernelTypeIdx_;
  int memPathIdx_;
  int numElementsIdx_;
  int workSizeIdx_;
  float gold_;
  double _reqDataSize;
  bool _dataSizeTooBig;

  // device attributes
  cl_uint maxComputeUnits_;
  cl_uint maxClockFrequency_;

  LARGE_INT numComputeUnits_;
  LARGE_INT numWorkGroupsPerComputeUnit_;
  LARGE_INT numThreads_;
  cl_uint work_dim_;
  size_t* global_work_size_;
  size_t* local_work_size_;

  // opencl objects
  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_program program_;
  cl_kernel kernel_;
  cl_int error_;

  // buffer sizes

  // kernel-specific values
  int flopsPerByte_;
  int matrixDim1_, matrixDim2_;

  // buffers
  size_t input1BufferSize_;
  size_t input2BufferSize_;
  size_t output1BufferSize_;
  cl_mem input1Buffer_;
  cl_mem input2Buffer_;
  cl_mem output1Buffer_;
  float* input1Ptr_;
  float* input2Ptr_;
  float* output1Ptr_;

  // performance results
  float bandwidth_;      // GB/s
  float gflops_;         // GFlop/s
  float avgKernelTime_;  // microseconds
};

#endif  // _OCL_KernelThroughput_H_
