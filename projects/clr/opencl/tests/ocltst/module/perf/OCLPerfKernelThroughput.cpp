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

#include "OCLPerfKernelThroughput.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <sstream>

#include "CL/cl.h"
#include "Timer.h"

#define DO_GPU_KERNELS 1

#if 0
#define ENTER(X) printf("Entering %s\n", X);
#define EXIT(X) printf("Exiting  %s\n", X);
#define PKT(X) X
#else
#define ENTER(X)
#define EXIT(X)
#define PKT(X)
#endif

// work with multiples of 128
#define ROUND_MULT(VAL, MULT) ((VAL / MULT) * MULT)
/*
int roundUp( int numToRound, int multiple)
{
    int r = numToRound % multiple;
    if (r == 0)
    {
        return numToRound;
    } else {
        return numToRound + multiple - remainder;
    }
}
*/
// quiety warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

#define WORK_GROUP_SIZE 256

/*******************************************************************************
 * Enumerated Types for Tests
 ******************************************************************************/

// memory operations
const LARGE_INT numKernelTypes = 2;
static const char* kernelType[numKernelTypes] = {"MatMul", "Madds"};

// source/read memory locations
const LARGE_INT numMemPaths = 2;
static const char* memPath[numMemPaths] = {"Host", "Device"};

// buffer size
const LARGE_INT numNumElements = 12;  // 15;
static const LARGE_INT numElements[numNumElements] = {
    4,       16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304,
    16777216  //,
              // 67108864,
              // 268435456
};

// flops/byte
const LARGE_INT numWorkSizes = 5;
static const LARGE_INT workSize[numWorkSizes] = {1, 4, 16, 64, 256};

const float initFloat = 0.001f;
const float zeroFloat = 0.0f;

#define WORK_GROUP_SIZE 256

/*******************************************************************************
 * Write the Matrix Multiply Shader Kernel
 ******************************************************************************/
void OCLPerfKernelThroughput::genShaderMatrixMultiply() {
  ENTER("genShaderMatrixMultiply");

  std::stringstream ss;
  ss.clear();
#if 0
    printf("%ix%i * %ix%i = %ix%i:\n",
            matrixDim1_, matrixDim2_,
            matrixDim2_, matrixDim1_,
            matrixDim1_, matrixDim1_
            );
#endif
  ss << "#define BLOCK_SIZE 16\n"
        "#define HA "
     << matrixDim1_
     << "\n"
        "#define WA "
     << matrixDim2_
     << "\n"
        "#define HB WA\n"
        "#define WB HA\n"
        "#define HC HA\n"
        "#define WC WB\n"
        "__kernel void\n"
        "__attribute__((reqd_work_group_size(16,16,1)))\n"
        "kernel1(\n"
        "       __global float * restrict C,\n"
        "       __global float * restrict A,\n"
        "       __global float * restrict B )\n"
        "{\n"
        "   int bx = get_group_id(0);\n"
        "   int by = get_group_id(1);\n"
        "   int tx = get_local_id(0);\n"
        "   int ty = get_local_id(1);\n"
        "   int aBegin = WA * BLOCK_SIZE * by;\n"
        "   int aEnd   = aBegin + WA - 1;\n"
        "   int aStep  = BLOCK_SIZE;\n"
        "   int bBegin = BLOCK_SIZE * bx;\n"
        "   int bStep  = BLOCK_SIZE * WB;\n"
        "   __private float c = 0.f;\n"
        "   __local float localA[BLOCK_SIZE][BLOCK_SIZE];\n"
        "   __local float localB[BLOCK_SIZE][BLOCK_SIZE];\n"
        "   for (\n"
        "           int a = aBegin, b = bBegin;\n"
        "           a <= aEnd;\n"
        "           a += aStep, b += bStep)\n"
        "   {\n"
        "       localA[ty][tx] = (get_global_id(0) < WA && get_global_id(1) < "
        "HA) ? A[a + WA * ty + tx] : 0;\n"
        "       localB[ty][tx] = (get_global_id(0) < WB && get_global_id(1) < "
        "HB) ? B[b + WB * ty + tx] : 0;\n"
        "       barrier(CLK_LOCAL_MEM_FENCE);\n"
        "       for (int k = 0; k < BLOCK_SIZE; ++k)\n"
        "           c += localA[ty][k] * localB[k][tx];\n"
        "       barrier(CLK_LOCAL_MEM_FENCE);\n"
        "   }\n"
        "   int cIdx = WB * BLOCK_SIZE * by + BLOCK_SIZE * bx + WB * ty + tx;\n"
        "   if (get_global_id(0) < WC && get_global_id(1) < WC)\n"
        "   {\n"
        "       C[cIdx] = c;\n"
        "   }\n"
        "}\n";

  shader_ = ss.str();
  gold_ = 0.f;
  for (int i = 0; i < matrixDim2_; i++) gold_ += initFloat * initFloat;
  // gold_ = initFloat * initFloat * matrixDim2_;
  // printf("shader:\n%s\n", shader_.c_str());
  // printf("gold_: %f\n", gold_);
  EXIT("genShaderMatrixMultiply");
}

/*******************************************************************************
 * Write the Madds Shader Kernel
 ******************************************************************************/
void OCLPerfKernelThroughput::genShaderMadds() {
  ENTER("genShaderMadds");

  int flopLoopIter = 2 * (flopsPerByte_ * 4 * 4) / 16;  // bytes, flops

  std::stringstream ss;
  ss.clear();
  float a, b;

  ss <<  // begin kernel
      "__kernel void\n"
      "__attribute__((reqd_work_group_size("
     << 256
     << ",1,1)))\n"
        "kernel1(\n"
        "   __global float4 * restrict input,\n"
        "   __global float4 * restrict output )\n"
        "{\n";

  // begin loop
  ss << "   for ( uint idx = get_global_id(0);\n"
        "         idx < "
     << numElements[numElementsIdx_]
     << ";\n"
        "         idx += get_global_size(0) )\n"
        "   {\n";

  // do load
  ss << "       float4 prefetch = input[ idx ];\n"
        "       float a0 = prefetch.x;\n"
        "       float a1 = prefetch.y;\n"
        "       float a2 = prefetch.z;\n"
        "       float a3 = prefetch.w;\n"
        "       float b0 = a0;\n"
        "       float b1 = a1;\n"
        "       float b2 = a2;\n"
        "       float b3 = a3;\n";
  a = initFloat;
  b = a;

  // do math
  for (int i = 0; i < flopLoopIter; i++) {
    ss << "       a0 += b3*b1;\n"
          "       a1 += b0*b2;\n"
          "       a2 += b1*b3;\n"
          "       a3 += b2*b0;\n"
          "       b0 += a3*a1;\n"
          "       b1 += a0*a2;\n"
          "       b2 += a1*a3;\n"
          "       b3 += a2*a0;\n";
    // printf("a += b*b; %f += %f*%f\n", a, b, b);
    a += b * b;
    // printf("b += a*a; %f += %f*%f\n", b, a, a);
    b += a * a;
  }

  // do write or accumulate
  ss << "       __private float4 tmp;\n"
        "       tmp.x = b0;\n"
        "       tmp.y = b1;\n"
        "       tmp.z = b2;\n"
        "       tmp.w = b3;\n"
        "       output[ idx ] = tmp;\n";
  gold_ = b;
  // printf("GPU gold_ Tmp: %f\n", gold_);

  // end loop
  ss << "   } // end loop\n";
  // end kernel
  ss << " } // end kernel\n\n";

  shader_ = ss.str();
  // printf("shader:\n%s\n", shader_.c_str());
  // printf("gold_: %f\n", gold_);
  EXIT("genShaderMadds");
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

/*******************************************************************************
 * Constructor
 ******************************************************************************/
OCLPerfKernelThroughput::OCLPerfKernelThroughput() {
  ENTER("constructor");
  _numSubTests = numKernelTypes * numMemPaths * numNumElements * numWorkSizes;

  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  context_ = 0;

  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
    // Get last for default
    platform = platforms[numPlatforms - 1];
    for (unsigned i = 0; i < numPlatforms; ++i) {
      char pbuf[100];
      error_ =
          _wrapper->clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, NULL);
      num_devices = 0;
      /* Get the number of requested devices */
      error_ = _wrapper->clGetDeviceIDs(platforms[i], type_, 0, NULL, &num_devices);
      // Runtime returns an error when no GPU devices are present
      // instead of just returning 0 devices
      // CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");
      // Choose platform with GPU devices
      if (num_devices > 0) {
        // printf("NumDevices: %i\n", num_devices);
        platform = platforms[i];
        break;
      }
    }
    delete platforms;
  }

  /*
   * If we could find our platform, use it, else die.
   */
  CHECK_RESULT(platform == 0, "Couldn't find AMD platform, cannot proceed");

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  CHECK_RESULT(devices == 0, "no devices");

  /* Get the requested device */
  error_ = _wrapper->clGetDeviceIDs(platform, type_, num_devices, devices, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");

  CHECK_RESULT(_deviceId >= num_devices, "Requested deviceID not available");
  device = devices[_deviceId];

  // get gpu speed
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                     sizeof(maxClockFrequency_), &maxClockFrequency_, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits_),
                                     &maxComputeUnits_, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if (maxComputeUnits_ > 8) {
    // printf("%i CUs reported; assuming 8 instead.", maxComputeUnits_);
    maxComputeUnits_ = 8;
  }
  // printf("Compute Units: %i\n", maxComputeUnits_);

  // printf("Subtests: %i\n", _numSubTests);

  // create context
  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  char charbuf[1024];
  size_t retsize;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1024, charbuf, &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  if (context_) {
    error_ = _wrapper->clReleaseContext(context_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }

  cl_uint tmp;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(tmp), &tmp, NULL);
  CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  // printf("NumComputeUnits: %u\n", tmp);
  maxComputeUnits_ = static_cast<LARGE_INT>(tmp);
  // printf("NumComputeUnits: %lld\n", maxComputeUnits_);
  EXIT("constructor");
}

OCLPerfKernelThroughput::~OCLPerfKernelThroughput() {}

/*******************************************************************************
 * Open - initializes test, compile GPU kernel
 ******************************************************************************/
void OCLPerfKernelThroughput::open(unsigned int test, char* units, double& conversion,
                                   unsigned int deviceId) {
  ENTER("open");
  /***********************************************************
   * select subtest
   **********************************************************/
  int testIdx = test + numKernelTypes * numMemPaths * numNumElements * numWorkSizes;
  memPathIdx_ = testIdx % numMemPaths;
  testIdx /= numMemPaths;
  numElementsIdx_ = testIdx % numNumElements;
  testIdx /= numNumElements;
  workSizeIdx_ = testIdx % numWorkSizes;
  testIdx /= numWorkSizes;
  kernelTypeIdx_ = testIdx % numKernelTypes;
  testIdx /= numKernelTypes;

  // float md1;

  // kernel values
  switch (kernelTypeIdx_) {
    case 0:  // Matrix Multiply
      // md1 = sqrt(1.f*numElements[numElementsIdx_]);
      // printf("MD1: sqrt(%f) = %f\n", 1.f*numElements[numElementsIdx_],md1);
      matrixDim1_ = static_cast<int>(sqrt(1.f * numElements[numElementsIdx_]));
      matrixDim2_ = matrixDim1_ * (int)workSize[workSizeIdx_];
      genShaderMatrixMultiply();
      work_dim_ = 2;
      global_work_size_ = new size_t[work_dim_];
      global_work_size_[0] =
          ((matrixDim1_ - 1) / 16 + 1) * 16;  // matrixDim1_ < 16 ? 16 : matrixDim1_;
      global_work_size_[1] = global_work_size_[0];
      local_work_size_ = new size_t[work_dim_];
      local_work_size_[0] = 16;
      local_work_size_[1] = local_work_size_[0];
      /*
      printf("Global: %ix%i; Local: %ix%i; Matrix: %ix%i\n",
              global_work_size_[0],
              global_work_size_[1],
              local_work_size_[0],
              local_work_size_[1],
              matrixDim1_,
              matrixDim2_
              );
      */
      input1BufferSize_ = static_cast<size_t>(matrixDim1_ * matrixDim2_ * sizeof(float));
      input2BufferSize_ = static_cast<size_t>(matrixDim2_ * matrixDim1_ * sizeof(float));
      output1BufferSize_ = static_cast<size_t>(matrixDim1_ * matrixDim1_ * sizeof(float));
      _reqDataSize = (1.0 * matrixDim1_ * matrixDim2_ * sizeof(float)) +
                     (1.0 * matrixDim2_ * matrixDim1_ * sizeof(float)) +
                     (1.0 * matrixDim1_ * matrixDim1_ * sizeof(float));
      break;
    case 1:                                         // Flops/Byte
      flopsPerByte_ = (int)workSize[workSizeIdx_];  // for kernelType == 0
      genShaderMadds();
      numWorkGroupsPerComputeUnit_ = 32;  // TODO
      numThreads_ = numWorkGroupsPerComputeUnit_ * maxComputeUnits_ * WORK_GROUP_SIZE;
      work_dim_ = 1;
      global_work_size_ = new size_t[work_dim_];
      local_work_size_ = new size_t[work_dim_];
      global_work_size_[0] = numThreads_;
      local_work_size_[0] = WORK_GROUP_SIZE;
      input1BufferSize_ = static_cast<size_t>(numElements[numElementsIdx_] * sizeof(float4));
      input2BufferSize_ = 0;
      output1BufferSize_ = static_cast<size_t>(numElements[numElementsIdx_] * sizeof(float4));
      _reqDataSize = 2.0 * numElements[numElementsIdx_] * sizeof(float4);
      break;
  }

  PKT(printf("Test Parameters:\n"
             "\tkernelTypeIdx: %i\n"
             "\tmemPathIdx: %i\n"
             "\tnumElementsIdx: %i\n"
             "\tworkSizeIdx: %i\n"
             "\n\n",
             kernelTypeIdx_, memPathIdx_, numElementsIdx_, workSizeIdx_);)

  /***********************************************************
   * get context and queue
   **********************************************************/
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  _crcword = 0;
  conversion = 1.0;
  _deviceId = deviceId;

  context_ = 0;
  cmd_queue_ = 0;
  program_ = 0;
  kernel_ = 0;
  input1Buffer_ = 0;
  output1Buffer_ = 0;
  _errorFlag = false;  // Reset error code so a single error
                       // doesn't prevent other subtests from running
  _errorMsg = "";

  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");

    platform = platforms[_platformIndex];
    char pbuf[100];
    error_ = _wrapper->clGetPlatformInfo(platforms[_platformIndex], CL_PLATFORM_VENDOR,
                                         sizeof(pbuf), pbuf, NULL);
    num_devices = 0;
    /* Get the number of requested devices */
    error_ = _wrapper->clGetDeviceIDs(platforms[_platformIndex], type_, 0, NULL, &num_devices);
    // Runtime returns an error when no GPU devices are present
    // instead of just returning 0 devices
    // CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");
    delete platforms;
  }

  /*
   * If we could find our platform, use it, else die.
   */
  CHECK_RESULT(platform == 0, "Couldn't find AMD platform, cannot proceed");

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  CHECK_RESULT(devices == 0, "no devices");

  /*
   * Get the requested device
   */
  error_ = _wrapper->clGetDeviceIDs(platform, type_, num_devices, devices, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");

  device = devices[0];

  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, CL_QUEUE_PROFILING_ENABLE, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  // Global memory size
  cl_ulong _maxMemoryAllocationSize;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
                                     &_maxMemoryAllocationSize, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs(CL_DEVICE_GLOBAL_MEM_SIZE) failed");
#if 0
    printf("Buffer Sizes: %i %i %i = %f\n",
            input1BufferSize_,
            input2BufferSize_,
            output1BufferSize_,
            _reqDataSize);
#endif
  _dataSizeTooBig = (_reqDataSize > _maxMemoryAllocationSize);
  if (_dataSizeTooBig) {
    // printf("DATA TOO LARGE FOR DEVICE !!!");
    return;
  }

  // create kernel
  char* tmp = (char*)shader_.c_str();
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&tmp, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  std::string args;
  args.clear();
  error_ = _wrapper->clBuildProgram(program_, 1, &device, args.c_str(), NULL, NULL);
  if (error_ != CL_SUCCESS) {
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);
    CHECK_RESULT(0, "clBuildProgram failed");
  }
  kernel_ = _wrapper->clCreateKernel(program_, "kernel1", &error_);
  CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");

  /***********************************************************
   * Allocate GPU Memory
   **********************************************************/
  cl_mem_flags inputBufferFlags = 0;
  cl_mem_flags outputBufferFlags = 0;

  // choose gpu source buffer type
  switch (memPathIdx_) {
    case 0:  // host memory
      // printf("Allocating Host Memories\n");
      // allocate "device" memory
      inputBufferFlags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
      outputBufferFlags = CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;
      input1Buffer_ =
          _wrapper->clCreateBuffer(context_, inputBufferFlags, input1BufferSize_, NULL, &error_);
      CHECK_RESULT(input1Buffer_ == 0, "clCreateBuffer Input failed");
      if (input1Buffer_ == 0) printf("Error: %i\n", error_);
      if (input2BufferSize_) {
        input2Buffer_ =
            _wrapper->clCreateBuffer(context_, inputBufferFlags, input2BufferSize_, NULL, &error_);
        CHECK_RESULT(input2Buffer_ == 0, "clCreateBuffer Input failed");
      }
      output1Buffer_ =
          _wrapper->clCreateBuffer(context_, outputBufferFlags, output1BufferSize_, NULL, &error_);
      CHECK_RESULT(output1Buffer_ == 0, "clCreateBuffer Input failed");
      if (output1Buffer_ == 0) printf("Error: %i\n", error_);

      // map host memory
      input1Ptr_ =
          (float*)_wrapper->clEnqueueMapBuffer(cmd_queue_, input1Buffer_, true, CL_MAP_WRITE, 0,
                                               input1BufferSize_, 0, NULL, NULL, &error_);
      if (input2BufferSize_) {
        input2Ptr_ =
            (float*)_wrapper->clEnqueueMapBuffer(cmd_queue_, input2Buffer_, true, CL_MAP_WRITE, 0,
                                                 input2BufferSize_, 0, NULL, NULL, &error_);
      }
      output1Ptr_ =
          (float*)_wrapper->clEnqueueMapBuffer(cmd_queue_, output1Buffer_, true, CL_MAP_READ, 0,
                                               output1BufferSize_, 0, NULL, NULL, &error_);
      _wrapper->clFinish(cmd_queue_);
      break;

    case 1:  // device memory
      // printf("Allocating Device Memories\n");
      // allocate device memory
      inputBufferFlags = CL_MEM_READ_WRITE;
      outputBufferFlags = CL_MEM_READ_WRITE;
      input1Buffer_ =
          _wrapper->clCreateBuffer(context_, inputBufferFlags, input1BufferSize_, NULL, &error_);
      CHECK_RESULT(input1Buffer_ == 0, "clCreateBuffer Input failed");
      if (input2BufferSize_) {
        input2Buffer_ =
            _wrapper->clCreateBuffer(context_, inputBufferFlags, input2BufferSize_, NULL, &error_);
        CHECK_RESULT(input2Buffer_ == 0, "clCreateBuffer Input failed");
      }
      output1Buffer_ =
          _wrapper->clCreateBuffer(context_, outputBufferFlags, output1BufferSize_, NULL, &error_);
      CHECK_RESULT(output1Buffer_ == 0, "clCreateBuffer Input failed");
      // printf("\tDone Allocating Device Memory\n");

      // allocate host memory
      input1Ptr_ = new float[input1BufferSize_ / sizeof(float)];
      if (input2BufferSize_) {
        input2Ptr_ = new float[input2BufferSize_ / sizeof(float)];
      }
      output1Ptr_ = new float[output1BufferSize_ / sizeof(float)];
      // printf("\tDone Allocating Host Memory\n");

      break;
    default:
      CHECK_RESULT(1, "Invalid Memory Path Idx");
      // invalid
  }
  for (unsigned int i = 0; i < input1BufferSize_ / sizeof(float); i++) {
    input1Ptr_[i] = initFloat;
  }
  for (unsigned int i = 0; i < input2BufferSize_ / sizeof(float); i++) {
    input2Ptr_[i] = initFloat;
  }
  for (unsigned int i = 0; i < output1BufferSize_ / sizeof(float); i++) {
    output1Ptr_[i] = zeroFloat;
  }

#if 0
    printf("Allocating GPU: %.0fMB, %.0fMB\n",
            static_cast<float>(1.f*input1BufferSize_/1024.f/1024.f),
            static_cast<float>(1.f*output1BufferSize_/1024.f/1024.f));
    input1Buffer_ = _wrapper->clCreateBuffer(
            context_, inputBufferFlags, input1BufferSize_, NULL, &error_);
    CHECK_RESULT(input1Buffer_ == 0, "clCreateBuffer Input failed");
    output1Buffer_ = _wrapper->clCreateBuffer(
            context_, outputBufferFlags, output1BufferSize_, NULL, &error_);
    CHECK_RESULT(output1Buffer_ == 0, "clCreateBuffer Output failed");
    error_ = /*_wrapper->*/clEnqueueFillBuffer(
            cmd_queue_, input1Buffer_, &initFloat, sizeof(initFloat),
            0, input1BufferSize_, 0, NULL, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueFillBuffer failed");
    error_ = /*_wrapper->*/clEnqueueFillBuffer(
            cmd_queue_, output1Buffer_, &zeroFloat, sizeof(zeroFloat),
            0, output1BufferSize_, 0, NULL, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueFillBuffer failed");

    /***********************************************************
     * Set Kernel Args
     **********************************************************/
    error_ = _wrapper->clSetKernelArg(
                kernel_, 0, sizeof(input1Buffer_), (void *) &input1Buffer_);
    CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg failed");
    error_ = _wrapper->clSetKernelArg(
                kernel_, 1, sizeof(output1Buffer_), (void *) &output1Buffer_);
    CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg failed");
#endif

  EXIT("open");
}

/*******************************************************************************
 * Run - execute full test once and return performance
 ******************************************************************************/
void OCLPerfKernelThroughput::run(void) {
  ENTER("run");
  CPerfCounter timer;
  if (!_dataSizeTooBig) {
    // set kernel args
#if 1
    switch (kernelTypeIdx_) {
      case 0:  // Matrix Multiply
        error_ =
            _wrapper->clSetKernelArg(kernel_, 0, sizeof(output1Buffer_), (void*)&output1Buffer_);
        CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg failed");
        error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(input1Buffer_), (void*)&input1Buffer_);
        CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg failed");
        error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(input2Buffer_), (void*)&input2Buffer_);
        CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg failed");
        break;
      case 1:  // Flops/Byte
        error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(input1Buffer_), (void*)&input1Buffer_);
        CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg failed");
        error_ =
            _wrapper->clSetKernelArg(kernel_, 1, sizeof(output1Buffer_), (void*)&output1Buffer_);
        CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg failed");
        break;
    }
#endif
    launchKernel();
    timer.Reset();
    timer.Start();
    for (int i = 0; i < MAX_LOOP_ITER; i++) {
      launchKernel();
    }
    timer.Stop();
  }  // data not too large
  double totalSec = _dataSizeTooBig ? 1 : timer.GetElapsedTime();
  // printf("Total Time: %f seconds\n", totalSec);
  // printf("Average Kernel Time: %f seconds\n", totalSec / MAX_LOOP_ITER);

  // analyze performance
  avgKernelTime_ = (float)(totalSec / MAX_LOOP_ITER * 1000000);  // microseconds
  double flopCount;
  switch (kernelTypeIdx_) {
    case 0:  // Matrix Multiply
      flopCount = (2.0 * matrixDim1_ * matrixDim1_ * matrixDim2_);
      // printf("FlopCount = 2*%i*%i*%i=%f\n",
      // matrixDim1_,matrixDim1_,matrixDim2_,flopCount);
      bandwidth_ = (float)(1.f * _reqDataSize / 1024.f / 1024.f / 1024.f) * 1000000.f /
                   avgKernelTime_;  // GB/s
      gflops_ = (float)(1000000.f * flopCount / avgKernelTime_ / 1000000000.0);
      break;
    case 1:  // Madds
      flopCount = _reqDataSize * flopsPerByte_;
      bandwidth_ = (float)(1.f * _reqDataSize / 1024.f / 1024.f / 1024.f) * 1000000.f /
                   avgKernelTime_;  // GB/s
      gflops_ = bandwidth_ * flopsPerByte_;
      break;
  }
  if (_dataSizeTooBig) {
    printf("REQUESTED DATA SIZE EXCEEDS GLOBAL MEMORY !!!\n");
    bandwidth_ = 0;
    gflops_ = 0;
    avgKernelTime_ = 0;
  }
  // here print out details
  char buf[512];
  int bytesWritten;
  bytesWritten = SNPRINTF(buf, sizeof(buf),
                          "Kernel:%7s; "
                          "Work:%4i; "
                          "Buff:%11.0f; "
                          "Path:%7s; "
                          "%10.5e GB/s; "
                          "%10.5e GFlop/s; ",
                          kernelType[kernelTypeIdx_], static_cast<int>(workSize[workSizeIdx_]),
                          _reqDataSize, memPath[memPathIdx_], bandwidth_, gflops_);
  testDescString = buf;
  _perfInfo = avgKernelTime_;
  if (!_dataSizeTooBig) checkData();
  EXIT("run");
}

void OCLPerfKernelThroughput::launchKernel(void) {
  ENTER("launchKernel")
  /***********************************************************
   * Copy Data To
   **********************************************************/
  // printf("Copying Data To Device\n");
  switch (memPathIdx_) {
    case 0:  // zero copy
      // do nothing
      // void *inputPtr = _wrapper->clEnqueueMapBuffer(
      //        cmd_queue_, input1Buffer_, true, CL_MAP_READ,
      //        0, input1BufferSize_, 0, NULL, NULL, &error_);
      // void *outputPtr = _wrapper->clEnqueueMapBuffer(
      //        cmd_queue_, output1Buffer_, true, CL_MAP_READ,
      //        0, output1BufferSize_, 0, NULL, NULL, &error_);
      //_wrapper->clFinish(cmd_queue_);
      break;
    case 1:  // explicit copy to device memory
      // printf("Queue:     %p\n", &cmd_queue_);
      // printf("devBuffer: %i\n", input1Buffer_);
      // printf("hstBuffer: %p\n", input1Ptr_);
      // printf("bufSize:   %i\n", input1BufferSize_);
      error_ = _wrapper->clEnqueueWriteBuffer(cmd_queue_, input1Buffer_, true, 0, input1BufferSize_,
                                              (const void*)input1Ptr_, 0, NULL, NULL);
      if (input2BufferSize_) {
        error_ =
            _wrapper->clEnqueueWriteBuffer(cmd_queue_, input2Buffer_, true, 0, input2BufferSize_,
                                           (const void*)input2Ptr_, 0, NULL, NULL);
      }
      // printf("Error: %i\n", error_);
      std::fflush(stdout);
      _wrapper->clFinish(cmd_queue_);
      CHECK_RESULT(error_ != CL_SUCCESS, "clWriteBuffer failed");
      //_error = _wrapper->clEnqueueWriteBuffer(
      //        cmd_queue_, output1Buffer_, true, 0, output1BufferSize_,
      //        (const void *)output1Ptr_, 0, NULL, NULL );
      // CHECK_RESULT(error_ != CL_SUCCESS, "clWriteBuffer failed");
      break;
  }

    /***********************************************************
     * Set Kernel Args
     **********************************************************/
#if 0
    error_ = _wrapper->clSetKernelArg(
                kernel_, 0, sizeof(input1Buffer_), (void *) &input1Buffer_);
    CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg failed");
    error_ = _wrapper->clSetKernelArg(
                kernel_, 1, sizeof(output1Buffer_), (void *) &output1Buffer_);
    CHECK_RESULT(error_ != CL_SUCCESS, "clSetKernelArg failed");
#endif

  // printf("Launching Kernel: %ix%i threads\n", global_work_size_[0],
  // local_work_size_[0]);

  /***********************************************************
   * Launch Kernel
   **********************************************************/
  error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, work_dim_, NULL,
                                            (const size_t*)global_work_size_,
                                            (const size_t*)local_work_size_, 0, NULL, NULL);
  // printf("Error: %i\n", error_);
  CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  _wrapper->clFinish(cmd_queue_);

  /***********************************************************
   * Copy Data From
   **********************************************************/
  // printf("Copying Data From Device\n");
  switch (memPathIdx_) {
    case 0:  // zero copy
      // do nothing
      // void *inputPtr = _wrapper->clEnqueueMapBuffer(
      //        cmd_queue_, input1Buffer_, true, CL_MAP_READ,
      //        0, input1BufferSize_, 0, NULL, NULL, &error_);
      // void *outputPtr = _wrapper->clEnqueueMapBuffer(
      //        cmd_queue_, output1Buffer_, true, CL_MAP_READ,
      //        0, output1BufferSize_, 0, NULL, NULL, &error_);
      //_wrapper->clFinish(cmd_queue_);
      break;
    case 1:  // explicit copy to device memory
      //_error = _wrapper->clEnqueueReadBuffer(
      //        cmd_queue_, input1Buffer_, true, 0, input1BufferSize_,
      //        (void *)input1Ptr_, 0, NULL, NULL );
      // CHECK_RESULT(error_ != CL_SUCCESS, "clWriteBuffer failed");
      // printf("VAL0 %p
      error_ = _wrapper->clEnqueueReadBuffer(cmd_queue_, output1Buffer_, true, 0,
                                             output1BufferSize_, (void*)output1Ptr_, 0, NULL, NULL);
      // printf("Error: %i\n", error_);
      CHECK_RESULT(error_ != CL_SUCCESS, "clWriteBuffer failed");
      break;
  }

  EXIT("launchKernel")
}

/*******************************************************************************
 * Check Data
 ******************************************************************************/
void OCLPerfKernelThroughput::checkData() {
  _wrapper->clFinish(cmd_queue_);
  float errorThreshhold = 0.00001f;
  float eqMax = gold_ + errorThreshhold * gold_;
  float eqMin = gold_ - errorThreshhold * gold_;
  /*
  printf("%ix%i * %ix%i = %ix%i:\n",
          matrixDim1_, matrixDim2_,
          matrixDim2_, matrixDim1_,
          matrixDim1_, matrixDim1_
          );
  */
  for (unsigned int i = 0; i < output1BufferSize_ / sizeof(float); i++) {
    float value = output1Ptr_[i];
    bool equal = (value > eqMin && value < eqMax);
    if (!equal) {
#if 0
            printf("Output[%i] = %.6e; gold_ = %.6e; %s\n",
                    i,
                    value,
                    gold_,
                    equal ? "Equal" : "NOT Equal");
#endif
      // printf("FAILURE\n");
      // CHECK_RESULT_NO_RETURN(1, "Data validation failed!\n");
      _errorFlag = true;
      break;
    } else {
      // printf("M[%i] = %.6e\n", i, output1Ptr_[i]);
    }
  }
}

/*******************************************************************************
 * Close - delete all data and release opencl objects
 ******************************************************************************/
unsigned int OCLPerfKernelThroughput::close(void) {
  ENTER("close");
  _wrapper->clFinish(cmd_queue_);

  if (global_work_size_) {
    delete[] global_work_size_;
    global_work_size_ = NULL;
  }
  if (local_work_size_) {
    delete[] local_work_size_;
    local_work_size_ = NULL;
  }
  // switch for memory type
  switch (memPathIdx_) {
    case 0:  // zero copy
      // unmap ptr
      if (input1Ptr_) {
        error_ = /*_wrapper->*/ clEnqueueUnmapMemObject(cmd_queue_, input1Buffer_, input1Ptr_, 0,
                                                        NULL, NULL);
        CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clEnqueueUnmapMemObject(input_) failed");
        _wrapper->clFinish(cmd_queue_);
        error_ = _wrapper->clReleaseMemObject(input1Buffer_);
        CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(input1Buffer_) failed");
        input1Buffer_ = 0;
      }
      if (input2Ptr_) {
        error_ = /*_wrapper->*/ clEnqueueUnmapMemObject(cmd_queue_, input2Buffer_, input2Ptr_, 0,
                                                        NULL, NULL);
        CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clEnqueueUnmapMemObject(input_) failed");
        _wrapper->clFinish(cmd_queue_);
        error_ = _wrapper->clReleaseMemObject(input2Buffer_);
        CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(input2Buffer_) failed");
        input2Buffer_ = 0;
      }
      if (output1Ptr_) {
        error_ = /*_wrapper->*/ clEnqueueUnmapMemObject(cmd_queue_, output1Buffer_, output1Ptr_, 0,
                                                        NULL, NULL);
        CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clEnqueueUnmapMemObject(output_) failed");
        _wrapper->clFinish(cmd_queue_);
        error_ = _wrapper->clReleaseMemObject(output1Buffer_);
        CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(input1Buffer_) failed");
        output1Buffer_ = 0;
      }
      break;
    case 1:  // explicit copy to device memory
      // release object
      if (input1Buffer_) {
        error_ = _wrapper->clReleaseMemObject(input1Buffer_);
        CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(input1Buffer_) failed");
        input1Buffer_ = 0;
      }
      if (input2Buffer_) {
        error_ = _wrapper->clReleaseMemObject(input2Buffer_);
        CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(input2Buffer_) failed");
        input2Buffer_ = 0;
      }
      if (output1Buffer_) {
        error_ = _wrapper->clReleaseMemObject(output1Buffer_);
        CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(input1Buffer_) failed");
        output1Buffer_ = 0;
      }
      if (input1Ptr_) {
        delete[] input1Ptr_;
        input1Ptr_ = 0;
      }
      if (input2Ptr_) {
        delete[] input2Ptr_;
        input2Ptr_ = 0;
      }
      if (output1Ptr_) {
        delete[] output1Ptr_;
        output1Ptr_ = 0;
      }
      break;
  }

  if (kernel_) {
    error_ = _wrapper->clReleaseKernel(kernel_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel failed");
    kernel_ = 0;
  }
  if (program_) {
    error_ = _wrapper->clReleaseProgram(program_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseProgram failed");
    program_ = 0;
  }
  if (cmd_queue_) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queue_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
    cmd_queue_ = 0;
  }
  if (context_) {
    error_ = _wrapper->clReleaseContext(context_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
    context_ = 0;
  }
  _wrapper->clFinish(cmd_queue_);

  EXIT("close");
  return _crcword;
}
