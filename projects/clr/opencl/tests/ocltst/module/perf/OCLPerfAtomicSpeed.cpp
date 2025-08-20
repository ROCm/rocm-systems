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

#include "OCLPerfAtomicSpeed.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "CL/cl.h"
#include "OCLPerfAtomicSpeedKernels.h"
#include "Timer.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

// Define the test suite tests.
testOCLPerfAtomicSpeedStruct testOCLPerfAtomicSpeedList[] = {
    {LocalHistogram, 1},
    {LocalHistogram, 2},
    {LocalHistogram, 4},
    {GlobalHistogram, 1},
    {GlobalHistogram, 2},
    {GlobalHistogram, 4},
    {Global4Histogram, 1},
    {Global4Histogram, 2},
    {Global4Histogram, 4},
    {LocalReductionNoAtomics, 1},
    {LocalReductionNoAtomics, 2},
    {LocalReductionNoAtomics, 4},
    {LocalReductionAtomics, 1},
    {LocalReductionAtomics, 2},
    {LocalReductionAtomics, 4},
    {Local4ReductionNoAtomics, 1},
    {Local4ReductionNoAtomics, 2},
    {Local4ReductionNoAtomics, 4},
    /*    {Local4ReductionAtomics, 1},
        {Local4ReductionAtomics, 2},
        {Local4ReductionAtomics, 4},*/
    {GlobalWGReduction, 1},
    {GlobalWGReduction, 2},
    {GlobalWGReduction, 4},
    {GlobalAllToZeroReduction, 1},
    {GlobalAllToZeroReduction, 2},
    {GlobalAllToZeroReduction, 4},
    {Global4WGReduction, 1},
    {Global4WGReduction, 2},
    {Global4WGReduction, 4},
    {Global4AllToZeroReduction, 1},
    {Global4AllToZeroReduction, 2},
    {Global4AllToZeroReduction, 4},
};

///////////////////////////////////////////////////////////////////////////////
// OCLPerfAtomicSpeed implementation.
///////////////////////////////////////////////////////////////////////////////
OCLPerfAtomicSpeed::OCLPerfAtomicSpeed() {
  _atomicsSupported = false;
  _dataSizeTooBig = false;
  _numSubTests = sizeof(testOCLPerfAtomicSpeedList) / sizeof(testOCLPerfAtomicSpeedStruct);
  _numLoops = 10;
  _nCurrentInputScale = 1;
  _maxMemoryAllocationSize = 0;

  _input = NULL;
  _output = NULL;
  _inputBuffer = NULL;
  _outputBuffer = NULL;
  _workgroupSize = 256;
  _programs.clear();
  _kernels.clear();
}

OCLPerfAtomicSpeed::~OCLPerfAtomicSpeed() {}

void OCLPerfAtomicSpeed::open(unsigned int test, char* units, double& conversion,
                              unsigned int deviceId) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_int status = CL_SUCCESS;

  device = NULL;
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
  _openTest = test;
  _cpuReductionSum = 0;
  _nCurrentInputScale = testOCLPerfAtomicSpeedList[_openTest].inputScale;
  AtomicType atomicType = testOCLPerfAtomicSpeedList[_openTest].atomicType;

  // Setup stuff...
  setupHistogram();
  calculateHostBin();

  context_ = 0;
  cmd_queue_ = 0;

  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
    // Get last for default
#if 0
        platform = platforms[numPlatforms-1];
        for (unsigned i = 0; i < numPlatforms; ++i) {
#endif
    char pbuf[100];
    error_ = _wrapper->clGetPlatformInfo(platforms[_platformIndex], CL_PLATFORM_VENDOR,
                                         sizeof(pbuf), pbuf, NULL);
#if 0
            if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) {
                platform = platforms[i];
                break;
            }
#endif
    num_devices = 0;
    /* Get the number of requested devices */
    error_ = _wrapper->clGetDeviceIDs(platforms[_platformIndex], type_, 0, NULL, &num_devices);
    // Runtime returns an error when no GPU devices are present instead of just
    // returning 0 devices
    // CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");
    // Choose platform with GPU devices
    if (num_devices > 0) {
#if 0
                if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) {
                    isAMD = true;
                }
#endif
      platform = platforms[_platformIndex];
    }
#if 0
        }
#endif
    delete platforms;
  }
  /*
   * If we could find our platform, use it. If not, die as we need the AMD
   * platform for these extensions.
   */
  CHECK_RESULT(platform == 0, "Couldn't find platform with GPU devices, cannot proceed");

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  CHECK_RESULT(devices == 0, "no devices");

  /* Get the requested device */
  error_ = _wrapper->clGetDeviceIDs(platform, type_, num_devices, devices, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");

  CHECK_RESULT(_deviceId >= num_devices, "Requested deviceID not available");
  device = devices[_deviceId];

  context_ = _wrapper->clCreateContext(NULL, 1, &device, NULL, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  char charbuf[1024];
  size_t retsize;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1024, charbuf, &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  // Global memory size
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
                                     &_maxMemoryAllocationSize, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs(CL_DEVICE_GLOBAL_MEM_SIZE) failed");

  // Check that the test size is not too big for the current GPU.
  _dataSizeTooBig = false;
  cl_ulong tenMB = 1024 * 10240;
  if (_inputNBytes >= (_maxMemoryAllocationSize - tenMB)) {
    _dataSizeTooBig = true;
    return;
  }

  char* p = strstr(charbuf, "cl_khr_global_int32_base_atomics");
  char* p2 = strstr(charbuf, "cl_khr_local_int32_base_atomics");

  _atomicsSupported = false;
  if (p || p2) _atomicsSupported = true;

  // Verify atomics are supported.
  if (!_atomicsSupported) return;

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  // Create buffers...
  _inputBuffer = clCreateBuffer(context_, CL_MEM_READ_ONLY, _inputNBytes, 0, &status);
  CHECK_RESULT(status, "clCreateBuffer failed. (inputBuffer)");

  // Create the programs/kernels for the current test type.
  CreateKernels(atomicType);

  _nThreadsPerGroup = _workgroupSize;
  _nGroups = _nThreads / _nThreadsPerGroup;
  _outputNBytes = _nGroups * NBINS * sizeof(cl_uint);
  if (IsReduction(atomicType)) _outputNBytes = _inputNBytes;

  _output = (cl_uint*)malloc(_outputNBytes);
  if (0 == _output) {
    _dataSizeTooBig = true;
    return;
  }

  // Create output Buffer
  _outputBuffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, _outputNBytes, 0, &status);
  CHECK_RESULT(status, "clCreateBuffer failed. (outputBuffer)");
}

// Create the programs/kernels for the current test type.
void OCLPerfAtomicSpeed::CreateKernels(const AtomicType atomicType) {
  char log[16384];
  cl_kernel kernel_;
  cl_program program_;
  char buildOptions[1000];
  cl_int status = CL_SUCCESS;

  SNPRINTF(buildOptions, sizeof(buildOptions), "-D NBINS=%d -D BITS_PER_PIX=%d -D NBANKS=%d", NBINS,
           BITS_PER_PIX, NBANKS);

  // Create the programs.
  switch (atomicType) {
    case LocalHistogram:
      program_ = _wrapper->clCreateProgramWithSource(
          context_, 1, (const char**)&local_atomics_histogram, NULL, &error_);
      CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");
      _programs.push_back(program_);
      program_ = _wrapper->clCreateProgramWithSource(
          context_, 1, (const char**)&local_atomics_reduce, NULL, &error_);
      CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");
      _programs.push_back(program_);
      break;
    case LocalReductionNoAtomics:
      program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&local_reduction,
                                                     NULL, &error_);
      CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");
      _programs.push_back(program_);
      break;
    case Local4ReductionNoAtomics:
      program_ = _wrapper->clCreateProgramWithSource(
          context_, 1, (const char**)&local_vec4_reduction, NULL, &error_);
      CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");
      _programs.push_back(program_);
      break;
    case LocalReductionAtomics:
      program_ = _wrapper->clCreateProgramWithSource(
          context_, 1, (const char**)&local_atomics_reduction, NULL, &error_);
      CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");
      _programs.push_back(program_);
      break;
    case Local4ReductionAtomics:
      program_ = _wrapper->clCreateProgramWithSource(
          context_, 1, (const char**)&local_vec4_atomics_reduction, NULL, &error_);
      CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");
      _programs.push_back(program_);
      break;
    case GlobalHistogram:
    case Global4Histogram:
      program_ = _wrapper->clCreateProgramWithSource(
          context_, 1, (const char**)&global_atomics_histogram, NULL, &error_);
      CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");
      _programs.push_back(program_);
      break;
    case GlobalWGReduction:
    case Global4WGReduction:
      program_ = _wrapper->clCreateProgramWithSource(
          context_, 1, (const char**)&global_atomics_sum_reduction_workgroup, NULL, &error_);
      CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");
      _programs.push_back(program_);
      break;
    case GlobalAllToZeroReduction:
    case Global4AllToZeroReduction:
      program_ = _wrapper->clCreateProgramWithSource(
          context_, 1, (const char**)&global_atomics_sum_reduction_all_to_zero, NULL, &error_);
      CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");
      _programs.push_back(program_);
      break;
    default:
      CHECK_RESULT(true, "Atomic type not supported (clCreateProgram)");
  }
  // Build the programs.
  for (size_t i = 0; i < _programs.size(); i++) {
    error_ = _wrapper->clBuildProgram(_programs[i], 1, &device, buildOptions, NULL, NULL);
    if (error_ != CL_SUCCESS) {
      status = _wrapper->clGetProgramBuildInfo(_programs[i], device, CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
      printf("Build error -> %s\n", log);

      CHECK_RESULT(0, "clBuildProgram failed");
    }
  }

  switch (atomicType) {
    case LocalHistogram:
      kernel_ = _wrapper->clCreateKernel(_programs[0], "local_atomics_histogram", &error_);
      CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");
      _kernels.push_back(kernel_);
      kernel_ = _wrapper->clCreateKernel(_programs[1], "local_atomics_reduce", &error_);
      CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");
      _kernels.push_back(kernel_);
      break;
    case LocalReductionNoAtomics:
    case Local4ReductionNoAtomics:
    case LocalReductionAtomics:
    case Local4ReductionAtomics:
      kernel_ = _wrapper->clCreateKernel(_programs[0], "local_reduction", &error_);
      CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");
      _kernels.push_back(kernel_);
      break;
    case GlobalHistogram:
    case Global4Histogram:
      kernel_ = _wrapper->clCreateKernel(_programs[0], "global_atomics_histogram", &error_);
      CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");
      _kernels.push_back(kernel_);
      break;
    case GlobalWGReduction:
    case Global4WGReduction:
      kernel_ =
          _wrapper->clCreateKernel(_programs[0], "global_atomics_sum_reduction_workgroup", &error_);
      CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");
      _kernels.push_back(kernel_);
      break;
    case GlobalAllToZeroReduction:
    case Global4AllToZeroReduction:
      kernel_ = _wrapper->clCreateKernel(_programs[0], "global_atomics_sum_reduction_all_to_zero",
                                         &error_);
      CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");
      _kernels.push_back(kernel_);
      break;
    default:
      CHECK_RESULT(true, "Atomic type not supported (clCreateKernel)");
  }
}

// Sets the kernel arguments based on the current test type.
void OCLPerfAtomicSpeed::SetKernelArguments(const AtomicType atomicType) {
  int Arg = 0;
  int localSize = 0;
  int itemsPerThread = 1;
  cl_int status = CL_SUCCESS;

  switch (atomicType) {
    case LocalHistogram:
      // Set arguments for the local atomics histogram kernel
      status = _wrapper->clSetKernelArg(_kernels[0], Arg++, sizeof(cl_mem), (void*)&_inputBuffer);
      CHECK_RESULT(status, "clSetKernelArg failed. (inputBuffer)");

      status |= _wrapper->clSetKernelArg(_kernels[0], Arg++, sizeof(cl_mem), (void*)&_outputBuffer);
      CHECK_RESULT(status, "clSetKernelArg failed. (outputBuffer)");

      status |= _wrapper->clSetKernelArg(_kernels[0], Arg++, sizeof(_n4VectorsPerThread),
                                         (void*)&_n4VectorsPerThread);
      CHECK_RESULT(status, "clSetKernelArg failed. (n4VectorsPerThread)");

      // Set arguments for the local atomics reduce kernel
      Arg = 0;
      status |= _wrapper->clSetKernelArg(_kernels[1], Arg++, sizeof(cl_mem), (void*)&_outputBuffer);
      CHECK_RESULT(status, "clSetKernelArg failed. (outputBuffer)");

      status |= _wrapper->clSetKernelArg(_kernels[1], Arg++, sizeof(_nGroups), (void*)&_nGroups);
      CHECK_RESULT(status, "clSetKernelArg failed. (nGroups)");
      break;
    case LocalReductionAtomics:
    case LocalReductionNoAtomics:
    case Local4ReductionNoAtomics:
    case Local4ReductionAtomics:
      status = _wrapper->clSetKernelArg(_kernels[0], Arg++, sizeof(cl_mem), (void*)&_inputBuffer);
      CHECK_RESULT(status, "clSetKernelArg failed. (inputBuffer)");

      status |= _wrapper->clSetKernelArg(_kernels[0], Arg++, sizeof(cl_mem), (void*)&_outputBuffer);
      CHECK_RESULT(status, "clSetKernelArg failed. (outputBuffer)");

      localSize = DEFAULT_WG_SIZE * sizeof(cl_uint);
      if ((Local4ReductionNoAtomics == atomicType) || (Local4ReductionAtomics == atomicType))
        localSize *= 4;
      status = _wrapper->clSetKernelArg(_kernels[0], Arg++, localSize, NULL);
      CHECK_RESULT(status, "clSetKernelArg failed. (local memory)");
      break;
    case GlobalHistogram:
    case Global4Histogram:
    case GlobalWGReduction:
    case Global4WGReduction:
    case GlobalAllToZeroReduction:
    case Global4AllToZeroReduction:
      // Set arguments for the global atomics histogram kernel
      if ((Global4Histogram == atomicType) || (Global4WGReduction == atomicType) ||
          (Global4AllToZeroReduction == atomicType))
        itemsPerThread = 4;

      status = _wrapper->clSetKernelArg(_kernels[0], Arg++, sizeof(itemsPerThread),
                                        (void*)&itemsPerThread);
      CHECK_RESULT(status, "clSetKernelArg failed. (itemsPerThread)");

      status = _wrapper->clSetKernelArg(_kernels[0], Arg++, sizeof(cl_mem), (void*)&_inputBuffer);
      CHECK_RESULT(status, "clSetKernelArg failed. (inputBuffer)");

      status |= _wrapper->clSetKernelArg(_kernels[0], Arg++, sizeof(cl_mem), (void*)&_outputBuffer);
      CHECK_RESULT(status, "clSetKernelArg failed. (outputBuffer)");
      break;
    default:
      CHECK_RESULT(true, "Atomic type not supported (clSetKernelArg)");
  }
}

// Since we write multiple times to the output in global atomics, need to
// reset the content every time.
void OCLPerfAtomicSpeed::ResetGlobalOutput() {
  cl_int status;

  memset(_output, 0, _outputNBytes);

  status = _wrapper->clEnqueueWriteBuffer(cmd_queue_, _outputBuffer, CL_TRUE, 0, _outputNBytes,
                                          _output, 0, NULL, NULL);
  CHECK_RESULT(status, "clEnqueueWriteBuffer failed.");

  status = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(status, "clFlush failed.");
}

// Run the local histogram kernels.
void OCLPerfAtomicSpeed::RunLocalHistogram() {
  cl_uint status;
  cl_event events[2];
  size_t globalThreads[3] = {1};
  size_t localThreads[3] = {1};
  size_t globalThreadsReduce = NBINS;
  size_t localThreadsReduce = _nThreadsPerGroup;

  globalThreads[0] = _nThreads;
  localThreads[0] = _nThreadsPerGroup;

  status = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, _kernels[0], 1, NULL, globalThreads,
                                            localThreads, 0, NULL, &events[0]);
  CHECK_RESULT(status, "clEnqueueNDRangeKernel failed. (histogram)");

  status = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, _kernels[1], 1, NULL, &globalThreadsReduce,
                                            &localThreadsReduce, 1, &events[0], &events[1]);
  CHECK_RESULT(status, "clEnqueueNDRangeKernel failed. (reduce)");

  status = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(status, "clFlush failed.");

  status = _wrapper->clWaitForEvents(1, &events[0]);
  status |= _wrapper->clWaitForEvents(1, &events[1]);
  CHECK_RESULT(status, "clWaitForEvents failed.");
}

// Run the local reduction kernel.
void OCLPerfAtomicSpeed::RunLocalReduction(const AtomicType atomicType) {
  cl_uint status;
  size_t globalThreads[3] = {1};
  size_t localThreads[3] = {1};

  globalThreads[0] = _inputNBytes / sizeof(cl_uint) / 2;
  localThreads[0] = _nThreadsPerGroup;
  if ((Local4ReductionNoAtomics == atomicType) || (Local4ReductionAtomics == atomicType))
    globalThreads[0] /= 4;

  status = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, _kernels[0], 1, NULL, globalThreads,
                                            localThreads, 0, NULL, NULL);
  CHECK_RESULT(status, "clEnqueueNDRangeKernel failed. (reduction)");

  status = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(status, "clFlush failed.");
}

// Run the global histogram kernel.
void OCLPerfAtomicSpeed::RunGlobalHistogram(AtomicType atomicType) {
  cl_uint status;
  size_t globalThreads[3] = {1};
  size_t localThreads[3] = {1};

  globalThreads[0] = _inputNBytes / sizeof(cl_uint);
  localThreads[0] = _nThreadsPerGroup;

  if ((Global4Histogram == atomicType) || (Global4WGReduction == atomicType) ||
      (Global4AllToZeroReduction == atomicType))
    globalThreads[0] /= 4;

  status = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, _kernels[0], 1, NULL, globalThreads,
                                            localThreads, 0, NULL, NULL);
  CHECK_RESULT(status, "clEnqueueNDRangeKernel failed.");

  status = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(status, "clFlush failed.");
}

// Run the AtomicSpeed logic.
void OCLPerfAtomicSpeed::run() {
  int Arg = 0;
  cl_uint status;
  AtomicType atomicType = testOCLPerfAtomicSpeedList[_openTest].atomicType;

  // Verify atomics are supported.
  if ((!_atomicsSupported) || (_dataSizeTooBig)) return;

  // Write data to the GPU
  status = _wrapper->clEnqueueWriteBuffer(cmd_queue_, _inputBuffer, CL_FALSE, 0, _inputNBytes,
                                          _input, 0, NULL, NULL);
  CHECK_RESULT(status, "clEnqueueWriteBuffer failed. (inputBuffer)");

  status = _wrapper->clFlush(cmd_queue_);
  CHECK_RESULT(status, "clFlush failed.");

  // Set the current arguments based on the test type.
  SetKernelArguments(atomicType);

  // Run the kernels.
  CPerfCounter timer;
  double totalTime = 0.0f;

  for (unsigned int k = 0; k < _numLoops + 1; k++) {
    // Since we run multiple times using global atomics the output
    // would get accumulated therefore first clean it.
    ResetGlobalOutput();

    timer.Reset();
    timer.Start();
    switch (atomicType) {
      case LocalHistogram:
        RunLocalHistogram();
        break;
      case LocalReductionAtomics:
      case LocalReductionNoAtomics:
      case Local4ReductionNoAtomics:
      case Local4ReductionAtomics:
        RunLocalReduction(atomicType);
        break;
      case GlobalHistogram:
      case Global4Histogram:
      case GlobalWGReduction:
      case Global4WGReduction:
      case GlobalAllToZeroReduction:
      case Global4AllToZeroReduction:
        RunGlobalHistogram(atomicType);
        break;
      default:
        CHECK_RESULT(true, "Atomic type not supported");
    }
    timer.Stop();
    // Don't count the warm-up
    if (0 != k) totalTime += timer.GetElapsedTime();
  }

  // Read the results back to the CPU - Only do it for the last run
  // of the test instead of for each iteration of _numLoops.
  status = _wrapper->clEnqueueReadBuffer(cmd_queue_, _outputBuffer, CL_FALSE, 0, _outputNBytes,
                                         _output, 0, NULL, NULL);
  CHECK_RESULT(status, "clEnqueueReadBuffer failed.");
  status = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(status, "clFlush failed.");

  // Print the results.
  PrintResults(atomicType, totalTime);

  // Check the results for the current test.
  _errorFlag = !(VerifyResults(atomicType));
}

// Compare the results and see if they match
bool OCLPerfAtomicSpeed::VerifyResults(const AtomicType atomicType) {
  cl_uint i = 0;
  bool flag = true;
  cl_uint calculatedValue = 0;
  cl_uint reductionElementCount = 0;
  switch (atomicType) {
    case LocalHistogram:
    case GlobalHistogram:
    case Global4Histogram:
      for (i = 0; i < NBINS; ++i) {
        if (_cpuhist[i] != _output[i]) {
          flag = false;
          break;
        }
      }
      break;
    case LocalReductionAtomics:
    case LocalReductionNoAtomics:
    case Local4ReductionNoAtomics:
    case Local4ReductionAtomics:
    case GlobalWGReduction:
    case Global4WGReduction:
      reductionElementCount = _inputNBytes / sizeof(cl_uint) / _nThreadsPerGroup;
      for (i = 0; i < reductionElementCount; i++) {
        calculatedValue += _output[i];
      }
      flag = (calculatedValue == _cpuReductionSum);
      break;
    case GlobalAllToZeroReduction:
    case Global4AllToZeroReduction:
      flag = (_output[0] == _cpuReductionSum);
      break;
    default:
      CHECK_RESULT_NO_RETURN(true, "Atomic type not supported (VerifyResults)");
      return false;
  }
  if (!flag) printf("WRONG VALUES!!!!!");
  return flag;
}

unsigned int OCLPerfAtomicSpeed::close() {
  size_t i = 0;
  for (; i < _kernels.size(); i++) {
    error_ = _wrapper->clReleaseKernel(_kernels[i]);
  }
  for (; i < _programs.size(); i++) {
    error_ = _wrapper->clReleaseProgram(_programs[i]);
  }
  if (_inputBuffer) {
    error_ = clReleaseMemObject(_inputBuffer);
    CHECK_RESULT_NO_RETURN(error_, "clReleaseMemObject failed.(inputBuffer )");
  }
  if (_outputBuffer) {
    error_ = clReleaseMemObject(_outputBuffer);
    CHECK_RESULT_NO_RETURN(error_, "clReleaseMemObject failed.(outputBuffer)");
  }

  if (cmd_queue_) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queue_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }
  if (context_) {
    error_ = _wrapper->clReleaseContext(context_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }

  // Free host memory.
  free(_input);
  free(_output);

  // Reset everything.
  _kernels.clear();
  _programs.clear();
  _inputBuffer = NULL;
  _outputBuffer = NULL;
  cmd_queue_ = NULL;
  context_ = NULL;
  _input = NULL;
  _output = NULL;

  return _crcword;
}

/* Helper functions */
void OCLPerfAtomicSpeed::calculateHostBin() {
  // compute CPU histogram
  cl_int* p = (cl_int*)_input;
  memset(_cpuhist, 0, NBINS * sizeof(cl_uint));
  _cpuReductionSum = 0;

  for (unsigned int i = 0; i < _inputNBytes / sizeof(cl_uint); i++) {
    _cpuhist[(p[i] >> 24) & 0xff]++;
    _cpuhist[(p[i] >> 16) & 0xff]++;
    _cpuhist[(p[i] >> 8) & 0xff]++;
    _cpuhist[(p[i] >> 0) & 0xff]++;
    _cpuReductionSum +=
        ((p[i] >> 24) & 0x3) + ((p[i] >> 16) & 0x3) + ((p[i] >> 8) & 0x3) + ((p[i] >> 0) & 0x3);
  }
}

void OCLPerfAtomicSpeed::setupHistogram() {
  cl_int status = 0;

  _nThreads = 64 * 1024;
#if defined(_WIN32) && !defined(_WIN64)
  _n4Vectors = 1024 * 1024;
#else
  _n4Vectors = 2048 * 2048;
#endif
  _n4Vectors *= _nCurrentInputScale;
  _n4VectorsPerThread = _n4Vectors / _nThreads;
  _inputNBytes = _n4Vectors * sizeof(cl_uint4);

  _input = (cl_uint*)malloc(_inputNBytes);
  if (0 == _input) {
    _dataSizeTooBig = true;
    return;
  }

  // random initialization of input
  time_t ltime;
  time(&ltime);
  cl_uint a = (cl_uint)ltime, b = (cl_uint)ltime;
  cl_uint* p = (cl_uint*)_input;

  for (unsigned int i = 0; i < _inputNBytes / sizeof(cl_uint); i++)
    p[i] = (b = (a * (b & 65535)) + (b >> 16));
}

// Print the results of the current test.
void OCLPerfAtomicSpeed::PrintResults(const AtomicType atomicType, double totalTime) {
  char buf[500];
  char sAtomicType[100];
  double inputInGB = (double)_inputNBytes * (double)(1e-09);
  // each cl_uint in _inputNBytes contributes 4 items.
  double totalHistogramDataInGB = (double)inputInGB * 4;
  double perf = totalTime / _numLoops;

  switch (atomicType) {
    case LocalHistogram:
      SNPRINTF(sAtomicType, sizeof(sAtomicType), "Local histogram");
      break;
    case GlobalHistogram:
      SNPRINTF(sAtomicType, sizeof(sAtomicType), "Global histogram");
      break;
    case Global4Histogram:
      SNPRINTF(sAtomicType, sizeof(sAtomicType), "Global vec 4 histogram");
      break;
    case LocalReductionNoAtomics:
      SNPRINTF(sAtomicType, sizeof(sAtomicType), "Local reduction NO atomics");
      break;
    case Local4ReductionNoAtomics:
      SNPRINTF(sAtomicType, sizeof(sAtomicType), "Local vec 4 reduction NO atomics");
      break;
    case LocalReductionAtomics:
      SNPRINTF(sAtomicType, sizeof(sAtomicType), "Local reduction with atomics");
      break;
    case Local4ReductionAtomics:
      SNPRINTF(sAtomicType, sizeof(sAtomicType), "Local vec 4 reduction with atomics");
      break;
    case GlobalWGReduction:
      SNPRINTF(sAtomicType, sizeof(sAtomicType), "Global work-group reduction");
      break;
    case Global4WGReduction:
      SNPRINTF(sAtomicType, sizeof(sAtomicType), "Global vec 4 work-group reduction");
      break;
    case GlobalAllToZeroReduction:
      SNPRINTF(sAtomicType, sizeof(sAtomicType), "Global all to zero reduction");
      break;
    case Global4AllToZeroReduction:
      SNPRINTF(sAtomicType, sizeof(sAtomicType), "Global vec 4 all to zero reduction");
      break;
    default:
      CHECK_RESULT(true, "Atomic type not supported (PrintResults)");
  }

  SNPRINTF(buf, sizeof(buf), "%45s: Input [%.3f GB], Time [%.3f sec]: GB/s", sAtomicType,
           totalHistogramDataInGB, perf);
  _perfInfo = (float)(totalHistogramDataInGB / perf);
  testDescString = buf;
}

bool OCLPerfAtomicSpeed::IsReduction(const AtomicType atomicType) {
  return ((atomicType >= LocalReductionNoAtomics) && (atomicType <= GlobalAllToZeroReduction));
}
