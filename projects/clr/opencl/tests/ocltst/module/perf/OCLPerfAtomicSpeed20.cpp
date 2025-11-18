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

#include "OCLPerfAtomicSpeed20.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "CL/cl.h"
#include "OCLPerfAtomicSpeed20Kernels.h"
#include "Timer.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

// Define the test suite tests.
testOCLPerfAtomicSpeed20Struct testOCLPerfAtomicSpeed20List[] = {
    {GlobalWGReduction, 1},         {GlobalWGReduction, 2},         {GlobalWGReduction, 4},
    {GlobalAllToZeroReduction, 1},  {GlobalAllToZeroReduction, 2},  {GlobalAllToZeroReduction, 4},
    {Global4WGReduction, 1},        {Global4WGReduction, 2},        {Global4WGReduction, 4},
    {Global4AllToZeroReduction, 1}, {Global4AllToZeroReduction, 2}, {Global4AllToZeroReduction, 4},
};

///////////////////////////////////////////////////////////////////////////////
// OCLPerfAtomicSpeed20 implementation.
///////////////////////////////////////////////////////////////////////////////
OCLPerfAtomicSpeed20::OCLPerfAtomicSpeed20() {
  _atomicsSupported = false;
  _dataSizeTooBig = false;
  _numSubTests = sizeof(testOCLPerfAtomicSpeed20List) / sizeof(testOCLPerfAtomicSpeed20Struct);
  _numLoops = 10;
  _nCurrentInputScale = 1;
  _maxMemoryAllocationSize = 0;

  _input = NULL;
  _output = NULL;
  _inputBuffer = NULL;
  _outputBuffer = NULL;

  skip_ = false;

  _workgroupSize = 256;
  _programs.clear();
  _kernels.clear();
}

OCLPerfAtomicSpeed20::~OCLPerfAtomicSpeed20() {}

void OCLPerfAtomicSpeed20::open(unsigned int test, char* units, double& conversion,
                                unsigned int deviceId) {
  error_ = CL_SUCCESS;

  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_RESULT((error_ != CL_SUCCESS), "Error opening test");

  program_ = 0;
  kernel_ = 0;

#if defined(CL_VERSION_2_0)
  cl_device_id device;
  cl_int status = CL_SUCCESS;

  conversion = 1.0f;
  _openTest = test;
  _cpuReductionSum = 0;
  _nCurrentInputScale = testOCLPerfAtomicSpeed20List[_openTest].inputScale;
  AtomicType atomicType = testOCLPerfAtomicSpeed20List[_openTest].atomicType;

  // Setup stuff...
  setupHistogram();
  calculateHostBin();

  device = devices_[_deviceId];

  cmd_queue_ = cmdQueues_[_deviceId];

  char charbuf[1024];
  size_t retsize;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1024, charbuf, &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  // Global memory size
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong),
                                     &_maxMemoryAllocationSize, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE) failed");

  // Check that the test size is not too big for the current GPU.
  _dataSizeTooBig = false;
  cl_ulong tenMB = 1024 * 10240;
  if (_inputNBytes >= (_maxMemoryAllocationSize - tenMB)) {
    _dataSizeTooBig = true;
    return;
  }

  char* p = strstr(charbuf, "cl_khr_global_int32_base_atomics");

  _atomicsSupported = false;
  if (p) _atomicsSupported = true;

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
  _outputNBytes = _inputNBytes;

  _output = (cl_uint*)malloc(_outputNBytes);
  if (0 == _output) {
    _dataSizeTooBig = true;
    return;
  }

  // Create output Buffer
  _outputBuffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, _outputNBytes, 0, &status);
  CHECK_RESULT(status, "clCreateBuffer failed. (outputBuffer)");
#else
  skip_ = true;
  testDescString = "OpenCL verion < 2.0. Test Skipped.";
  return;
#endif
}

// Create the programs/kernels for the current test type.
void OCLPerfAtomicSpeed20::CreateKernels(const AtomicType atomicType) {
  char log[16384];
  cl_kernel kernel_;
  cl_program program_;
  char buildOptions[1000];
  cl_int status = CL_SUCCESS;
  cl_device_id device = devices_[_deviceId];

  SNPRINTF(buildOptions, sizeof(buildOptions),
           "-cl-std=CL2.0 -D NBINS=%d -D BITS_PER_PIX=%d -D NBANKS=%d", NBINS, BITS_PER_PIX,
           NBANKS);

  // Create the programs.
  switch (atomicType) {
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
void OCLPerfAtomicSpeed20::SetKernelArguments(const AtomicType atomicType) {
  int Arg = 0;
  int localSize = 0;
  int itemsPerThread = 1;
  cl_int status = CL_SUCCESS;

  switch (atomicType) {
    case GlobalWGReduction:
    case Global4WGReduction:
    case GlobalAllToZeroReduction:
    case Global4AllToZeroReduction:
      // Set arguments for the global atomics histogram kernel
      if ((Global4WGReduction == atomicType) || (Global4AllToZeroReduction == atomicType))
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
void OCLPerfAtomicSpeed20::ResetGlobalOutput() {
  cl_int status;

  memset(_output, 0, _outputNBytes);

  status = _wrapper->clEnqueueWriteBuffer(cmd_queue_, _outputBuffer, CL_TRUE, 0, _outputNBytes,
                                          _output, 0, NULL, NULL);
  CHECK_RESULT(status, "clEnqueueWriteBuffer failed.");

  status = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(status, "clFlush failed.");
}

// Run the global histogram kernel.
void OCLPerfAtomicSpeed20::RunGlobalHistogram(AtomicType atomicType) {
  cl_uint status;
  size_t globalThreads[3] = {1};
  size_t localThreads[3] = {1};

  globalThreads[0] = _inputNBytes / sizeof(cl_uint);
  localThreads[0] = _nThreadsPerGroup;

  if ((Global4WGReduction == atomicType) || (Global4AllToZeroReduction == atomicType))
    globalThreads[0] /= 4;

  status = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, _kernels[0], 1, NULL, globalThreads,
                                            localThreads, 0, NULL, NULL);
  CHECK_RESULT(status, "clEnqueueNDRangeKernel failed.");

  status = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(status, "clFlush failed.");
}

// Run the AtomicSpeed logic.
void OCLPerfAtomicSpeed20::run() {
  if (skip_) {
    return;
  }

#if defined(CL_VERSION_2_0)
  int Arg = 0;
  cl_uint status;
  AtomicType atomicType = testOCLPerfAtomicSpeed20List[_openTest].atomicType;

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

  status = _wrapper->clEnqueueReadBuffer(cmd_queue_, _outputBuffer, CL_FALSE, 0, _outputNBytes,
                                         _output, 0, NULL, NULL);
  CHECK_RESULT(status, "clEnqueueReadBuffer failed.");
  status = _wrapper->clFinish(cmd_queue_);
  CHECK_RESULT(status, "clFlush failed.");

  // Print the results.
  PrintResults(atomicType, totalTime);

  // Check the results for the current test.
  _errorFlag = !(VerifyResults(atomicType));
#endif
}

// Compare the results and see if they match
bool OCLPerfAtomicSpeed20::VerifyResults(const AtomicType atomicType) {
  cl_uint i = 0;
  bool flag = true;
  cl_uint calculatedValue = 0;
  cl_uint reductionElementCount = 0;
  switch (atomicType) {
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

unsigned int OCLPerfAtomicSpeed20::close() {
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

  // Free host memory.
  free(_input);
  free(_output);

  // Reset everything.
  _kernels.clear();
  _programs.clear();

  _inputBuffer = NULL;
  _outputBuffer = NULL;

  _input = NULL;
  _output = NULL;

  return OCLTestImp::close();
}

/* Helper functions */
void OCLPerfAtomicSpeed20::calculateHostBin() {
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

void OCLPerfAtomicSpeed20::setupHistogram() {
  cl_int status = 0;

  _nThreads = 64 * 1024;
  _n4Vectors = 2048 * 2048;
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
void OCLPerfAtomicSpeed20::PrintResults(const AtomicType atomicType, double totalTime) {
  char buf[500];
  char sAtomicType[100];
  double inputInGB = (double)_inputNBytes * (double)(1e-09);
  // each cl_uint in _inputNBytes contributes 4 items.
  double totalHistogramDataInGB = (double)inputInGB * 4;
  double perf = totalTime / _numLoops;

  switch (atomicType) {
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
