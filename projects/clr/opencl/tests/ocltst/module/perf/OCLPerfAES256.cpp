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

#include "OCLPerfAES256.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "Timer.h"

static const char* aes256_kernel =
    "// NOTE: THIS KERNEL WAS ADOPTED FROM SISOFT SANDRA: DO NOT "
    "REDISTRIBUTE!!\n"
    "inline uint Load(__global uint* pData, const uint iX, const uint iY)\n"
    "{\n"
    "   return pData[iX | (iY << 8)];\n"
    "}\n"
    "\n"
    "\n"
    "inline uint4 Load4(__global uint* pData, const uint4 uX, const uint iY)\n"
    "{\n"
    "   uint  uExtent = iY << 8;\n"
    "   uint4 uNdx = uX + uExtent;\n"
    "   \n"
    "   return (uint4)(pData[uNdx.x], pData[uNdx.y], pData[uNdx.z], "
    "pData[uNdx.w]);\n"
    "}\n"
    "\n"
    "\n"
    "__kernel \n"
    "__attribute__((vec_type_hint(uint4))) \n"
    "void CryptThread(__global uint4* pInput, __global uint4* pOutput,\n"
    "                       __global uint* pTables,\n"
    "                       __global uint4* pKey, const uint iRounds)\n"
    "{\n"
    "   const uint iNdx = get_global_id(0);\n"
    "   \n"
    "   uint4 state, istate, tstate;\n"
    "   state = pInput[iNdx] ^ pKey[iRounds];\n"
    "   \n"
    "   for (uint i = iRounds-1; i; i--)\n"
    "   {\n"
    "       istate = state & 0xFF;\n"
    "       tstate = Load4(pTables, istate.xyzw, 0);\n"
    "\n"
    "       istate = (state >> 8) & 0xFF;\n"
    "       tstate^= Load4(pTables, istate.wxyz, 1);\n"
    "\n"
    "       istate = (state >> 16) & 0xFF;\n"
    "       tstate^= Load4(pTables, istate.zwxy, 2);\n"
    "\n"
    "       istate = state >> 24;\n"
    "       tstate^= Load4(pTables, istate.yzwx, 3);\n"
    "\n"
    "       state = tstate ^ pKey[i];\n"
    "   }\n"
    "\n"
    "   istate = state & 0xFF;\n"
    "   tstate = Load4(pTables, istate.xyzw, 4);\n"
    "\n"
    "   istate = (state >> 8) & 0xFF;\n"
    "   tstate |= Load4(pTables, istate.wxyz, 4) << 8;\n"
    "\n"
    "   istate = (state >> 16) & 0xFF;\n"
    "   tstate |= Load4(pTables, istate.zwxy, 4) << 16;\n"
    "\n"
    "   istate = state >> 24;\n"
    "   tstate |= Load4(pTables, istate.yzwx, 4) << 24;\n"
    "\n"
    "   pOutput[iNdx] = tstate ^ pKey[0];\n"
    "}\n";

static const char* aes256_kernel2 =
    "// NOTE: THIS KERNEL WAS ADOPTED FROM SISOFT SANDRA: DO NOT "
    "REDISTRIBUTE!!\n"
    "#define AES_BLOCK_SIZE      16\n"
    "#define AES_TABLE_SIZE      256\n"
    "\n"
    "#define AES_TABLE_MAX       5\n"
    "#define AES_CONST_SIZE      (AES_TABLE_SIZE*AES_TABLE_MAX)\n"
    "\n"
    "#define AES_ROUND_128       10\n"
    "#define AES_ROUND_192       12\n"
    "#define AES_ROUND_256       14\n"
    "#define AES_ROUNDKEY_MAX    (AES_BLOCK_SIZE/4*(AES_ROUND_256+1))\n"
    "#define _IS_GPU_\n"
    "\n"
    "\n"
    "inline uint Load(\n"
    "#ifdef _IS_GPU_\n"
    "    __local uint* pData,\n"
    "#else\n"
    "    __constant uint* pData,\n"
    "#endif\n"
    "    const uint iX, const uint iY)\n"
    "{\n"
    "    const uint uNdx = iX + iY*AES_TABLE_SIZE;\n"
    "    return pData[uNdx];\n"
    "}\n"
    "\n"
    "\n"
    "inline uint4 Load4(\n"
    "#ifdef _IS_GPU_\n"
    "    __local uint* pData,\n"
    "#else\n"
    "    __constant uint* pData,\n"
    "#endif\n"
    "    const uint4 uX, const uint iY)\n"
    "{\n"
    "    const uint  uExtent = iY*AES_TABLE_SIZE;\n"
    "    const uint4 uNdx = uX + uExtent;\n"
    "    \n"
    "    return (uint4)(pData[uNdx.x], pData[uNdx.y], pData[uNdx.z], "
    "pData[uNdx.w]);\n"
    "}\n"
    "\n"
    "\n"
    "__kernel \n"
    "__attribute__((vec_type_hint(uint4)))\n"
    "#ifdef KERNEL_MAX_THREADS\n"
    "__attribute__((work_group_size_hint(KERNEL_MAX_THREADS, 1, 1)))\n"
    "#endif\n"
    "void CryptThread(__global const uint4* pInput, __global uint4* pOutput,\n"
    "                        __constant uint* pTables,\n"
    "                        __constant uint4* pKey, const uint iRounds)\n"
    "{\n"
    "    const size_t iNdx = get_global_id(0);\n"
    "\n"
    "#ifdef _IS_GPU_\n"
    "    #define Load4T(x, y)    Load4(ulTables, x, y)\n"
    "\n"
    "    __local uint  ulTables[AES_CONST_SIZE];\n"
    "\n"
    "    const uint iLdx = get_local_id(0);\n"
    "    if (iLdx < AES_TABLE_SIZE) {\n"
    "        const uint iGrps = get_local_size(0);\n"
    "        const uint iLSize = min(iGrps, (uint)AES_TABLE_SIZE);\n"
    "        const uint iBpL = AES_CONST_SIZE/iLSize;\n"
    "\n"
    "        const uint iStart = iLdx*iBpL;\n"
    "        const uint iEnd   = iStart + iBpL;\n"
    "\n"
    "        for (uint i=iStart; i<iEnd; i++) {\n"
    "            ulTables[i] = pTables[i];\n"
    "        }\n"
    "    }\n"
    "\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "#else\n"
    "    #define Load4T(x, y)    Load4(pTables, x, y)\n"
    "#endif\n"
    "    \n"
    "    uint4 state, istate, tstate;\n"
    "    state = pInput[iNdx] ^ pKey[0];\n"
    "    \n"
    "    for (uint i = 1; i < iRounds; i++)\n"
    "    {\n"
    "        istate = state & 0xFF;\n"
    "        tstate = Load4T(istate.xyzw, 0);\n"
    "\n"
    "        istate = (state >> 8) & 0xFF;\n"
    "        tstate^= Load4T(istate.yzwx, 1);\n"
    "\n"
    "        istate = (state >> 16) & 0xFF;\n"
    "        tstate^= Load4T(istate.zwxy, 2);\n"
    "\n"
    "        istate = state >> 24;\n"
    "        tstate^= Load4T(istate.wxyz, 3);\n"
    "\n"
    "        state = tstate ^ pKey[i];\n"
    "    }\n"
    "\n"
    "    istate = state & 0xFF;\n"
    "    tstate = Load4T(istate.xyzw, 4);\n"
    "\n"
    "    istate = (state >> 8) & 0xFF;\n"
    "    tstate |= Load4T(istate.yzwx, 4) << 8;\n"
    "\n"
    "    istate = (state >> 16) & 0xFF;\n"
    "    tstate |= Load4T(istate.zwxy, 4) << 16;\n"
    "\n"
    "    istate = state >> 24;\n"
    "    tstate |= Load4T(istate.wxyz, 4) << 24;\n"
    "\n"
    "    pOutput[iNdx] = tstate ^ pKey[iRounds];\n"
    "}\n";

OCLPerfAES256::OCLPerfAES256() { _numSubTests = 2; }

OCLPerfAES256::~OCLPerfAES256() {}

void OCLPerfAES256::setData(cl_mem buffer, unsigned int val) {
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_, buffer, true, CL_MAP_WRITE, 0, bufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < bufSize_ / sizeof(unsigned int); i++) data[i] = val;
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
  _wrapper->clFinish(cmd_queue_);
}

void OCLPerfAES256::checkData(cl_mem buffer) {
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_, buffer, true, CL_MAP_READ, 0, bufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < bufSize_ / sizeof(unsigned int); i++) {
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
  _wrapper->clFinish(cmd_queue_);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfAES256::open(unsigned int test, char* units, double& conversion,
                         unsigned int deviceId) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
  _openTest = test;

  context_ = 0;
  cmd_queue_ = 0;
  program_ = 0;
  kernel_ = 0;
  inBuffer_ = 0;
  outBuffer_ = 0;
  tableBuffer_ = 0;
  keyBuffer_ = 0;
  blockSize_ = 1024;
  maxIterations = 50;

  bufSize_ = 5592320 * sizeof(cl_uint4);

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

  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  char charbuf[1024];
  size_t retsize;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1024, charbuf, &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  // Increase iterations for devices with many CUs
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &numCUs,
                                     &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  maxIterations *= (unsigned int)(1 + 10 * numCUs / 20);

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  inBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY, bufSize_, NULL, &error_);
  CHECK_RESULT(inBuffer_ == 0, "clCreateBuffer(inBuffer) failed");

  outBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bufSize_, NULL, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer) failed");

  tableBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY, 5120, NULL, &error_);
  CHECK_RESULT(tableBuffer_ == 0, "clCreateBuffer(tableBuffer) failed");

  keyBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY, 240, NULL, &error_);
  CHECK_RESULT(keyBuffer_ == 0, "clCreateBuffer(keyBuffer) failed");

  if (_openTest == 0) {
    program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&aes256_kernel, NULL,
                                                   &error_);
    CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");
    testDescString += "orig";
  } else {
    program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&aes256_kernel2, NULL,
                                                   &error_);
    CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");
    testDescString += " new";
  }

  const char* buildOps = NULL;
  error_ = _wrapper->clBuildProgram(program_, 1, &device, buildOps, NULL, NULL);

  if (error_ != CL_SUCCESS) {
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);

    CHECK_RESULT(0, "clBuildProgram failed");
  }
  kernel_ = _wrapper->clCreateKernel(program_, "CryptThread", &error_);
  CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");

  cl_uint rounds = 14;

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&inBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), (void*)&outBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_mem), (void*)&tableBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel_, 3, sizeof(cl_mem), (void*)&keyBuffer_);
  error_ = _wrapper->clSetKernelArg(kernel_, 4, sizeof(cl_uint), (void*)&rounds);
  setData(inBuffer_, 0xdeadbeef);
  setData(outBuffer_, 0xdeadbeef);
}

void OCLPerfAES256::run(void) {
  int global = bufSize_ / sizeof(cl_uint4);
  int local = 64;

  size_t global_work_size[1] = {(size_t)global};
  size_t local_work_size[1] = {(size_t)local};

  CPerfCounter timer;

  timer.Reset();
  timer.Start();
  for (unsigned int i = 0; i < maxIterations; i++) {
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);
  }

  CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  _wrapper->clFinish(cmd_queue_);

  timer.Stop();
  double sec = timer.GetElapsedTime();

  // No idea what data should be in here
  // checkData(outBuffer_);
  // Compute GB/s
  double perf = ((double)bufSize_ * (double)maxIterations * (double)(1e-09)) / sec;

  _perfInfo = (float)perf;
}

unsigned int OCLPerfAES256::close(void) {
  _wrapper->clFinish(cmd_queue_);

  if (inBuffer_) {
    error_ = _wrapper->clReleaseMemObject(inBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(inBuffer_) failed");
  }
  if (outBuffer_) {
    error_ = _wrapper->clReleaseMemObject(outBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(outBuffer_) failed");
  }
  if (tableBuffer_) {
    error_ = _wrapper->clReleaseMemObject(tableBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(tableBuffer_) failed");
  }
  if (keyBuffer_) {
    error_ = _wrapper->clReleaseMemObject(keyBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(keyBuffer_) failed");
  }
  if (kernel_) {
    error_ = _wrapper->clReleaseKernel(kernel_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel failed");
  }
  if (program_) {
    error_ = _wrapper->clReleaseProgram(program_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseProgram failed");
  }
  if (cmd_queue_) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queue_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }
  if (context_) {
    error_ = _wrapper->clReleaseContext(context_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }

  return _crcword;
}
