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

#include "OCLPerfGenoilSiaMiner.h"

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

#define NUM_INTENSITY 15

static const unsigned int intensities[NUM_INTENSITY] = {
    DEFAULT_INTENSITY, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31};

static const char* siaKernel =
    "   inline static uint2 ror64(const uint2 x, const uint y)                 "
    "                                   \n"
    "   {                                                                      "
    "                                   \n"
    "       return "
    "(uint2)(((x).x>>y)^((x).y<<(32-y)),((x).y>>y)^((x).x<<(32-y)));           "
    "                     \n"
    "   }                                                                      "
    "                                   \n"
    "   inline static uint2 ror64_2(const uint2 x, const uint y)               "
    "                                   \n"
    "   {                                                                      "
    "                                   \n"
    "       return "
    "(uint2)(((x).y>>(y-32))^((x).x<<(64-y)),((x).x>>(y-32))^((x).y<<(64-y))); "
    "                     \n"
    "   }                                                                      "
    "                                   \n"
    "   __constant static const uchar blake2b_sigma[12][16] = {                "
    "                                   \n"
    "       { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } "
    ",                                  \n"
    "       { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  } "
    ",                                  \n"
    "       { 11, 8,  12, 0,  5,  2,  15, 13, 10, 14, 3,  6,  7,  1,  9,  4  } "
    ",                                  \n"
    "       { 7,  9,  3,  1,  13, 12, 11, 14, 2,  6,  5,  10, 4,  0,  15, 8  } "
    ",                                  \n"
    "       { 9,  0,  5,  7,  2,  4,  10, 15, 14, 1,  11, 12, 6,  8,  3,  13 } "
    ",                                  \n"
    "       { 2,  12, 6,  10, 0,  11, 8,  3,  4,  13, 7,  5,  15, 14, 1,  9  } "
    ",                                  \n"
    "       { 12, 5,  1,  15, 14, 13, 4,  10, 0,  7,  6,  3,  9,  2,  8,  11 } "
    ",                                  \n"
    "       { 13, 11, 7,  14, 12, 1,  3,  9,  5,  0,  15, 4,  8,  6,  2,  10 } "
    ",                                  \n"
    "       { 6,  15, 14, 9,  11, 3,  0,  8,  12, 2,  13, 7,  1,  4,  10, 5  } "
    ",                                  \n"
    "       { 10, 2,  8,  4,  7,  6,  1,  5,  15, 11, 9,  14, 3,  12, 13, 0  } "
    ",                                  \n"
    "       { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } "
    ",                                  \n"
    "       { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  } "
    "};                                 \n"
    "   // Target is passed in via headerIn[32 - 29]                           "
    "                                   \n"
    "   __kernel void nonceGrind(__global ulong *headerIn, __global ulong "
    "*nonceOut) {                            \n"
    "       ulong target = headerIn[4];                                        "
    "                                   \n"
    "       ulong m[16] = {    headerIn[0], headerIn[1],                       "
    "                                   \n"
    "                       headerIn[2], headerIn[3],                          "
    "                                   \n"
    "                       (ulong)get_global_id(0), headerIn[5],              "
    "                                   \n"
    "                       headerIn[6], headerIn[7],                          "
    "                                   \n"
    "                       headerIn[8], headerIn[9], 0, 0, 0, 0, 0, 0 };      "
    "                                   \n"
    "       ulong v[16] = { 0x6a09e667f2bdc928, 0xbb67ae8584caa73b, "
    "0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,       \n"
    "                       0x510e527fade682d1, 0x9b05688c2b3e6c1f, "
    "0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,       \n"
    "                       0x6a09e667f3bcc908, 0xbb67ae8584caa73b, "
    "0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,       \n"
    "                       0x510e527fade68281, 0x9b05688c2b3e6c1f, "
    "0xe07c265404be4294, 0x5be0cd19137e2179 };     \n"
    "   #define G(r,i,a,b,c,d) \\\n"
    "       a = a + b + m[ blake2b_sigma[r][2*i] ]; \\\n"
    "       ((uint2*)&d)[0] = ((uint2*)&d)[0].yx ^ ((uint2*)&a)[0].yx; \\\n"
    "       c = c + d; \\\n"
    "       ((uint2*)&b)[0] = ror64( ((uint2*)&b)[0] ^ ((uint2*)&c)[0], 24U); "
    "\\\n"
    "       a = a + b + m[ blake2b_sigma[r][2*i+1] ]; \\\n"
    "       ((uint2*)&d)[0] = ror64( ((uint2*)&d)[0] ^ ((uint2*)&a)[0], 16U); "
    "\\\n"
    "       c = c + d; \\\n"
    "       ((uint2*)&b)[0] = ror64_2( ((uint2*)&b)[0] ^ ((uint2*)&c)[0], "
    "63U);\n"
    "   #define ROUND(r)                    \\\n"
    "       G(r,0,v[ 0],v[ 4],v[ 8],v[12]); \\\n"
    "       G(r,1,v[ 1],v[ 5],v[ 9],v[13]); \\\n"
    "       G(r,2,v[ 2],v[ 6],v[10],v[14]); \\\n"
    "       G(r,3,v[ 3],v[ 7],v[11],v[15]); \\\n"
    "       G(r,4,v[ 0],v[ 5],v[10],v[15]); \\\n"
    "       G(r,5,v[ 1],v[ 6],v[11],v[12]); \\\n"
    "       G(r,6,v[ 2],v[ 7],v[ 8],v[13]); \\\n"
    "       G(r,7,v[ 3],v[ 4],v[ 9],v[14]);                                    "
    "                                   \n"
    "       ROUND( 0 );                                                        "
    "                                   \n"
    "       ROUND( 1 );                                                        "
    "                                   \n"
    "       ROUND( 2 );                                                        "
    "                                   \n"
    "       ROUND( 3 );                                                        "
    "                                   \n"
    "       ROUND( 4 );                                                        "
    "                                   \n"
    "       ROUND( 5 );                                                        "
    "                                   \n"
    "       ROUND( 6 );                                                        "
    "                                   \n"
    "       ROUND( 7 );                                                        "
    "                                   \n"
    "       ROUND( 8 );                                                        "
    "                                   \n"
    "       ROUND( 9 );                                                        "
    "                                   \n"
    "       ROUND( 10 );                                                       "
    "                                   \n"
    "       ROUND( 11 );                                                       "
    "                                   \n"
    "   #undef G                                                               "
    "                                   \n"
    "   #undef ROUND                                                           "
    "                                   \n"
    "       if (as_ulong(as_uchar8(0x6a09e667f2bdc928 ^ v[0] ^ "
    "v[8]).s76543210) < target) {                       \n"
    "           *nonceOut = m[4];                                              "
    "                                   \n"
    "           return;                                                        "
    "                                   \n"
    "       }                                                                  "
    "                                   \n"
    "   }\n";

OCLPerfGenoilSiaMiner::OCLPerfGenoilSiaMiner() { _numSubTests = NUM_INTENSITY; }

OCLPerfGenoilSiaMiner::~OCLPerfGenoilSiaMiner() {}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfGenoilSiaMiner::setHeader(uint32_t* ptr) {
  ptr[0] = 0x10;
  for (unsigned int i = 1; i < 9; i++) {
    ptr[i] = 0;
  }
  ptr[9] = 0x4a5e1e4b;
  ptr[10] = 0xaab89f3a;
  ptr[11] = 0x32518a88;
  ptr[12] = 0xc31bc87f;
  ptr[13] = 0x618f7667;
  ptr[14] = 0x3e2cc77a;
  ptr[15] = 0xb2127b7a;
  ptr[16] = 0xfdeda33b;
  ptr[17] = 0x495fab29;
  ptr[18] = 0x1d00ffff;
  ptr[19] = 0x7c2bac1d;
}

void OCLPerfGenoilSiaMiner::open(unsigned int test, char* units, double& conversion,
                                 unsigned int deviceId) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  cl_device_id device = NULL;
  _crcword = 0;
  _deviceId = deviceId;
  _openTest = test;

  context_ = 0;
  cmd_queue_ = 0;
  // Parse args.
  isAMD = false;

  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
#if 0
        // Get last for default
        platform = platforms[numPlatforms-1];
        for (unsigned i = 0; i < numPlatforms; ++i) {
#endif
    platform = platforms[_platformIndex];
    char pbuf[100];
    error_ = _wrapper->clGetPlatformInfo(platforms[_platformIndex], CL_PLATFORM_VENDOR,
                                         sizeof(pbuf), pbuf, NULL);
    num_devices = 0;
    /* Get the number of requested devices */
    error_ = _wrapper->clGetDeviceIDs(platforms[_platformIndex], type_, 0, NULL, &num_devices);
    // Runtime returns an error when no GPU devices are present instead of just
    // returning 0 devices
    // CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");
    // Choose platform with GPU devices
    if (num_devices > 0) {
      if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) {
        isAMD = true;
      }
      // platform = platforms[_platformIndex];
      // break;
    }
#if 0
        }
#endif
    delete platforms;
  }

  char getVersion[128];
  error_ = _wrapper->clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(getVersion),
                                       getVersion, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformInfo failed");
  platformVersion[0] = getVersion[7];
  platformVersion[1] = getVersion[8];
  platformVersion[2] = getVersion[9];
  platformVersion[3] = '\0';

  /*
   * If we could find our platform, use it. If not, die as we need the AMD
   * platform for these extensions.
   */
  CHECK_RESULT(platform == 0, "Couldn't find AMD platform, cannot proceed");

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  CHECK_RESULT(devices == 0, "no devices");

  /* Get the requested device */
  error_ = _wrapper->clGetDeviceIDs(platform, type_, num_devices, devices, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");

  CHECK_RESULT(_deviceId >= num_devices, "Requested deviceID not available");
  device = devices[_deviceId];

  // Make sure the device can handle our local item size.
  size_t max_group_size = 0;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                                     &max_group_size, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");
  if (local_item_size > max_group_size) {
    char buf[256];
    SNPRINTF(buf, sizeof(buf), "Selected device cannot handle work groups larger than %zu.\n",
             local_item_size);
    local_item_size = max_group_size;
    testDescString = buf;
  }

  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  // Create Buffer Objects.
  blockHeadermobj_ =
      _wrapper->clCreateBuffer(context_, CL_MEM_READ_ONLY, 80 * sizeof(uint8_t), NULL, &error_);
  CHECK_RESULT(blockHeadermobj_ == 0, "clCreateBuffer(outBuffer) failed");
  nonceOutmobj_ =
      _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, 8 * sizeof(uint8_t), NULL, &error_);
  CHECK_RESULT(nonceOutmobj_ == 0, "clCreateBuffer(outBuffer) failed");

  // Create kernel program from source file.
  program_ =
      _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&siaKernel, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &device, NULL, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);

    CHECK_RESULT(0, "clBuildProgram failed");
  }
  // Create data parallel OpenCL kernel.
  kernel_ = _wrapper->clCreateKernel(program_, "nonceGrind", &error_);
  CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");

  // Set OpenCL kernel arguments.
  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&blockHeadermobj_);
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), (void*)&nonceOutmobj_);
}

void OCLPerfGenoilSiaMiner::run(void) {
  CPerfCounter timer;

  uint8_t blockHeader[80];
  uint8_t target[32] = {255};
  uint8_t nonceOut[8] = {0};

  setHeader((uint32_t*)blockHeader);
  intensity = intensities[_openTest % NUM_INTENSITY];
  size_t global_item_size = 1ULL << intensity;

  timer.Reset();
  timer.Start();

  // By doing a bunch of low intensity calls, we prevent freezing
  // By splitting them up inside this function, we also avoid calling
  // get_block_for_work too often.
  for (unsigned int i = 0; i < cycles_per_iter; i++) {
    // Offset global ids so that each loop call tries a different set of
    // hashes.
    size_t globalid_offset = i * global_item_size;

    // Copy input data to the memory buffer.
    error_ = clEnqueueWriteBuffer(cmd_queue_, blockHeadermobj_, CL_TRUE, 0, 80 * sizeof(uint8_t),
                                  blockHeader, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueWriteBuffer failed");

    error_ = clEnqueueWriteBuffer(cmd_queue_, nonceOutmobj_, CL_TRUE, 0, 8 * sizeof(uint8_t),
                                  nonceOut, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueWriteBuffer failed");

    // Run the kernel.
    error_ = clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, &globalid_offset, &global_item_size,
                                    &local_item_size, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");

    // Copy result to host and see if a block was found.
    error_ = clEnqueueReadBuffer(cmd_queue_, nonceOutmobj_, CL_TRUE, 0, 8 * sizeof(uint8_t),
                                 nonceOut, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueReadBuffer failed");

    // if (nonceOut[0] != 0) {
    //    // Copy nonce to header.
    //    memcpy(blockHeader + 32, nonceOut, 8);
    //    break;
    //}
  }
  _wrapper->clFinish(cmd_queue_);

  timer.Stop();
  double sec = timer.GetElapsedTime();

  // Hash rate calculation MH/s
  double hash_rate = cycles_per_iter * global_item_size / (sec * 1000000);

  _perfInfo = (float)hash_rate;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " (%4d cycles) Work_items:%10zu Intensity:%d (MH/s) ", cycles_per_iter,
           global_item_size, intensity);
  testDescString = buf;
}

unsigned int OCLPerfGenoilSiaMiner::close(void) {
  if (blockHeadermobj_) {
    error_ = _wrapper->clReleaseMemObject(blockHeadermobj_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(blockHeadermobj_) failed");
  }
  if (nonceOutmobj_) {
    error_ = _wrapper->clReleaseMemObject(nonceOutmobj_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(nonceOutmobj_) failed");
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
