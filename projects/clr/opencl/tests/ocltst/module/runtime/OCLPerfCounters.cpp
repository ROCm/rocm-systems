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

#include "OCLPerfCounters.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "Timer.h"

#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

struct PerfCounterInfo {
  cl_long blockIdx;    //!< Block Index
  cl_long counterIdx;  //!< Counter Index
  cl_long eventIdx;    //!< Event Index
};

struct DeviceCounterInfo {
  const char* deviceName_;          //!< Device name
  unsigned int devId_;              //!< Device id
  PerfCounterInfo perfCounter_[2];  //!< Perforamnce counter array
};

static const DeviceCounterInfo DeviceInfo[]{
#ifdef _WIN32
    // GFX11 supports performance counter on Windows only.
    {"gfx1100", 11, {{139, 0, 4}, {74, 0, 13}}},  // {SQWGP, reg 0, SQ_PERF_SEL_WAVES}, {CPC,
                                                  // reg 0, Me1 busy for packet decode}
    {"gfx1101", 11, {{139, 0, 4}, {74, 0, 13}}},  // {SQWGP, reg 0, SQ_PERF_SEL_WAVES}, {CPC,
                                                  // reg 0, Me1 busy for packet decode}
    {"gfx1102", 11, {{139, 0, 4}, {74, 0, 13}}},  // {SQWGP, reg 0, SQ_PERF_SEL_WAVES}, {CPC,
                                                  // reg 0, Me1 busy for packet decode}
    {"gfx1103", 11, {{139, 0, 4}, {74, 0, 13}}},  // {SQWGP, reg 0, SQ_PERF_SEL_WAVES}, {CPC,
                                                  // reg 0, Me1 busy for packet decode}
#endif
    // GFX10
    {"gfx1000", 10, {{15, 0, 4}, {74, 0, 13}}},  // {SQ, reg 0, SQ_PERF_SEL_WAVES}, {CPC,
                                                 // reg 0, Me1 busy for packet decode}
    {"gfx1010", 10, {{15, 0, 4}, {74, 0, 13}}},  // {SQ, reg 0, SQ_PERF_SEL_WAVES}, {CPC,
                                                 // reg 0, Me1 busy for packet decode}
    {"gfx1011", 10, {{15, 0, 4}, {74, 0, 13}}},  // {SQ, reg 0, SQ_PERF_SEL_WAVES}, {CPC,
                                                 // reg 0, Me1 busy for packet decode}
    {"gfx1012", 10, {{15, 0, 4}, {74, 0, 13}}},  // {SQ, reg 0, SQ_PERF_SEL_WAVES}, {CPC,
                                                 // reg 0, Me1 busy for packet decode}
    // GFX9
    {"gfx900", 9, {{14, 0, 4}, {97, 1, 2}}},  // {SQ, reg 0, SQ_PERF_SEL_WAVES}, {MCVML2_l,
                                              // reg 0, BigK bank 0 hits}
    {"gfx901", 9, {{14, 0, 4}, {97, 1, 2}}},  // {SQ, reg 0, SQ_PERF_SEL_WAVES}, {MCVML2_l,
                                              // reg 0, BigK bank 0 hits}
    {"gfx902", 9, {{14, 0, 4}, {97, 1, 2}}},  // {SQ, reg 0, SQ_PERF_SEL_WAVES}, {MCVML2_l,
                                              // reg 0, BigK bank 0 hits}
    {"gfx903", 9, {{14, 0, 4}, {97, 1, 2}}},  // {SQ, reg 0, SQ_PERF_SEL_WAVES}, {MCVML2_l,
                                              // reg 0, BigK bank 0 hits}
    {"gfx904", 9, {{14, 0, 4}, {97, 1, 2}}},  // {SQ, reg 0, SQ_PERF_SEL_WAVES}, {MCVML2_l,
                                              // reg 0, BigK bank 0 hits}
    {"gfx905", 9, {{14, 0, 4}, {97, 1, 2}}},  // {SQ, reg 0, SQ_PERF_SEL_WAVES}, {MCVML2_l,
                                              // reg 0, BigK bank 0 hits}
    {"gfx906", 9, {{14, 0, 4}, {97, 1, 2}}},  // {SQ, reg 0, SQ_PERF_SEL_WAVES}, {MCVML2_l,
                                              // reg 0, BigK bank 0 hits}
    {"gfx907", 9, {{14, 0, 4}, {97, 1, 2}}},  // {SQ, reg 0, SQ_PERF_SEL_WAVES}, {MCVML2_l,
                                              // reg 0, BigK bank 0 hits}
};
const int DeviceCounterSize = sizeof(DeviceInfo) / sizeof(DeviceCounterInfo);

static const char* sha256_kernel =
    "typedef uint UINT;\n"
    "\n"
    "#define VECTOR_LEN 1\n"
    "\n"
    "#ifdef LITTLE_E\n"
    "\n"
    "inline UINT byteswap(UINT x)\n"
    "{\n"
    "    UINT res = 0;\n"
    "    \n"
    "    for (uint i=0; i<4; i++)\n"
    "    {\n"
    "        res <<= 8;\n"
    "        res |= (x & 0xff);\n"
    "        x >>= 8;\n"
    "    }\n"
    "    \n"
    "    return res;\n"
    "}\n"
    "\n"
    "#else\n"
    "\n"
    "inline UINT byteswap(const UINT x)\n"
    "{\n"
    "    return x;\n"
    "}\n"
    "\n"
    "#endif\n"
    "\n"
    "\n"
    "void sha256_step( const UINT data[16], UINT *state )\n"
    "{\n"
    "   UINT W[64], temp1, temp2;\n"
    "   UINT A, B, C, D, E, F, G, H;\n"
    "\n"
    "   for( int i = 0; i < 16; i++)\n"
    "   {\n"
    "      W[i] = byteswap(data[i]);\n"
    "   }\n"
    "\n"
    "#define SHR(x,n)  ((x & 0xFFFFFFFF) >> n)\n"
    "#define ROTR(x,n) (SHR(x,n) | (x << (32 - n)))\n"
    "\n"
    "#define S0(x) (ROTR(x, 7) ^ ROTR(x,18) ^  SHR(x, 3))\n"
    "#define S1(x) (ROTR(x,17) ^ ROTR(x,19) ^  SHR(x,10))\n"
    "\n"
    "#define S2(x) (ROTR(x, 2) ^ ROTR(x,13) ^ ROTR(x,22))\n"
    "#define S3(x) (ROTR(x, 6) ^ ROTR(x,11) ^ ROTR(x,25))\n"
    "\n"
    "#define F0(x,y,z) ((x & y) | (z & (x | y)))\n"
    "#define F1(x,y,z) (z ^ (x & (y ^ z)))\n"
    "\n"
    "#define R(t)                                    \\\n"
    "(                                               \\\n"
    "    W[t] = S1(W[t -  2]) + W[t -  7] +          \\\n"
    "           S0(W[t - 15]) + W[t - 16]            \\\n"
    ")\n"
    "\n"
    "#define P(a,b,c,d,e,f,g,h,x,K)                  \\\n"
    "{                                               \\\n"
    "    temp1 = h + S3(e) + F1(e,f,g) + K + x;      \\\n"
    "    temp2 = S2(a) + F0(a,b,c);                  \\\n"
    "    d += temp1; h = temp1 + temp2;              \\\n"
    "}\n"
    "\n"
    "    A = state[0];\n"
    "    B = state[1];\n"
    "    C = state[2];\n"
    "    D = state[3];\n"
    "    E = state[4];\n"
    "    F = state[5];\n"
    "    G = state[6];\n"
    "    H = state[7];\n"
    "\n"
    "    P( A, B, C, D, E, F, G, H, W[ 0], 0x428A2F98 );\n"
    "    P( H, A, B, C, D, E, F, G, W[ 1], 0x71374491 );\n"
    "    P( G, H, A, B, C, D, E, F, W[ 2], 0xB5C0FBCF );\n"
    "    P( F, G, H, A, B, C, D, E, W[ 3], 0xE9B5DBA5 );\n"
    "    P( E, F, G, H, A, B, C, D, W[ 4], 0x3956C25B );\n"
    "    P( D, E, F, G, H, A, B, C, W[ 5], 0x59F111F1 );\n"
    "    P( C, D, E, F, G, H, A, B, W[ 6], 0x923F82A4 );\n"
    "    P( B, C, D, E, F, G, H, A, W[ 7], 0xAB1C5ED5 );\n"
    "    P( A, B, C, D, E, F, G, H, W[ 8], 0xD807AA98 );\n"
    "    P( H, A, B, C, D, E, F, G, W[ 9], 0x12835B01 );\n"
    "    P( G, H, A, B, C, D, E, F, W[10], 0x243185BE );\n"
    "    P( F, G, H, A, B, C, D, E, W[11], 0x550C7DC3 );\n"
    "    P( E, F, G, H, A, B, C, D, W[12], 0x72BE5D74 );\n"
    "    P( D, E, F, G, H, A, B, C, W[13], 0x80DEB1FE );\n"
    "    P( C, D, E, F, G, H, A, B, W[14], 0x9BDC06A7 );\n"
    "    P( B, C, D, E, F, G, H, A, W[15], 0xC19BF174 );\n"
    "    P( A, B, C, D, E, F, G, H, R(16), 0xE49B69C1 );\n"
    "    P( H, A, B, C, D, E, F, G, R(17), 0xEFBE4786 );\n"
    "    P( G, H, A, B, C, D, E, F, R(18), 0x0FC19DC6 );\n"
    "    P( F, G, H, A, B, C, D, E, R(19), 0x240CA1CC );\n"
    "    P( E, F, G, H, A, B, C, D, R(20), 0x2DE92C6F );\n"
    "    P( D, E, F, G, H, A, B, C, R(21), 0x4A7484AA );\n"
    "    P( C, D, E, F, G, H, A, B, R(22), 0x5CB0A9DC );\n"
    "    P( B, C, D, E, F, G, H, A, R(23), 0x76F988DA );\n"
    "    P( A, B, C, D, E, F, G, H, R(24), 0x983E5152 );\n"
    "    P( H, A, B, C, D, E, F, G, R(25), 0xA831C66D );\n"
    "    P( G, H, A, B, C, D, E, F, R(26), 0xB00327C8 );\n"
    "    P( F, G, H, A, B, C, D, E, R(27), 0xBF597FC7 );\n"
    "    P( E, F, G, H, A, B, C, D, R(28), 0xC6E00BF3 );\n"
    "    P( D, E, F, G, H, A, B, C, R(29), 0xD5A79147 );\n"
    "    P( C, D, E, F, G, H, A, B, R(30), 0x06CA6351 );\n"
    "    P( B, C, D, E, F, G, H, A, R(31), 0x14292967 );\n"
    "    P( A, B, C, D, E, F, G, H, R(32), 0x27B70A85 );\n"
    "    P( H, A, B, C, D, E, F, G, R(33), 0x2E1B2138 );\n"
    "    P( G, H, A, B, C, D, E, F, R(34), 0x4D2C6DFC );\n"
    "    P( F, G, H, A, B, C, D, E, R(35), 0x53380D13 );\n"
    "    P( E, F, G, H, A, B, C, D, R(36), 0x650A7354 );\n"
    "    P( D, E, F, G, H, A, B, C, R(37), 0x766A0ABB );\n"
    "    P( C, D, E, F, G, H, A, B, R(38), 0x81C2C92E );\n"
    "    P( B, C, D, E, F, G, H, A, R(39), 0x92722C85 );\n"
    "    P( A, B, C, D, E, F, G, H, R(40), 0xA2BFE8A1 );\n"
    "    P( H, A, B, C, D, E, F, G, R(41), 0xA81A664B );\n"
    "    P( G, H, A, B, C, D, E, F, R(42), 0xC24B8B70 );\n"
    "    P( F, G, H, A, B, C, D, E, R(43), 0xC76C51A3 );\n"
    "    P( E, F, G, H, A, B, C, D, R(44), 0xD192E819 );\n"
    "    P( D, E, F, G, H, A, B, C, R(45), 0xD6990624 );\n"
    "    P( C, D, E, F, G, H, A, B, R(46), 0xF40E3585 );\n"
    "    P( B, C, D, E, F, G, H, A, R(47), 0x106AA070 );\n"
    "    P( A, B, C, D, E, F, G, H, R(48), 0x19A4C116 );\n"
    "    P( H, A, B, C, D, E, F, G, R(49), 0x1E376C08 );\n"
    "    P( G, H, A, B, C, D, E, F, R(50), 0x2748774C );\n"
    "    P( F, G, H, A, B, C, D, E, R(51), 0x34B0BCB5 );\n"
    "    P( E, F, G, H, A, B, C, D, R(52), 0x391C0CB3 );\n"
    "    P( D, E, F, G, H, A, B, C, R(53), 0x4ED8AA4A );\n"
    "    P( C, D, E, F, G, H, A, B, R(54), 0x5B9CCA4F );\n"
    "    P( B, C, D, E, F, G, H, A, R(55), 0x682E6FF3 );\n"
    "    P( A, B, C, D, E, F, G, H, R(56), 0x748F82EE );\n"
    "    P( H, A, B, C, D, E, F, G, R(57), 0x78A5636F );\n"
    "    P( G, H, A, B, C, D, E, F, R(58), 0x84C87814 );\n"
    "    P( F, G, H, A, B, C, D, E, R(59), 0x8CC70208 );\n"
    "    P( E, F, G, H, A, B, C, D, R(60), 0x90BEFFFA );\n"
    "    P( D, E, F, G, H, A, B, C, R(61), 0xA4506CEB );\n"
    "    P( C, D, E, F, G, H, A, B, R(62), 0xBEF9A3F7 );\n"
    "    P( B, C, D, E, F, G, H, A, R(63), 0xC67178F2 );\n"
    "\n"
    "    state[0] += A;\n"
    "    state[1] += B;\n"
    "    state[2] += C;\n"
    "    state[3] += D;\n"
    "    state[4] += E;\n"
    "    state[5] += F;\n"
    "    state[6] += G;\n"
    "    state[7] += H;\n"
    "}\n"
    "\n"
    "\n"
    "#define choose_temp(x) ((x)/16)\n"
    "\n"
    "#define STORE_TO_TEMP(i) tb[((i)/16)][((i)%16)]\n"
    "\n"
    "\n"
    "__kernel void CryptThread(__global const uint *buffer, __global uint "
    "*state, const uint blockLen, const uint foo)\n"
    "{\n"
    "    const uint init[8] = {\n"
    "        0x6a09e667,\n"
    "        0xbb67ae85,\n"
    "        0x3c6ef372,\n"
    "        0xa54ff53a,\n"
    "        0x510e527f,\n"
    "        0x9b05688c,\n"
    "        0x1f83d9ab,\n"
    "        0x5be0cd19\n"
    "    };\n"
    "    \n"
    "    const uint id = get_global_id(0);\n"
    "    uint len = blockLen;\n"
    "    uint i, j;\n"
    "    const uint startPosInDWORDs = (len*id*foo)/4;\n"
    "    const uint msgLenInBitsl = len * 8;\n"
    "    const uint msgLenInBitsh = (len) >> (32-3);\n"
    "    UINT localState[8];\n"
    "\n"
    "    for (j=0; j<8; j++) {\n"
    "        localState[j] = init[j];\n"
    "    }\n"
    "\n"
    "    i = 0;\n"
    "    while (len >=64)\n"
    "    {\n"
    "        UINT data[16];\n"
    "        for (j=0; j<16; j++) {\n"
    "            data[j] = buffer[j + startPosInDWORDs + i];\n"
    "        }\n"
    "\n"
    "        sha256_step(data, localState);\n"
    "        i += 16;\n"
    "        len -= 64;\n"
    "    }\n"
    "\n"
    "    len /= 4;\n"
    "\n"
    "    UINT tb[2][16];\n"
    "\n"
    "    for (j=0; j<len; j++) \n"
    "    {\n"
    "        STORE_TO_TEMP(j) = buffer[j + startPosInDWORDs + i];\n"
    "    }\n"
    "\n"
    "#ifdef LITTLE_E\n"
    "    STORE_TO_TEMP(len) = 0x80;\n"
    "#else\n"
    "    STORE_TO_TEMP(len) = byteswap(0x80000000);\n"
    "#endif\n"
    "\n"
    "    i = len+1;\n"
    "\n"
    "    while ((i % (512/32)) != (448/32))\n"
    "    {\n"
    "        STORE_TO_TEMP(i) = 0;\n"
    "        i++;\n"
    "    }\n"
    "\n"
    "#ifdef LITTLE_E\n"
    "    {\n"
    "        STORE_TO_TEMP(i) = byteswap(msgLenInBitsh);\n"
    "        STORE_TO_TEMP(i + 1) = byteswap(msgLenInBitsl);\n"
    "        i += 2;\n"
    "    }\n"
    "\n"
    "#else\n"
    "#endif\n"
    "    \n"
    "    sha256_step(tb[0], localState);\n"
    "    if (32 == i)\n"
    "    {\n"
    "        sha256_step(tb[1], localState);\n"
    "    }\n"
    "    \n"
    "    for (j=0; j<8; j++)\n"
    "    {\n"
    "        state[id*8 + j] = localState[j];\n"
    "    }\n"
    "}\n";

#define NUM_COUNTERS 2

cl_device_id global_device;

OCLPerfCounters::OCLPerfCounters() { _numSubTests = NUM_COUNTERS; }

OCLPerfCounters::~OCLPerfCounters() {}

bool OCLPerfCounters::setData(cl_mem buffer, unsigned int val) {
  bool retVal = false;
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_, buffer, true, CL_MAP_WRITE, 0, bufSize_, 0, NULL, NULL, &error_);

  if (error_ != CL_SUCCESS) {
    printf("\nError code : %d\n", error_);
  } else {
    for (unsigned int i = 0; i < width_; i++) data[i] = val;
    error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
    if (error_ == CL_SUCCESS) retVal = true;
  }
  return retVal;
}

void OCLPerfCounters::checkData(cl_mem buffer) {
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_, buffer, true, CL_MAP_READ, 0, bufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < width_; i++) {
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfCounters::open(unsigned int test, char* units, double& conversion,
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
  num_input_buf_ = 1;
  num_output_buf_ = 1;
  blockSize_ = 1024;
  isAMD = false;

  if (type_ != CL_DEVICE_TYPE_GPU) {
    char msg[256];
    SNPRINTF(msg, sizeof(msg), "No GPU devices present. Exiting!\t");
    testDescString = msg;
    return;
  }

  width_ = 22347776;
  // We compute a square domain
  bufSize_ = width_ * sizeof(cl_uint);

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

  global_device = device;

  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  char charbuf[1024];
  size_t retsize;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1024, charbuf, &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  inBuffer_ = new cl_mem[4];
  outBuffer_ = new cl_mem[4];

  for (int i = 0; i < num_input_buf_; ++i) {
    inBuffer_[i] = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
    CHECK_RESULT(inBuffer_[i] == 0, "clCreateBuffer(inBuffer) failed");
    bool result = setData(inBuffer_[i], 0xdeadbeef);
    CHECK_RESULT(result != true, "clEnqueueMapBuffer buffer failed");
  }

  for (int i = 0; i < num_output_buf_; ++i) {
    outBuffer_[i] = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
    CHECK_RESULT(outBuffer_[i] == 0, "clCreateBuffer(outBuffer) failed");
    bool result = setData(outBuffer_[i], 0xdeadbeef);
    CHECK_RESULT(result != true, "clEnqueueMapBuffer buffer failed");
  }

  program_ =
      _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&sha256_kernel, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  const char* buildOps = NULL;
  if (isAMD) {
    // Enable caching
    buildOps = "-fno-alias";
  }
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

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&inBuffer_[0]);
  error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), (void*)&outBuffer_[0]);
  error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_uint), (void*)&blockSize_);
  // Foo is not part of the original test, this can be used to see how much of
  // the performance is limited by fetch. Set foo to 0 and all threads will
  // fetch the same 1k block.  This way they will all be in cache and hit max
  // fetch speed.
  unsigned int foo = 1;
  error_ = _wrapper->clSetKernelArg(kernel_, 3, sizeof(cl_uint), (void*)&foo);
}

void OCLPerfCounters::run(void) {
  // Test runs only on GPU
  if (type_ != CL_DEVICE_TYPE_GPU) return;

  size_t global = bufSize_ / blockSize_;
  // 32 gives the best result due to memory thrashing.  Need to optimize and
  // give feedback to SiSoft.
  size_t local = 64;
  char buf[256];

  size_t global_work_size[1] = {global};
  size_t local_work_size[1] = {local};

  cl_int err = 0;
  cl_perfcounter_amd perfCounter;
  cl_perfcounter_property properties[4][2];
  cl_event perfEvent;
  cl_ulong result;
  char deviceName[1024];

  properties[0][0] = CL_PERFCOUNTER_GPU_BLOCK_INDEX;
  properties[1][0] = CL_PERFCOUNTER_GPU_COUNTER_INDEX;
  properties[2][0] = CL_PERFCOUNTER_GPU_EVENT_INDEX;
  properties[3][0] = CL_PERFCOUNTER_NONE;

  err = _wrapper->clGetDeviceInfo(global_device, CL_DEVICE_NAME, 1024, deviceName, NULL);
  CHECK_RESULT(err != CL_SUCCESS, "clGetDeviceInfo failed");

  // Remove target ID features
  char* targetIdColon = strchr(deviceName, ':');
  if (targetIdColon != nullptr) {
    size_t idx = targetIdColon - deviceName;
    deviceName[idx] = '\0';
  }

  bool found = false;
  unsigned int devId = 0;
  for (int idx = 0; !found && idx < DeviceCounterSize; idx++) {
    if (strcmp(deviceName, DeviceInfo[idx].deviceName_) == 0) {
      devId = DeviceInfo[idx].devId_;
      properties[0][1] = DeviceInfo[idx].perfCounter_[_openTest].blockIdx;
      properties[1][1] = DeviceInfo[idx].perfCounter_[_openTest].counterIdx;
      properties[2][1] = DeviceInfo[idx].perfCounter_[_openTest].eventIdx;
      found = true;
    }
  }

  if (!found) {
    char msg[256];
    SNPRINTF(msg, sizeof(msg), "Unsupported device(%s) for the test!\t", deviceName);
    testDescString = msg;
    return;
  }

  perfCounter = _wrapper->clCreatePerfCounterAMD(global_device, &properties[0][0], &err);
  CHECK_RESULT(err != CL_SUCCESS, "Create PerfCounter failed\n");

  // set clock mode
  cl_set_device_clock_mode_input_amd setClockModeInput;
  setClockModeInput.clock_mode = CL_DEVICE_CLOCK_MODE_PROFILING_AMD;
  cl_set_device_clock_mode_output_amd setClockModeOutput = {};
  _wrapper->clSetDeviceClockModeAMD(global_device, setClockModeInput, &setClockModeOutput);

  _wrapper->clEnqueueBeginPerfCounterAMD(cmd_queue_, 1, &perfCounter, 0, NULL, NULL);

  for (unsigned int i = 0; i < MAX_ITERATIONS; i++) {
    error_ =
        _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&inBuffer_[i % num_input_buf_]);
    error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem),
                                      (void*)&outBuffer_[i % num_output_buf_]);

    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);
  }

  CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueNDRangeKernel failed");

  _wrapper->clEnqueueEndPerfCounterAMD(cmd_queue_, 1, &perfCounter, 0, NULL, &perfEvent);
  clWaitForEvents(1, &perfEvent);

  // set clock mode to default
  setClockModeInput.clock_mode = CL_DEVICE_CLOCK_MODE_DEFAULT_AMD;
  _wrapper->clSetDeviceClockModeAMD(global_device, setClockModeInput, &setClockModeOutput);

  error_ = _wrapper->clGetPerfCounterInfoAMD(perfCounter, CL_PERFCOUNTER_DATA, sizeof(cl_ulong),
                                             &result, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS,
               "clGetPerfCounterInfoAMD failed (Hint (Linux): install hsa-amd-aqlprofile)\n");

  err = _wrapper->clReleasePerfCounterAMD(perfCounter);
  CHECK_RESULT(err != CL_SUCCESS, "Release PerfCounter failed\n");

  switch (_openTest) {
    case 0:
      SNPRINTF(buf, sizeof(buf), "SQ Number of Waves: %lu  ", (long)result);
      break;
    case 1:
      if (devId >= 9) {
        SNPRINTF(buf, sizeof(buf), "Me1 busy for packet decode: %lu  ", (long)result);
      } else {
        SNPRINTF(buf, sizeof(buf), "GRBM CP Busy: %lu  ", (long)result);
      }
      break;
  }

  testDescString = buf;
  CHECK_RESULT(!(result > 0), "Perf counter value read is zero!\n");
}

unsigned int OCLPerfCounters::close(void) {
  _wrapper->clFinish(cmd_queue_);

  if (inBuffer_) {
    for (int i = 0; i < num_input_buf_; ++i) {
      error_ = _wrapper->clReleaseMemObject(inBuffer_[i]);
      CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(inBuffer_) failed");
    }
    delete[] inBuffer_;
  }
  if (outBuffer_) {
    for (int i = 0; i < num_output_buf_; ++i) {
      error_ = _wrapper->clReleaseMemObject(outBuffer_[i]);
      CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(outBuffer_) failed");
    }
    delete[] outBuffer_;
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
