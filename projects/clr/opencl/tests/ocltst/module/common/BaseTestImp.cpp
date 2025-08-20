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

#include "BaseTestImp.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <cstring>

/////////////////////////////////////////////////////////////////////////////

static unsigned int crcinit(unsigned int crc);
static int initializeSeed(void);

/////////////////////////////////////////////////////////////////////////////

BaseTestImp::BaseTestImp() : _numSubTests(0), _openTest(0), _deviceName(NULL), _architecture(0) {
  _cpu = false;
  unsigned int i;
  for (i = 0; i < 256; i++) {
    _crctab[i] = crcinit(i << 24);
  }
  _crcword = ~0;
  _deviceId = 0;
  _platformIndex = 0;
  _perfInfo = 0.0f;

#ifdef __linux__    //
  _useThreads = 0;  // disable threads on linux
#else
  _useThreads = 1;  // if available on platform
#endif

  clearError();
}

void BaseTestImp::checkComplib(unsigned int test, const char* deviceName,
                               unsigned int architecture) {
  BaseTestImp::open();
  devices_ = 0;
  deviceCount_ = 0;
  context_ = 0;
  program_ = 0;
  kernel_ = 0;
  type_ = CL_DEVICE_TYPE_GPU;

  cl_uint numPlatforms = 0;
  error_ = clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT((error_ != CL_SUCCESS), "clGetPlatformIDs failed");
  CHECK_RESULT((numPlatforms == 0), "No platform found");

  cl_platform_id* platforms = new cl_platform_id[numPlatforms];
  error_ = clGetPlatformIDs(numPlatforms, platforms, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");

  cl_platform_id platform = 0;
#if 0
  for(unsigned int i = 0; i < numPlatforms; ++i)
  {
    char buff[200];
    error_ = clGetPlatformInfo(platforms[i],CL_PLATFORM_VENDOR, sizeof(buff), buff, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformInfo failed");
    if(strcmp(buff, "Advanced Micro Devices, Inc.") == 0)
    {
      platform = platforms[i];
      break;
    }
  }
#endif
  platform = platforms[_platformIndex];

  delete[] platforms;

  CHECK_RESULT((platform == 0), "AMD Platform not found");

  error_ = clGetDeviceIDs(platform, type_, 0, NULL, &deviceCount_);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs() failed");

  devices_ = new cl_device_id[deviceCount_];
  error_ = clGetDeviceIDs(platform, type_, deviceCount_, devices_, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs() failed");

  char device_string[200];
  clGetDeviceInfo(devices_[_deviceId], CL_DRIVER_VERSION, sizeof(device_string), &device_string,
                  NULL);
  if (strstr(device_string, "LC")) {
    printf("Skipping test since it does not run with LC\n");
    failed_ = true;
    return;
  }
  return;
}

BaseTestImp::~BaseTestImp() {}

void BaseTestImp::open() {
  _crcword = 0;
  clearError();
}
void BaseTestImp::open(unsigned int test, const char* deviceName, unsigned int architecture) {
  open();
}

unsigned int BaseTestImp::close() { return _crcword; }

unsigned int BaseTestImp::getThreadUsage(void) { return _useThreads; }

int BaseTestImp::getNumSubTests(void) { return _numSubTests; }

void BaseTestImp::setDeviceName(const char* name) { _deviceName = name; }

const char* BaseTestImp::getDeviceName() { return _deviceName; }

float BaseTestImp::getPerfInfo(void) { return _perfInfo; }

void BaseTestImp::clearPerfInfo(void) { _perfInfo = 0.0; }

void BaseTestImp::setDeviceId(unsigned int deviceId) { _deviceId = deviceId; }

void BaseTestImp::setIterationCount(int cnt) { _iterationCnt = cnt; }

unsigned int BaseTestImp::getDeviceId() { return _deviceId; }

void BaseTestImp::setPlatformIndex(unsigned int platformIndex) { _platformIndex = platformIndex; }

unsigned int BaseTestImp::getPlatformIndex() { return _platformIndex; }

void BaseTestImp::setErrorMsg(const char* error) {
  _errorFlag = true;
  _errorMsg.assign((const char*)error);
}

const char* BaseTestImp::getErrorMsg() { return _errorMsg.c_str(); }

bool BaseTestImp::hasErrorOccured() { return _errorFlag; }

void BaseTestImp::clearError() {
  _errorFlag = false;
  _errorMsg.clear();
}

/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
//
// Same CRC32 as used by ogtst
//
static const unsigned int CRCMASK = 0x04c11db7;

static unsigned int crcinit(unsigned int crc) {
  int i;
  unsigned int ans = crc;

  for (i = 0; i < 8; i++) {
    if (ans & 0x80000000) {
      ans = (ans << 1) ^ CRCMASK;
    } else {
      ans <<= 1;
    }
  }
  return (ans);
}
