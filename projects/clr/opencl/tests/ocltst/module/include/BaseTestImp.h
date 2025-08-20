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

#ifndef _BaseTestImp_H_
#define _BaseTestImp_H_

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

#include "OCLTest.h"
#include "OCLWrapper.h"

#define EXIT_SILENT_FAILURE 2
#define KERNEL(...) #__VA_ARGS__

#ifdef _MSC_VER
#define NOMINMAX
#define snprintf sprintf_s
#endif

#define CHECK_ERROR(error, msg)                                                                    \
  if (error != CL_SUCCESS) {                                                                       \
    _errorFlag = true;                                                                             \
    printf("\n\n%s\nError code: %d\n\n", msg, error);                                              \
    _errorMsg = msg;                                                                               \
    _crcword += 1;                                                                                 \
    return;                                                                                        \
  }

#define CHECK_ERROR_NO_RETURN(error, msg)                                                          \
  if (error != CL_SUCCESS) {                                                                       \
    _errorFlag = true;                                                                             \
    printf("\n\n%s\nError code: %d\n\n", msg, error);                                              \
    _errorMsg = msg;                                                                               \
    _crcword += 1;                                                                                 \
  }

#define CHECK_RESULT(test, msg, ...)                                                               \
  if ((test)) {                                                                                    \
    char* buf = (char*)malloc(4096);                                                               \
    _errorFlag = true;                                                                             \
    int rc = snprintf(buf, 4096, msg, ##__VA_ARGS__);                                              \
    assert(rc >= 0 && rc < (int)4096);                                                             \
    printf("%s:%d - %s\n", __FILE__, __LINE__, buf);                                               \
    _errorMsg = std::string(buf);                                                                  \
    _crcword += 1;                                                                                 \
    free(buf);                                                                                     \
    return;                                                                                        \
  }

#define CHECK_RESULT_ARGS CHECK_RESULT

#define CHECK_RESULT_NO_RETURN(test, msg, ...)                                                     \
  if ((test)) {                                                                                    \
    char* buf = (char*)malloc(4096);                                                               \
    _errorFlag = true;                                                                             \
    int rc = snprintf(buf, 4096, msg, ##__VA_ARGS__);                                              \
    assert(rc >= 0 && rc < (int)4096);                                                             \
    printf("%s:%d - %s\n", __FILE__, __LINE__, buf);                                               \
    _errorMsg = std::string(msg);                                                                  \
    _crcword += 1;                                                                                 \
    free(buf);                                                                                     \
  }

#define CHECK_RESULT_NO_RETURN_ARGS CHECK_RESULT_NO_RETURN

#define CHECK_RESULT_SHUTDOWN(test, msg)                                                           \
  if ((test)) {                                                                                    \
    _errorFlag = true;                                                                             \
    printf("%s\n", msg);                                                                           \
    _errorMsg = msg;                                                                               \
    _crcword += 1;                                                                                 \
    close();                                                                                       \
    return;                                                                                        \
  }

#define CHECK_RESULT_CL(test, msg)                                                                 \
  if ((test)) {                                                                                    \
    _errorFlag = true;                                                                             \
    printf("%s\n", msg);                                                                           \
    _errorMsg = msg;                                                                               \
    _crcword += 1;                                                                                 \
    return 1;                                                                                      \
  }

class BaseTestImp : public OCLTest {
 public:
  BaseTestImp();
  virtual ~BaseTestImp();

 public:
  virtual unsigned int getThreadUsage(void);
  virtual int getNumSubTests(void);

  //! Abstract functions being defined here
  virtual void open();
  virtual void open(unsigned int test, const char* deviceName, unsigned int architecture);

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId,
                    unsigned int platformIndex) {
    return open(test, "", platformIndex);
  }

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
    return open(test, "", 0);
  }

  virtual void run(void) = 0;
  virtual unsigned int close(void);

  //! Functions to set class members
  virtual void checkComplib(unsigned int test, const char* deviceName, unsigned int architecture);
  virtual void setDeviceName(const char*);
  virtual const char* getDeviceName();
  virtual void setErrorMsg(const char* error);
  virtual const char* getErrorMsg(void);
  virtual bool hasErrorOccured(void);
  virtual void clearError();
  BaseTestImp* toBaseTestImp() { return this; }
  virtual OCLTestImp* toOCLTestImp() { return NULL; }
  virtual void useCPU() { _cpu = true; }
  virtual void setIterationCount(int cnt);
  virtual void setDeviceId(unsigned int deviceId);
  virtual unsigned int getDeviceId();
  virtual void setPlatformIndex(unsigned int platformIndex);
  virtual unsigned int getPlatformIndex();
  virtual float getPerfInfo();
  virtual void clearPerfInfo();

 protected:
  unsigned int _numSubTests;
  unsigned int _openTest;
  unsigned int _useThreads;
  int _iterationCnt;
  float _perfInfo;
  bool _cpu;

  unsigned int _crcword;
  unsigned int _crctab[256];

  bool _errorFlag;
  std::string _errorMsg;

  const char* _deviceName;
  unsigned int _architecture;
  unsigned int _deviceId;
  unsigned int _platformIndex;
  bool failed_ = false;
  cl_int error_;
  cl_uint type_;
  cl_uint deviceCount_;
  cl_device_id* devices_;
  cl_context context_;

  cl_program program_;
  cl_kernel kernel_;
};

// enum to keep track of different memory types
enum MemType { LOOCL, REMOTE_CACHED, REMOTE_UNCACHED };

class DataType {
  cl_image_format f;
  const char* str;
  unsigned int size;

 public:
  DataType() {}

  DataType(cl_image_format f, const char* str, unsigned int size) {
    this->f = f;
    this->str = str;
    this->size = size;
  }
  operator const char*() { return str; }

  operator unsigned int() { return size; }
  operator cl_image_format() { return f; }
};

// useful for initialization of an array of data types for a test
#define DTYPE(x, y) DataType(x, #x, (unsigned int)y)

#endif
