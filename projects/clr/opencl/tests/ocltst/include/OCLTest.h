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

#ifndef _OCLTEST_H_
#define _OCLTEST_H_

#include <string>

#include "OCLWrapper.h"

class BaseTestImp;
class OCLTestImp;
class OCLTest {
 public:
  virtual unsigned int getThreadUsage(void) = 0;
  virtual int getNumSubTests(void) = 0;
  virtual void open() = 0;
  virtual void open(unsigned int test, const char* deviceName, unsigned int architecture) = 0;

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId,
                    unsigned int platformIndex) = 0;

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId) = 0;

  virtual void run(void) = 0;
  virtual unsigned int close(void) = 0;
  virtual void setErrorMsg(const char* error) = 0;
  virtual const char* getErrorMsg(void) = 0;
  virtual bool hasErrorOccured(void) = 0;
  virtual void clearError() = 0;
  virtual void setDeviceId(unsigned int deviceId) = 0;
  virtual void setPlatformIndex(unsigned int platformIndex) = 0;
  virtual OCLTestImp* toOCLTestImp() = 0;
  virtual BaseTestImp* toBaseTestImp() = 0;
  virtual float getPerfInfo() = 0;
  virtual void clearPerfInfo(void) = 0;

  virtual void setIterationCount(int cnt) = 0;
  virtual void useCPU() = 0;
  // Having this return true will allow the creation of the
  // test to be cached in between runs and will only be
  // deleted after all the tests are finished running.
  // This defaults to false as not many tests are modified
  // to use it.
  // FIXME: Switch all tests to support caching.
  virtual bool cache_test() { return true; }

  std::string testDescString;
  void resetDescString(void) { testDescString.clear(); }

  virtual ~OCLTest() {};
};

#endif  // _OCLTEST_H_
