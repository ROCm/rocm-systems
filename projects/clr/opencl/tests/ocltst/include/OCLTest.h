/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
