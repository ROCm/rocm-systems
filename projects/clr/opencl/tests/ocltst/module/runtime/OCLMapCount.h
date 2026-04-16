/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_MAP_COUNT_H_
#define _OCL_MAP_COUNT_H_

#include "OCLTestImp.h"

class OCLMapCount : public OCLTestImp {
 public:
  OCLMapCount();
  virtual ~OCLMapCount();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
};

#endif  // _OCL_MAP_COUNT_H_

class clMemWrapper {
 public:
  clMemWrapper() { mMem = NULL; }
  clMemWrapper(cl_mem mem) { mMem = mem; }
  ~clMemWrapper() {
    if (mMem != NULL) clReleaseMemObject(mMem);
  }

  clMemWrapper& operator=(const cl_mem& rhs) {
    mMem = rhs;
    return *this;
  }
  operator cl_mem() { return mMem; }

  cl_mem* operator&() { return &mMem; }

  bool operator==(const cl_mem& rhs) { return mMem == rhs; }

 protected:
  cl_mem mMem;
};
