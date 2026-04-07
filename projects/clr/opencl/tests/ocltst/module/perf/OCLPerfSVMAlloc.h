/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERF_SVM_ALLOC_H_
#define _OCL_PERF_SVM_ALLOC_H_

#include "OCLTestImp.h"

class OCLPerfSVMAlloc : public OCLTestImp {
 public:
  OCLPerfSVMAlloc();
  virtual ~OCLPerfSVMAlloc();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int testSize_;
  bool FGSystem_;
  unsigned int testCGFlag_;
  unsigned int testFGFlag_;
  bool skip_;
};

#endif  // _OCL_PERF_SVM_ALLOC_H_
