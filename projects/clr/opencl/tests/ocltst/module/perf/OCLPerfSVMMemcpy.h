/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERF_SVM_MEMCPY_H_
#define _OCL_PERF_SVM_MEMCPY_H_

#include "OCLTestImp.h"

class OCLPerfSVMMemcpy : public OCLTestImp {
 public:
  OCLPerfSVMMemcpy();
  virtual ~OCLPerfSVMMemcpy();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int testSize_;
  unsigned int testSrcFlag_;
  unsigned int testDstFlag_;
  unsigned int testFGFlag_;
  bool FGSystem_;
  bool skip_;
};

#endif  // _OCL_PERF_SVM_MEMCPY_H_
