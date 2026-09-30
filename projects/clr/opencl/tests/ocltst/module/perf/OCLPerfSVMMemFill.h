/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERF_SVM_MEMFILL_H_
#define _OCL_PERF_SVM_MEMFILL_H_

#include "OCLTestImp.h"

class OCLPerfSVMMemFill : public OCLTestImp {
 public:
  OCLPerfSVMMemFill();
  virtual ~OCLPerfSVMMemFill();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  unsigned int num_typeSize_;
  unsigned int num_elements_;
  bool FGSystem_;
  size_t testTypeSize_;
  unsigned int testCGFlag_;
  unsigned int testFGFlag_;
  unsigned int testNumEle_;
  bool atomic_;
  bool failed_;
  bool skip_;
};

#endif  // _OCL_PERF_SVM_MEMFILL_H_
