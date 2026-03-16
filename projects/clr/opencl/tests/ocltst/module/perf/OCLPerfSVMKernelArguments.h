/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERF_SVM_KERNEL_ARGUMENTS_H_
#define _OCL_PERF_SVM_KERNEL_ARGUMENTS_H_

#include <vector>

#include "OCLTestImp.h"

class OCLPerfSVMKernelArguments : public OCLTestImp {
 public:
  OCLPerfSVMKernelArguments();
  virtual ~OCLPerfSVMKernelArguments();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int test_;
  bool skip_;
  void** inOutBuffer;
  unsigned int numBufs_;
};

#endif  // _OCL_PERF_SVM_KERNEL_ARGUMENTS_H_
