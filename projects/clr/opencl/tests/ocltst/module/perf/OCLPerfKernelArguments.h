/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERF_KERNEL_ARGUMENTS_H_
#define _OCL_PERF_KERNEL_ARGUMENTS_H_

#include "OCLTestImp.h"

class OCLPerfKernelArguments : public OCLTestImp {
 public:
  OCLPerfKernelArguments();
  virtual ~OCLPerfKernelArguments();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int test_;
  bool perBatch_;
};

#endif  // _OCL_PERF_KERNEL_ARGUMENTS_H_
