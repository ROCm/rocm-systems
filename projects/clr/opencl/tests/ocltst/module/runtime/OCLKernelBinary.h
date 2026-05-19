/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_KERNEL_BINARY_H_
#define _OCL_KERNEL_BINARY_H_

#include "OCLTestImp.h"

class OCLKernelBinary : public OCLTestImp {
 public:
  OCLKernelBinary();
  virtual ~OCLKernelBinary();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
};

#endif  // _OCL_KERNEL_BINARY_H_
