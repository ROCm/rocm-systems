/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PLATFORM_ATOMICS_H_
#define _OCL_PLATFORM_ATOMICS_H_

#include "OCLTestImp.h"

class OCLPlatformAtomics : public OCLTestImp {
 public:
  OCLPlatformAtomics();
  virtual ~OCLPlatformAtomics();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  bool failed_;
  unsigned long long svmCaps_;
};

#endif  // _OCL_KERNEL_BINARY_H_
