/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_BLIT_KERNEL_H_
#define _OCL_BLIT_KERNEL_H_

#include "OCLTestImp.h"

class OCLBlitKernel : public OCLTestImp {
 public:
  OCLBlitKernel();
  virtual ~OCLBlitKernel();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  float time_;
};

#endif  // _OCL_BLIT_KERNEL_H_
