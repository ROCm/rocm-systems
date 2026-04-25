/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_MEM_DEPENDENCY_H_
#define _OCL_MEM_DEPENDENCY_H_

#include "OCLTestImp.h"

class OCLMemDependency : public OCLTestImp {
 public:
  OCLMemDependency();
  virtual ~OCLMemDependency();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
};

#endif  // _OCL_MEM_DEPENDENCY_H_
