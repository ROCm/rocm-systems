/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_CreateContext_H_
#define _OCL_CreateContext_H_

#include "OCLTestImp.h"

class OCLCreateContext : public OCLTestImp {
 public:
  OCLCreateContext();
  virtual ~OCLCreateContext();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
};

#endif  // _OCL_CreateContext_H_
