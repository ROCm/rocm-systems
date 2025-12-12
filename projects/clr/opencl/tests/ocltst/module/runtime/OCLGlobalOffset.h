/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_GLOBAL_OFFSET_H_
#define _OCL_GLOBAL_OFFSET_H_

#include "OCLTestImp.h"

class OCLGlobalOffset : public OCLTestImp {
 public:
  OCLGlobalOffset();
  virtual ~OCLGlobalOffset();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
};

#endif  // _OCL_GLOBAL_OFFSET_H_
