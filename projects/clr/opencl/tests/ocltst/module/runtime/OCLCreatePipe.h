/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_OCLCreatePipe_H_
#define _OCL_OCLCreatePipe_H_

#include "OCLTestImp.h"

class OCLCreatePipe : public OCLTestImp {
 public:
  OCLCreatePipe();
  virtual ~OCLCreatePipe();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
};

#endif  // _OCL_OCLCreatePipe_H_
