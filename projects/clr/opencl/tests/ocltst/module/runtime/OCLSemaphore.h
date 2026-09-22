/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_SEMAPHORE_H_
#define _OCL_SEMAPHORE_H_

#include "OCLTestImp.h"

class OCLSemaphore : public OCLTestImp {
 public:
  OCLSemaphore();
  virtual ~OCLSemaphore();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
  bool hasSemaphore;
};

#endif  // _OCL_SEMAPHORE_H_
