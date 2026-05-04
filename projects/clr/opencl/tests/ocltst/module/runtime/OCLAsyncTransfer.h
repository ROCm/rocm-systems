/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_ASYNC_TRANSFER_H_
#define _OCL_ASYNC_TRANSFER_H_

#include "OCLTestImp.h"

class OCLAsyncTransfer : public OCLTestImp {
 public:
  OCLAsyncTransfer();
  virtual ~OCLAsyncTransfer();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
};

#endif  // _OCL_ASYNC_TRANSFER_H_
