/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_MULTI_QUEUE_H_
#define _OCL_MULTI_QUEUE_H_

#include "OCLTestImp.h"

class OCLMultiQueue : public OCLTestImp {
 public:
  OCLMultiQueue();
  virtual ~OCLMultiQueue();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool test(cl_kernel kernel, cl_uint numRuns, cl_uint numQueues);
  bool failed_;
  unsigned int test_;
};

#endif  // _OCL_ASYNC_TRANSFER_H_
