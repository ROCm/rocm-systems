/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERF_COMMAND_QUEUE_H_
#define _OCL_PERF_COMMAND_QUEUE_H_

#include "OCLTestImp.h"

class OCLPerfCommandQueue : public OCLTestImp {
 public:
  OCLPerfCommandQueue();
  virtual ~OCLPerfCommandQueue();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int test_;
};

#endif  // _OCL_PERF_COMMAND_QUEUE_H_
