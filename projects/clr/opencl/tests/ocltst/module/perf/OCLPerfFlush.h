/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERF_FLUSH_H_
#define _OCL_PERF_FLUSH_H_

#include "OCLTestImp.h"

class OCLPerfFlush : public OCLTestImp {
 public:
  OCLPerfFlush();
  virtual ~OCLPerfFlush();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int test_;
};

#endif  // _OCL_PERF_FLUSH_H_
