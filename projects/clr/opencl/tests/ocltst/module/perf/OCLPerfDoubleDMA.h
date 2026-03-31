/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERF_DOUBLE_DMA_H_
#define _OCL_PERF_DOUBLE_DMA_H_

#include "OCLTestImp.h"

class OCLPerfDoubleDMA : public OCLTestImp {
 public:
  OCLPerfDoubleDMA();
  virtual ~OCLPerfDoubleDMA();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int test_;
};

#endif  // _OCL_PERF_DOUBLE_DMA_H_
