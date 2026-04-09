/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERF_DOUBLE_DMA_SEQ_H_
#define _OCL_PERF_DOUBLE_DMA_SEQ_H_

#include "OCLTestImp.h"

class OCLPerfDoubleDMASeq : public OCLTestImp {
 public:
  OCLPerfDoubleDMASeq();
  virtual ~OCLPerfDoubleDMASeq();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int test_;
  bool events_;
};

#endif  // _OCL_PERF_DOUBLE_DMA_SEQ_H_
