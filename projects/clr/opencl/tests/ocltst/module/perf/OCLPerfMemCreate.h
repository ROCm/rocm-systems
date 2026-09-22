/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERF_MEM_CREATE_H_
#define _OCL_PERF_MEM_CREATE_H_

#include "OCLTestImp.h"

class OCLPerfMemCreate : public OCLTestImp {
 public:
  OCLPerfMemCreate();
  virtual ~OCLPerfMemCreate();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int test_;
  bool useSubBuf_;
};

#endif  // _OCL_PERF_MEM_CREATE_H_
