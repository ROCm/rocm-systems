/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERF_SVM_MAP_H_
#define _OCL_PERF_SVM_MAP_H_

#include "OCLTestImp.h"

class OCLPerfSVMMap : public OCLTestImp {
 public:
  OCLPerfSVMMap();
  virtual ~OCLPerfSVMMap();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int testSize_;
  unsigned int testFlag_;
  bool skip_;
};

#endif  // _OCL_PERF_SVM_MAP_H_
