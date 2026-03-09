/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERF_FILL_IMAGE_H_
#define _OCL_PERF_FILL_IMAGE_H_

#include "OCLTestImp.h"

class OCLPerfFillImage : public OCLTestImp {
 public:
  OCLPerfFillImage();
  virtual ~OCLPerfFillImage();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  cl_mem buffer_;
  unsigned int bufSize_;
  unsigned int num_sizes_;
  bool failed_;
  bool skip_;
};

#endif  // _OCL_PERF_FILL_IMAGE_H_
