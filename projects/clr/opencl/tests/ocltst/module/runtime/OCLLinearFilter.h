/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_LINEAR_FILTER_H_
#define _OCL_LINEAR_FILTER_H_

#include "OCLTestImp.h"

class OCLLinearFilter : public OCLTestImp {
 public:
  OCLLinearFilter();
  virtual ~OCLLinearFilter();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool done_;
};

#endif  // _OCL_LINEAR_FILTER_H_
