/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_MEMORY_INFO_H_
#define _OCL_MEMORY_INFO_H_

#include "OCLTestImp.h"

class OCLMemoryInfo : public OCLTestImp {
 public:
  OCLMemoryInfo();
  virtual ~OCLMemoryInfo();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
  uint32_t test_;
};

#endif  // _OCL_MEMORY_INFO_H_
