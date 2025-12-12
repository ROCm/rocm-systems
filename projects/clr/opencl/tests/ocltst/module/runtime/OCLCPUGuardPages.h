/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_CPU_GUARD_PAGES_H_
#define _OCL_CPU_GUARD_PAGES_H_

#include "OCLTestImp.h"

typedef struct {
  bool useGuardPages;
  bool shouldFail;
  int items;
  int in_offset;
  int out_offset;
} testOCLCPUGuardPagesStruct;

class OCLCPUGuardPages : public OCLTestImp {
 public:
  OCLCPUGuardPages();
  virtual ~OCLCPUGuardPages();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  testOCLCPUGuardPagesStruct testValues;
};

#endif  // _OCL_CPU_GUARD_PAGES_H_
