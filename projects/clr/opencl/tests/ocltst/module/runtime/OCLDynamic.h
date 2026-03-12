/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_DYNAMIC_H_
#define _OCL_DYNAMIC_H_

#include "OCLTestImp.h"

class OCLDynamic : public OCLTestImp {
 public:
  OCLDynamic();
  virtual ~OCLDynamic();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  cl_command_queue deviceQueue_;
  bool failed_;
  unsigned int testID_;
};

#endif  // _OCL_MEM_DEPENDENCY_H_
