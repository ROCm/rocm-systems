/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_Mem_Objs_H_
#define _OCL_Mem_Objs_H_

#include "CL/cl.h"
#include "OCLTestImp.h"

class OCLMemObjs : public OCLTestImp {
 public:
  OCLMemObjs();
  virtual ~OCLMemObjs();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
  int test(void);

  static const char* kernel_src;

 private:
  cl_int error;
};

#endif  // _OCL_Mem_Objs_H_
