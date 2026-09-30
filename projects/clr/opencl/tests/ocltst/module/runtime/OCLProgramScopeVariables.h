/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_ProgramScopeVariables_H_
#define _OCL_ProgramScopeVariables_H_

#include "OCLTestImp.h"

class OCLProgramScopeVariables : public OCLTestImp {
 public:
  OCLProgramScopeVariables();
  virtual ~OCLProgramScopeVariables();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  void test0(void);
  void test1(void);
  void test2(void);
  bool silentFailure;
  cl_kernel kernel1_;
  cl_kernel kernel2_;
};

#endif  // _OCL_ProgramScopeVariables_H_
