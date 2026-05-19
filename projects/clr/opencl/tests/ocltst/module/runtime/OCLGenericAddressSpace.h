/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_GenericAddressSpace_H_
#define _OCL_GenericAddressSpace_H_

#include "OCLTestImp.h"

class OCLGenericAddressSpace : public OCLTestImp {
 public:
  OCLGenericAddressSpace();
  virtual ~OCLGenericAddressSpace();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  void test0(void);
  void test1(void);
  void test2(void);
  void test3(void);
  void test4(void);
  void test5(void);
  void test6(void);
  bool silentFailure;
  cl_kernel kernel_;
  size_t arrSize;
};

#endif  // _OCL_GenericAddressSpace_H_
