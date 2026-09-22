/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_MemCombine_H_
#define _OCL_MemCombine_H_

#include "OCLTestImp.h"

class OCLPerfMemCombine : public OCLTestImp {
  enum { inSize_ = 4096U * 1024U };
  enum { outSize_ = 4096U * 1024U };
  enum { loopSize_ = 8192 };

 public:
  OCLPerfMemCombine();
  virtual ~OCLPerfMemCombine();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  static const unsigned int NUM_ITER = 1000;

  const char* dataType_;
  unsigned int numCombine_;
  unsigned int dataRange_;
  unsigned char input[inSize_];
  unsigned char output[outSize_];

 private:
  void createKernel(const char* type, int numCombine);
  void setData(cl_mem buffer, unsigned int bufSize, unsigned char val);
  void checkData(cl_mem buffer, unsigned int bufSize, unsigned int limit, unsigned char defVal);
};

#endif  // _OCL_MemCombine_H_
