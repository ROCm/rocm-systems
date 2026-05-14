/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_CREATE_BUFFER_H_
#define _OCL_CREATE_BUFFER_H_

#include "OCLTestImp.h"
#define PATTERN_20_08BIT 0x20
#define PATTERN_20_64BIT 0x2020202020202020
#define PATTERN_2A_08BIT 0x2a
#define PATTERN_2A_64BIT 0x2a2a2a2a2a2a2a2a

class OCLCreateBuffer : public OCLTestImp {
 public:
  OCLCreateBuffer();
  virtual ~OCLCreateBuffer();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual void writeBuffer(size_t tmpMaxSize, void* dataBuf);
  virtual void checkResult(size_t tmpMaxSize, void* resultBuf, cl_ulong pattern);
  virtual unsigned int close(void);

 private:
  bool failed_;
  unsigned int testID_;
  cl_ulong maxSize_;
};

#endif  // _OCL_CREATE_BUFFER_H_
