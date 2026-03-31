/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_ImageReadWrite
#define _OCL_ImageReadWrite

#include "OCLTestImp.h"

class OCLPerfImageReadWrite : public OCLTestImp {
 public:
  OCLPerfImageReadWrite();
  virtual ~OCLPerfImageReadWrite();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
  virtual void setData(void* ptr, unsigned int size, unsigned int value);

  cl_command_queue cmd_queue_;
  cl_mem imageBuffer_;

  unsigned int bufSize_;
  unsigned int bufnum_;
  unsigned int numIter;
  char* memptr;
  unsigned int memSize;
  unsigned int testId_;

  bool skip_;
};

#endif  // _OCL_ImageReadWrite
