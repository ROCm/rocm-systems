/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_ImageReadsRGBA_H_
#define _OCL_ImageReadsRGBA_H_

#include "OCLTestImp.h"

class OCLPerfImageReadsRGBA : public OCLTestImp {
 public:
  OCLPerfImageReadsRGBA();
  virtual ~OCLPerfImageReadsRGBA();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
  virtual void setData(void* ptr, unsigned int size, float value);

  cl_command_queue cmd_queue_;
  cl_mem imageBuffer_;
  cl_mem valueBuffer_;

  unsigned int bufSize_;
  unsigned int bufnum_;
  unsigned int numIter;
  char* memptr;
  unsigned int memSize;
  unsigned int testId_;

  bool skip_;
};

#endif  // _OCL_ImageReadsRGBA_H_
