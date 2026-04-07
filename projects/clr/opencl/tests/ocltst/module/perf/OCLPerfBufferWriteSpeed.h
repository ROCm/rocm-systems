/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_BufferWriteSpeed_H_
#define _OCL_BufferWriteSpeed_H_

#include "OCLTestImp.h"

class OCLPerfBufferWriteSpeed : public OCLTestImp {
 public:
  OCLPerfBufferWriteSpeed();
  virtual ~OCLPerfBufferWriteSpeed();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  static const unsigned int NUM_ITER = 1000;

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_mem outBuffer_;
  cl_int error_;

  unsigned int bufSize_;
  bool persistent;
  bool allocHostPtr;
  bool useHostPtr;
  unsigned int numIter;
  char* hostMem;
  char* alignedMem;
  size_t alignment;
  unsigned int offset;
  bool isAMD;
  char platformVersion[32];
};

class OCLPerfBufferWriteRectSpeed : public OCLPerfBufferWriteSpeed {
 public:
  OCLPerfBufferWriteRectSpeed() : OCLPerfBufferWriteSpeed() {}

 public:
  virtual void run(void);
};

#endif  // _OCL_BufferWriteSpeed_H_
