/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_BufferCopySpeed_H_
#define _OCL_BufferCopySpeed_H_

#include "OCLTestImp.h"

class OCLPerfBufferCopySpeed : public OCLTestImp {
 public:
  OCLPerfBufferCopySpeed();
  virtual ~OCLPerfBufferCopySpeed();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  static const unsigned int NUM_ITER = 1000;

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_mem srcBuffer_;
  cl_mem dstBuffer_;
  cl_int error_;

  unsigned int bufSize_;
  bool persistent[2];
  bool allocHostPtr[2];
  bool useHostPtr[2];
  unsigned int numIter;
  bool isAMD;
  char platformVersion[32];
  void setData(void* ptr, unsigned int size, unsigned int value);
  void checkData(void* ptr, unsigned int size, unsigned int value);
  void* memptr[2];
  void* alignedmemptr[2];
};

class OCLPerfBufferCopyRectSpeed : public OCLPerfBufferCopySpeed {
 public:
  OCLPerfBufferCopyRectSpeed() : OCLPerfBufferCopySpeed() {}

 public:
  virtual void run(void);
};
#endif  // _OCL_BufferCopySpeed_H_
