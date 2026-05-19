/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_BufferCopyOverhead_H_
#define _OCL_BufferCopyOverhead_H_

#include "OCLTestImp.h"

class OCLPerfBufferCopyOverhead : public OCLTestImp {
 public:
  OCLPerfBufferCopyOverhead();
  virtual ~OCLPerfBufferCopyOverhead();

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
  bool sleep;
  bool srcHost;
};

#endif  // _OCL_BufferCopyOverhead_H_
