/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PipeCopySpeed_H_
#define _OCL_PipeCopySpeed_H_

#include "OCLTestImp.h"

class OCLPerfPipeCopySpeed : public OCLTestImp {
 public:
  OCLPerfPipeCopySpeed();
  virtual ~OCLPerfPipeCopySpeed();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  static const unsigned int NUM_ITER = 100;
  void setData(cl_mem buffer);
  void checkData(cl_mem buffer);

  cl_command_queue cmd_queue_;
  cl_mem srcBuffer_;
  cl_mem pipe_[2];
  cl_mem dstBuffer_;
  cl_program program_;
  cl_kernel initPipe_;
  cl_kernel copyPipe_;
  cl_kernel readPipe_;

  unsigned int bufSize_;
  unsigned int typeIdx_;
  unsigned int numElements;
  unsigned int numIter;
  unsigned int testIdx_;
  std::string testName_;
  bool subgroupSupport_;
  bool failed_;
};

#endif  // _OCL_PipeCopySpeed_H_
