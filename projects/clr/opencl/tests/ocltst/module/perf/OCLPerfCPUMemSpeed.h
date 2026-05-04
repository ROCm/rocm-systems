/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_CPUMemSpeed_H_
#define _OCL_CPUMemSpeed_H_

#include "OCLTestImp.h"

class OCLPerfCPUMemSpeed : public OCLTestImp {
 public:
  OCLPerfCPUMemSpeed();
  virtual ~OCLPerfCPUMemSpeed();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  static const unsigned int NUM_ITER = 100;

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_mem outBuffer_;
  cl_int error_;

  unsigned int bufSize_;
  bool persistent;
  bool allocHostPtr;
  bool useHostPtr;
  unsigned int numIter;
  bool testMemset;
  char* hostMem;
  char* alignedMem;
  size_t alignment;
  unsigned int offset;
  bool isAMD;
  bool gpuSrc;
  cl_map_flags mapFlags;
};

#endif  // _OCL_CPUMemSpeed_H_
