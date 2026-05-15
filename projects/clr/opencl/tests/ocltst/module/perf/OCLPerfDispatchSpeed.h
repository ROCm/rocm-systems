/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_DispatchSpeed_H_
#define _OCL_DispatchSpeed_H_

#include "OCLTestImp.h"

class OCLPerfDispatchSpeed : public OCLTestImp {
 public:
  OCLPerfDispatchSpeed();
  virtual ~OCLPerfDispatchSpeed();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void genShader(void);

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_program program_;
  cl_kernel kernel_;
  cl_mem outBuffer_;
  cl_int error_;
  bool doWarmup;

  unsigned int bufSize_;
  bool sleep;
  unsigned int testListSize;
};

class OCLPerfMapDispatchSpeed : public OCLPerfDispatchSpeed {
 public:
  OCLPerfMapDispatchSpeed();
  virtual void run(void);
};
#endif  // _OCL_DispatchSpeed_H_
