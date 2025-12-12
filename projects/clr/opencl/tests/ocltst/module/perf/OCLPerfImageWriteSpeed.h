/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_ImageWriteSpeed_H_
#define _OCL_ImageWriteSpeed_H_

#include "OCLTestImp.h"

class OCLPerfImageWriteSpeed : public OCLTestImp {
 public:
  OCLPerfImageWriteSpeed();
  virtual ~OCLPerfImageWriteSpeed();

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
  unsigned int bufnum_;
  unsigned int numIter;
  char* memptr;
  bool skip_;
};

class OCLPerfPinnedImageWriteSpeed : public OCLPerfImageWriteSpeed {
 public:
  OCLPerfPinnedImageWriteSpeed();
  virtual ~OCLPerfPinnedImageWriteSpeed();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual unsigned int close(void);

  cl_mem inBuffer_;
};

#endif  // _OCL_ImageWriteSpeed_H_
