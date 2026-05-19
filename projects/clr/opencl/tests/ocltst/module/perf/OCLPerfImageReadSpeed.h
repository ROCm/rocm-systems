/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_ImageReadSpeed_H_
#define _OCL_ImageReadSpeed_H_

#include "OCLTestImp.h"

class OCLPerfImageReadSpeed : public OCLTestImp {
 public:
  OCLPerfImageReadSpeed();
  virtual ~OCLPerfImageReadSpeed();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  static const unsigned int NUM_ITER = 100;

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_mem outBuffer_;
  cl_int error_;
  bool skip_;
  unsigned int bufSize_;
  unsigned int bufnum_;
  unsigned int numIter;
  char* memptr;
};

class OCLPerfPinnedImageReadSpeed : public OCLPerfImageReadSpeed {
 public:
  OCLPerfPinnedImageReadSpeed();
  virtual ~OCLPerfPinnedImageReadSpeed();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual unsigned int close(void);

  cl_mem inBuffer_;
};
#endif  // _OCL_ImageReadSpeed_H_
