/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_3DImageWriteSpeed_H_
#define _OCL_3DImageWriteSpeed_H_

#include "OCLTestImp.h"

class OCLPerf3DImageWriteSpeed : public OCLTestImp {
 public:
  OCLPerf3DImageWriteSpeed();
  virtual ~OCLPerf3DImageWriteSpeed();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  cl_command_queue cmd_queue_;
  cl_mem imageBuffer_;

  unsigned int bufSize_;
  unsigned int bufnum_;
  char* memptr;
  unsigned int memSize_;
  unsigned int testId_;

  bool skip_;
};

#endif  // _OCL_3DImageWriteSpeed_H_
