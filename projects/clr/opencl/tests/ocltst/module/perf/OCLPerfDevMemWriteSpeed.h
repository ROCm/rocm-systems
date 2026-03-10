/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_DevMemWriteSpeed_H_
#define _OCL_DevMemWriteSpeed_H_

#include "OCLTestImp.h"

class OCLPerfDevMemWriteSpeed : public OCLTestImp {
 public:
  OCLPerfDevMemWriteSpeed();
  virtual ~OCLPerfDevMemWriteSpeed();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  cl_mem dstBuffer_;
  unsigned int nWorkItems;  // number of GPU work items
  unsigned int wgs;         // work group size
  unsigned int nBytes;      // output buffer size
  unsigned int nIter;       // overall number of timing loops
  cl_uint inputData;        // input data to fill the input buffer
  bool skip_;
};

#endif  // _OCL_DevMemWriteSpeed_H_
