/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_DYNAMIC_BLINES_H_
#define _OCL_DYNAMIC_BLINES_H_

#include "OCLTestImp.h"

class OCLDynamicBLines : public OCLTestImp {
 public:
  OCLDynamicBLines();
  virtual ~OCLDynamicBLines();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  struct BezierLine {
    cl_float2 CP[3];
    long long vertexPos;
    int nVertices;
    int reserved;
  };

  cl_command_queue deviceQueue_;
  bool failed_;
  unsigned int testID_;
  BezierLine* bLines_;
  cl_float2* hostArray_;
  cl_kernel kernel2_;
  cl_kernel kernel3_;
};

#endif  // _OCL_DYNAMIC_BLINES__H_
