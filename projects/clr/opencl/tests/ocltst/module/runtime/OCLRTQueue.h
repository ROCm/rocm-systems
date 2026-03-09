/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_RT_QUEUE_H_
#define _OCL_RT_QUEUE_H_

#include "OCLTestImp.h"

class OCLRTQueue : public OCLTestImp {
 public:
  OCLRTQueue();
  virtual ~OCLRTQueue();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  cl_command_queue rtQueue_;
  cl_command_queue rtQueue1_;
  cl_kernel kernel2_;
  unsigned int testID_;
  bool failed_;
  cl_uint cu_;
  cl_uint maxCUs_;
  cl_uint rtCUs_;
  cl_uint rtCUsGranularity_;
};

#endif  // _OCL_RT_QUEUE_H_
