/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCLPERF_DEVICE_ENQUEUE_H_
#define _OCLPERF_DEVICE_ENQUEUE_H_

#include "OCLTestImp.h"

class OCLPerfDeviceEnqueue : public OCLTestImp {
 public:
  OCLPerfDeviceEnqueue();
  virtual ~OCLPerfDeviceEnqueue();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  cl_command_queue deviceQueue_;
  bool failed_;
  unsigned int testID_;
  cl_kernel kernel2_;
  unsigned int testListSize;
  unsigned int threads;
  cl_uint queueSize;
};

#endif  // _OCLPERF_DEVICE_ENQUEUE_H_
