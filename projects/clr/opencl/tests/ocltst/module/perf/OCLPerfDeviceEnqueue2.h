/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCLPERF_DEVICE_ENQUEUE2_H_
#define _OCLPERF_DEVICE_ENQUEUE2_H_

#include "OCLTestImp.h"

class OCLPerfDeviceEnqueue2 : public OCLTestImp {
 public:
  OCLPerfDeviceEnqueue2();
  virtual ~OCLPerfDeviceEnqueue2();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  cl_command_queue deviceQueue_;
  unsigned int testID_;
  cl_kernel kernel2_;
  unsigned int testListSize;
  unsigned int threads;
  cl_uint queueSize;
  unsigned int subTests_level;
  unsigned int subTests_qsize;
  unsigned int subTests_thread;
  unsigned int level;
  unsigned int lws_value;

  bool failed_;
  bool skip_;
};

#endif  // _OCLPERF_DEVICE_ENQUEUE2_H_
