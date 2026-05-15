/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCLPERF_DEVICE_ENQUEUE_SIER_H_
#define _OCLPERF_DEVICE_ENQUEUE_SIER_H_

#include "OCLTestImp.h"

class OCLPerfDeviceEnqueueSier : public OCLTestImp {
 public:
  OCLPerfDeviceEnqueueSier();
  virtual ~OCLPerfDeviceEnqueueSier();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  cl_command_queue deviceQueue_;
  unsigned int testID_;
  unsigned int testListSize;
  // unsigned int        threads;
  cl_uint queueSize;
  unsigned int image_size;

  bool failed_;
  bool skip_;
};

#endif  // _OCLPERF_DEVICE_ENQUEUE_SIER_H_
