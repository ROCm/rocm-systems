/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_DEVICE_ATOMIC_H_
#define _OCL_DEVICE_ATOMIC_H_

#include "OCLTestImp.h"

class OCLDeviceAtomic : public OCLTestImp {
 public:
  OCLDeviceAtomic();
  virtual ~OCLDeviceAtomic();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  cl_command_queue hostQueue_;
  bool failed_;
  cl_kernel kernel2_;
  unsigned int testID_;
};

#endif  // _OCL_DEVICE_ATOMIC_H_
