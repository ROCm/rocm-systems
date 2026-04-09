/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_DEVICE_QUERIES_H_
#define _OCL_DEVICE_QUERIES_H_

#include "OCLTestImp.h"

class OCLDeviceQueries : public OCLTestImp {
 public:
  OCLDeviceQueries();
  virtual ~OCLDeviceQueries();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
};

#endif  // _OCL_DEVICE_QUERIES_H_
