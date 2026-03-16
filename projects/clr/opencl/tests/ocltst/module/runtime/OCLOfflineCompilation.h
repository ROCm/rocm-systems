/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_OFFLINE_COMPILATION_H_
#define _OCL_OFFLINE_COMPILATION_H_

#include "OCLTestImp.h"

class OCLOfflineCompilation : public OCLTestImp {
 public:
  OCLOfflineCompilation();
  virtual ~OCLOfflineCompilation();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);
};

#endif  // _OCL_OFFLINE_COMPILATION_H_
