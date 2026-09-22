/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PARTIAL_WRKGRP_H_
#define _OCL_PARTIAL_WRKGRP_H_

#include "OCLTestImp.h"

class OCLPartialWrkgrp : public OCLTestImp {
 public:
  OCLPartialWrkgrp();
  virtual ~OCLPartialWrkgrp();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool isOCL2_;
};

#endif  // _OCL_PARTIAL_WRKGRP_H_
