/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_UNALIGNED_COPY_H_
#define _OCL_UNALIGNED_COPY_H_

#include "OCLTestImp.h"

class OCLUnalignedCopy : public OCLTestImp {
 public:
  OCLUnalignedCopy();
  virtual ~OCLUnalignedCopy();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool failed_;
};

#endif  // _OCL_UNALIGNED_COPY_H_
