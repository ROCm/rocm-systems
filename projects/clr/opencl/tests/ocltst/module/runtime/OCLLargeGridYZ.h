/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_LARGE_GRID_YZ_H_
#define _OCL_LARGE_GRID_YZ_H_

#include "OCLTestImp.h"

class OCLLargeGridYZ : public OCLTestImp {
 public:
  OCLLargeGridYZ();
  virtual ~OCLLargeGridYZ();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  bool enqueue(cl_kernel kernel, cl_uint dims, const size_t* gws, const size_t* lws,
               cl_int expected);
  bool verify(size_t count);

  bool isGfx12_;
};

#endif  // _OCL_LARGE_GRID_YZ_H_
