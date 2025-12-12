/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERSISTENT_H_
#define _OCL_PERSISTENT_H_

#include "OCLTestImp.h"

class OCLPersistent : public OCLTestImp {
 public:
  OCLPersistent();
  virtual ~OCLPersistent();
  static const unsigned int c_dimSize = 510;
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  ////////////////////
  // test functions //
  ////////////////////

  bool validateImage(unsigned int* image, size_t pitch, unsigned int dimSize);
  /////////////////////
  // private members //
  /////////////////////

  // CL identifiers
  cl_mem clImage_;
};

#endif  // _OCL_GL_BUFFER_H_
