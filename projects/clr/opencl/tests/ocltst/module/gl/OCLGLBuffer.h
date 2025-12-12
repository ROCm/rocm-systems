/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_GL_BUFFER_H_
#define _OCL_GL_BUFFER_H_

#include "OCLGLCommon.h"

class OCLGLBuffer : public OCLGLCommon {
 public:
  OCLGLBuffer();
  virtual ~OCLGLBuffer();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  static const unsigned int c_numOfElements = 1024;
  GLuint inGLBuffer_;
  GLuint outGLBuffer_;
};

#endif  // _OCL_GL_BUFFER_H_
