/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_GL_MULTI_CONTEXT_H_
#define _OCL_GL_MULTI_CONTEXT_H_

#include "OCLGLCommon.h"

class OCLGLMultiContext : public OCLGLCommon {
 public:
  OCLGLMultiContext();
  virtual ~OCLGLMultiContext();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  static const unsigned int c_glContextCount = 3;
  static const unsigned int c_numOfElements = 128;

  struct GLContextDataSet {
    OCLGLHandle glContext;
    cl_context clContext;
    cl_command_queue clCmdQueue;
    cl_program clProgram;
    cl_kernel clKernel;
    cl_mem inputBuffer;
    cl_mem outputBuffer;
  };
  GLContextDataSet contextData_[c_glContextCount];

  bool failed_;
};

#endif  // _OCL_GL_MULTI_CONTEXT_H_
