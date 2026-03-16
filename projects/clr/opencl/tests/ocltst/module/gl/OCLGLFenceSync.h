/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_GL_FENCE_SYNC_H_
#define _OCL_GL_FENCE_SYNC_H_

#include "OCLGLCommon.h"

class OCLGLFenceSync : public OCLGLCommon {
 public:
  OCLGLFenceSync();
  virtual ~OCLGLFenceSync();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  static const unsigned int c_glContextCount = 1;
  static const unsigned int c_numOfElements = 8192;

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
  bool extensionSupported_;
};

#endif  // _OCL_GL_FENCE_SYNC_H_
