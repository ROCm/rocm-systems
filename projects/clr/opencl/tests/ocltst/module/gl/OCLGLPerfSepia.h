/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_PERF_SEPIA_H_
#define _OCL_PERF_SEPIA_H_

#include "OCLGLCommon.h"
#include "Timer.h"

class OCLGLPerfSepia : public OCLGLCommon {
 public:
  OCLGLPerfSepia();
  virtual ~OCLGLPerfSepia();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  void runGL(void);
  void runCL(void);
  void populateData(void);
  void verifyResult(void);
  void GetKernelExecDimsForImage(unsigned int work_group_size, unsigned int w, unsigned int h,
                                 size_t* global, size_t* local);

  bool silentFailure_;
  cl_uint iterations_;
  cl_image_format format_;
  cl_uchar* data_;
  cl_uchar* result_;
  bool bVerify_;
  cl_uint width_;
  cl_uint height_;
  cl_uint bpr_;
  GLuint texId;
  CPerfCounter timer_;
};

#endif  // _OCL_PERF_SEPIA_H_
