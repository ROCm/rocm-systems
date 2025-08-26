/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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
