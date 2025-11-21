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

#ifndef _OCL_SVMSAMPLERATE_H_
#define _OCL_SVMSAMPLERATE_H_

#include "OCLTestImp.h"

class OCLPerfSVMSampleRate : public OCLTestImp {
 public:
  OCLPerfSVMSampleRate();
  virtual ~OCLPerfSVMSampleRate();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void setData(void* buffer, unsigned int data);
  void checkData(void* buffer);
  void setKernel(void);

  cl_command_queue cmd_queue_;
  cl_program program_;
  cl_kernel kernel_;
  void** inBuffer_;
  void* outBuffer_;

  unsigned int width_;
  unsigned int bufSize_;
  unsigned int outBufSize_;
  static const unsigned int MAX_ITERATIONS = 25;
  unsigned int numBufs_;
  unsigned int typeIdx_;
  unsigned int svmMode_;

  bool skip_;
  bool coarseGrainBuffer_;
  bool fineGrainBuffer_;
  bool fineGrainSystem_;
  std::string testdesc;
};

#endif  // _OCL_SVMSAMPLERATE_H_
