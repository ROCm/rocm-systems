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

#ifndef _OCL_ScalarReplArrayElem_H_
#define _OCL_ScalarReplArrayElem_H_

#include "OCLTestImp.h"

class OCLPerfScalarReplArrayElem : public OCLTestImp {
 public:
  OCLPerfScalarReplArrayElem();
  virtual ~OCLPerfScalarReplArrayElem();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

  std::string shader_;
  void genShader(unsigned int idx);
  void setData(cl_mem buffer, float data);
  void checkData(cl_mem buffer);

  static const unsigned int NUM_ITER = 100;

  cl_context context_;
  cl_command_queue cmd_queue_;
  cl_program program_;
  cl_kernel kernel_;
  cl_mem outBuffer_;
  cl_int error_;

  unsigned int width_;
  unsigned int bufSize_;
  unsigned int numReads_;
  unsigned int shaderIdx_;
  unsigned int itemWidth_;
  unsigned int vecTypeIdx_;
  unsigned int vecSizeIdx_;
};

#endif  // _OCL_ScalarReplArrayElem_H_
