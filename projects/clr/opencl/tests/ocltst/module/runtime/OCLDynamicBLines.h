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

#ifndef _OCL_DYNAMIC_BLINES_H_
#define _OCL_DYNAMIC_BLINES_H_

#include "OCLTestImp.h"

class OCLDynamicBLines : public OCLTestImp {
 public:
  OCLDynamicBLines();
  virtual ~OCLDynamicBLines();

 public:
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceID);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  struct BezierLine {
    cl_float2 CP[3];
    long long vertexPos;
    int nVertices;
    int reserved;
  };

  cl_command_queue deviceQueue_;
  bool failed_;
  unsigned int testID_;
  BezierLine* bLines_;
  cl_float2* hostArray_;
  cl_kernel kernel2_;
  cl_kernel kernel3_;
};

#endif  // _OCL_DYNAMIC_BLINES__H_
